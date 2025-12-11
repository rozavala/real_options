"""Functions for interacting with the Interactive Brokers (IB) TWS or Gateway.

This module abstracts the `ib_insync` library calls for operations specific
to this trading bot, such as fetching contracts, building option chains,
placing complex combo orders, and managing the order lifecycle.
"""

import asyncio
import logging
import uuid
from datetime import datetime

from ib_insync import *

from trading_bot.logging_config import setup_logging
from trading_bot.utils import price_option_black_scholes, log_trade_to_ledger

# --- Logging Setup ---
setup_logging()


async def get_option_market_data(ib: IB, contract: Contract, underlying_future: Contract) -> dict | None:
    """
    Fetches live market data for a single option contract, including bid, ask,
    and implied volatility.
    """
    logging.info(f"Fetching market data for option: {contract.localSymbol}")
    # Generic tick list 106 provides model-based option greeks (modelOptionImpliedVol)
    # FIX: Remove invalid generic ticks 104 and 24. Use only 106 (Implied Vol).
    ticker = ib.reqMktData(contract, '106', False, False)
    try:
        await asyncio.sleep(2)  # Allow time for data to arrive

        # Extract data from the ticker
        bid = ticker.bid if not util.isNan(ticker.bid) else None
        ask = ticker.ask if not util.isNan(ticker.ask) else None

        # Priority for IV: Model Option IV > Model Greeks IV
        iv = None
        # 1. Try User-specified 'modelOptionImpliedVol' (Generic 106)
        if hasattr(ticker, 'modelOptionImpliedVol') and not util.isNan(ticker.modelOptionImpliedVol):
            iv = ticker.modelOptionImpliedVol
            logging.info(f"Using IBKR Model Option IV: {iv:.2%}")
        # 2. Try standard ib_insync 'modelGreeks.impliedVol' (Generic 106 standard mapping)
        elif ticker.modelGreeks and not util.isNan(ticker.modelGreeks.impliedVol):
            iv = ticker.modelGreeks.impliedVol
            logging.info(f"Using IBKR Model Greeks IV: {iv:.2%}")

    finally:
        ib.cancelMktData(contract) # Clean up the market data subscription

    # FIX: Strict Safety Check.
    # Do not return 0 or fallback to Historical Volatility if live data is missing.
    if bid is None or ask is None or iv is None:
        logging.error(f"Insufficient live market data for {contract.localSymbol}. Bid: {bid}, Ask: {ask}, IV: {iv}")
        return None # Abort signal

    return {
        'bid': bid,
        'ask': ask,
        'implied_volatility': iv,
        'risk_free_rate': 0.04  # Assuming a constant risk-free rate
    }

async def get_active_futures(ib: IB, symbol: str, exchange: str, count: int = 5) -> list[Contract]:
    """Fetches the next N active futures contracts for a given symbol."""
    logging.info(f"Fetching {count} active futures contracts for {symbol} on {exchange}...")
    try:
        cds = await ib.reqContractDetailsAsync(Future(symbol, exchange=exchange))
        active = sorted([cd.contract for cd in cds if cd.contract.lastTradeDateOrContractMonth > datetime.now().strftime('%Y%m')], key=lambda c: c.lastTradeDateOrContractMonth)
        logging.info(f"Found {len(active)} active contracts. Returning the first {count}.")
        return active[:count]
    except Exception as e:
        logging.error(f"Error fetching active futures: {e}"); return []


async def build_option_chain(ib: IB, future_contract: Contract) -> dict | None:
    """Fetches the full option chain for a given futures contract."""
    logging.info(f"Fetching option chain for future {future_contract.localSymbol}...")
    try:
        chains = await ib.reqSecDefOptParamsAsync(future_contract.symbol, future_contract.exchange, 'FUT', future_contract.conId)
        if not chains: return None
        chain = next((c for c in chains if c.exchange == future_contract.exchange), chains[0])
        return {
            'exchange': chain.exchange,
            'tradingClass': chain.tradingClass,
            'expirations': sorted(chain.expirations),
            'strikes_by_expiration': {exp: sorted(chain.strikes) for exp in chain.expirations}
        }
    except Exception as e:
        logging.error(f"Failed to build option chain for {future_contract.localSymbol}: {e}"); return None


async def create_combo_order_object(ib: IB, config: dict, strategy_def: dict) -> tuple[Contract, Order] | None:
    """
    Prices a combo strategy and creates qualified Contract and Order objects without placing them.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        config (dict): The application configuration dictionary.
        strategy_def (dict): A dictionary containing the strategy parameters.

    Returns:
        A tuple of (Contract, Order) or None if creation fails.
    """
    logging.info("--- Pricing individual legs and creating order object ---")
    action = strategy_def['action']
    legs_def = strategy_def['legs_def']
    exp_details = strategy_def['exp_details']
    chain = strategy_def['chain']
    underlying_price = strategy_def['underlying_price']

    # 1. Create all leg contract objects first
    leg_contracts = []
    for right, _, strike in legs_def:
        contract = FuturesOption(
            symbol=config['symbol'],
            lastTradeDateOrContractMonth=exp_details['exp_date'],
            strike=strike,
            right=right,
            exchange=chain['exchange'],
            # FIX: Use the dynamically fetched tradingClass instead of a hardcoded value.
            # The tradingClass for Coffee (KC) options is 'OKC', not 'OK'.
            tradingClass=chain['tradingClass'],
            multiplier="37500"
        )
        leg_contracts.append(contract)

    # 2. Qualify all leg contracts in a single batch
    try:
        qualified_legs = await ib.qualifyContractsAsync(*leg_contracts)
    except Exception as e:
        logging.error(f"Exception during contract qualification: {e}")
        for lc in leg_contracts:
            logging.error(f"Contract that failed qualification: {lc}")
        return None

    # 3. Validate all legs were qualified successfully before pricing
    validated_legs = []
    for leg in qualified_legs:
        if leg.conId == 0:
            logging.error(f"Failed to qualify contract, conId is 0. Contract details: {leg}")
            return None
        validated_legs.append(leg)

    if len(validated_legs) != len(leg_contracts):
        logging.error("Mismatch between number of requested and qualified legs. Aborting.")
        return None

    # 4. Price each leg theoretically and get live market spread
    net_theoretical_price = 0.0
    combo_bid_price = 0.0
    combo_ask_price = 0.0
    for i, q_leg in enumerate(validated_legs):
        leg_action = legs_def[i][1]  # 'BUY' or 'SELL'

        market_data = await get_option_market_data(ib, q_leg, strategy_def['future_contract'])
        if not market_data:
            logging.error(f"Failed to get market data for {q_leg.localSymbol}. Aborting order.")
            return None

        # Calculate theoretical price using Black-Scholes
        pricing_result = price_option_black_scholes(
            S=underlying_price,
            K=q_leg.strike,
            T=exp_details['days_to_exp'] / 365,
            r=market_data['risk_free_rate'],
            sigma=market_data['implied_volatility'],
            option_type=q_leg.right
        )
        if not pricing_result:
            logging.error(f"Failed to price leg {leg_action} {q_leg.localSymbol}. Aborting."); return None

        price = pricing_result['price']
        net_theoretical_price += price if leg_action == 'BUY' else -price
        logging.info(f"  -> Leg Theoretical Price ({leg_action}): {q_leg.localSymbol} @ {price:.2f}")

        # Aggregate combo bid/ask from live market data
        leg_bid = market_data['bid']
        leg_ask = market_data['ask']
        if leg_action == 'BUY':
            combo_bid_price += leg_bid
            combo_ask_price += leg_ask
        else:  # 'SELL'
            combo_bid_price -= leg_ask
            combo_ask_price -= leg_bid

    # 5. Add Liquidity Filter and Calculate Limit Price (Ceiling/Floor) and Initial Price (Start)
    tuning_params = config.get('strategy_tuning', {})
    max_spread_pct = tuning_params.get('max_liquidity_spread_percentage', 0.25)
    fixed_slippage = tuning_params.get('fixed_slippage_cents', 0.5)

    market_spread = combo_ask_price - combo_bid_price

    # Liquidity Filter: Check if the market spread is too wide relative to the theoretical price
    if net_theoretical_price > 0 and (market_spread / net_theoretical_price) > max_spread_pct:
        logging.warning(
            f"LIQUIDITY FILTER FAILED: Market spread ({market_spread:.2f}) is "
            f"{(market_spread / net_theoretical_price):.1%} of theoretical price ({net_theoretical_price:.2f}), "
            f"which exceeds the max of {max_spread_pct:.1%}. Aborting order."
        )
        return None

    # Calculate Ceiling/Floor Price (Theoretical Max/Min)
    start_offset = 0.05
    if action == 'BUY':
        ceiling_price = round(net_theoretical_price + fixed_slippage, 2)
        # Start at Bid + 1 tick (assuming tick is start_offset, or just small increment)
        # If Bid is 0/invalid, we can't start properly, but we checked market_data earlier.
        # Actually combo_bid_price is the synthetic bid.
        initial_price = round(combo_bid_price + start_offset, 2)
        # Ensure initial price does not exceed ceiling
        initial_price = min(initial_price, ceiling_price)
    else:  # SELL
        floor_price = round(net_theoretical_price - fixed_slippage, 2)
        # Start at Ask - 1 tick
        initial_price = round(combo_ask_price - start_offset, 2)
        # Ensure initial price is not below floor
        initial_price = max(initial_price, floor_price)

    logging.info(f"Net Theoretical: {net_theoretical_price:.2f}, Market Spread: {market_spread:.2f}")
    logging.info(f"Adaptive Strategy: Start @ {initial_price:.2f}, Cap/Floor @ {ceiling_price if action == 'BUY' else floor_price:.2f}")

    # 6. Build the Bag contract using qualified leg conIds
    combo = Bag(symbol=config['symbol'], exchange=chain['exchange'], currency='USD')
    for i, q_leg in enumerate(validated_legs):
        leg_action = legs_def[i][1]
        combo.comboLegs.append(ComboLeg(conId=q_leg.conId, ratio=1, action=leg_action, exchange=chain['exchange']))

    # The Bag contract itself does not need to be qualified if the legs are.

    order_type = config.get('strategy_tuning', {}).get('order_type', 'LMT').upper()

    if order_type == 'MKT':
        order = MarketOrder(action, config['strategy']['quantity'], tif="DAY")
        logging.info(f"Creating Market Order for {action} {config['strategy']['quantity']}.")
    else: # Default to Limit Order
        # We set the initial price as the limit price
        order = LimitOrder(action, config['strategy']['quantity'], initial_price, tif="DAY")
        # Store the ceiling/floor in the order object for the manager to use
        if action == 'BUY':
            order.adaptive_limit_price = ceiling_price
        else:
            order.adaptive_limit_price = floor_price
        logging.info(f"Creating Limit Order for {action} {config['strategy']['quantity']} @ {initial_price:.2f} (Adaptive Cap: {order.adaptive_limit_price:.2f}).")

    # Switch to IBKR Adaptive Algo (Still keeping this as user didn't say to remove it, but said "IBKR Adaptive Algo doesn't apply... need to build it".
    # User said "It seems that Adaptive Algo strategy from IBKR doesn't apply to this type of trading so we'll probably need to build it in our bot."
    # So I should REMOVE the IBKR Adaptive Algo settings.)
    # order.algoStrategy = 'Adaptive'
    # order.algoParams = [TagValue('adaptivePriority', 'Normal')]
    logging.info("Using Custom Adaptive Logic (IBKR Algo Disabled).")

    # Assign a unique reference ID to the parent order.
    # IB will propagate this ID to all execution reports for the individual legs.
    order.orderRef = str(uuid.uuid4())
    logging.info(f"Assigned OrderRef: {order.orderRef}")

    return (combo, order)


def place_order(ib: IB, contract: Contract, order: Order) -> Trade:
    """
    Places a pre-constructed order, ensuring it has a unique `orderRef`.

    If the order does not already have an `orderRef`, this function assigns a
    new UUID to it. This ensures that all orders, including those created for
    risk management or position closing, have a unique identifier that can be
    tracked.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        contract (Contract): The qualified contract to be traded.
        order (Order): The order object to be placed.

    Returns:
        The `ib_insync.Trade` object for the placed order.
    """
    if not order.orderRef:
        order.orderRef = str(uuid.uuid4())
        logging.info(f"Assigned new unique OrderRef for tracking: {order.orderRef}")

    logging.info(f"Placing {order.action} order for {contract.localSymbol}...")
    trade = ib.placeOrder(contract, order)
    logging.info(f"Successfully placed order ID {trade.order.orderId} for {contract.localSymbol}.")
    return trade

