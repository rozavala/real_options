"""Functions for interacting with the Interactive Brokers (IB) TWS or Gateway.

This module abstracts the `ib_insync` library calls for operations specific
to this trading bot, such as fetching contracts, building option chains,
placing complex combo orders, and managing the order lifecycle.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta

from ib_insync import *

from trading_bot.logging_config import setup_logging
from trading_bot.utils import (
    price_option_black_scholes, log_trade_to_ledger, round_to_tick,
    get_tick_size, get_contract_multiplier
)

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

    # Price data is required
    if bid is None or ask is None:
        logging.error(f"Insufficient price data for {contract.localSymbol}. Bid: {bid}, Ask: {ask}")
        return None

    # IV is optional - proceed without it if unavailable
    if iv is None:
        logging.warning(f"IV data missing for {contract.localSymbol}, proceeding with price-only execution.")
        iv = 0.35  # Default fallback IV for coffee (~35%)

    return {
        'bid': bid,
        'ask': ask,
        'implied_volatility': iv,
        'risk_free_rate': 0.04  # Assuming a constant risk-free rate
    }

async def get_underlying_iv_metrics(ib: IB, future_contract: Contract) -> dict:
    """
    Fetches IV metrics from IBKR for the underlying.
    Returns dict with iv_rank, iv_percentile, current_iv (approximate from ATM option).
    """
    try:
        # Get ATM option for IV proxy
        chains = await ib.reqSecDefOptParamsAsync(
            future_contract.symbol,
            future_contract.exchange,
            future_contract.secType,
            future_contract.conId
        )

        if not chains:
            return {'iv_rank': 'N/A', 'iv_percentile': 'N/A', 'current_iv': 'N/A'}

        # Get ticker with model greeks (generic tick 106)
        ticker = ib.reqMktData(future_contract, '106', False, False)
        await asyncio.sleep(2)

        iv_data = {
            'iv_rank': 'N/A',
            'iv_percentile': 'N/A',
            'current_iv': 'N/A'
        }

        # IBKR provides impliedVolatility on the underlying ticker for index options
        # For futures, we approximate from near-term ATM option
        if hasattr(ticker, 'modelGreeks') and ticker.modelGreeks:
            if not util.isNan(ticker.modelGreeks.impliedVol):
                iv_data['current_iv'] = f"{ticker.modelGreeks.impliedVol:.1%}"

        ib.cancelMktData(future_contract)
        return iv_data

    except Exception as e:
        logging.warning(f"Failed to fetch IV metrics: {e}")
        return {'iv_rank': 'N/A', 'iv_percentile': 'N/A', 'current_iv': 'N/A'}

async def get_active_futures(ib: IB, symbol: str, exchange: str, count: int = 5) -> list[Contract]:
    """
    Fetches the next N active futures contracts for a given symbol.
    Excludes contracts expiring within 45 days.
    """
    logging.info(f"Fetching {count} active futures contracts for {symbol} on {exchange}...")
    try:
        cds = await ib.reqContractDetailsAsync(Future(symbol, exchange=exchange))
        now = datetime.now()
        filtered_contracts = []

        for cd in cds:
            contract = cd.contract
            # Check basic future expiration (must be in future)
            if contract.lastTradeDateOrContractMonth > now.strftime('%Y%m'):
                # Apply 45-day rule
                # Parse expiration date. Typically YYYYMMDD for specific contracts.
                try:
                    exp_str = contract.lastTradeDateOrContractMonth
                    # Handle YYYYMMDD format
                    if len(exp_str) == 8:
                        exp_date = datetime.strptime(exp_str, '%Y%m%d')
                    # Handle YYYYMM format (less precise, assume end of month?)
                    # Usually reqContractDetails returns specific contracts with full dates.
                    elif len(exp_str) == 6:
                        # Assume roughly 15th for safety or just pass if YYYYMM > current
                        # But user wants 45-day rule. Let's skip imprecise ones or parse as 1st?
                        # Better to be strict.
                        exp_date = datetime.strptime(exp_str + "01", '%Y%m%d')
                    else:
                        continue # Invalid format

                    if exp_date >= (now + timedelta(days=45)):
                        filtered_contracts.append(contract)
                    else:
                        logging.info(f"Skipping contract {contract.localSymbol} (Exp: {exp_str}): Expires < 45 days.")
                except ValueError:
                    logging.warning(f"Could not parse expiration for {contract.localSymbol}: {contract.lastTradeDateOrContractMonth}")
                    continue

        active = sorted(filtered_contracts, key=lambda c: c.lastTradeDateOrContractMonth)
        logging.info(f"Found {len(active)} active tradeable contracts (>=45d exp). Returning the first {count}.")
        return active[:count]
    except Exception as e:
        logging.error(f"Error fetching active futures: {e}"); return []


async def build_option_chain(ib: IB, future_contract: Contract) -> dict | None:
    """Fetches the full option chain for a given futures contract."""
    logging.info(f"Fetching option chain for future {future_contract.localSymbol}...")
    try:
        chains = await ib.reqSecDefOptParamsAsync(future_contract.symbol, future_contract.exchange, 'FUT', future_contract.conId)
        if not chains:
            logging.warning(f"No option chains found for future {future_contract.localSymbol} (conId: {future_contract.conId})")
            return None
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

    # Get profile-driven specs
    tick_size = get_tick_size(config)
    # IBKR expects multiplier as string (e.g. "37500")
    contract_multiplier = str(get_contract_multiplier(config))

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
            multiplier=contract_multiplier
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
    start_offset = tick_size  # Start 1 tick into the book
    market_mid = (combo_bid_price + combo_ask_price) / 2

    if action == 'BUY':
        # Theoretical ceiling with slippage - TICK ALIGNED
        theoretical_ceiling = round_to_tick(net_theoretical_price + fixed_slippage, tick_size, 'BUY')
        # Market-based ceiling: don't exceed the current ask - TICK ALIGNED
        market_ceiling = round_to_tick(combo_ask_price + (market_spread * 0.1), tick_size, 'BUY')
        # Use the more conservative (lower) ceiling
        ceiling_price = min(theoretical_ceiling, market_ceiling)
        # But ensure we can at least reach the market mid - TICK ALIGNED
        ceiling_price = max(ceiling_price, round_to_tick(market_mid, tick_size, 'BUY'))

        # Initial price - TICK ALIGNED (round down for buys)
        initial_price = round_to_tick(combo_bid_price + start_offset, tick_size, 'BUY')
        initial_price = min(initial_price, ceiling_price)

        logging.info(f"BUY Cap Calc: Theoretical={theoretical_ceiling:.2f}, Market={market_ceiling:.2f}, Mid={market_mid:.2f}, Final Cap={ceiling_price:.2f}")

    else:  # SELL
        # Theoretical floor with slippage - TICK ALIGNED (round up for sells)
        theoretical_floor = round_to_tick(net_theoretical_price - fixed_slippage, tick_size, 'SELL')
        # Market-based floor - TICK ALIGNED
        market_floor = round_to_tick(combo_bid_price - (market_spread * 0.1), tick_size, 'SELL')
        # Use the more aggressive floor
        floor_price = min(theoretical_floor, market_floor)
        floor_price = min(floor_price, round_to_tick(market_mid, tick_size, 'SELL'))

        # Initial price - TICK ALIGNED (round up for sells)
        initial_price = round_to_tick(combo_ask_price - start_offset, tick_size, 'SELL')
        initial_price = max(initial_price, floor_price)

        logging.info(f"SELL Floor Calc: Theoretical={theoretical_floor:.2f}, Market={market_floor:.2f}, Mid={market_mid:.2f}, Final Floor={floor_price:.2f}")

    logging.info(f"Net Theoretical: {net_theoretical_price:.2f}, Market Spread: {market_spread:.2f}")
    logging.info(f"Adaptive Strategy: Start @ {initial_price:.2f}, Cap/Floor @ {ceiling_price if action == 'BUY' else floor_price:.2f}")

    # 6. Build the Bag contract using qualified leg conIds
    combo = Bag(symbol=config['symbol'], exchange=chain['exchange'], currency='USD')
    for i, q_leg in enumerate(validated_legs):
        leg_action = legs_def[i][1]
        combo.comboLegs.append(ComboLeg(conId=q_leg.conId, ratio=1, action=leg_action, exchange=chain['exchange']))

    # The Bag contract itself does not need to be qualified if the legs are.

    order_type = config.get('strategy_tuning', {}).get('order_type', 'LMT').upper()

    # Determine Quantity (Use override from strategy_def if available, else config)
    quantity = strategy_def.get('quantity', config['strategy']['quantity'])

    if order_type == 'MKT':
        order = MarketOrder(action, quantity, tif="DAY")
        logging.info(f"Creating Market Order for {action} {quantity}.")
    else: # Default to Limit Order
        # We set the initial price as the limit price
        order = LimitOrder(action, quantity, initial_price, tif="DAY")
        # Store the ceiling/floor in the order object for the manager to use
        if action == 'BUY':
            order.adaptive_limit_price = ceiling_price
        else:
            order.adaptive_limit_price = floor_price
        logging.info(f"Creating Limit Order for {action} {quantity} @ {initial_price:.2f} (Adaptive Cap: {order.adaptive_limit_price:.2f}).")

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


async def place_directional_spread_with_protection(
    ib: IB,
    combo_contract: Contract,
    combo_order: Order,
    underlying_contract: Contract,
    entry_price: float,
    stop_distance_pct: float = 0.03,
    is_bullish_strategy: bool = True
) -> tuple[Trade, Trade | None]:
    """
    Places a directional spread with an exchange-native stop order on the underlying.

    The stop order acts as a "circuit breaker" if price gaps through our thesis.
    It hedges delta exposure, buying time to close the options position.

    Args:
        combo_contract: The BAG contract for the spread
        combo_order: The order for the spread
        underlying_contract: The underlying future (e.g., KCH6)
        entry_price: Current underlying price at entry
        stop_distance_pct: Distance for stop trigger (default 3%)
        is_bullish_strategy: True for Bull Spreads (Protects Downside), False for Bear (Protects Upside)

    Returns:
        Tuple of (spread_trade, stop_trade)
    """
    if not combo_order.orderRef:
        combo_order.orderRef = str(uuid.uuid4())

    # 1. Place the spread order
    logging.info(f"Placing protected spread order for {combo_contract.localSymbol}...")
    spread_trade = ib.placeOrder(combo_contract, combo_order)

    # 2. Determine stop parameters
    if is_bullish_strategy:
        # Bull spread (Long Delta): Protect against Drop
        stop_price = entry_price * (1 - stop_distance_pct)
        stop_action = 'SELL'  # Sell future to hedge
    else:
        # Bear spread (Short Delta): Protect against Rise
        stop_price = entry_price * (1 + stop_distance_pct)
        stop_action = 'BUY'  # Buy future to hedge

    # 3. Create stop order on underlying
    # NOTE: This creates a DELTA MISMATCH by design.
    stop_order = StopOrder(
        action=stop_action,
        totalQuantity=1,  # Single future hedges ~100 delta (approx for 1 spread?)
        # Ideally this should match spread delta, but spec says "Single future".
        # Spread is usually size 1. Future size 1.
        stopPrice=round(stop_price, 2),
        tif='GTC',  # Good-til-cancelled
        outsideRth=True  # Active outside regular hours
    )
    stop_order.orderRef = f"CATASTROPHE_{spread_trade.order.orderRef}"

    # 4. Place the stop order
    stop_trade = ib.placeOrder(underlying_contract, stop_order)

    logging.info(
        f"Catastrophe protection placed: {stop_action} {underlying_contract.localSymbol} "
        f"@ {stop_price:.2f} (stop) for spread order {spread_trade.order.orderId}"
    )

    return spread_trade, stop_trade


async def close_spread_with_protection_cleanup(
    ib: IB,
    spread_trade: Trade, # Pass the trade or finding the stop by ref?
    # The spec used spread_position, stop_order_ref.
    # We will use stop_order_ref approach.
    stop_order_ref: str
):
    """Cancels the associated catastrophe stop when a spread is closed."""
    if not stop_order_ref:
        return

    # Find and cancel the orphaned stop order
    open_orders = await ib.reqAllOpenOrdersAsync()
    for trade in open_orders:
        if trade.order.orderRef == stop_order_ref:
            ib.cancelOrder(trade.order)
            logging.info(f"Cancelled orphaned catastrophe stop: {stop_order_ref}")
            break
