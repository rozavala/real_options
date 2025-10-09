"""Functions for interacting with the Interactive Brokers (IB) TWS or Gateway.

This module abstracts the `ib_insync` library calls for operations specific
to this trading bot, such as fetching contracts, building option chains,
placing complex combo orders, and managing the order lifecycle.
"""

import asyncio
import logging
from datetime import datetime

from ib_insync import *

from logging_config import setup_logging
from trading_bot.utils import price_option_black_scholes, log_trade_to_ledger, normalize_strike

# --- Logging Setup ---
setup_logging()


async def get_option_market_data(ib: IB, contract: Contract) -> dict | None:
    """Fetches live market data for a single option contract."""
    logging.info(f"Fetching market data for option: {contract.localSymbol}")
    ticker = ib.reqMktData(contract, '106', False, False)
    await asyncio.sleep(2)
    iv = ticker.modelGreeks.impliedVol if ticker.modelGreeks and not util.isNan(ticker.modelGreeks.impliedVol) else 0.25
    ib.cancelMktData(contract)
    return {'implied_volatility': iv, 'risk_free_rate': 0.04}


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
            tradingClass='OK',  # Corrected Trading Class for Coffee Options
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

    # 4. Price each qualified leg
    net_theoretical_price = 0.0
    for i, q_leg in enumerate(validated_legs):
        leg_action = legs_def[i][1] # 'BUY' or 'SELL'

        market_data = await get_option_market_data(ib, q_leg)
        if not market_data:
            logging.error(f"Failed to get market data for {q_leg.localSymbol}. Aborting."); return None

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
        logging.info(f"Theoretical price calculated for {q_leg.right} @ {q_leg.strike}: {pricing_result}")
        net_theoretical_price += price if leg_action == 'BUY' else -price

    # 5. Calculate final limit price
    slippage_allowance = config.get('strategy_tuning', {}).get('slippage_usd_per_contract', 5)
    limit_price = round(net_theoretical_price + slippage_allowance if action == 'BUY' else net_theoretical_price - slippage_allowance, 2)
    logging.info(f"Net theoretical price: {net_theoretical_price:.2f}, Slippage: {slippage_allowance:.2f}, Final Limit Price: {limit_price:.2f}")

    # 6. Build the Bag contract using qualified leg conIds
    combo = Bag(symbol=config['symbol'], exchange=chain['exchange'], currency='USD')
    for i, q_leg in enumerate(validated_legs):
        leg_action = legs_def[i][1]
        combo.comboLegs.append(ComboLeg(conId=q_leg.conId, ratio=1, action=leg_action, exchange=chain['exchange']))

    # The Bag contract itself does not need to be qualified if the legs are.
    order = LimitOrder(action, config['strategy']['quantity'], limit_price, tif="DAY")

    return (combo, order)


def place_order(ib: IB, contract: Contract, order: Order) -> Trade:
    """
    Places a pre-constructed order.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        contract (Contract): The qualified contract to be traded.
        order (Order): The order object to be placed.

    Returns:
        The `ib_insync.Trade` object for the placed order.
    """
    logging.info(f"Placing {order.action} order for {contract.localSymbol}...")
    trade = ib.placeOrder(contract, order)
    logging.info(f"Successfully placed order ID {trade.order.orderId} for {contract.localSymbol}.")
    return trade