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
            'strikes_by_expiration': {exp: sorted([normalize_strike(s) for s in chain.strikes]) for exp in chain.expirations}
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

    net_theoretical_price = 0.0
    for right, leg_action, strike in legs_def:
        leg_contract = FuturesOption(config['symbol'], exp_details['exp_date'], strike, right, chain['exchange'], tradingClass=chain['tradingClass'])
        market_data = await get_option_market_data(ib, leg_contract)
        if market_data:
            pricing_result = price_option_black_scholes(S=underlying_price, K=strike, T=exp_details['days_to_exp'] / 365, r=market_data['risk_free_rate'], sigma=market_data['implied_volatility'], option_type=right)
            if pricing_result:
                price = pricing_result['price']
                net_theoretical_price += price if leg_action == 'BUY' else -price
            else:
                logging.error(f"Failed to price leg {leg_action} {strike}{right}. Aborting."); return None
        else:
            logging.error(f"Failed to get market data for {leg_action} {strike}{right}. Aborting."); return None

    slippage_allowance = config.get('strategy_tuning', {}).get('slippage_usd_per_contract', 0.01)
    limit_price = round(net_theoretical_price + slippage_allowance if action == 'BUY' else net_theoretical_price - slippage_allowance, 2)

    logging.info(f"Net theoretical price: {net_theoretical_price:.2f}, Slippage: {slippage_allowance:.2f}, Final Limit Price: {limit_price:.2f}")

    combo = Bag(symbol=config['symbol'], exchange=chain['exchange'], currency='USD')
    for right, leg_action, strike in legs_def:
        contract = FuturesOption(config['symbol'], exp_details['exp_date'], strike, right, chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(contract)
        combo.comboLegs.append(ComboLeg(conId=contract.conId, ratio=1, action=leg_action, exchange=chain['exchange']))

    await ib.qualifyContractsAsync(combo)
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