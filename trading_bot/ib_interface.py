"""Functions for interacting with the Interactive Brokers (IB) TWS or Gateway.

This module abstracts the `ib_insync` library calls for operations specific
to this trading bot, such as fetching contracts, building option chains,
placing complex combo orders, and waiting for orders to fill.
"""

import asyncio
import logging
import time
from datetime import datetime

from ib_insync import *

from logging_config import setup_logging
from trading_bot.utils import price_option_black_scholes, log_trade_to_ledger, normalize_strike

# --- Logging Setup ---
setup_logging()


async def get_option_market_data(ib: IB, contract: Contract) -> dict | None:
    """Fetches live market data for a single option contract.

    This function requests market data, specifically the model greeks, to get
    the implied volatility. It uses a short sleep to allow data to arrive.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        contract (Contract): The option contract to fetch data for.

    Returns:
        A dictionary containing the 'implied_volatility' and a hardcoded
        'risk_free_rate'. Returns None if data cannot be fetched.
    """
    logging.info(f"Fetching market data for option: {contract.localSymbol}")
    ticker = ib.reqMktData(contract, '106', False, False)  # 106 is for greeks
    await asyncio.sleep(2)
    # Use a default IV if not available from the ticker
    iv = ticker.modelGreeks.impliedVol if ticker.modelGreeks and not util.isNan(ticker.modelGreeks.impliedVol) else 0.25
    ib.cancelMktData(contract)
    return {'implied_volatility': iv, 'risk_free_rate': 0.04}


async def wait_for_fill(ib: IB, trade: Trade, config: dict, timeout: int = 180, reason: str = "Strategy Execution"):
    """Waits for a submitted trade to be filled and logs it.

    This function monitors a `Trade` object until its status is final (e.g.,
    Filled, Cancelled). If the order does not fill within the timeout period,
    it is automatically cancelled. If filled, the trade is logged to the
    trade ledger.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        trade (Trade): The trade object returned by `ib.placeOrder`.
        config (dict): The application's configuration dictionary.
        timeout (int): The maximum number of seconds to wait for the fill.
        reason (str): The reason for the trade, to be logged in the ledger.
    """
    logging.info(f"Waiting for order {trade.order.orderId} to fill...")
    start_time = time.time()
    while not trade.isDone():
        await asyncio.sleep(1)
        if (time.time() - start_time) > timeout:
            logging.warning(f"Order {trade.order.orderId} not filled within {timeout}s. Canceling.")
            ib.cancelOrder(trade.order)
            break
    if trade.orderStatus.status == OrderStatus.Filled:
        log_trade_to_ledger(trade, reason)


async def get_active_futures(ib: IB, symbol: str, exchange: str, count: int = 5) -> list[Contract]:
    """Fetches the next N active futures contracts for a given symbol.

    It requests all contract details for a futures symbol and returns a
    chronologically sorted list of contracts that have not yet expired.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        symbol (str): The underlying symbol (e.g., 'KC').
        exchange (str): The exchange where the future is traded (e.g., 'NYBOT').
        count (int): The maximum number of active contracts to return.

    Returns:
        A list of `ib_insync.Contract` objects for the active futures, sorted
        by expiration date. Returns an empty list on failure.
    """
    logging.info(f"Fetching {count} active futures contracts for {symbol} on {exchange}...")
    try:
        cds = await ib.reqContractDetailsAsync(Future(symbol, exchange=exchange))
        # Filter for contracts that expire in the future and sort them
        active = sorted([cd.contract for cd in cds if cd.contract.lastTradeDateOrContractMonth > datetime.now().strftime('%Y%m')], key=lambda c: c.lastTradeDateOrContractMonth)
        logging.info(f"Found {len(active)} active contracts. Returning the first {count}.")
        return active[:count]
    except Exception as e:
        logging.error(f"Error fetching active futures: {e}"); return []


async def build_option_chain(ib: IB, future_contract: Contract) -> dict | None:
    """Fetches the full option chain for a given futures contract.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        future_contract (Contract): The underlying futures contract.

    Returns:
        A dictionary containing the exchange, trading class, a sorted list of
        expirations, and a dictionary of strikes keyed by expiration.
        Returns None if the chain cannot be fetched.
    """
    logging.info(f"Fetching option chain for future {future_contract.localSymbol}...")
    try:
        chains = await ib.reqSecDefOptParamsAsync(future_contract.symbol, future_contract.exchange, 'FUT', future_contract.conId)
        if not chains: return None
        # Find the chain corresponding to the future's exchange
        chain = next((c for c in chains if c.exchange == future_contract.exchange), chains[0])
        return {
            'exchange': chain.exchange,
            'tradingClass': chain.tradingClass,
            'expirations': sorted(chain.expirations),
            'strikes_by_expiration': {exp: sorted([normalize_strike(s) for s in chain.strikes]) for exp in chain.expirations}
        }
    except Exception as e:
        logging.error(f"Failed to build option chain for {future_contract.localSymbol}: {e}"); return None


async def place_combo_order(ib: IB, config: dict, action: str, legs_def: list, exp_details: dict, chain: dict, underlying_price: float) -> Trade | None:
    """Prices, builds, and places a multi-leg combo (bag) order.

    This function performs several steps:
    1. Prices each leg individually using the Black-Scholes model.
    2. Calculates a net theoretical price for the combo.
    3. Adds a slippage allowance to create a workable limit price.
    4. Constructs a `Bag` contract with the specified legs.
    5. Places a `LimitOrder` and waits for it to be filled.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        config (dict): The application configuration dictionary.
        action (str): The overall action for the combo ('BUY' or 'SELL').
        legs_def (list): A list of tuples defining the combo legs. Each tuple
            is (right, leg_action, strike), e.g., ('C', 'BUY', 3.5).
        exp_details (dict): Expiration details from `get_expiration_details`.
        chain (dict): The option chain data from `build_option_chain`.
        underlying_price (float): The current price of the underlying future.

    Returns:
        The `ib_insync.Trade` object for the placed order, or None if pricing
        or order placement fails.
    """
    logging.info("--- Pricing individual legs before execution ---")

    net_theoretical_price = 0.0
    leg_prices = []

    for right, leg_action, strike in legs_def:
        leg_contract = FuturesOption(config['symbol'], exp_details['exp_date'], strike, right, chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(leg_contract)
        market_data = await get_option_market_data(ib, leg_contract)
        if market_data:
            pricing_result = price_option_black_scholes(S=underlying_price, K=strike, T=exp_details['days_to_exp'] / 365, r=market_data['risk_free_rate'], sigma=market_data['implied_volatility'], option_type=right)
            if pricing_result:
                price = pricing_result['price']
                leg_prices.append(price)
                if leg_action == 'BUY':
                    net_theoretical_price += price
                else: # SELL
                    net_theoretical_price -= price

    if len(leg_prices) != len(legs_def):
        logging.error("Failed to price all legs. Aborting combo order.")
        return None

    logging.info("--- End of pricing section ---")

    slippage_allowance = config.get('strategy_tuning', {}).get('slippage_usd_per_contract', 5.0)

    # For a debit (BUY), we are willing to pay more. For a credit (SELL), we accept less.
    if action == 'BUY': # Debit
        limit_price = round(net_theoretical_price + slippage_allowance, 2)
    else: # Credit
        limit_price = round(net_theoretical_price - slippage_allowance, 2)

    logging.info(f"Net theoretical price: {net_theoretical_price:.2f}, Slippage allowance: {slippage_allowance:.2f}, Final Limit Price: {limit_price:.2f}")

    combo = Bag(symbol=config['symbol'], exchange=chain['exchange'], currency='USD')
    for right, leg_action, strike in legs_def:
        contract = FuturesOption(config['symbol'], exp_details['exp_date'], strike, right, chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(contract)
        combo.comboLegs.append(ComboLeg(conId=contract.conId, ratio=1, action=leg_action, exchange=chain['exchange']))

    order = LimitOrder(action, config['strategy']['quantity'], limit_price)
    trade = ib.placeOrder(combo, order)
    await wait_for_fill(ib, trade, config, timeout=180)
    return trade