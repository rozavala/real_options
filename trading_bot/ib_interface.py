import asyncio
import logging
import time
from datetime import datetime

from ib_insync import *

from trading_bot.utils import price_option_black_scholes, log_trade_to_ledger


async def get_option_market_data(ib: IB, contract: Contract) -> dict | None:
    """Fetches live market data for a single option contract, including implied volatility."""
    logging.info(f"Fetching market data for option: {contract.localSymbol}")
    ticker = ib.reqMktData(contract, '106', False, False)
    await asyncio.sleep(2)
    iv = ticker.modelGreeks.impliedVol if ticker.modelGreeks and not util.isNan(ticker.modelGreeks.impliedVol) else 0.25
    ib.cancelMktData(contract)
    return {'implied_volatility': iv, 'risk_free_rate': 0.04}


async def wait_for_fill(ib: IB, trade: Trade, config: dict, timeout: int = 180, reason: str = "Strategy Execution"):
    logging.info(f"Waiting for order {trade.order.orderId} to fill...")
    start_time = time.time()
    while not trade.isDone():
        await asyncio.sleep(1)
        if (time.time() - start_time) > timeout:
            logging.warning(f"Order {trade.order.orderId} not filled within {timeout}s. Canceling.")
            # send_notification(config, "Order Canceled", f"Order {trade.order.orderId} for {trade.contract.localSymbol} was canceled due to timeout.")
            ib.cancelOrder(trade.order)
            break
    if trade.orderStatus.status == OrderStatus.Filled:
        log_trade_to_ledger(trade, reason)


async def get_active_futures(ib: IB, symbol: str, exchange: str, count: int = 5) -> list[Contract]:
    logging.info(f"Fetching {count} active futures contracts for {symbol} on {exchange}...")
    try:
        cds = await ib.reqContractDetailsAsync(Future(symbol, exchange=exchange))
        active = sorted([cd.contract for cd in cds if cd.contract.lastTradeDateOrContractMonth > datetime.now().strftime('%Y%m')], key=lambda c: c.lastTradeDateOrContractMonth)
        logging.info(f"Found {len(active)} active contracts. Returning the first {count}.")
        return active[:count]
    except Exception as e:
        logging.error(f"Error fetching active futures: {e}"); return []


async def build_option_chain(ib: IB, future_contract: Contract):
    logging.info(f"Fetching option chain for future {future_contract.localSymbol}...")
    try:
        chains = await ib.reqSecDefOptParamsAsync(future_contract.symbol, future_contract.exchange, 'FUT', future_contract.conId)
        if not chains: return None
        chain = next((c for c in chains if c.exchange == future_contract.exchange), chains[0])
        return {'exchange': chain.exchange, 'tradingClass': chain.tradingClass, 'expirations': sorted(chain.expirations), 'strikes_by_expiration': {exp: sorted(chain.strikes) for exp in chain.expirations}}
    except Exception as e:
        logging.error(f"Failed to build option chain for {future_contract.localSymbol}: {e}"); return None


async def place_combo_order(ib: IB, config: dict, action: str, legs_def: list, exp_details: dict, chain: dict, underlying_price: float) -> Trade | None:
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

    # Add a slippage buffer to the theoretical price to create a workable limit price
    slippage_allowance = config.get('strategy_tuning', {}).get('slippage_usd_per_contract', 5.0)

    # For a debit (BUY), we are willing to pay a bit more. For a credit (SELL), we are willing to accept a bit less.
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