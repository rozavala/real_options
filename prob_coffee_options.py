import asyncio
import csv
import json
import os
import random
import time
from datetime import datetime, time as dt_time, timedelta
import numpy as np
from scipy.stats import norm
import requests
import pytz
from ib_insync import *
import logging
import traceback

# --- Logging Setup ---
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_bot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ])
ib_logger = logging.getLogger('ib_insync')
ib_logger.setLevel(logging.INFO)


# --- 1. API Integration & Notifications ---

def send_notification(config: dict, title: str, message: str):
    notif_config = config.get('notifications', {})
    if not notif_config.get('enabled'): return
    user_key = notif_config.get('pushover_user_key')
    api_token = notif_config.get('pushover_api_token')
    if not user_key or not api_token:
        logging.warning("Pushover keys not configured.")
        return
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": api_token, "user": user_key, "title": f"Coffee Trader Alert: {title}",
            "message": message, "sound": "persistent", "html": 1
        }, timeout=10).raise_for_status()
        logging.info(f"Sent notification: '{title}'")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send notification: {e}")

def get_prediction_from_api(api_url: str) -> list | None:
    logging.info("Requesting new trading signals from prediction API...")
    try:
        mock_signals = [ # Replace with actual API call
            {"contract_month": "202512", "prediction_type": "DIRECTIONAL", "direction": "BULLISH", "confidence": 0.65},
            {"contract_month": "202603", "prediction_type": "DIRECTIONAL", "direction": "BEARISH", "confidence": 0.88},
            {"contract_month": "202605", "prediction_type": "VOLATILITY", "level": "LOW", "confidence": 0.92},
            {"contract_month": "202607", "prediction_type": "VOLATILITY", "level": "HIGH", "confidence": 0.85},
            {"contract_month": "202609", "prediction_type": "DIRECTIONAL", "direction": "BEARISH", "confidence": 0.79}
        ]
        logging.info(f"Multi-contract signals received: {json.dumps(mock_signals, indent=2)}")
        return mock_signals
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching prediction from API: {e}")
        return None

# --- 2. Core Trading Logic & Strategy Execution ---

def price_option_black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict | None:
    if T <= 0 or sigma <= 0: return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price, delta = 0.0, 0.0
    if option_type.upper() == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type.upper() == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    else:
        return None

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type.upper() == 'C' else -d2)) / 365
    
    results = {"price": round(price, 4), "delta": round(delta, 4), "gamma": round(gamma, 4), "vega": round(vega, 4), "theta": round(theta, 4)}
    logging.info(f"Theoretical price/greeks calculated for {option_type} @ {K}: {results}")
    return results

async def get_option_market_data(ib: IB, contract: Contract) -> dict | None:
    ticker = ib.reqMktData(contract, '106', False, False)
    await asyncio.sleep(2)
    iv = ticker.modelGreeks.impliedVol if ticker.modelGreeks and not util.isNan(ticker.modelGreeks.impliedVol) else 0.25
    ib.cancelMktData(contract)
    return {'implied_volatility': iv, 'risk_free_rate': 0.04}

async def manage_existing_positions(ib: IB, config: dict, signal: dict, future_contract: Contract):
    target_future_conId = future_contract.conId
    logging.info(f"--- Managing Positions for Future: {future_contract.localSymbol} ---")
    
    positions = await ib.reqPositionsAsync()
    positions_for_future = []
    for p in positions:
        if p.contract.symbol != config['symbol'] or p.position == 0: continue
        try:
            details = (await ib.reqContractDetailsAsync(Contract(conId=p.contract.conId)))[0]
            if hasattr(details, 'underConId') and details.underConId == target_future_conId:
                positions_for_future.append(p)
        except Exception as e:
            logging.warning(f"Could not resolve underlying for position {p.contract.localSymbol}: {e}")

    if not positions_for_future: return True

    target_strategy = get_strategy_name_from_signal(signal)
    
    trades_to_close, position_is_aligned = [], False
    for pos in positions_for_future:
        pos_strategy = get_strategy_name_from_contract(pos.contract)
        logging.info(f"Found open position: {pos.position} of {pos_strategy} for {future_contract.localSymbol}")
        
        if pos_strategy != target_strategy:
            logging.info(f"Position MISALIGNED. Target: {target_strategy}. Closing.")
            trades_to_close.append(pos)
        else:
            position_is_aligned = True

    for pos in trades_to_close:
        try:
            contract_to_close = Contract(conId=pos.contract.conId)
            await ib.qualifyContractsAsync(contract_to_close)
            order = MarketOrder('BUY' if pos.position < 0 else 'SELL', abs(pos.position))
            trade = ib.placeOrder(contract_to_close, order)
            await wait_for_fill(ib, trade, config, future_contract, None, reason="Position Misaligned")
        except Exception as e:
            logging.error(f"Failed to close position for conId {pos.contract.conId}: {e}")
    
    if position_is_aligned: logging.info(f"An aligned position for {future_contract.localSymbol} already exists."); return False
    return True

async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    logging.info(f"--- Executing {signal['direction']} Spread for {future_contract.localSymbol} ---")
    spread_width = config['strategy_tuning'].get('vertical_spread_strikes_apart', 2)
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None
    
    atm_idx = min(range(len(exp_details['strikes'])), key=lambda i: abs(exp_details['strikes'][i] - underlying_price))
    
    if signal['direction'] == 'BULLISH':
        if atm_idx + spread_width >= len(exp_details['strikes']): return None
        legs_def = [('C', 'BUY', exp_details['strikes'][atm_idx]), ('C', 'SELL', exp_details['strikes'][atm_idx + spread_width])]
    else: # BEARISH
        if atm_idx - spread_width < 0: return None
        legs_def = [('P', 'BUY', exp_details['strikes'][atm_idx]), ('P', 'SELL', exp_details['strikes'][atm_idx - spread_width])]

    return await place_combo_order(ib, config, 'BUY', legs_def, exp_details, chain, underlying_price, signal, future_contract)

async def execute_volatility_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    logging.info(f"--- Executing {signal['level']} Volatility Strategy for {future_contract.localSymbol} ---")
    tuning = config['strategy_tuning']
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None
    
    atm_idx = min(range(len(exp_details['strikes'])), key=lambda i: abs(exp_details['strikes'][i] - underlying_price))
    
    legs_def, order_action = [], ''
    if signal['level'] == 'HIGH':
        legs_def, order_action = [('C', 'BUY', exp_details['strikes'][atm_idx]), ('P', 'BUY', exp_details['strikes'][atm_idx])], 'BUY'
    elif signal['level'] == 'LOW':
        short_dist, wing_width = tuning.get('iron_condor_short_strikes_from_atm', 2), tuning.get('iron_condor_wing_strikes_apart', 2)
        if not (atm_idx - short_dist - wing_width >= 0 and atm_idx + short_dist + wing_width < len(exp_details['strikes'])): return None
        s = { 'lp': exp_details['strikes'][atm_idx - short_dist - wing_width], 'sp': exp_details['strikes'][atm_idx - short_dist], 
              'sc': exp_details['strikes'][atm_idx + short_dist], 'lc': exp_details['strikes'][atm_idx + short_dist + wing_width] }
        legs_def, order_action = [('P', 'BUY', s['lp']), ('P', 'SELL', s['sp']), ('C', 'SELL', s['sc']), ('C', 'BUY', s['lc'])], 'SELL'
    
    return await place_combo_order(ib, config, order_action, legs_def, exp_details, chain, underlying_price, signal, future_contract)

async def place_combo_order(ib: IB, config: dict, action: str, legs_def: list, exp_details: dict, chain: dict, underlying_price: float, signal: dict, future: Contract) -> Trade | None:
    net_theoretical_price = 0.0
    for right, leg_action, strike in legs_def:
        market_data = await get_option_market_data(ib, FuturesOption(config['symbol'], exp_details['exp_date'], strike, right, chain['exchange'], tradingClass=chain['tradingClass']))
        if market_data:
            pricing_result = price_option_black_scholes(S=underlying_price, K=strike, T=exp_details['days_to_exp'] / 365, r=market_data['risk_free_rate'], sigma=market_data['implied_volatility'], option_type=right)
            if pricing_result:
                net_theoretical_price += pricing_result['price'] if leg_action == 'BUY' else -pricing_result['price']
    
    slippage = config['strategy_tuning'].get('slippage_usd_per_contract', 5.0)
    limit_price = round(net_theoretical_price + (slippage if action == 'BUY' else -slippage), 2)
    logging.info(f"Net theoretical: {net_theoretical_price:.2f}, Limit Price: {limit_price:.2f}")

    combo = Bag(symbol=config['symbol'], exchange=chain['exchange'], currency='USD')
    for right, leg_action, strike in legs_def:
        contract = FuturesOption(config['symbol'], exp_details['exp_date'], strike, right, chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(contract)
        combo.comboLegs.append(ComboLeg(conId=contract.conId, ratio=1, action=leg_action, exchange=chain['exchange']))
    
    order = LimitOrder(action, config['strategy']['quantity'], limit_price)
    trade = ib.placeOrder(combo, order)
    await wait_for_fill(ib, trade, config, future, signal, timeout=180)
    return trade

# --- 3. Helper & Utility Functions ---

def log_trade_to_ledger(trade: Trade, future: Contract | None, signal: dict | None, reason: str):
    if trade.orderStatus.status != OrderStatus.Filled: return
    
    contract = trade.contract
    strategy = get_strategy_name_from_contract(contract) if not signal else get_strategy_name_from_signal(signal)
    strikes = '/'.join(map(str, sorted([leg.strike for leg in contract.comboLegs]))) if isinstance(contract, Bag) else str(contract.strike)

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'underlying_symbol': future.localSymbol if future else '',
        'strategy_type': strategy, 'strikes': strikes, 'action': trade.order.action,
        'quantity': trade.orderStatus.filled, 'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * 100,
        'confidence': signal.get('confidence', '') if signal else '', 'reason': reason,
        'order_id': trade.order.orderId
    }
    
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv')
    try:
        with open(ledger_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not os.path.isfile(ledger_path) or os.path.getsize(ledger_path) == 0: writer.writeheader()
            writer.writerow(row)
        logging.info(f"Logged trade to ledger ({reason})")
    except Exception as e:
        logging.error(f"Error writing to trade ledger: {e}")

async def wait_for_fill(ib: IB, trade: Trade, config: dict, future: Contract | None, signal: dict | None, timeout: int = 180, reason: str = "Strategy Execution"):
    logging.info(f"Waiting for order {trade.order.orderId} to fill...")
    start_time = time.time()
    while not trade.isDone():
        await asyncio.sleep(1)
        if (time.time() - start_time) > timeout:
            logging.warning(f"Order {trade.order.orderId} not filled. Canceling.")
            send_notification(config, "Order Canceled", f"Order {trade.contract.localSymbol} timed out.")
            ib.cancelOrder(trade.order)
            break
    if trade.orderStatus.status == OrderStatus.Filled:
        log_trade_to_ledger(trade, future, signal, reason)

def get_strategy_name_from_signal(signal: dict) -> str:
    if signal['prediction_type'] == 'DIRECTIONAL':
        return 'BULL_CALL_SPREAD' if signal['direction'] == 'BULLISH' else 'BEAR_PUT_SPREAD'
    elif signal['prediction_type'] == 'VOLATILITY':
        return 'LONG_STRADDLE' if signal['level'] == 'HIGH' else 'IRON_CONDOR'
    return "UNKNOWN"

def get_strategy_name_from_contract(contract: Contract) -> str:
    if not isinstance(contract, Bag): return "SINGLE_LEG"
    actions = ''.join(sorted([leg.action for leg in contract.comboLegs]))
    rights = ''.join(sorted([leg.right for leg in contract.comboLegs]))
    if len(contract.comboLegs) == 2:
        if rights == 'CC' and actions == 'BS': return 'BULL_CALL_SPREAD'
        if rights == 'PP' and actions == 'BS': return 'BEAR_PUT_SPREAD'
        if rights == 'CP' and actions == 'BB': return 'LONG_STRADDLE'
    if len(contract.comboLegs) == 4 and rights == 'CCPP' and actions == 'BBSS':
        return 'IRON_CONDOR'
    return "CUSTOM_COMBO"

def get_expiration_details(chain: dict, future_exp: str) -> dict | None:
    valid_exp = [exp for exp in sorted(chain['expirations']) if exp <= future_exp] or \
                [exp for exp in sorted(chain['expirations']) if exp > datetime.now().strftime('%Y%m%d')]
    if not valid_exp: return None
    chosen_exp = valid_exp[-1] if future_exp[:6] in [exp[:6] for exp in valid_exp] else valid_exp[0]
    days = (datetime.strptime(chosen_exp, '%Y%m%d').date() - datetime.now().date()).days
    return {'exp_date': chosen_exp, 'days_to_exp': days, 'strikes': chain['strikes_by_expiration'][chosen_exp]}

def is_market_open(contract_details, exchange_timezone_str: str):
    if not contract_details or not contract_details.liquidHours: return False
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str: continue
        try:
            start_str, end_str = session_str.split('-')
            start_dt = datetime.strptime(start_str.split(':')[0], '%Y%m%d').date()
            start_hm = datetime.strptime(start_str.split(':')[1], '%H%M').time()
            end_hm = datetime.strptime(end_str.split(':')[1], '%H%M').time()
            if now_tz.date() == start_dt and start_hm <= now_tz.time() < end_hm:
                return True
        except (ValueError, IndexError): continue
    return False

def calculate_wait_until_market_open(contract_details, exchange_timezone_str: str) -> float:
    if not contract_details or not contract_details.liquidHours: return 3600
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    next_open_dt = None
    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str: continue
        try:
            session_start_dt = tz.localize(datetime.strptime(session_str.split('-')[0], '%Y%m%d:%H%M'))
            if session_start_dt > now_tz and (next_open_dt is None or session_start_dt < next_open_dt):
                next_open_dt = session_start_dt
        except (ValueError, IndexError): continue
    if next_open_dt:
        wait_seconds = (next_open_dt - now_tz).total_seconds() + 60
        logging.info(f"Market closed. Waiting {wait_seconds / 3600:.2f} hours.")
        return wait_seconds
    return 3600

async def get_active_futures(ib: IB, symbol: str, exchange: str, count: int = 5) -> list[Contract]:
    logging.info(f"Fetching {count} active futures contracts...")
    try:
        cds = await ib.reqContractDetailsAsync(Future(symbol, exchange=exchange))
        active = sorted([cd.contract for cd in cds if cd.contract.lastTradeDateOrContractMonth > datetime.now().strftime('%Y%m')], key=lambda c: c.lastTradeDateOrContractMonth)
        return active[:count]
    except Exception as e:
        logging.error(f"Error fetching active futures: {e}"); return []

async def build_option_chain(ib: IB, future: Contract):
    logging.info(f"Fetching option chain for {future.localSymbol}...")
    try:
        chains = await ib.reqSecDefOptParamsAsync(future.symbol, '', 'FUT', future.conId)
        chain = next(c for c in chains if c.exchange == future.exchange)
        return {'exchange': chain.exchange, 'tradingClass': chain.tradingClass, 'expirations': sorted(chain.expirations), 'strikes_by_expiration': {exp: sorted(chain.strikes) for exp in chain.expirations}}
    except Exception as e:
        logging.error(f"Failed to build option chain for {future.localSymbol}: {e}"); return None

def is_trade_sane(signal, config): 
    min_confidence = config.get('risk_management', {}).get('min_confidence_threshold', 0.0)
    if signal['confidence'] < min_confidence:
        logging.warning(f"Sanity Check FAILED: Confidence {signal['confidence']} < {min_confidence}.")
        return False
    return True

async def monitor_positions_for_risk(ib: IB, config: dict):
    risk_params = config.get('risk_management', {})
    stop_loss, take_profit, interval = risk_params.get('stop_loss_per_contract_usd'), risk_params.get('take_profit_per_contract_usd'), risk_params.get('check_interval_seconds', 300)
    if not stop_loss and not take_profit: return
    logging.info(f"Starting risk monitor. Stop: ${stop_loss}, Profit: ${take_profit}")
    closed_ids = set()
    while True:
        try:
            await asyncio.sleep(interval)
            if not ib.isConnected(): continue
            positions = [p for p in await ib.reqPositionsAsync() if p.position != 0 and p.contract.conId not in closed_ids]
            if not positions: continue
            account = (await ib.managedAccounts())[0]
            for p in positions:
                pnl = await ib.reqPnLSingleAsync(account, '', p.contract.conId)
                pnl_per_contract = pnl.unrealizedPnL / abs(p.position)
                reason = "Stop-Loss" if stop_loss and pnl_per_contract < -abs(stop_loss) else "Take-Profit" if take_profit and pnl_per_contract > abs(take_profit) else None
                if reason:
                    logging.info(f"{reason} TRIGGERED for {p.contract.localSymbol}. PnL/contract: ${pnl_per_contract:.2f}")
                    send_notification(config, reason, f"{p.contract.localSymbol} closed. PnL/contract: ${pnl_per_contract:.2f}")
                    order = MarketOrder('BUY' if p.position < 0 else 'SELL', abs(p.position))
                    trade = ib.placeOrder(p.contract, order)
                    await wait_for_fill(ib, trade, config, None, None, reason=reason)
                    closed_ids.add(p.contract.conId)
        except Exception as e:
            logging.error(f"Error in risk monitor loop: {e}")

async def send_heartbeat(ib: IB):
    while True:
        try:
            await asyncio.sleep(300)
            if ib.isConnected():
                await ib.reqCurrentTimeAsync()
        except Exception as e:
            logging.error(f"Error in heartbeat loop: {e}")

# --- 5. Main Execution Loop ---

async def main_runner():
    ib = IB()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    monitor_task, heartbeat_task = None, None
    
    while True:
        try:
            with open(config_path, 'r') as f: config = json.load(f)
            if not ib.isConnected():
                await ib.connectAsync(config['connection']['host'], config['connection']['port'], clientId=config['connection']['clientId'])
                send_notification(config, "Script Started", "Trading script connected.")
                if monitor_task is None or monitor_task.done(): monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
                if heartbeat_task is None or heartbeat_task.done(): heartbeat_task = asyncio.create_task(send_heartbeat(ib))

            active_futures = await get_active_futures(ib, config['symbol'], config['exchange'])
            if not active_futures: raise ConnectionError("No active futures found.")

            details = (await ib.reqContractDetailsAsync(active_futures[0]))[0]
            if not is_market_open(details, config['exchange_timezone']):
                await asyncio.sleep(calculate_wait_until_market_open(details, config['exchange_timezone']))
                continue

            signals = get_prediction_from_api(config.get('api_base_url'))
            if not signals: raise ValueError("No valid signals from API.")
            
            for signal in signals:
                future = next((f for f in active_futures if f.lastTradeDateOrContractMonth.startswith(signal.get("contract_month", ""))), None)
                if not future: continue

                logging.info(f"\n===== Processing Signal for {future.localSymbol} =====")
                ticker = ib.reqMktData(future, '', False, False); await asyncio.sleep(1)
                price = ticker.marketPrice()
                if util.isNan(price): continue
                
                if await manage_existing_positions(ib, config, signal, future) and is_trade_sane(signal, config):
                    chain = await build_option_chain(ib, future)
                    if not chain: continue
                    
                    await (execute_directional_strategy if signal['prediction_type'] == 'DIRECTIONAL' else execute_volatility_strategy)(ib, config, signal, chain, price, future)

            ny_tz = pytz.timezone(config['exchange_timezone'])
            now_ny = datetime.now(ny_tz)
            h, m = map(int, config.get('trade_execution_time_ny', '04:26').split(':'))
            next_trade = now_ny.replace(hour=h, minute=m, second=0) + timedelta(days=1 if now_ny.time() > dt_time(h, m) else 0)
            wait = (next_trade - now_ny).total_seconds()
            logging.info(f"\nCycle complete. Waiting {wait / 3600:.2f} hours.")
            await asyncio.sleep(wait)

        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            logging.error(f"Connection error: {e}. Reconnecting...")
            if monitor_task and not monitor_task.done(): monitor_task.cancel()
            if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)
        except Exception as e:
            logging.critical(f"Critical error: {e}\n{traceback.format_exc()}")
            if monitor_task and not monitor_task.done(): monitor_task.cancel()
            if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        util.run(main_runner())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Script stopped manually.")

