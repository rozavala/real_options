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
# This ensures all messages, including from ib_insync, are captured.
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_bot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ])
# Capture ib_insync's internal logs
ib_logger = logging.getLogger('ib_insync')
ib_logger.setLevel(logging.INFO)


# --- 1. API Integration & Notifications ---

def send_notification(config: dict, title: str, message: str):
    """Sends a push notification via Pushover if enabled in the config."""
    notif_config = config.get('notifications', {})
    if not notif_config.get('enabled'):
        return

    user_key = notif_config.get('pushover_user_key')
    api_token = notif_config.get('pushover_api_token')

    if not user_key or not api_token or 'YOUR_KEY' in user_key or 'YOUR_TOKEN' in api_token:
        logging.warning("Notification sending skipped: Pushover keys not configured.")
        return

    try:
        response = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": api_token,
            "user": user_key,
            "title": f"Coffee Trader Alert: {title}",
            "message": message,
            "sound": "persistent",
            "html": 1
        }, timeout=10)
        response.raise_for_status()
        logging.info(f"Sent notification: '{title}'")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send notification: {e}")


def get_prediction_from_api(api_url: str) -> list | None:
    """Calls the external prediction API to get trading signals."""
    logging.info("Requesting new trading signals from prediction API...")
    try:
        # MOCK DATA - Replace with actual API call
        mock_signals = [
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
    """Calculates the theoretical price and Greeks of an option using the Black-Scholes model."""
    if T <= 0 or sigma <= 0:
        logging.warning(f"Invalid input for B-S model: T={T}, sigma={sigma}. Cannot price option.")
        return None

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = 0.0
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
    logging.info(f"Theoretical price calculated for {option_type} @ {K}: {results}")
    return results

async def get_option_market_data(ib: IB, contract: Contract) -> dict | None:
    """Fetches live market data for a single option contract, including implied volatility."""
    logging.info(f"Fetching market data for option: {contract.localSymbol}")
    ticker = ib.reqMktData(contract, '106', False, False)
    await asyncio.sleep(2)
    iv = ticker.modelGreeks.impliedVol if ticker.modelGreeks and not util.isNan(ticker.modelGreeks.impliedVol) else 0.25
    ib.cancelMktData(contract)
    return {'implied_volatility': iv, 'risk_free_rate': 0.04}

def get_position_details(position: Position) -> dict:
    """Determines the strategy type from a position object."""
    contract = position.contract
    details = {'type': 'UNKNOWN', 'key_strikes': []}
    if isinstance(contract, FuturesOption):
        details['type'] = 'SINGLE_LEG'
        details['key_strikes'].append(contract.strike)
    elif isinstance(contract, Bag):
        legs = contract.comboLegs
        actions = ''.join(sorted([leg.action for leg in legs]))
        rights = ''.join(sorted([leg.right for leg in legs]))
        strikes = sorted([leg.strike for leg in legs])
        if len(legs) == 2:
            details['key_strikes'] = strikes
            if rights == 'CC' and actions == 'BS': details['type'] = 'BULL_CALL_SPREAD'
            elif rights == 'PP' and actions == 'BS': details['type'] = 'BEAR_PUT_SPREAD'
            elif rights == 'CP' and actions == 'BB': details['type'] = 'LONG_STRADDLE'
        if len(legs) == 4 and rights == 'CCPP' and actions == 'BBSS':
            details['type'] = 'IRON_CONDOR'
            details['key_strikes'] = [strikes[1], strikes[2]]
    return details

async def manage_existing_positions(ib: IB, config: dict, signal: dict, underlying_price: float, future_contract: Contract):
    """Checks for existing positions for a given future and closes them if misaligned."""
    target_future_conId = future_contract.conId
    logging.info(f"--- Managing Positions for Future Contract: {future_contract.localSymbol} (conId: {target_future_conId}) ---")
    
    all_positions = await ib.reqPositionsAsync()
    positions_for_this_future = []

    for p in all_positions:
        if p.contract.symbol != config['symbol'] or p.position == 0: continue
        try:
            details_list = await ib.reqContractDetailsAsync(Contract(conId=p.contract.conId))
            if details_list and hasattr(details_list[0], 'underConId') and details_list[0].underConId == target_future_conId:
                positions_for_this_future.append(p)
        except Exception as e:
            logging.warning(f"Could not resolve underlying for position {p.contract.localSymbol}: {e}")

    if not positions_for_this_future:
        logging.info(f"No existing positions found for future {future_contract.localSymbol}.")
        return True

    target_strategy = ''
    if signal['prediction_type'] == 'DIRECTIONAL':
        target_strategy = 'BULL_CALL_SPREAD' if signal['direction'] == 'BULLISH' else 'BEAR_PUT_SPREAD'
    elif signal['prediction_type'] == 'VOLATILITY':
        target_strategy = 'LONG_STRADDLE' if signal['level'] == 'HIGH' else 'IRON_CONDOR'
    
    trades_to_close, position_is_aligned = [], False
    for pos in positions_for_this_future:
        pos_details = get_position_details(pos)
        current_strategy = pos_details['type']
        logging.info(f"Found open position for {future_contract.localSymbol}: {pos.position} of {current_strategy}")
        
        if current_strategy != target_strategy:
            logging.info(f"Position MISALIGNED. Current: {current_strategy}, Target: {target_strategy}. Closing.")
            trades_to_close.append(pos)
        else:
            logging.info("Position is ALIGNED. Holding.")
            position_is_aligned = True

    if trades_to_close:
        for pos_to_close in trades_to_close:
            try:
                contract_to_close = Contract(conId=pos_to_close.contract.conId)
                await ib.qualifyContractsAsync(contract_to_close)
                if contract_to_close.strike > 100: contract_to_close.strike /= 100.0
                order = MarketOrder('BUY' if pos_to_close.position < 0 else 'SELL', abs(pos_to_close.position))
                trade = ib.placeOrder(contract_to_close, order)
                await wait_for_fill(ib, trade, config, reason="Position Misaligned")
            except Exception as e:
                logging.error(f"Failed to close position for conId {pos_to_close.contract.conId}: {e}\n{traceback.format_exc()}")
        return True
    
    if position_is_aligned:
        logging.info(f"An aligned position for {future_contract.localSymbol} already exists.")
        return False
    return True

async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    logging.info(f"--- Executing {signal['direction']} Spread for {future_contract.localSymbol} ---")
    strategy_params, tuning = config['strategy'], config.get('strategy_tuning', {})
    spread_width = tuning.get('vertical_spread_strikes_apart', 2)
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None
    strikes = exp_details['strikes']
    try:
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
    except (ValueError, IndexError):
        logging.error("Could not find ATM strike in the chain."); return None

    if signal['direction'] == 'BULLISH':
        if atm_idx + spread_width >= len(strikes): logging.warning("Not enough strikes for Bull Call Spread."); return None
        legs_def = [('C', 'BUY', strikes[atm_idx]), ('C', 'SELL', strikes[atm_idx + spread_width])]
    else: # BEARISH
        if atm_idx - spread_width < 0: logging.warning("Not enough strikes for Bear Put Spread."); return None
        legs_def = [('P', 'BUY', strikes[atm_idx]), ('P', 'SELL', strikes[atm_idx - spread_width])]

    return await place_combo_order(ib, config, 'BUY', legs_def, exp_details, chain, underlying_price)

async def execute_volatility_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    logging.info(f"--- Executing {signal['level']} Volatility Strategy for {future_contract.localSymbol} ---")
    tuning = config.get('strategy_tuning', {})
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None
    strikes = exp_details['strikes']
    try:
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
    except (ValueError, IndexError):
        logging.error("Could not find ATM strike in the chain."); return None
    
    legs_def, order_action = [], ''
    if signal['level'] == 'HIGH': # Long Straddle
        legs_def, order_action = [('C', 'BUY', strikes[atm_idx]), ('P', 'BUY', strikes[atm_idx])], 'BUY'
    elif signal['level'] == 'LOW': # Iron Condor
        short_dist, wing_width = tuning.get('iron_condor_short_strikes_from_atm', 2), tuning.get('iron_condor_wing_strikes_apart', 2)
        if not (atm_idx - short_dist - wing_width >= 0 and atm_idx + short_dist + wing_width < len(strikes)):
            logging.warning("Not enough strikes for Iron Condor."); return None
        s = { 'lp': strikes[atm_idx - short_dist - wing_width], 'sp': strikes[atm_idx - short_dist], 
              'sc': strikes[atm_idx + short_dist], 'lc': strikes[atm_idx + short_dist + wing_width] }
        legs_def, order_action = [('P', 'BUY', s['lp']), ('P', 'SELL', s['sp']), ('C', 'SELL', s['sc']), ('C', 'BUY', s['lc'])], 'SELL'
    
    return await place_combo_order(ib, config, order_action, legs_def, exp_details, chain, underlying_price)

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

# --- 3. Helper & Utility Functions ---

def log_trade_to_ledger(trade: Trade, reason: str = "Strategy Execution"):
    if trade.orderStatus.status != OrderStatus.Filled: return
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'local_symbol': trade.contract.localSymbol,
        'action': trade.order.action, 'quantity': trade.orderStatus.filled, 'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * (100 if isinstance(trade.contract, Bag) else 37500),
        'order_id': trade.order.orderId, 'reason': reason
    }
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv'), 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not os.path.isfile(f.name): writer.writeheader()
            writer.writerow(row)
        logging.info(f"Logged trade to ledger ({reason}): {row['action']} {row['quantity']} @ {row['avg_fill_price']}")
    except Exception as e:
        logging.error(f"Error writing to trade ledger: {e}")

async def wait_for_fill(ib: IB, trade: Trade, config: dict, timeout: int = 180, reason: str = "Strategy Execution"):
    logging.info(f"Waiting for order {trade.order.orderId} to fill...")
    start_time = time.time()
    while not trade.isDone():
        await asyncio.sleep(1)
        if (time.time() - start_time) > timeout:
            logging.warning(f"Order {trade.order.orderId} not filled within {timeout}s. Canceling.")
            send_notification(config, "Order Canceled", f"Order {trade.order.orderId} for {trade.contract.localSymbol} was canceled due to timeout.")
            ib.cancelOrder(trade.order)
            break
    if trade.orderStatus.status == OrderStatus.Filled:
        log_trade_to_ledger(trade, reason)


def is_market_open(contract_details, exchange_timezone_str: str):
    if not contract_details or not contract_details.liquidHours: return False
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str: continue
        try:
            start_str, end_str = session_str.split('-')
            start_dt, start_hm, end_hm = datetime.strptime(start_str.split(':')[0], '%Y%m%d').date(), datetime.strptime(start_str.split(':')[1], '%H%M').time(), datetime.strptime(end_str.split(':')[1], '%H%M').time()
            if now_tz.date() == start_dt and start_hm <= now_tz.time() < end_hm:
                logging.info("Market is open."); return True
        except (ValueError, IndexError): continue
    logging.info("Market is closed."); return False

def calculate_wait_until_market_open(contract_details, exchange_timezone_str: str) -> float:
    if not contract_details or not contract_details.liquidHours: return 3600
    tz, now_tz, next_open_dt = pytz.timezone(exchange_timezone_str), datetime.now(tz), None
    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str: continue
        try:
            session_start_dt = tz.localize(datetime.strptime(session_str.split('-')[0], '%Y%m%d:%H%M'))
            if session_start_dt > now_tz and (next_open_dt is None or session_start_dt < next_open_dt):
                next_open_dt = session_start_dt
        except (ValueError, IndexError): continue
    if next_open_dt:
        wait_seconds = (next_open_dt - now_tz).total_seconds() + 60
        logging.info(f"Next market open: {next_open_dt:%Y-%m-%d %H:%M:%S}. Waiting {wait_seconds / 3600:.2f} hours.")
        return wait_seconds
    return 3600


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

def get_expiration_details(chain: dict, future_exp: str) -> dict | None:
    valid_exp = [exp for exp in sorted(chain['expirations']) if exp <= future_exp]
    if not valid_exp:
        logging.warning(f"No option expiration before future {future_exp}. Using nearest available.")
        valid_exp = [exp for exp in sorted(chain['expirations']) if exp > datetime.now().strftime('%Y%m%d')]
        if not valid_exp: logging.error(f"No suitable option expirations for future {future_exp}."); return None
    chosen_exp = valid_exp[-1] if future_exp in [exp[:6] for exp in valid_exp] else valid_exp[0]
    days = (datetime.strptime(chosen_exp, '%Y%m%d').date() - datetime.now().date()).days
    return {'exp_date': chosen_exp, 'days_to_exp': days, 'strikes': chain['strikes_by_expiration'][chosen_exp]}


# --- 4. Risk, Sanity Checks & Connection Heartbeat ---

def is_trade_sane(signal: dict, config: dict) -> bool:
    min_confidence = config.get('risk_management', {}).get('min_confidence_threshold', 0.0)
    if signal['confidence'] < min_confidence:
        logging.warning(f"Sanity Check FAILED: Signal confidence {signal['confidence']} is below threshold {min_confidence}.")
        return False
    logging.info("Sanity Check PASSED."); return True

async def monitor_positions_for_risk(ib: IB, config: dict):
    risk_params = config.get('risk_management', {})
    stop_loss, take_profit, interval = risk_params.get('stop_loss_per_contract_usd'), risk_params.get('take_profit_per_contract_usd'), risk_params.get('check_interval_seconds', 300)
    if not stop_loss and not take_profit: return
    logging.info(f"Starting intraday risk monitor. Stop: ${stop_loss}, Profit: ${take_profit}")
    closed_ids = set()
    while True:
        try:
            await asyncio.sleep(interval)
            if not ib.isConnected(): continue
            positions = [p for p in await ib.reqPositionsAsync() if p.position != 0 and p.contract.conId not in closed_ids]
            if not positions: continue
            account = (await ib.managedAccounts())[0]
            logging.info("--- Risk Monitor: Checking open positions ---")
            for p in positions:
                pnl = await ib.reqPnLSingleAsync(account, '', p.contract.conId)
                pnl_per_contract = pnl.unrealizedPnL / abs(p.position)
                logging.info(f"Position {p.contract.localSymbol}: Unrealized PnL/Contract = ${pnl_per_contract:.2f}")
                reason = "Stop-Loss" if stop_loss and pnl_per_contract < -abs(stop_loss) else "Take-Profit" if take_profit and pnl_per_contract > abs(take_profit) else None
                if reason:
                    logging.info(f"{reason.upper()} TRIGGERED for {p.contract.localSymbol}. PnL/contract: ${pnl_per_contract:.2f}")
                    send_notification(config, reason, f"{p.contract.localSymbol} closed. PnL/contract: ${pnl_per_contract:.2f}")
                    order = MarketOrder('BUY' if p.position < 0 else 'SELL', abs(p.position))
                    trade = ib.placeOrder(p.contract, order)
                    await wait_for_fill(ib, trade, config, reason=reason)
                    closed_ids.add(p.contract.conId)
        except Exception as e:
            logging.error(f"Error in risk monitor loop: {e}")

async def send_heartbeat(ib: IB):
    while True:
        try:
            await asyncio.sleep(300)
            if ib.isConnected():
                server_time = await ib.reqCurrentTimeAsync()
                logging.info(f"Heartbeat sent. Server time: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
            conn_settings = config.get('connection', {})
            host, port = conn_settings.get('host', '127.0.0.1'), conn_settings.get('port', 7497)

            if not ib.isConnected():
                logging.info(f"Connecting to {host}:{port}...")
                await ib.connectAsync(host, port, clientId=conn_settings.get('clientId', random.randint(1, 1000)))
                send_notification(config, "Script Started", "Trading script has successfully connected to IBKR.")
                if monitor_task is None or monitor_task.done(): monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
                if heartbeat_task is None or heartbeat_task.done(): heartbeat_task = asyncio.create_task(send_heartbeat(ib))

            active_futures = await get_active_futures(ib, config['symbol'], config['exchange'])
            if not active_futures: raise ConnectionError("Could not find any active futures contracts.")

            front_month_details = (await ib.reqContractDetailsAsync(active_futures[0]))[0]
            if not is_market_open(front_month_details, config['exchange_timezone']):
                await asyncio.sleep(calculate_wait_until_market_open(front_month_details, config['exchange_timezone']))
                continue

            signals = get_prediction_from_api(config.get('api_base_url'))
            if not signals: raise ValueError("Could not get a valid signal list from the API.")
            
            trades_summary = {'filled': [], 'cancelled': []}
            for signal in signals:
                future = next((f for f in active_futures if f.lastTradeDateOrContractMonth.startswith(signal.get("contract_month", ""))), None)
                if not future: logging.warning(f"No active future for signal month {signal.get('contract_month')}."); continue

                logging.info(f"\n===== Processing Signal for {future.localSymbol} =====")
                ticker = ib.reqMktData(future, '', False, False); await asyncio.sleep(1)
                price = ticker.marketPrice()
                if util.isNan(price): logging.error(f"Failed to get market price for {future.localSymbol}."); continue
                
                logging.info(f"Current price for {future.localSymbol}: {price}")
                
                if await manage_existing_positions(ib, config, signal, price, future) and is_trade_sane(signal, config):
                    chain = await build_option_chain(ib, future)
                    if not chain: continue
                    
                    trade = await (execute_directional_strategy if signal['prediction_type'] == 'DIRECTIONAL' else execute_volatility_strategy)(ib, config, signal, chain, price, future)
                    if trade: trades_summary['filled' if trade.orderStatus.status == OrderStatus.Filled else 'cancelled'].append(trade)

            # End of cycle summary notification
            summary_msg = "<b>-- Trading Cycle Summary --</b>"
            if trades_summary['filled']: summary_msg += "\n<b>Positions Opened:</b>\n" + "".join([f"- {t.contract.localSymbol}: {t.order.action} {t.orderStatus.filled} @ ${t.orderStatus.avgFillPrice:.2f}\n" for t in trades_summary['filled']])
            if trades_summary['cancelled']: summary_msg += "\n<b>Orders Cancelled:</b>\n" + "".join([f"- {t.contract.localSymbol}: {t.order.action} {t.order.totalQuantity}\n" for t in trades_summary['cancelled']])
            if not trades_summary['filled'] and not trades_summary['cancelled']: summary_msg += "\nNo new positions were opened or attempted."
            send_notification(config, "Trading Cycle Complete", summary_msg)

            ny_tz = pytz.timezone(config['exchange_timezone'])
            now_ny = datetime.now(ny_tz)
            h, m = map(int, config.get('trade_execution_time_ny', '04:26').split(':'))
            next_trade = now_ny.replace(hour=h, minute=m, second=0, microsecond=0)
            if now_ny >= next_trade: next_trade += timedelta(days=1)
            wait = (next_trade - now_ny).total_seconds()
            logging.info(f"\nTrading cycle complete. Waiting {wait / 3600:.2f} hours until {next_trade:%Y-%m-%d %H:%M:%S} NY time.")
            await asyncio.sleep(wait)

        except (ConnectionError, OSError, asyncio.TimeoutError, asyncio.CancelledError) as e:
            msg = f"Connection error: {e}. Reconnecting..."
            logging.error(msg)
            send_notification(config, "Connection Error", msg)
            if monitor_task and not monitor_task.done(): monitor_task.cancel()
            if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
            monitor_task, heartbeat_task = None, None
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)
        except Exception as e:
            msg = f"An unexpected critical error occurred: {e}. Restarting logic in 1 min..."
            logging.critical(f"{msg}\n{traceback.format_exc()}")
            send_notification(config, "Critical Error", msg)
            if monitor_task and not monitor_task.done(): monitor_task.cancel()
            if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
            monitor_task, heartbeat_task = None, None
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        util.run(main_runner())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Script stopped manually.")

