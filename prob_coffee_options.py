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

# --- 1. API Integration & Notifications ---

def send_notification(config: dict, title: str, message: str):
    """Sends a push notification via Pushover if enabled in the config."""
    notif_config = config.get('notifications', {})
    if not notif_config.get('enabled'):
        return

    user_key = notif_config.get('pushover_user_key')
    api_token = notif_config.get('pushover_api_token')

    if not user_key or not api_token or 'YOUR_KEY' in user_key or 'YOUR_TOKEN' in api_token:
        log_with_timestamp("Notification sending skipped: Pushover keys not configured.")
        return

    try:
        # This is a blocking call, so it will be run in a separate thread
        response = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": api_token,
            "user": user_key,
            "title": f"Coffee Trader Alert: {title}",
            "message": message,
            "sound": "persistent" # Critical alert sound
        }, timeout=10)
        response.raise_for_status()
        log_with_timestamp(f"Sent notification: '{title}'")
    except requests.exceptions.RequestException as e:
        log_with_timestamp(f"Failed to send notification: {e}")


def get_prediction_from_api(api_url: str) -> list | None:
    """
    Calls the external prediction API to get the latest trading signals
    for multiple contracts.
    """
    log_with_timestamp("Requesting new trading signals from prediction API...")
    try:
        mock_signals = [
            {"contract_month": "202512", "prediction_type": "DIRECTIONAL", "direction": "BULLISH", "confidence": 0.65}, # Example below threshold
            {"contract_month": "202603", "prediction_type": "DIRECTIONAL", "direction": "BEARISH", "confidence": 0.88},
            {"contract_month": "202605", "prediction_type": "DIRECTIONAL", "direction": "BEARISH", "confidence": 0.75},
            {"contract_month": "202607", "prediction_type": "DIRECTIONAL", "direction": "BEARISH", "confidence": 0.75},
            {"contract_month": "202609", "prediction_type": "DIRECTIONAL", "direction": "BEARISH", "confidence": 0.79}
        ]
        log_with_timestamp(f"Multi-contract signals received: {json.dumps(mock_signals, indent=2)}")
        return mock_signals
    except requests.exceptions.RequestException as e:
        log_with_timestamp(f"Error fetching prediction from API: {e}")
        return None

# --- 2. Core Trading Logic & Strategy Execution ---

def price_option_black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict | None:
    """
    Calculates the theoretical price and Greeks of an option using the Black-Scholes model.
    """
    if T <= 0 or sigma <= 0:
        log_with_timestamp(f"Invalid input for B-S model: T={T}, sigma={sigma}. Cannot price option.")
        return None

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

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

    results = {
        "price": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "vega": round(vega, 4),
        "theta": round(theta, 4)
    }
    log_with_timestamp(f"Theoretical price calculated for {option_type} @ {K}: {results}")
    return results

async def get_option_market_data(ib: IB, contract: Option) -> dict | None:
    """
    Fetches live market data for a single option contract, including implied volatility.
    """
    log_with_timestamp(f"Fetching market data for option: {contract.localSymbol}")
    ticker = ib.reqMktData(contract, '106', False, False)
    
    # Wait for the ticker to populate with the necessary data
    await asyncio.sleep(2) # Allow time for data to stream
    
    iv = ticker.modelGreeks.impliedVol if ticker.modelGreeks else None
    
    if not iv or util.isNan(iv):
        log_with_timestamp(f"Could not get implied volatility for {contract.localSymbol}. Using fallback.")
        iv = 0.25 # Fallback volatility
    
    # The risk-free rate can also be derived from the ticker if available
    risk_free_rate = ticker.modelGreeks.optPrice if ticker.modelGreeks and ticker.modelGreeks.optPrice else 0.04 # Fallback
    
    ib.cancelMktData(contract) # Clean up the data subscription

    return {
        'implied_volatility': iv,
        'risk_free_rate': risk_free_rate
    }

def get_position_details(position: Position) -> dict:
    contract = position.contract
    details = {'type': 'UNKNOWN', 'key_strikes': []}
    if not isinstance(contract, Bag):
        details['type'] = 'SINGLE_LEG'
        details['key_strikes'].append(contract.strike)
        return details
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

async def manage_existing_positions(ib: IB, config: dict, signal: dict, underlying_price: float, contract_month: str):
    log_with_timestamp(f"--- Managing Positions for Contract Month: {contract_month} ---")
    all_positions = await ib.reqPositionsAsync()
    positions = [
        p for p in all_positions
        if p.contract.symbol == config['symbol'] and p.position != 0 and p.contract.secType == 'BAG' and
           p.contract.comboLegs and p.contract.comboLegs[0].lastTradeDateOrContractMonth.startswith(contract_month)
    ]
    if not positions:
        log_with_timestamp(f"No existing combo positions for {contract_month}.")
        return True
    target_strategy = ''
    if signal['prediction_type'] == 'DIRECTIONAL':
        target_strategy = 'BULL_CALL_SPREAD' if signal['direction'] == 'BULLISH' else 'BEAR_PUT_SPREAD'
    elif signal['prediction_type'] == 'VOLATILITY':
        target_strategy = 'LONG_STRADDLE' if signal['level'] == 'HIGH' else 'IRON_CONDOR'
    trades_to_close, position_is_aligned = [], False
    for pos in positions:
        pos_details = get_position_details(pos)
        current_strategy = pos_details['type']
        log_with_timestamp(f"Found open position for {contract_month}: {pos.position} of {current_strategy}")
        if current_strategy != target_strategy:
            log_with_timestamp(f"Position MISALIGNED. Current: {current_strategy}, Target: {target_strategy}. Closing.")
            trades_to_close.append(pos)
            continue
        log_with_timestamp("Position type is aligned. Checking viability...")
        viable = not (current_strategy == 'IRON_CONDOR' and not (pos_details['key_strikes'][0] < underlying_price < pos_details['key_strikes'][1]))
        if viable:
            log_with_timestamp("Position is ALIGNED and VIABLE. Holding.")
            position_is_aligned = True
        else:
            log_with_timestamp("Position no longer viable. Closing.")
            trades_to_close.append(pos)
    if trades_to_close:
        for pos_to_close in trades_to_close:
            order = MarketOrder('BUY' if pos_to_close.position < 0 else 'SELL', abs(pos_to_close.position))
            trade = ib.placeOrder(pos_to_close.contract, order)
            await wait_for_fill(trade, config, reason="Position Misaligned")
        return True
    if position_is_aligned:
        log_with_timestamp(f"An aligned position for {contract_month} already exists.")
        return False
    return True

async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract):
    log_with_timestamp(f"--- Executing {signal['direction']} Spread for {future_contract.localSymbol} ---")
    strategy_params = config['strategy']
    tuning = config.get('strategy_tuning', {})
    spread_width_strikes = tuning.get('vertical_spread_strikes_apart', 1)
    exp_details = get_expiration_details(chain, config, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return
    strikes = exp_details['strikes']
    try:
        atm_strike_index = strikes.index(min(strikes, key=lambda s: abs(s - underlying_price)))
    except (ValueError, IndexError):
        return log_with_timestamp("Could not find ATM strike in the chain. Aborting.")
    if signal['direction'] == 'BULLISH':
        if atm_strike_index + spread_width_strikes >= len(strikes): return log_with_timestamp("Not enough strikes for Bull Call Spread.")
        long_leg, short_leg = strikes[atm_strike_index], strikes[atm_strike_index + spread_width_strikes]
        legs_def = [('C', 'BUY', long_leg), ('C', 'SELL', short_leg)]
    else: # BEARISH
        if atm_strike_index - spread_width_strikes < 0: return log_with_timestamp("Not enough strikes for Bear Put Spread.")
        long_leg, short_leg = strikes[atm_strike_index], strikes[atm_strike_index - spread_width_strikes]
        legs_def = [('P', 'BUY', long_leg), ('P', 'SELL', short_leg)]
    
    # --- Black-Scholes Pricing Section ---
    log_with_timestamp("--- Pricing individual legs before execution ---")
    for right, _, strike in legs_def:
        leg_contract = Option(
            future_contract.localSymbol, exp_details['exp_date'], strike, right,
            chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(leg_contract)
        
        market_data = await get_option_market_data(ib, leg_contract)
        if market_data:
            price_option_black_scholes(
                S=underlying_price,
                K=strike,
                T=exp_details['days_to_exp'] / 365,
                r=market_data['risk_free_rate'],
                sigma=market_data['implied_volatility'],
                option_type=right
            )
    log_with_timestamp("--- End of pricing section ---")

    combo_contract = Bag(
        symbol=config['symbol'],
        exchange=chain['exchange'],
        currency='USD')

    combo_legs = []
    for right, action, strike in legs_def:
        contract = Option(
            future_contract.localSymbol, exp_details['exp_date'], strike, right,
            chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(contract)
        leg = ComboLeg(
            conId=contract.conId,
            ratio=1,
            action=action,
            exchange=chain['exchange'])
        combo_legs.append(leg)
    combo_contract.comboLegs = combo_legs

    order = MarketOrder('BUY', strategy_params['quantity'])
    trade = ib.placeOrder(combo_contract, order)
    await wait_for_fill(trade, config)

async def execute_volatility_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract):
    log_with_timestamp(f"--- Executing {signal['level']} Volatility Strategy for {future_contract.localSymbol} ---")
    strategy_params = config['strategy']
    tuning = config.get('strategy_tuning', {})
    exp_details = get_expiration_details(chain, config, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return
    strikes = exp_details['strikes']
    try:
        atm_strike_index = strikes.index(min(strikes, key=lambda s: abs(s - underlying_price)))
    except (ValueError, IndexError):
        return log_with_timestamp("Could not find ATM strike in the chain. Aborting.")
    combo_legs_def, order_action = [], ''
    if signal['level'] == 'HIGH':
        atm_strike = strikes[atm_strike_index]
        order_action = 'BUY'
        combo_legs_def = [('C', 'BUY', atm_strike), ('P', 'BUY', atm_strike)]
    elif signal['level'] == 'LOW':
        order_action = 'SELL'
        short_dist = tuning.get('iron_condor_short_strikes_from_atm', 1)
        wing_width = tuning.get('iron_condor_wing_strikes_apart', 1)
        short_put_idx, long_put_idx = atm_strike_index - short_dist, atm_strike_index - short_dist - wing_width
        short_call_idx, long_call_idx = atm_strike_index + short_dist, atm_strike_index + short_dist + wing_width
        if not (long_put_idx >= 0 and long_call_idx < len(strikes)):
            return log_with_timestamp("Not enough strikes for Iron Condor. Aborting.")
        strikes_map = {'lp': strikes[long_put_idx], 'sp': strikes[short_put_idx], 'sc': strikes[short_call_idx], 'lc': strikes[long_call_idx]}
        combo_legs_def = [('P', 'BUY', strikes_map['lp']), ('P', 'SELL', strikes_map['sp']), ('C', 'SELL', strikes_map['sc']), ('C', 'BUY', strikes_map['lc'])]
    if not combo_legs_def: return log_with_timestamp("No legs defined for combo. Aborting.")
    
    # --- Black-Scholes Pricing Section ---
    log_with_timestamp("--- Pricing individual legs before execution ---")
    for right, _, strike in combo_legs_def:
        leg_contract = Option(
            future_contract.localSymbol, exp_details['exp_date'], strike, right,
            chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(leg_contract)
        
        market_data = await get_option_market_data(ib, leg_contract)
        if market_data:
            price_option_black_scholes(
                S=underlying_price,
                K=strike,
                T=exp_details['days_to_exp'] / 365,
                r=market_data['risk_free_rate'],
                sigma=market_data['implied_volatility'],
                option_type=right
            )
    log_with_timestamp("--- End of pricing section ---")

    combo_contract = Bag(
        symbol=config['symbol'],
        exchange=chain['exchange'],
        currency='USD')
    
    combo_legs = []
    for right, action, strike in combo_legs_def:
        contract = Option(
            future_contract.localSymbol, exp_details['exp_date'], strike, right,
            chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(contract)
        leg = ComboLeg(
            conId=contract.conId,
            ratio=1,
            action=action,
            exchange=chain['exchange'])
        combo_legs.append(leg)
    combo_contract.comboLegs = combo_legs

    order = MarketOrder(order_action, strategy_params['quantity'])
    trade = ib.placeOrder(combo_contract, order)
    await wait_for_fill(trade, config)

# --- 3. Helper & Utility Functions ---

def log_with_timestamp(message: str):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def log_trade_to_ledger(trade: Trade, reason: str = "Strategy Execution"):
    if trade.orderStatus.status != OrderStatus.Filled: return
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'local_symbol': trade.contract.localSymbol,
        'action': trade.order.action, 'quantity': trade.orderStatus.filled, 'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * 100, # Combos are priced per share
        'order_id': trade.order.orderId, 'reason': reason
    }
    try:
        ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv')
        file_exists = os.path.isfile(ledger_path)
        with open(ledger_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(row)
        log_with_timestamp(f"Logged trade to ledger ({reason}): {row['action']} {row['quantity']} @ {row['avg_fill_price']}")
    except Exception as e:
        log_with_timestamp(f"Error writing to trade ledger: {e}")

async def wait_for_fill(trade: Trade, config: dict, timeout: int = 60, reason: str = "Strategy Execution"):
    log_with_timestamp(f"Waiting for order {trade.order.orderId} to fill...")
    start_time = time.time()
    while not trade.isDone():
        await asyncio.sleep(1)
        if timeout > 0 and (time.time() - start_time) > timeout:
            log_with_timestamp(f"Order {trade.order.orderId} not filled. Canceling.")
            # MODIFIED: Run notification in a non-blocking task
            asyncio.create_task(asyncio.to_thread(send_notification, config, "Order Canceled", f"Order {trade.order.orderId} for {trade.contract.localSymbol} was canceled due to timeout."))
            trade.ib.cancelOrder(trade.order)
            break
    if trade.orderStatus.status == OrderStatus.Filled:
        log_trade_to_ledger(trade, reason)

def is_market_open(contract_details, exchange_timezone_str: str):
    if not contract_details or not contract_details.liquidHours: return None, False
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    for session_str in contract_details.liquidHours.split(';'):
        if not session_str or 'CLOSED' in session_str: continue
        try:
            start_str, end_str = session_str.split('-')
            start_dt = datetime.strptime(start_str.split(':')[0], '%Y%m%d').date()
            start_hm, end_hm = datetime.strptime(start_str.split(':')[1], '%H%M').time(), datetime.strptime(end_str.split(':')[1], '%H%M').time()
            if now_tz.date() == start_dt and start_hm <= now_tz.time() < end_hm:
                return log_with_timestamp("Market is open."), True
        except (ValueError, IndexError): continue
    return log_with_timestamp("Market is closed."), False

def calculate_wait_until_market_open(contract_details, exchange_timezone_str: str) -> float:
    """Calculates the seconds to wait until the next market open session."""
    if not contract_details or not contract_details.liquidHours:
        return 3600  # Fallback to 1 hour if no details

    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    next_open_dt = None

    # Find the earliest upcoming session start time
    for session_str in contract_details.liquidHours.split(';'):
        if not session_str or 'CLOSED' in session_str:
            continue
        try:
            start_str = session_str.split('-')[0]
            session_start_dt = tz.localize(datetime.strptime(start_str, '%Y%m%d:%H%M'))
            if session_start_dt > now_tz:
                if next_open_dt is None or session_start_dt < next_open_dt:
                    next_open_dt = session_start_dt
        except (ValueError, IndexError):
            continue

    if next_open_dt:
        wait_seconds = (next_open_dt - now_tz).total_seconds() + 60 # Add a 1-min buffer
        log_with_timestamp(f"Next market open is at {next_open_dt.strftime('%Y-%m-%d %H:%M:%S')}. Waiting for {wait_seconds / 3600:.2f} hours.")
        return wait_seconds
    else:
        # If no future session is found for today/tomorrow (e.g., weekend), default to a longer wait
        log_with_timestamp("No upcoming market session found in contract details. Defaulting to 1-hour wait.")
        return 3600


async def get_active_futures(ib: IB, symbol: str, exchange: str, count: int = 5) -> list[Contract]:
    log_with_timestamp(f"Fetching {count} active futures contracts for {symbol} on {exchange}...")
    try:
        cds = await ib.reqContractDetailsAsync(Future(symbol, exchange=exchange))
        today_str = datetime.now().strftime('%Y%m')
        active_contracts = sorted(
            [cd.contract for cd in cds if cd.contract.lastTradeDateOrContractMonth > today_str],
            key=lambda c: c.lastTradeDateOrContractMonth
        )
        log_with_timestamp(f"Found {len(active_contracts)} active contracts. Returning the first {count}.")
        return active_contracts[:count]
    except Exception as e:
        log_with_timestamp(f"Error fetching active futures: {e}")
        return []

async def build_option_chain(ib: IB, future_contract: Contract):
    log_with_timestamp(f"Fetching option chain for future {future_contract.localSymbol}...")
    try:
        chains = await ib.reqSecDefOptParamsAsync(future_contract.symbol, future_contract.exchange, future_contract.secType, future_contract.conId)
        if not chains: return None
        chain = next((c for c in chains if c.exchange == future_contract.exchange), chains[0])
        return {
            'exchange': chain.exchange,
            'tradingClass': chain.tradingClass,
            'expirations': sorted(chain.expirations),
            'strikes_by_expiration': {exp: sorted(chain.strikes) for exp in chain.expirations}
        }
    except Exception as e:
        log_with_timestamp(f"Failed to build option chain for {future_contract.localSymbol}: {e}")
        return None

def get_expiration_details(chain: dict, config: dict, future_exp: str) -> dict | None:
    # Find the option expiration that is closest to the future's expiration, but not after it.
    valid_expirations = [exp for exp in sorted(chain['expirations']) if exp <= future_exp]
    if not valid_expirations:
        # As a fallback, if no options expire on or before the future, pick the nearest one.
        today_str = datetime.now().strftime('%Y%m%d')
        valid_expirations = [exp for exp in sorted(chain['expirations']) if exp > today_str]
        if not valid_expirations:
            return log_with_timestamp(f"No suitable option expirations found for future expiring {future_exp}."), None
        chosen_expiration = valid_expirations[0]
        log_with_timestamp(f"Warning: No option expiration found before future {future_exp}. Using nearest available: {chosen_expiration}")
    else:
        # From the valid ones, pick the latest one (closest to the future's expiry)
        chosen_expiration = valid_expirations[-1]

    days_to_exp = (datetime.strptime(chosen_expiration, '%Y%m%d').date() - datetime.now().date()).days
    return {'exp_date': chosen_expiration, 'days_to_exp': days_to_exp, 'strikes': chain['strikes_by_expiration'][chosen_expiration]}

# --- 4. Risk, Sanity Checks & Connection Heartbeat ---

def is_trade_sane(signal: dict, config: dict) -> bool:
    """Performs pre-trade sanity checks."""
    risk_params = config.get('risk_management', {})
    min_confidence = risk_params.get('min_confidence_threshold', 0.0)

    if signal['confidence'] < min_confidence:
        log_with_timestamp(f"Sanity Check FAILED: Signal confidence {signal['confidence']} is below threshold {min_confidence}.")
        return False

    log_with_timestamp("Sanity Check PASSED.")
    return True

async def monitor_positions_for_risk(ib: IB, config: dict):
    risk_params = config.get('risk_management', {})
    stop_loss_usd = risk_params.get('stop_loss_per_contract_usd')
    take_profit_usd = risk_params.get('take_profit_per_contract_usd')
    check_interval_sec = risk_params.get('check_interval_seconds', 300)
    if not stop_loss_usd and not take_profit_usd: return
    log_with_timestamp(f"Starting intraday risk monitor. Stop: ${stop_loss_usd}, Profit: ${take_profit_usd}")
    closed_by_monitor = set()
    while True:
        try:
            await asyncio.sleep(check_interval_sec)
            if not ib.isConnected(): continue
            positions = [p for p in await ib.reqPositionsAsync() if p.position != 0 and p.contract.secType == 'BAG' and p.contract.conId not in closed_by_monitor]
            if not positions: continue
            account = (await ib.managedAccounts())[0]
            for p in positions:
                pnl = await ib.reqPnLSingleAsync(account, '', p.contract.conId)
                unrealized_pnl_per_contract = pnl.unrealizedPnL / abs(p.position)
                reason = None
                if stop_loss_usd and unrealized_pnl_per_contract < -abs(stop_loss_usd): reason = "Stop-Loss"
                elif take_profit_usd and unrealized_pnl_per_contract > abs(take_profit_usd): reason = "Take-Profit"
                if reason:
                    log_with_timestamp(f"{reason.upper()} TRIGGERED for {p.contract.localSymbol}. PnL/contract: ${unrealized_pnl_per_contract:.2f}")
                    # MODIFIED: Run notification in a non-blocking task
                    asyncio.create_task(asyncio.to_thread(send_notification, config, reason, f"{p.contract.localSymbol} closed. PnL/contract: ${unrealized_pnl_per_contract:.2f}"))
                    order = MarketOrder('BUY' if p.position < 0 else 'SELL', abs(p.position))
                    trade = ib.placeOrder(p.contract, order)
                    await wait_for_fill(trade, config, reason=reason)
                    closed_by_monitor.add(p.contract.conId)
        except Exception as e:
            log_with_timestamp(f"Error in risk monitor loop: {e}")

async def send_heartbeat(ib: IB):
    """
    NEW: Sends a lightweight request every 5 minutes to keep the connection alive.
    """
    while True:
        try:
            await asyncio.sleep(300) # Sleep for 5 minutes
            if ib.isConnected():
                current_time = await ib.reqCurrentTimeAsync()
                log_with_timestamp(f"Heartbeat sent. Server time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            log_with_timestamp(f"Error in heartbeat loop: {e}")

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
                log_with_timestamp(f"Connecting to {host}:{port}...")
                client_id = conn_settings.get('clientId') or random.randint(1, 1000)
                await ib.connectAsync(host, port, clientId=client_id)
                asyncio.create_task(asyncio.to_thread(send_notification, config, "Script Started", "Trading script has successfully connected to IBKR."))

            if monitor_task is None or monitor_task.done():
                monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
            if heartbeat_task is None or heartbeat_task.done():
                heartbeat_task = asyncio.create_task(send_heartbeat(ib))

            active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=5)
            if not active_futures: raise ConnectionError("Could not find any active futures contracts.")

            front_month_details_list = await ib.reqContractDetailsAsync(active_futures[0])
            front_month_details = front_month_details_list[0] if front_month_details_list else None
            _, market_is_open = is_market_open(front_month_details, config['exchange_timezone'])

            if not market_is_open:
                wait_time = calculate_wait_until_market_open(front_month_details, config['exchange_timezone'])
                await asyncio.sleep(wait_time)
                continue

            signals = get_prediction_from_api(api_url="http://127.0.0.1:8000")
            if not signals: raise ValueError("Could not get a valid signal list from the API.")

            for signal in signals:
                signal_month = signal.get("contract_month")
                if not signal_month:
                    continue

                future_to_trade = next((f for f in active_futures if f.lastTradeDateOrContractMonth.startswith(signal_month)), None)

                if not future_to_trade:
                    log_with_timestamp(f"No active future contract found for signal month {signal_month}. Skipping.")
                    continue

                log_with_timestamp(f"\n===== Processing Signal for {future_to_trade.localSymbol} =====")
                ticker = ib.reqMktData(future_to_trade, '', False, False)
                await asyncio.sleep(1)
                price = ticker.marketPrice()
                if util.isNan(price):
                    log_with_timestamp(f"Failed to retrieve valid market price for {future_to_trade.localSymbol}. Skipping signal.")
                    continue
                log_with_timestamp(f"Current price for {future_to_trade.localSymbol}: {price}")
                can_open_new = await manage_existing_positions(ib, config, signal, price, signal.get("contract_month"))
                if can_open_new:
                    if not is_trade_sane(signal, config): continue
                    chain = await build_option_chain(ib, future_to_trade)
                    if not chain: continue
                    if signal['prediction_type'] == 'DIRECTIONAL':
                        await execute_directional_strategy(ib, config, signal, chain, price, future_to_trade)
                    elif signal['prediction_type'] == 'VOLATILITY':
                        await execute_volatility_strategy(ib, config, signal, chain, price, future_to_trade)

            ny_tz = pytz.timezone(config['exchange_timezone'])
            now_ny = datetime.now(ny_tz)
            hour, minute = map(int, config.get('trade_execution_time_ny', '04:26').split(':'))
            next_trade_dt_ny = now_ny.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now_ny >= next_trade_dt_ny: next_trade_dt_ny += timedelta(days=1)
            seconds_to_wait = (next_trade_dt_ny - now_ny).total_seconds()
            log_with_timestamp(f"\nTrading cycle complete. Waiting {seconds_to_wait / 3600:.2f} hours until {next_trade_dt_ny.strftime('%Y-%m-%d %H:%M:%S')} NY time.")
            await asyncio.sleep(seconds_to_wait)

        except (ConnectionError, OSError, asyncio.TimeoutError, asyncio.CancelledError) as e:
            msg = f"Connection error: {e}. Reconnecting..."
            log_with_timestamp(msg)
            send_notification(config, "Connection Error", msg)
            if monitor_task and not monitor_task.done(): monitor_task.cancel()
            if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
            monitor_task, heartbeat_task = None, None
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)
        except Exception as e:
            msg = f"An unexpected critical error occurred: {e}. Restarting logic in 1 min..."
            log_with_timestamp(msg)
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
        log_with_timestamp("Script stopped manually.")

