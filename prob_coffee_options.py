import asyncio
import csv
import json
import os
import random
import time
from datetime import datetime
import requests # Added for API calls

import pytz
from ib_insync import *

# --- 1. API Integration & Model Placeholders ---

def get_prediction_from_api(api_url: str) -> dict:
    """
    Calls the external prediction API to get the latest trading signal.
    """
    log_with_timestamp("Requesting new trading signal from prediction API...")
    try:
        # Simulate a random signal for demonstration
        if random.random() > 0.5:
            direction = "BULLISH" if random.random() > 0.5 else "BEARISH"
            signal = {
                "prediction_type": "DIRECTIONAL",
                "direction": direction,
                "confidence": round(random.uniform(0.6, 0.95), 2)
            }
        else:
            level = "HIGH" if random.random() > 0.5 else "LOW"
            signal = {
                "prediction_type": "VOLATILITY",
                "level": level,
                "confidence": round(random.uniform(0.6, 0.95), 2)
            }
        
        log_with_timestamp(f"Signal received: {signal}")
        return signal

    except requests.exceptions.RequestException as e:
        log_with_timestamp(f"Error fetching prediction from API: {e}")
        return None


def get_theoretical_option_price(underlying_price: float, strike_price: float, days_to_expiration: int, option_type: str) -> float:
    """
    Placeholder for your model that calculates the "fair value" of an option.
    """
    price = 0.0
    time_premium_per_day = 0.05
    if option_type == 'C':
        intrinsic_value = max(0, underlying_price - strike_price)
        price = intrinsic_value + (days_to_expiration * time_premium_per_day)
    elif option_type == 'P':
        intrinsic_value = max(0, strike_price - underlying_price)
        price = intrinsic_value + (days_to_expiration * time_premium_per_day)
    return price if price > 0 else 0.01

# --- 2. Core Trading Logic & Strategy Execution ---

def get_position_details(position: Position) -> dict:
    """
    Analyzes a position's contract to identify the strategy and its key strikes.
    Returns a dictionary like {'type': 'IRON_CONDOR', 'key_strikes': [210.0, 225.0]}
    """
    contract = position.contract
    details = {'type': 'UNKNOWN', 'key_strikes': []}

    if not isinstance(contract, Bag):
        details['type'] = 'SINGLE_LEG'
        details['key_strikes'].append(contract.strike)
        return details

    legs = contract.comboLegs
    num_legs = len(legs)
    actions = ''.join(sorted([leg.action for leg in legs]))
    rights = ''.join(sorted([leg.right for leg in legs]))
    strikes = sorted([leg.strike for leg in legs])

    if num_legs == 2:
        details['key_strikes'] = strikes
        if rights == 'CC' and actions == 'BS': details['type'] = 'BULL_CALL_SPREAD'
        elif rights == 'PP' and actions == 'BS': details['type'] = 'BEAR_PUT_SPREAD'
        elif rights == 'CP' and actions == 'BB': details['type'] = 'LONG_STRADDLE'
    
    if num_legs == 4 and rights == 'CCPP' and actions == 'BBSS':
        details['type'] = 'IRON_CONDOR'
        # For a condor, the key strikes are the two inner (short) strikes
        details['key_strikes'] = [strikes[1], strikes[2]]

    return details


async def manage_existing_positions(ib: IB, config: dict, new_signal: dict, underlying_price: float):
    """
    Intelligently checks current positions against the new signal, including strike viability.
    """
    log_with_timestamp("--- Managing Existing Positions ---")
    positions = [
        p for p in await ib.reqPositionsAsync()
        if p.contract.symbol == config['symbol'] and p.position != 0
    ]

    if not positions:
        log_with_timestamp("No existing positions to manage.")
        return True # Return True indicating it's safe to open a new position

    # Determine the target strategy based on the new signal
    target_strategy = ''
    if new_signal['prediction_type'] == 'DIRECTIONAL':
        target_strategy = 'BULL_CALL_SPREAD' if new_signal['direction'] == 'BULLISH' else 'BEAR_PUT_SPREAD'
    elif new_signal['prediction_type'] == 'VOLATILITY':
        target_strategy = 'LONG_STRADDLE' if new_signal['level'] == 'HIGH' else 'IRON_CONDOR'

    trades_to_close = []
    position_is_aligned = False

    for pos in positions:
        pos_details = get_position_details(pos)
        current_strategy = pos_details['type']
        key_strikes = pos_details['key_strikes']
        
        log_with_timestamp(f"Found open position: {pos.position} of {current_strategy} ({pos.contract.localSymbol}) with key strikes {key_strikes}")

        # Primary Check: Is the strategy type aligned with the signal?
        if current_strategy != target_strategy:
            log_with_timestamp(f"Position MISALIGNED. Current type: {current_strategy}, Target type: {target_strategy}. Closing.")
            trades_to_close.append(pos)
            continue # Move to next position

        # Secondary Check: If type is aligned, is the position still viable?
        log_with_timestamp("Position type is aligned with signal. Checking viability...")
        
        viable = True
        if current_strategy == 'IRON_CONDOR':
            lower_short_strike, upper_short_strike = key_strikes[0], key_strikes[1]
            if not (lower_short_strike < underlying_price < upper_short_strike):
                viable = False
                log_with_timestamp(f"Iron Condor is no longer viable. Price {underlying_price} has breached the short strikes {key_strikes}. Closing.")
        
        if viable:
            log_with_timestamp("Position is ALIGNED and VIABLE. Holding position.")
            position_is_aligned = True
        else:
            trades_to_close.append(pos)
            
    if trades_to_close:
        log_with_timestamp(f"Closing {len(trades_to_close)} misaligned/unviable position(s)...")
        for pos_to_close in trades_to_close:
            action = 'BUY' if pos_to_close.position < 0 else 'SELL'
            order = MarketOrder(action, abs(pos_to_close.position))
            trade = ib.placeOrder(pos_to_close.contract, order)
            await wait_for_fill(trade)
        return True # It's now clear to open a new position

    if position_is_aligned:
        log_with_timestamp("An aligned position already exists. No new trade will be placed.")
        return False # Do not open a new position

    return True


async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float):
    """
    Executes a risk-defined vertical spread as a single COMBO order using configurable widths.
    """
    log_with_timestamp(f"--- Executing Directional Spread Strategy: {signal['direction']} ---")
    strategy_params = config['strategy']
    tuning = config.get('strategy_tuning', {})
    spread_width_strikes = tuning.get('vertical_spread_strikes_apart', 1)

    exp_details = get_expiration_details(chain, config)
    if not exp_details: return

    strikes = exp_details['strikes']
    
    try:
        atm_strike_index = strikes.index(min(strikes, key=lambda s: abs(s - underlying_price)))
    except ValueError:
        return log_with_timestamp("Could not find ATM strike in the chain. Aborting.")

    if signal['direction'] == 'BULLISH':
        if atm_strike_index + spread_width_strikes >= len(strikes):
            return log_with_timestamp("Not enough strikes available for Bull Call Spread width. Aborting.")
        long_leg_strike = strikes[atm_strike_index]
        short_leg_strike = strikes[atm_strike_index + spread_width_strikes]
        leg1_right, leg2_right, leg1_action, leg2_action = 'C', 'C', 'BUY', 'SELL'

    elif signal['direction'] == 'BEARISH':
        if atm_strike_index - spread_width_strikes < 0:
            return log_with_timestamp("Not enough strikes available for Bear Put Spread width. Aborting.")
        long_leg_strike = strikes[atm_strike_index]
        short_leg_strike = strikes[atm_strike_index - spread_width_strikes]
        leg1_right, leg2_right, leg1_action, leg2_action = 'P', 'P', 'BUY', 'SELL'

    combo_contract = Bag(
        symbol=config['symbol'], exchange=chain['exchange'], currency='USD',
        comboLegs=[
            ComboLeg(conId=0, ratio=1, action=leg1_action, exchange=chain['exchange'], right=leg1_right, strike=long_leg_strike, lastTradeDateOrContractMonth=exp_details['exp_date']),
            ComboLeg(conId=0, ratio=1, action=leg2_action, exchange=chain['exchange'], right=leg2_right, strike=short_leg_strike, lastTradeDateOrContractMonth=exp_details['exp_date'])
        ]
    )
    await ib.qualifyContractsAsync(combo_contract)
    log_with_timestamp(f"Placing Combo Order for {signal['direction']} Spread ({long_leg_strike}/{short_leg_strike}).")
    order = MarketOrder('BUY', strategy_params['quantity'])
    trade = ib.placeOrder(combo_contract, order)
    await wait_for_fill(trade)


async def execute_volatility_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float):
    """
    Executes a defined-risk volatility strategy as a single COMBO order using configurable widths.
    """
    log_with_timestamp(f"--- Executing Volatility Strategy for {signal['level']} Volatility ---")
    strategy = config['strategy']
    tuning = config.get('strategy_tuning', {})
    exp_details = get_expiration_details(chain, config)
    if not exp_details: return

    strikes = exp_details['strikes']
    try:
        atm_strike_index = strikes.index(min(strikes, key=lambda s: abs(s - underlying_price)))
    except ValueError:
        return log_with_timestamp("Could not find ATM strike in the chain. Aborting.")
        
    combo_legs, order_action = [], ''
    
    if signal['level'] == 'HIGH':
        atm_strike = strikes[atm_strike_index]
        log_with_timestamp(f"Strategy: Long Straddle at {atm_strike} strike.")
        order_action = 'BUY'
        combo_legs.extend([
            ComboLeg(conId=0, ratio=1, action='BUY', exchange=chain['exchange'], right='C', strike=atm_strike, lastTradeDateOrContractMonth=exp_details['exp_date']),
            ComboLeg(conId=0, ratio=1, action='BUY', exchange=chain['exchange'], right='P', strike=atm_strike, lastTradeDateOrContractMonth=exp_details['exp_date'])
        ])

    elif signal['level'] == 'LOW':
        log_with_timestamp("Strategy: Iron Condor.")
        order_action = 'SELL'
        
        short_dist = tuning.get('iron_condor_short_strikes_from_atm', 1)
        wing_width = tuning.get('iron_condor_wing_strikes_apart', 1)

        short_put_idx = atm_strike_index - short_dist
        long_put_idx = short_put_idx - wing_width
        short_call_idx = atm_strike_index + short_dist
        long_call_idx = short_call_idx + wing_width

        if not (long_put_idx >= 0 and long_call_idx < len(strikes)):
            return log_with_timestamp("Not enough strikes to build configured Iron Condor. Aborting.")

        long_put_strike = strikes[long_put_idx]
        short_put_strike = strikes[short_put_idx]
        short_call_strike = strikes[short_call_idx]
        long_call_strike = strikes[long_call_idx]

        log_with_timestamp(f"Condor Legs: BUY {long_put_strike}P, SELL {short_put_strike}P, SELL {short_call_strike}C, BUY {long_call_strike}C")
            
        combo_legs.extend([
            ComboLeg(conId=0, ratio=1, action='BUY', exchange=chain['exchange'], right='P', strike=long_put_strike, lastTradeDateOrContractMonth=exp_details['exp_date']),
            ComboLeg(conId=0, ratio=1, action='SELL', exchange=chain['exchange'], right='P', strike=short_put_strike, lastTradeDateOrContractMonth=exp_details['exp_date']),
            ComboLeg(conId=0, ratio=1, action='SELL', exchange=chain['exchange'], right='C', strike=short_call_strike, lastTradeDateOrContractMonth=exp_details['exp_date']),
            ComboLeg(conId=0, ratio=1, action='BUY', exchange=chain['exchange'], right='C', strike=long_call_strike, lastTradeDateOrContractMonth=exp_details['exp_date'])
        ])

    if not combo_legs: return log_with_timestamp("No legs defined for combo order. Aborting.")

    combo_contract = Bag(symbol=config['symbol'], exchange=chain['exchange'], currency='USD', comboLegs=combo_legs)
    await ib.qualifyContractsAsync(combo_contract)
    log_with_timestamp(f"Placing {len(combo_legs)}-Leg Combo Order.")
    order = MarketOrder(order_action, strategy['quantity'])
    trade = ib.placeOrder(combo_contract, order)
    await wait_for_fill(trade)


# --- 3. Helper & Utility Functions ---

def log_with_timestamp(message: str):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def log_trade_to_ledger(trade: Trade):
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv')
    file_exists = os.path.isfile(ledger_path)
    if trade.orderStatus.status != OrderStatus.Filled: return
    is_combo = isinstance(trade.contract, Bag)
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'local_symbol': trade.contract.localSymbol, 
        'action': trade.order.action, 'quantity': trade.orderStatus.filled, 'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * (100 if is_combo else 37500),
        'order_id': trade.order.orderId
    }
    try:
        with open(ledger_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(row)
        log_with_timestamp(f"Logged trade to ledger: {row['action']} {row['quantity']} {row['local_symbol']} @ {row['avg_fill_price']}")
    except Exception as e:
        log_with_timestamp(f"Error writing to trade ledger: {e}")

async def wait_for_fill(trade: Trade, timeout: int = 60):
    log_with_timestamp(f"Waiting for order {trade.order.orderId} to fill...")
    start_time = time.time()
    while not trade.isDone():
        await asyncio.sleep(1)
        if timeout > 0 and (time.time() - start_time) > timeout:
            log_with_timestamp(f"Order {trade.order.orderId} not filled within {timeout}s. Canceling.")
            trade.ib.cancelOrder(trade.order)
            break
    if trade.orderStatus.status == OrderStatus.Filled:
        log_trade_to_ledger(trade)

def is_market_open(contract_details, exchange_timezone_str: str):
    if not contract_details or not contract_details.liquidHours: return None, False
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    for session_str in contract_details.liquidHours.split(';'):
        if not session_str or 'CLOSED' in session_str: continue
        try:
            start_str, end_str = session_str.split('-')
            start_dt = datetime.strptime(f"{start_str.split(':')[0]}{start_str.split(':')[1]}", '%Y%m%d%H%M').date()
            start_hm, end_hm = datetime.strptime(start_str.split(':')[1], '%H%M').time(), datetime.strptime(end_str.split(':')[1], '%H%M').time()
            if now_tz.date() == start_dt and tz.localize(datetime.combine(now_tz.date(), start_hm)) <= now_tz < tz.localize(datetime.combine(now_tz.date(), end_hm)):
                return log_with_timestamp("Market is open."), True
        except (ValueError, IndexError): continue
    return log_with_timestamp("Market is closed."), False

async def get_full_contract_details(ib: IB, contract):
    details = await ib.reqContractDetailsAsync(contract)
    return details[0] if details else None

async def build_option_chain(ib: IB, symbol: str, exchange: str):
    log_with_timestamp(f"Fetching option chain for {symbol} on {exchange}...")
    try:
        contracts_details = await ib.reqContractDetailsAsync(FuturesOption(symbol, exchange=exchange, currency='USD', multiplier='37500'))
        if not contracts_details: return None
        strikes_by_expiration = {exp: sorted(list({c.contract.strike for c in contracts_details if c.contract.lastTradeDateOrContractMonth == exp}))
                                 for exp in sorted({c.contract.lastTradeDateOrContractMonth for c in contracts_details})}
        return {'exchange': exchange, 'expirations': sorted(strikes_by_expiration.keys()), 'strikes_by_expiration': strikes_by_expiration}
    except Exception as e:
        log_with_timestamp(f"Failed to build option chain on {exchange}: {e}")
        return None

def get_expiration_details(chain: dict, config: dict) -> dict | None:
    today_str = datetime.now().strftime('%Y%m%d')
    upcoming_expirations = [exp for exp in sorted(chain['expirations']) if exp > today_str]
    exp_choice = config.get('expiration_to_use', 'next')
    exp_index = 1 if exp_choice == 'next' and len(upcoming_expirations) > 1 else 0
    if not upcoming_expirations or len(upcoming_expirations) <= exp_index:
        log_with_timestamp(f"Not enough upcoming expirations for '{exp_choice}'.")
        return None
    
    chosen_expiration = upcoming_expirations[exp_index]
    days_to_exp = (datetime.strptime(chosen_expiration, '%Y%m%d').date() - datetime.now().date()).days
    return {'exp_date': chosen_expiration, 'days_to_exp': days_to_exp, 'strikes': chain['strikes_by_expiration'][chosen_expiration]}


# --- 4. Main Execution Loop ---

async def main_runner():
    ib = IB()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

    while True:
        try:
            with open(config_path, 'r') as f: config = json.load(f)
            conn_settings = config.get('connection', {})
            host, port = conn_settings.get('host', '127.0.0.1'), conn_settings.get('port', 7497)
            
            if not ib.isConnected():
                client_id = conn_settings.get('clientId') or random.randint(1, 1000)
                await ib.connectAsync(host, port, clientId=client_id)

            future_details = await get_full_contract_details(ib, Future(config['symbol'], exchange=config['exchange']))
            if not future_details: raise ConnectionError("Could not find a valid front-month future.")

            _, market_is_open = is_market_open(future_details, config['exchange_timezone'])
            if market_is_open:
                ticker_list = await ib.reqTickersAsync(future_details.contract)
                if not ticker_list or util.isNan(ticker_list[0].marketPrice()):
                    log_with_timestamp("Could not get valid market price for underlying. Skipping cycle.")
                else:
                    current_underlying_price = ticker_list[0].marketPrice()
                    signal = get_prediction_from_api(api_url="http://127.0.0.1:8000")
                    if not signal: raise ValueError("Could not get a valid signal from the API.")

                    can_open_new_trade = await manage_existing_positions(ib, config, signal, current_underlying_price)

                    if can_open_new_trade:
                        log_with_timestamp("Proceeding to execute new trade based on signal.")
                        chain = await build_option_chain(ib, config['symbol'], config['exchange'])
                        if not chain: chain = await build_option_chain(ib, config['symbol'], 'ICE')
                        if not chain: raise ConnectionError("Could not build option chain.")
                        
                        if signal['prediction_type'] == 'DIRECTIONAL':
                            await execute_directional_strategy(ib, config, signal, chain, current_underlying_price)
                        elif signal['prediction_type'] == 'VOLATILITY':
                            await execute_volatility_strategy(ib, config, signal, chain, current_underlying_price)
                    else:
                        log_with_timestamp("Holding existing aligned position. No new trade will be placed.")

            wait_hours = config.get('trade_interval_hours', 2)
            log_with_timestamp(f"\nTrading cycle complete. Waiting for {wait_hours} hours...")
            end_time = time.time() + (wait_hours * 3600)
            while time.time() < end_time:
                await asyncio.sleep(min(300, end_time - time.time()))
                if time.time() < end_time and ib.isConnected():
                    await ib.reqCurrentTimeAsync(), log_with_timestamp("Heartbeat sent. Connection alive.")

        except (ConnectionError, OSError) as e:
            log_with_timestamp(f"Connection error: {e}. Handling reconnect...")
            if ib.isConnected(): ib.disconnect()
            log_with_timestamp("Attempting to reconnect every 2 minutes... ACTION REQUIRED: Please log in to IBKR Gateway.")
            while True:
                await asyncio.sleep(120)
                try:
                    client_id = conn_settings.get('clientId') or random.randint(1, 1000)
                    log_with_timestamp(f"Attempting reconnect to {host}:{port}...")
                    await ib.connectAsync(host, port, clientId=client_id, timeout=10)
                    if ib.isConnected(): log_with_timestamp("Reconnection successful!"); break
                except Exception as reconnect_e:
                    log_with_timestamp(f"Reconnect failed: {reconnect_e}. Retrying...")

        except Exception as e:
            log_with_timestamp(f"An unexpected critical error occurred: {e}. Restarting logic in 1 min...")
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        util.run(main_runner())
    except (KeyboardInterrupt, SystemExit):
        log_with_timestamp("Script stopped manually.")

