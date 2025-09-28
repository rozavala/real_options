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
    This replaces the old placeholder function.
    
    The expected signal format is a dictionary, e.g.:
    { "prediction_type": "DIRECTIONAL", "direction": "BULLISH", "confidence": 0.75 }
    OR
    { "prediction_type": "VOLATILITY", "level": "HIGH", "confidence": 0.82 }
    """
    log_with_timestamp("Requesting new trading signal from prediction API...")
    try:
        # In a real scenario, this would poll the result of a job started by send_data_to_api.py
        # For this example, we'll simulate a direct response for simplicity.
        # This function now returns a more complex signal.
        
        # Simulate a random signal for demonstration
        if random.random() > 0.5:
            # Directional Signal
            direction = "BULLISH" if random.random() > 0.5 else "BEARISH"
            signal = {
                "prediction_type": "DIRECTIONAL",
                "direction": direction,
                "confidence": round(random.uniform(0.6, 0.95), 2)
            }
        else:
            # Volatility Signal
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
    Replace this with your Black-Scholes or other pricing model.
    """
    # This remains a placeholder for your sophisticated pricing model.
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

async def manage_existing_positions(ib: IB, config: dict, new_signal: dict):
    """
    Checks current positions against the new signal and closes those that are misaligned.
    """
    log_with_timestamp("--- Managing Existing Positions ---")
    positions = await ib.reqPositionsAsync()
    trades_to_close = []

    # Filter for relevant coffee option positions
    coffee_positions = [
        p for p in positions 
        if isinstance(p.contract, (FuturesOption, Bag)) and p.contract.symbol == config['symbol'] and p.position != 0
    ]

    if not coffee_positions:
        log_with_timestamp("No existing positions to manage.")
        return

    for pos in coffee_positions:
        # For simplicity, we'll close any complex position before opening a new one.
        # A more advanced system could analyze the greeks of the whole portfolio.
        log_with_timestamp(f"Closing existing position in {pos.contract.localSymbol} to make way for new signal.")
        action = 'BUY' if pos.position < 0 else 'SELL'
        order = MarketOrder(action, abs(pos.position))
        trade = ib.placeOrder(pos.contract, order)
        trades_to_close.append(trade)

    if trades_to_close:
        log_with_timestamp(f"Waiting for {len(trades_to_close)} position(s) to close...")
        for trade in trades_to_close:
            await wait_for_fill(trade)
    else:
        log_with_timestamp("All existing positions are aligned with the new signal.")


async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float):
    """
    Executes a risk-defined vertical spread as a single COMBO order.
    """
    log_with_timestamp(f"--- Executing Directional Spread Strategy: {signal['direction']} ---")
    strategy_params = config['strategy']
    exp_details = get_expiration_details(chain, config)
    if not exp_details: return

    strikes = exp_details['strikes']
    atm_strike = min(strikes, key=lambda s: abs(s - underlying_price))
    
    # Define legs for the spread
    if signal['direction'] == 'BULLISH':
        # Bull Call Spread: BUY ATM Call, SELL OTM Call
        long_leg_strike = atm_strike
        short_leg_strike = next((s for s in strikes if s > long_leg_strike), None)
        if not short_leg_strike:
            log_with_timestamp("Could not find OTM strike for Bull Call Spread. Aborting.")
            return
        leg1_right, leg2_right = 'C', 'C'
        leg1_action, leg2_action = 'BUY', 'SELL'

    elif signal['direction'] == 'BEARISH':
        # Bear Put Spread: BUY ATM Put, SELL OTM Put
        long_leg_strike = atm_strike
        short_leg_strike = next((s for s in reversed(strikes) if s < long_leg_strike), None)
        if not short_leg_strike:
            log_with_timestamp("Could not find OTM strike for Bear Put Spread. Aborting.")
            return
        leg1_right, leg2_right = 'P', 'P'
        leg1_action, leg2_action = 'BUY', 'SELL'

    # Create the combo contract (Bag)
    combo_contract = Bag(
        symbol=config['symbol'],
        exchange=chain['exchange'],
        currency='USD',
        comboLegs=[
            ComboLeg(conId=0, ratio=1, action=leg1_action, exchange=chain['exchange'], right=leg1_right, strike=long_leg_strike, lastTradeDateOrContractMonth=exp_details['exp_date']),
            ComboLeg(conId=0, ratio=1, action=leg2_action, exchange=chain['exchange'], right=leg2_right, strike=short_leg_strike, lastTradeDateOrContractMonth=exp_details['exp_date'])
        ]
    )
    # IBKR needs to resolve the conIds for the legs
    await ib.qualifyContractsAsync(combo_contract)

    log_with_timestamp(f"Placing Combo Order for {signal['direction']} Spread.")
    # For spreads, we typically pay a debit, so we use a Market Order to get filled quickly
    order = MarketOrder('BUY', strategy_params['quantity'])
    trade = ib.placeOrder(combo_contract, order)
    
    await wait_for_fill(trade)


async def execute_volatility_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float):
    """
    Executes a defined-risk volatility strategy as a single COMBO order.
    """
    log_with_timestamp(f"--- Executing Volatility Strategy for {signal['level']} Volatility ---")
    strategy = config['strategy']
    exp_details = get_expiration_details(chain, config)
    if not exp_details: return

    strikes = exp_details['strikes']
    atm_strike = min(strikes, key=lambda s: abs(s - underlying_price))

    combo_legs = []
    order_action = ''
    
    if signal['level'] == 'HIGH':
        log_with_timestamp(f"Strategy: Long Straddle at {atm_strike} strike.")
        order_action = 'BUY'
        # Long Straddle: BUY ATM Call + BUY ATM Put
        combo_legs.append(ComboLeg(conId=0, ratio=1, action='BUY', exchange=chain['exchange'], right='C', strike=atm_strike, lastTradeDateOrContractMonth=exp_details['exp_date']))
        combo_legs.append(ComboLeg(conId=0, ratio=1, action='BUY', exchange=chain['exchange'], right='P', strike=atm_strike, lastTradeDateOrContractMonth=exp_details['exp_date']))

    elif signal['level'] == 'LOW':
        log_with_timestamp("Strategy: Iron Condor.")
        order_action = 'SELL' # We sell condors for a credit
        # Iron Condor: BUY OTM Put, SELL closer OTM Put, SELL closer OTM Call, BUY OTM Call
        short_put_strike = next((s for s in reversed(strikes) if s < atm_strike), None)
        short_call_strike = next((s for s in strikes if s > atm_strike), None)
        if not short_put_strike or not short_call_strike:
            log_with_timestamp("Not enough strikes to build Iron Condor. Aborting.")
            return
        long_put_strike = next((s for s in reversed(strikes) if s < short_put_strike), None)
        long_call_strike = next((s for s in strikes if s > short_call_strike), None)
        if not long_put_strike or not long_call_strike:
            log_with_timestamp("Not enough strikes to build wings for Iron Condor. Aborting.")
            return
            
        combo_legs.append(ComboLeg(conId=0, ratio=1, action='BUY', exchange=chain['exchange'], right='P', strike=long_put_strike, lastTradeDateOrContractMonth=exp_details['exp_date']))
        combo_legs.append(ComboLeg(conId=0, ratio=1, action='SELL', exchange=chain['exchange'], right='P', strike=short_put_strike, lastTradeDateOrContractMonth=exp_details['exp_date']))
        combo_legs.append(ComboLeg(conId=0, ratio=1, action='SELL', exchange=chain['exchange'], right='C', strike=short_call_strike, lastTradeDateOrContractMonth=exp_details['exp_date']))
        combo_legs.append(ComboLeg(conId=0, ratio=1, action='BUY', exchange=chain['exchange'], right='C', strike=long_call_strike, lastTradeDateOrContractMonth=exp_details['exp_date']))

    if not combo_legs:
        log_with_timestamp("No legs defined for combo order. Aborting.")
        return

    # Create the combo contract (Bag)
    combo_contract = Bag(
        symbol=config['symbol'],
        exchange=chain['exchange'],
        currency='USD',
        comboLegs=combo_legs
    )
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
    
    # For combo orders, the multiplier is part of the leg definitions, not the parent Bag contract
    # We will log the net price of the combo. A more detailed log could break out each leg.
    is_combo = isinstance(trade.contract, Bag)
    
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'local_symbol': trade.contract.localSymbol, 
        'action': trade.order.action, 
        'quantity': trade.orderStatus.filled,
        'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * (37500 if not is_combo else 100), # Combos are priced differently
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

async def wait_for_fill(trade: Trade, timeout: int = 60): # Default timeout of 60s
    """Waits for a trade to complete, with an optional timeout."""
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
    if not contract_details or not contract_details.liquidHours: return False
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    sessions_str = contract_details.liquidHours.split(';')
    for session_str in sessions_str:
        if not session_str or 'CLOSED' in session_str: continue
        try:
            start_str, end_str = session_str.split('-')
            start_dt = datetime.strptime(f"{start_str.split(':')[0]}{start_str.split(':')[1]}", '%Y%m%d%H%M').date()
            start_hm = datetime.strptime(start_str.split(':')[1], '%H%M').time()
            end_hm = datetime.strptime(end_str.split(':')[1], '%H%M').time()
            
            if now_tz.date() == start_dt:
                session_start_time = tz.localize(datetime.combine(now_tz.date(), start_hm))
                session_end_time = tz.localize(datetime.combine(now_tz.date(), end_hm))
                if session_start_time <= now_tz < session_end_time:
                    log_with_timestamp("Market is open.")
                    return True
        except (ValueError, IndexError): continue
    log_with_timestamp("Market is closed.")
    return False

async def get_full_contract_details(ib: IB, contract):
    details = await ib.reqContractDetailsAsync(contract)
    return details[0] if details else None

async def build_option_chain(ib: IB, symbol: str, exchange: str):
    log_with_timestamp(f"Fetching option chain for {symbol} on {exchange}...")
    try:
        # We need conIds to build combo legs, so we request full details
        contracts_details = await ib.reqContractDetailsAsync(FuturesOption(symbol, exchange=exchange, currency='USD', multiplier='37500'))
        if not contracts_details: return None
        
        strikes_by_expiration = {}
        for c in contracts_details:
            exp, strike = c.contract.lastTradeDateOrContractMonth, c.contract.strike
            if exp not in strikes_by_expiration: strikes_by_expiration[exp] = set()
            strikes_by_expiration[exp].add(strike)
            
        for exp in strikes_by_expiration:
            strikes_by_expiration[exp] = sorted(list(strikes_by_expiration[exp]))
        expirations = sorted(strikes_by_expiration.keys())
        
        return {'exchange': exchange, 'expirations': expirations, 'strikes_by_expiration': strikes_by_expiration}
    except Exception as e:
        log_with_timestamp(f"Failed to build option chain on {exchange}: {e}")
        return None

def get_expiration_details(chain: dict, config: dict) -> dict | None:
    """Extracts expiration date, days to expiry, and strikes from the chain."""
    today_str = datetime.now().strftime('%Y%m%d')
    upcoming_expirations = [exp for exp in sorted(chain['expirations']) if exp > today_str]
    exp_choice = config.get('expiration_to_use', 'next')
    exp_index = 1 if exp_choice == 'next' and len(upcoming_expirations) > 1 else 0
    if not upcoming_expirations or len(upcoming_expirations) <= exp_index:
        log_with_timestamp(f"Not enough upcoming expirations for '{exp_choice}'.")
        return None
    
    chosen_expiration = upcoming_expirations[exp_index]
    days_to_exp = (datetime.strptime(chosen_expiration, '%Y%m%d').date() - datetime.now().date()).days
    strikes = chain['strikes_by_expiration'][chosen_expiration]
    
    return {'exp_date': chosen_expiration, 'days_to_exp': days_to_exp, 'strikes': strikes}


# --- 4. Main Execution Loop ---

async def main_runner():
    ib = IB()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

    while True:
        try:
            with open(config_path, 'r') as f: config = json.load(f)
            conn_settings = config.get('connection', {})
            host = conn_settings.get('host', '127.0.0.1')
            port = conn_settings.get('port', 7497)
            
            if not ib.isConnected():
                client_id = conn_settings.get('clientId') or random.randint(1, 1000)
                log_with_timestamp(f"Connecting to IBKR at {host}:{port} with Client ID: {client_id}...")
                await ib.connectAsync(host, port, clientId=client_id)
                await ib.reqCurrentTimeAsync()

            future_details = await get_full_contract_details(ib, Future(config['symbol'], exchange=config['exchange']))
            if not future_details: raise ConnectionError("Could not find a valid front-month future.")

            if is_market_open(future_details, config['exchange_timezone']):
                signal = get_prediction_from_api(api_url="http://127.0.0.1:8000")
                if not signal: raise ValueError("Could not get a valid signal from the API.")

                await manage_existing_positions(ib, config, signal)

                ticker_list = await ib.reqTickersAsync(future_details.contract)
                if not ticker_list or util.isNan(ticker_list[0].marketPrice()):
                    log_with_timestamp("Could not get valid market price for underlying. Skipping cycle.")
                    await asyncio.sleep(60)
                    continue
                current_underlying_price = ticker_list[0].marketPrice()

                chain = await build_option_chain(ib, config['symbol'], config['exchange'])
                if not chain: chain = await build_option_chain(ib, config['symbol'], 'ICE')
                if not chain: raise ConnectionError("Could not build option chain.")
                
                if signal['prediction_type'] == 'DIRECTIONAL':
                    await execute_directional_strategy(ib, config, signal, chain, current_underlying_price)
                elif signal['prediction_type'] == 'VOLATILITY':
                    await execute_volatility_strategy(ib, config, signal, chain, current_underlying_price)

            wait_hours = config.get('trade_interval_hours', 2)
            log_with_timestamp(f"\nTrading cycle complete. Waiting for {wait_hours} hours...")
            
            wait_seconds = wait_hours * 3600
            heartbeat_interval = 300
            end_time = time.time() + wait_seconds
            
            while time.time() < end_time:
                sleep_duration = min(heartbeat_interval, end_time - time.time())
                if sleep_duration <= 0: break
                await asyncio.sleep(sleep_duration)
                if time.time() < end_time and ib.isConnected():
                    await ib.reqCurrentTimeAsync()
                    log_with_timestamp("Heartbeat sent. Connection alive.")

        except (ConnectionError, OSError) as e:
            log_with_timestamp(f"Connection error: {e}. Handling reconnect...")
            if ib.isConnected(): ib.disconnect()
            
            log_with_timestamp("Attempting to reconnect every 2 minutes...")
            log_with_timestamp("ACTION REQUIRED: Please ensure the IBKR Gateway is running and you are logged in.")
            
            while True:
                await asyncio.sleep(120)
                try:
                    client_id = conn_settings.get('clientId') or random.randint(1, 1000)
                    log_with_timestamp(f"Attempting to reconnect to {host}:{port} with Client ID: {client_id}...")
                    await ib.connectAsync(host, port, clientId=client_id, timeout=10)
                    if ib.isConnected():
                        log_with_timestamp("Reconnection successful!")
                        break
                except Exception as reconnect_e:
                    log_with_timestamp(f"Reconnect attempt failed: {reconnect_e}. Will try again in 2 minutes...")

        except Exception as e:
            log_with_timestamp(f"An unexpected critical error occurred: {e}. Restarting logic in 1 min...")
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        util.run(main_runner())
    except (KeyboardInterrupt, SystemExit):
        log_with_timestamp("Script stopped manually.")

