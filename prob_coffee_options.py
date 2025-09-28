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
        if isinstance(p.contract, FuturesOption) and p.contract.symbol == config['symbol'] and p.position != 0
    ]

    if not coffee_positions:
        log_with_timestamp("No existing positions to manage.")
        return

    for pos in coffee_positions:
        misaligned = False
        reason = ""
        
        # This is a simplified check. A real system might check the net delta of all positions.
        # For now, we close any position that is overtly against the new signal's direction.
        if new_signal['prediction_type'] == 'DIRECTIONAL':
            # If we have a bullish signal, we don't want to hold bearish single-leg positions (like long puts or short calls).
            if new_signal['direction'] == 'BULLISH' and ((pos.contract.right == 'P' and pos.position > 0) or (pos.contract.right == 'C' and pos.position < 0)):
                misaligned = True
                reason = "New signal is BULLISH, closing bearish position."
            # If we have a bearish signal, we don't want to hold bullish single-leg positions.
            elif new_signal['direction'] == 'BEARISH' and ((pos.contract.right == 'C' and pos.position > 0) or (pos.contract.right == 'P' and pos.position < 0)):
                misaligned = True
                reason = "New signal is BEARISH, closing bullish position."
        
        if misaligned:
            log_with_timestamp(f"Position {pos.contract.localSymbol} ({pos.position}) is misaligned. {reason}")
            action = 'BUY' if pos.position < 0 else 'SELL'
            order = MarketOrder(action, abs(pos.position))
            trade = ib.placeOrder(pos.contract, order)
            trades_to_close.append(trade)

    if trades_to_close:
        log_with_timestamp(f"Waiting for {len(trades_to_close)} misaligned position(s) to close...")
        for trade in trades_to_close:
            await wait_for_fill(trade)
    else:
        log_with_timestamp("All existing positions are aligned with the new signal.")


async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float):
    """
    Executes a risk-defined vertical spread based on the signal.
    - BULLISH: Bull Call Spread (Buy ATM Call, Sell OTM Call)
    - BEARISH: Bear Put Spread (Buy ATM Put, Sell OTM Put)
    """
    log_with_timestamp(f"--- Executing Directional Spread Strategy: {signal['direction']} ---")
    strategy_params = config['strategy']
    
    exp_details = get_expiration_details(chain, config)
    if not exp_details: return

    strikes = exp_details['strikes']
    
    # Find the At-The-Money (ATM) strike
    atm_strike = min(strikes, key=lambda s: abs(s - underlying_price))
    
    long_leg_strike = atm_strike
    short_leg_strike = None

    if signal['direction'] == 'BULLISH':
        # For a Bull Call Spread, we sell a call with a HIGHER strike
        otm_strikes = [s for s in strikes if s > long_leg_strike]
        if not otm_strikes:
            log_with_timestamp("Could not find OTM strike for Bull Call Spread. Aborting.")
            return
        short_leg_strike = otm_strikes[0]
        long_leg_right, short_leg_right = 'C', 'C'
        
    elif signal['direction'] == 'BEARISH':
        # For a Bear Put Spread, we sell a put with a LOWER strike
        otm_strikes = [s for s in strikes if s < long_leg_strike]
        if not otm_strikes:
            log_with_timestamp("Could not find OTM strike for Bear Put Spread. Aborting.")
            return
        short_leg_strike = otm_strikes[-1] # Farthest OTM
        long_leg_right, short_leg_right = 'P', 'P'

    log_with_timestamp(f"Constructing {signal['direction']} spread: BUY {long_leg_strike}{long_leg_right}, SELL {short_leg_strike}{short_leg_right}")

    # Create contracts for both legs of the spread
    long_leg_contract = FuturesOption(
        config['symbol'], exp_details['exp_date'], long_leg_strike, long_leg_right,
        chain['exchange'], currency='USD', multiplier='37500'
    )
    short_leg_contract = FuturesOption(
        config['symbol'], exp_details['exp_date'], short_leg_strike, short_leg_right,
        chain['exchange'], currency='USD', multiplier='37500'
    )

    await ib.qualifyContractsAsync(long_leg_contract, short_leg_contract)

    # Place orders for both legs
    trades = []
    
    # Leg 1: Buy the long leg
    long_order = MarketOrder('BUY', strategy_params['quantity'])
    long_trade = ib.placeOrder(long_leg_contract, long_order)
    trades.append(long_trade)
    log_with_timestamp(f"Placed order {long_trade.order.orderId}: BUY {long_leg_contract.localSymbol}")

    # Leg 2: Sell the short leg
    short_order = MarketOrder('SELL', strategy_params['quantity'])
    short_trade = ib.placeOrder(short_leg_contract, short_order)
    trades.append(short_trade)
    log_with_timestamp(f"Placed order {short_trade.order.orderId}: SELL {short_leg_contract.localSymbol}")

    for trade in trades:
        await wait_for_fill(trade)


async def execute_volatility_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float):
    """
    Executes a defined-risk volatility strategy.
    - HIGH Volatility: Long Straddle (Buy ATM Call & Put) - Defined Risk
    - LOW Volatility: Iron Condor (Sell OTM Put Spread & Call Spread) - Defined Risk
    """
    log_with_timestamp(f"--- Executing Volatility Strategy for {signal['level']} Volatility ---")
    strategy = config['strategy']
    exp_details = get_expiration_details(chain, config)
    if not exp_details: return

    strikes = exp_details['strikes']
    atm_strike = min(strikes, key=lambda s: abs(s - underlying_price))

    trades = []
    
    if signal['level'] == 'HIGH':
        log_with_timestamp(f"Strategy: Long Straddle at {atm_strike} strike.")
        contracts = [
            FuturesOption(config['symbol'], exp_details['exp_date'], atm_strike, r, chain['exchange'], currency='USD', multiplier='37500')
            for r in ['C', 'P']
        ]
        await ib.qualifyContractsAsync(*contracts)
        for contract in contracts:
            order = MarketOrder('BUY', strategy['quantity'])
            trade = ib.placeOrder(contract, order)
            trades.append(trade)
            log_with_timestamp(f"Placed Straddle Leg: BUY {strategy['quantity']} {contract.localSymbol}")

    elif signal['level'] == 'LOW':
        log_with_timestamp("Strategy: Iron Condor.")
        # --- Iron Condor Logic ---
        # 1. Find strikes for the short legs (e.g., next strike out from ATM)
        short_put_strike = next((s for s in reversed(strikes) if s < atm_strike), None)
        short_call_strike = next((s for s in strikes if s > atm_strike), None)

        if not short_put_strike or not short_call_strike:
            log_with_timestamp("Not enough strikes to build Iron Condor. Aborting.")
            return

        # 2. Find strikes for the long legs (wings) to define risk
        long_put_strike = next((s for s in reversed(strikes) if s < short_put_strike), None)
        long_call_strike = next((s for s in strikes if s > short_call_strike), None)

        if not long_put_strike or not long_call_strike:
            log_with_timestamp("Not enough strikes to build wings for Iron Condor. Aborting.")
            return

        log_with_timestamp(f"Condor Legs: BUY {long_put_strike}P, SELL {short_put_strike}P, SELL {short_call_strike}C, BUY {long_call_strike}C")

        # 3. Define the 4 contracts
        condor_contracts = {
            'long_put': FuturesOption(config['symbol'], exp_details['exp_date'], long_put_strike, 'P', chain['exchange'], currency='USD', multiplier='37500'),
            'short_put': FuturesOption(config['symbol'], exp_details['exp_date'], short_put_strike, 'P', chain['exchange'], currency='USD', multiplier='37500'),
            'short_call': FuturesOption(config['symbol'], exp_details['exp_date'], short_call_strike, 'C', chain['exchange'], currency='USD', multiplier='37500'),
            'long_call': FuturesOption(config['symbol'], exp_details['exp_date'], long_call_strike, 'C', chain['exchange'], currency='USD', multiplier='37500')
        }
        
        await ib.qualifyContractsAsync(*condor_contracts.values())

        # 4. Place the 4 orders
        orders = {
            'long_put': MarketOrder('BUY', strategy['quantity']),
            'short_put': MarketOrder('SELL', strategy['quantity']),
            'short_call': MarketOrder('SELL', strategy['quantity']),
            'long_call': MarketOrder('BUY', strategy['quantity'])
        }

        for name, contract in condor_contracts.items():
            order = orders[name]
            trade = ib.placeOrder(contract, order)
            trades.append(trade)
            log_with_timestamp(f"Placed Condor Leg: {order.action} {strategy['quantity']} {contract.localSymbol}")

    # Wait for all legs to fill
    for trade in trades:
        await wait_for_fill(trade)


# --- 3. Helper & Utility Functions ---

def log_with_timestamp(message: str):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def log_trade_to_ledger(trade: Trade):
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv')
    file_exists = os.path.isfile(ledger_path)
    if trade.orderStatus.status != OrderStatus.Filled: return
    contract_multiplier = int(trade.contract.multiplier) if trade.contract.multiplier and trade.contract.multiplier.isdigit() else 37500
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'local_symbol': trade.contract.localSymbol, 'action': trade.order.action, 'quantity': trade.orderStatus.filled,
        'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * contract_multiplier,
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

async def wait_for_fill(trade: Trade, timeout: int = 0):
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
            
            # Use current date for comparison if the session date matches today
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
        contracts = await ib.reqContractDetailsAsync(FuturesOption(symbol, exchange=exchange, currency='USD', multiplier='37500'))
        if not contracts: return None
        strikes_by_expiration = {}
        for c in contracts:
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


async def find_best_mispriced_option(ib: IB, config: dict, strategy: dict, option_type: str, all_strikes: list, days_to_exp: int, underlying_price: float, expiration_date: str):
    log_with_timestamp(f"--- Searching for best overpriced {option_type} to sell ---")
    best_option, highest_overpricing = None, -1
    contracts_to_check = [
        FuturesOption(config['symbol'], expiration_date, s, option_type, config['exchange'], currency='USD', multiplier='37500')
        for s in all_strikes
    ]
    await ib.qualifyContractsAsync(*contracts_to_check)
    tickers = await ib.reqTickersAsync(*contracts_to_check)
    for ticker in tickers:
        if util.isNan(ticker.bid) or ticker.bid <= 0: continue
        market_price, strike_price = ticker.bid, ticker.contract.strike
        theoretical_price = get_theoretical_option_price(underlying_price, strike_price, days_to_exp, option_type)
        if theoretical_price > 0:
            overpricing_pct = (market_price / theoretical_price) - 1
            if overpricing_pct > highest_overpricing:
                highest_overpricing = overpricing_pct
                best_option = {'contract': ticker.contract, 'market_price': market_price, 'overpricing_pct': overpricing_pct}
    
    if best_option and best_option['overpricing_pct'] >= strategy.get('min_overpricing_pct', 0.15):
        log_with_timestamp(f"Found best option to sell: {best_option['contract'].localSymbol} is {best_option['overpricing_pct']:.2%} overpriced.")
        return best_option['contract'], round(best_option['market_price'], 4)
    
    log_with_timestamp("No sufficiently overpriced options found.")
    return None, 0


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
                # 1. Get Signal
                signal = get_prediction_from_api(api_url="http://127.0.0.1:8000") # URL from prediction_api.py
                if not signal: raise ValueError("Could not get a valid signal from the API.")

                # 2. Manage Positions based on new signal
                await manage_existing_positions(ib, config, signal)

                # 3. Get Market Data
                ticker_list = await ib.reqTickersAsync(future_details.contract)
                current_underlying_price = ticker_list[0].marketPrice()
                chain = await build_option_chain(ib, config['symbol'], config['exchange'])
                if not chain: chain = await build_option_chain(ib, config['symbol'], 'ICE') # Fallback
                if not chain: raise ConnectionError("Could not build option chain.")
                
                # 4. Execute New Strategy
                if signal['prediction_type'] == 'DIRECTIONAL':
                    await execute_directional_strategy(ib, config, signal, chain, current_underlying_price)
                elif signal['prediction_type'] == 'VOLATILITY':
                    await execute_volatility_strategy(ib, config, signal, chain, current_underlying_price)
                else:
                    log_with_timestamp(f"Unknown prediction type: {signal['prediction_type']}")

            wait_hours = config.get('trade_interval_hours', 2)
            log_with_timestamp(f"\nTrading cycle complete. Waiting for {wait_hours} hours...")
            
            # --- FIX: Replace ib.sleep() with a more robust heartbeat loop ---
            wait_seconds = wait_hours * 3600
            heartbeat_interval = 300  # 5 minutes
            end_time = time.time() + wait_seconds
            
            while time.time() < end_time:
                sleep_duration = min(heartbeat_interval, end_time - time.time())
                if sleep_duration <= 0:
                    break
                await asyncio.sleep(sleep_duration)
                # Send a heartbeat request to keep the connection alive
                if time.time() < end_time and ib.isConnected():
                    await ib.reqCurrentTimeAsync()
                    log_with_timestamp("Heartbeat sent. Connection alive.")
            # --- END FIX ---

        except (ConnectionError, OSError) as e:
            log_with_timestamp(f"Connection error: {e}. Handling reconnect...")
            if ib.isConnected():
                ib.disconnect()
            
            # --- IMPROVED RECONNECT LOGIC ---
            # This loop will patiently wait for the user to manually log back into the Gateway.
            log_with_timestamp("Attempting to reconnect every 2 minutes...")
            log_with_timestamp("ACTION REQUIRED: Please ensure the IBKR Gateway is running and you are logged in.")
            
            while True:
                await asyncio.sleep(120) # Wait for 2 minutes between attempts
                try:
                    client_id = conn_settings.get('clientId') or random.randint(1, 1000)
                    log_with_timestamp(f"Attempting to reconnect to {host}:{port} with Client ID: {client_id}...")
                    await ib.connectAsync(host, port, clientId=client_id, timeout=10)
                    if ib.isConnected():
                        log_with_timestamp("Reconnection successful!")
                        break # Exit the reconnect loop and continue the main script
                except Exception as reconnect_e:
                    log_with_timestamp(f"Reconnect attempt failed: {reconnect_e}. Will try again in 2 minutes...")
            # --- END IMPROVED LOGIC ---

        except Exception as e:
            log_with_timestamp(f"An unexpected critical error occurred: {e}. Restarting logic in 1 min...")
            if ib.isConnected():
                ib.disconnect()
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        util.run(main_runner())
    except (KeyboardInterrupt, SystemExit):
        log_with_timestamp("Script stopped manually.")




