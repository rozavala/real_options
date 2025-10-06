import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, timedelta
import pytz

from ib_insync import *

# --- Custom Modules ---
from trading.connection import connect_to_ibkr, disconnect_from_ibkr, send_heartbeat
from trading.contracts import get_active_futures, build_option_chain, get_expiration_details
from trading.execution import place_combo_order, wait_for_fill
from trading.risk import is_trade_sane, monitor_positions_for_risk
from notifications import send_pushover_notification
from signal_provider import get_trading_signals

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

# --- Strategy Execution Logic ---

async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, future_contract: Contract) -> Trade | None:
    """
    Constructs the legs for a directional spread (Bull Call or Bear Put)
    and places the order.
    """
    logging.info(f"--- Preparing {signal['direction']} Spread for {future_contract.localSymbol} ---")
    
    # Get strategy parameters from config
    tuning = config.get('strategy_tuning', {})
    spread_width = tuning.get('vertical_spread_strikes_apart', 2)

    # Get expiration and strike data
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details:
        logging.error("Could not get valid expiration details. Skipping strategy.")
        return None
    strikes = exp_details['strikes']

    # Get underlying price to find ATM strike
    ticker = ib.reqMktData(future_contract, '', False, False)
    await asyncio.sleep(1)
    underlying_price = ticker.marketPrice()
    ib.cancelMktData(future_contract)
    
    if util.isNan(underlying_price):
        logging.error(f"Failed to get market price for {future_contract.localSymbol}. Cannot determine ATM strike.")
        return None
        
    logging.info(f"Current price for {future_contract.localSymbol} is {underlying_price}")
    
    try:
        # Find the index of the strike closest to the underlying price
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
    except (ValueError, IndexError):
        logging.error("Could not find ATM strike in the provided chain.")
        return None

    # Define the legs for the spread
    legs_def = []
    if signal['direction'] == 'BULLISH': # Bull Call Spread
        if atm_idx + spread_width >= len(strikes):
            logging.warning("Not enough strikes available above the ATM strike for a Bull Call Spread.")
            return None
        # Buy ATM call, Sell OTM call
        legs_def = [('C', 'BUY', strikes[atm_idx]), ('C', 'SELL', strikes[atm_idx + spread_width])]
        action = 'BUY' # It's a debit spread
    
    elif signal['direction'] == 'BEARISH': # Bear Put Spread
        if atm_idx - spread_width < 0:
            logging.warning("Not enough strikes available below the ATM strike for a Bear Put Spread.")
            return None
        # Buy ATM put, Sell OTM put
        legs_def = [('P', 'BUY', strikes[atm_idx]), ('P', 'SELL', strikes[atm_idx - spread_width])]
        action = 'BUY' # It's a debit spread

    if not legs_def:
        logging.error("Could not define legs for the strategy.")
        return None

    # Place the combo order
    trade = await place_combo_order(ib, config, action, legs_def, exp_details, chain)
    if trade:
        await wait_for_fill(ib, trade, config, reason=f"{signal['direction']} Spread")
    return trade


# --- Market Hours Utilities ---

def is_market_open(contract_details: ContractDetails, exchange_timezone_str: str) -> bool:
    """
    Checks if the market for a given contract is currently open.
    """
    if not contract_details or not contract_details.liquidHours:
        logging.warning("No liquid hours data available. Assuming market is closed.")
        return False

    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)

    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str:
            continue
        try:
            # Format is YYYYMMDD:HHMM-YYYYMMDD:HHMM
            start_str, end_str = session_str.split('-')
            start_dt_str, start_hm_str = start_str.split(':')
            end_hm_str = end_str.split(':')[1]

            start_dt = datetime.strptime(start_dt_str, '%Y%m%d').date()
            start_hm = datetime.strptime(start_hm_str, '%H%M').time()
            end_hm = datetime.strptime(end_hm_str, '%H%M').time()

            # Check if today is the session day and current time is within the session hours
            if now_tz.date() == start_dt and start_hm <= now_tz.time() < end_hm:
                logging.info(f"Market is OPEN. Current time {now_tz.time()} is within session {start_hm}-{end_hm}.")
                return True
        except (ValueError, IndexError) as e:
            logging.error(f"Could not parse liquid hours segment: '{session_str}'. Error: {e}")
            continue

    logging.info("Market is CLOSED.")
    return False

def calculate_wait_until_market_open(contract_details: ContractDetails, exchange_timezone_str: str) -> float:
    """
    Calculates the number of seconds to wait until the next market open.
    """
    if not contract_details or not contract_details.liquidHours:
        return 3600 # Default to 1 hour if no data

    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    next_open_dt = None

    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str:
            continue
        try:
            session_start_str = session_str.split('-')[0]
            # Create a timezone-aware datetime object for the session start
            session_start_dt = tz.localize(datetime.strptime(session_start_str, '%Y%m%d:%H%M'))

            # If this session starts in the future and is earlier than the next one we've found
            if session_start_dt > now_tz and (next_open_dt is None or session_start_dt < next_open_dt):
                next_open_dt = session_start_dt
        except (ValueError, IndexError):
            continue

    if next_open_dt:
        wait_seconds = (next_open_dt - now_tz).total_seconds() + 60 # Add a 1-minute buffer
        logging.info(f"Next market open is at {next_open_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}. Waiting for {wait_seconds / 3600:.2f} hours.")
        return wait_seconds

    # If no future session found for today, wait for the scheduled time tomorrow
    return None


# --- Main Application Logic ---

async def main_runner():
    """
    The main execution loop of the trading bot.
    """
    ib = IB()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

    # Task handles for background processes
    monitor_task = None
    heartbeat_task = None
    
    while True:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # --- Connection Phase ---
            if not ib.isConnected():
                # Cancel old tasks if they exist
                if monitor_task and not monitor_task.done(): monitor_task.cancel()
                if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()

                if not await connect_to_ibkr(ib, config):
                    logging.info("Will retry connection in 1 minute.")
                    await asyncio.sleep(60)
                    continue

                # Start background tasks once connected
                monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
                heartbeat_task = asyncio.create_task(send_heartbeat(ib))

            # --- Market Hours Check ---
            active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=1)
            if not active_futures:
                logging.error("Could not find any active futures contracts. Waiting and retrying.")
                await asyncio.sleep(3600)
                continue

            front_month_details_list = await ib.reqContractDetailsAsync(active_futures[0])
            if not front_month_details_list:
                 logging.error("Could not get contract details for front month future. Waiting and retrying.")
                 await asyncio.sleep(3600)
                 continue

            front_month_details = front_month_details_list[0]

            if not is_market_open(front_month_details, config['exchange_timezone']):
                wait_time = calculate_wait_until_market_open(front_month_details, config['exchange_timezone'])
                if wait_time:
                    await asyncio.sleep(wait_time)
                else:
                    # If we can't calculate wait time, just wait for the scheduled time
                    pass # The logic below will handle this
                # Continue to the next loop iteration to re-check market hours
                continue

            # --- Signal & Trading Phase ---
            # This section will only run if the market is open.
            
            logging.info("Market is open. Starting trading cycle.")

            # Get fresh trading signals from our integrated provider
            signals = get_trading_signals(config)
            if not signals:
                logging.error("Could not get trading signals. Ending cycle and will retry later.")
                # Wait for a bit before retrying to avoid spamming a failing API
                await asyncio.sleep(300)
                continue

            # Get all active futures for matching with signals
            all_active_futures = await get_active_futures(ib, config['symbol'], config['exchange'])

            for signal in signals:
                # Find the future contract that matches the signal's contract month
                future = next((f for f in all_active_futures if f.lastTradeDateOrContractMonth.startswith(signal.get("contract_month", ""))), None)
                if not future:
                    logging.warning(f"No active future found for signal month {signal.get('contract_month')}. Skipping.")
                    continue

                logging.info(f"\n===== Processing Signal for {future.localSymbol} =====")
                
                # Pre-trade sanity check
                if not is_trade_sane(signal, config):
                    continue

                # Build the option chain for the target future
                chain = await build_option_chain(ib, future)
                if not chain:
                    logging.error(f"Could not build option chain for {future.localSymbol}. Skipping.")
                    continue
                
                # Execute strategy based on signal type
                if signal['prediction_type'] == 'DIRECTIONAL':
                    await execute_directional_strategy(ib, config, signal, chain, future)
                # Volatility strategies can be added here later
                # elif signal['prediction_type'] == 'VOLATILITY':
                #     await execute_volatility_strategy(...)
                else:
                    logging.warning(f"Signal type '{signal['prediction_type']}' is not yet supported.")

            # --- End of Cycle ---
            logging.info("\nTrading cycle complete.")

            # Calculate wait time until next scheduled execution
            tz = pytz.timezone(config['exchange_timezone'])
            now_tz = datetime.now(tz)
            h, m = map(int, config.get('trade_execution_time_ny', '09:30').split(':'))

            next_trade_time = now_tz.replace(hour=h, minute=m, second=0, microsecond=0)
            if now_tz >= next_trade_time:
                next_trade_time += timedelta(days=1)

            wait_seconds = (next_trade_time - now_tz).total_seconds()
            logging.info(f"Waiting {wait_seconds / 3600:.2f} hours until the next scheduled run at {next_trade_time.strftime('%Y-%m-%d %H:%M:%S %Z')}.")
            await asyncio.sleep(wait_seconds)

        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            logging.error(f"A connection error occurred: {e}. Attempting to reconnect...")
            send_pushover_notification(config, "Bot Status: Connection Error", str(e))
            disconnect_from_ibkr(ib)
            await asyncio.sleep(60) # Wait before trying to reconnect
        except asyncio.CancelledError:
            logging.info("Main task was cancelled. Shutting down.")
            break
        except Exception as e:
            msg = f"An unexpected critical error occurred in the main loop: {e}"
            logging.critical(f"{msg}\n{traceback.format_exc()}")
            send_pushover_notification(config, "Bot Status: CRITICAL ERROR", msg)
            disconnect_from_ibkr(ib)
            logging.info("Restarting main loop in 1 minute...")
            await asyncio.sleep(60)

    # --- Cleanup ---
    if monitor_task and not monitor_task.done(): monitor_task.cancel()
    if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
    disconnect_from_ibkr(ib)
    logging.info("Script has shut down.")


if __name__ == "__main__":
    try:
        # util.run() is a convenient wrapper for asyncio.run()
        # that works well in different environments.
        util.run(main_runner())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Script stopped manually by user.")