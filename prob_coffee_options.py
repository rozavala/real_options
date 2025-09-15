import asyncio
import csv
import json
import os
import random
import time
from datetime import datetime

import pytz
from ib_insync import *


def log_with_timestamp(message: str):
    """Prints a message with a timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def log_trade_to_ledger(trade: Trade):
    """Appends a record of a filled trade to the trade_ledger.csv file."""
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv')
    file_exists = os.path.isfile(ledger_path)

    if trade.orderStatus.status != OrderStatus.Filled:
        return

    contract_multiplier = int(trade.contract.multiplier) if trade.contract.multiplier and trade.contract.multiplier.isdigit() else 37500

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'local_symbol': trade.contract.localSymbol,
        'action': trade.order.action,
        'quantity': trade.orderStatus.filled,
        'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * contract_multiplier / 100,
        'order_id': trade.order.orderId
    }

    try:
        with open(ledger_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        log_with_timestamp(f"Logged trade to ledger: {row['action']} {row['quantity']} {row['local_symbol']} @ {row['avg_fill_price']}")
    except Exception as e:
        log_with_timestamp(f"Error writing to trade ledger: {e}")


def is_market_open(contract_details, exchange_timezone_str: str):
    """
    Checks if the market for a given contract is currently open.
    """
    if not contract_details or not contract_details.liquidHours:
        log_with_timestamp("Warning: Could not get liquid hours for contract.")
        return False

    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    
    sessions_str = contract_details.liquidHours.split(';')

    for session_str in sessions_str:
        if not session_str: continue
        try:
            start_str, end_str = session_str.split('-')
            start_dt_str, start_hm_str = start_str.split(':')
            end_dt_str, end_hm_str = end_str.split(':')
            start_datetime = tz.localize(datetime.strptime(f"{start_dt_str}{start_hm_str}", '%Y%m%d%H%M'))
            end_datetime = tz.localize(datetime.strptime(f"{end_dt_str}{end_hm_str}", '%Y%m%d%H%M'))
            if start_datetime <= now_tz < end_datetime:
                log_with_timestamp(f"Market is open. Current session: {start_datetime.strftime('%Y-%m-%d %H:%M')} to {end_datetime.strftime('%Y-%m-%d %H:%M')} {exchange_timezone_str}")
                return True
        except ValueError as e:
            log_with_timestamp(f"Warning: Could not parse trading session '{session_str}'. Error: {e}")
    
    log_with_timestamp("Market is closed. No active session found for the current time.")
    return False


async def get_full_contract_details(ib: IB, contract):
    """Gets the full contract details for a given contract."""
    details = await ib.reqContractDetailsAsync(contract)
    return details[0] if details else None


async def find_liquid_option(ib: IB, config: dict, strategy: dict, target_strike: float, option_type: str, available_strikes: list):
    """
    Finds the nearest liquid option to a target strike.
    """
    log_with_timestamp(f"\n--- Searching for a liquid option near strike {target_strike:.2f} ---")
    sorted_strikes = sorted(available_strikes, key=lambda x: abs(x - target_strike))
    option_exchange = strategy.get('option_exchange', config['exchange'])

    for strike in sorted_strikes[:10]:
        log_with_timestamp(f"Checking liquidity for strike: {strike}...")
        contract = FuturesOption(
            symbol=config['symbol'], lastTradeDateOrContractMonth=strategy['expiration_date'],
            strike=strike, right=option_type, exchange=option_exchange
        )
        await ib.qualifyContractsAsync(contract)
        ticker_list = await ib.reqTickersAsync(contract)
        ticker = ticker_list[0] if ticker_list else None
        
        if ticker:
            log_with_timestamp(f"Market data for strike {strike}: Bid={ticker.bid}, Ask={ticker.ask}, Last={ticker.last}")
            if not util.isNan(ticker.bid) and not util.isNan(ticker.ask) and ticker.bid > 0:
                limit_price = ticker.bid
                log_with_timestamp(f"Found liquid option! Strike: {strike}, Using reliable BID price: {limit_price:.2f}")
                return contract, round(limit_price, 2)
            else:
                log_with_timestamp(f"Strike {strike} is illiquid (no valid bid/ask spread). Trying next closest.")
        else:
            log_with_timestamp(f"No ticker data for strike {strike}. Trying next closest.")
    
    log_with_timestamp("--- Could not find any liquid options with an active bid/ask spread. ---")
    return None, 0


async def build_option_chain(ib: IB, symbol: str, exchange: str):
    """
    Builds a precise option chain by mapping strikes to their valid expirations.
    """
    log_with_timestamp(f"Fetching contract details for {symbol} on {exchange}...")
    try:
        contracts = await ib.reqContractDetailsAsync(FuturesOption(symbol, exchange=exchange))
        if not contracts: return None
        strikes_by_expiration = {}
        for c in contracts:
            exp, strike = c.contract.lastTradeDateOrContractMonth, c.contract.strike
            if exp not in strikes_by_expiration: strikes_by_expiration[exp] = set()
            strikes_by_expiration[exp].add(strike)
        for exp in strikes_by_expiration:
            strikes_by_expiration[exp] = sorted(list(strikes_by_expiration[exp]))
        expirations = sorted(strikes_by_expiration.keys())
        log_with_timestamp(f"Successfully built option chain from {len(contracts)} contracts.")
        return {'exchange': exchange, 'expirations': expirations, 'strikes_by_expiration': strikes_by_expiration}
    except Exception as e:
        log_with_timestamp(f"Failed to build option chain on {exchange}: {e}")
        return None


async def main_runner():
    """The main, continuous loop that connects, runs logic, and handles errors."""
    ib = IB()
    config = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')

    while True:
        try:
            if not ib.isConnected():
                with open(config_path, 'r') as f: config = json.load(f)
                client_id = random.randint(1, 1000)
                log_with_timestamp(f"Connecting to IBKR with Client ID: {client_id}...")
                await ib.connectAsync('127.0.0.1', 7497, clientId=client_id)
                log_with_timestamp("Successfully connected.")
                log_with_timestamp("Synchronizing with server...")
                await ib.reqCurrentTimeAsync()
                log_with_timestamp("Connection synced.")

            # --- 0. CANCEL ALL ORDERS GLOBALLY ---
            log_with_timestamp("--- Issuing Global Cancel for All Open Orders ---")
            ib.reqGlobalCancel()
            log_with_timestamp("Waiting for order cancellations to be confirmed...")
            for _ in range(10): # Try for up to 5 seconds
                if not await ib.reqAllOpenOrdersAsync(): break
                await asyncio.sleep(0.5)
            log_with_timestamp("All stale orders successfully cancelled.")
            log_with_timestamp("-------------------------------------------------\n")

            # --- 1. MARKET ANALYSIS (Moved Up) ---
            log_with_timestamp("--- Finding front-month future for KC ---")
            future_details = await get_full_contract_details(ib, Future(config['symbol'], exchange=config['exchange']))
            if not future_details: raise ConnectionError("Could not find a valid front-month future.")
            log_with_timestamp(f"Found front-month future: {future_details.contract.localSymbol}")

            # --- 2. CHECK MARKET HOURS AND DECIDE ACTION ---
            if is_market_open(future_details, config['exchange_timezone']):
                # --- 2a. MARKET IS OPEN: Close Positions and Trade ---
                log_with_timestamp("--- Closing All Open Positions ---")
                positions = await ib.reqPositionsAsync()
                close_trades = []
                for pos in positions:
                    if isinstance(pos.contract, FuturesOption) and pos.contract.symbol == config['symbol'] and pos.position != 0:
                        contract = pos.contract
                        if not contract.exchange: contract.exchange = config.get('option_exchange', config['exchange'])
                        action = 'BUY' if pos.position < 0 else 'SELL'
                        quantity = abs(pos.position)
                        log_with_timestamp(f"Closing position: {action} {quantity} of {contract.localSymbol}")
                        order = MarketOrder(action, quantity)
                        trade = ib.placeOrder(contract, order)
                        close_trades.append(trade)
                
                if not close_trades:
                    log_with_timestamp("No open positions to close.")
                else:
                    log_with_timestamp(f"Waiting for {len(close_trades)} closing trade(s) to fill...")
                    for trade in close_trades:
                        while not trade.isDone():
                            await asyncio.sleep(1)
                        log_with_timestamp(f"Trade {trade.order.orderId} is done with status: {trade.orderStatus.status}")
                        if trade.orderStatus.status == OrderStatus.Filled:
                            log_trade_to_ledger(trade)
                log_with_timestamp("--------------------------------\n")

                # --- 2b. EXECUTE TRADING LOGIC ---
                strategy = config['strategy']
                ticker_list = await ib.reqTickersAsync(future_details.contract)
                current_underlying_price = ticker_list[0].marketPrice()
                log_with_timestamp(f"Current underlying price is: {current_underlying_price}")

                primary_exchange, fallback_exchange = config['exchange'], 'ICE'
                chain = await build_option_chain(ib, config['symbol'], primary_exchange)
                option_exchange = primary_exchange
                if not chain:
                    log_with_timestamp(f"No chain found on {primary_exchange}. Trying fallback: {fallback_exchange}...")
                    chain = await build_option_chain(ib, config['symbol'], fallback_exchange)
                    if chain: option_exchange = fallback_exchange
                if not chain: raise ConnectionError("Could not build option chain on primary or fallback exchange.")
                
                log_with_timestamp(f"Using option chain from exchange: {option_exchange}")
                strategy['option_exchange'] = option_exchange

                today_str = datetime.now().strftime('%Y%m%d')
                upcoming_expirations = [exp for exp in sorted(chain['expirations']) if exp > today_str]
                exp_choice = config.get('expiration_to_use', 'nearest')
                exp_index = 1 if exp_choice == 'next' and len(upcoming_expirations) > 1 else 0
                if not upcoming_expirations or len(upcoming_expirations) <= exp_index:
                    raise ValueError(f"Not enough upcoming expirations to select the '{exp_choice}' one.")
                chosen_expiration = upcoming_expirations[exp_index]
                strategy['expiration_date'] = chosen_expiration
                log_with_timestamp(f"Using the '{exp_choice}' option expiration: {chosen_expiration}")

                prob, offset = random.uniform(0, 1), strategy['strike_offset']
                option_type = 'P' if prob > 0.5 else 'C'
                target_strike = current_underlying_price + offset if option_type == 'P' else current_underlying_price - offset
                log_with_timestamp(f"Targeting {option_type} sell at strike ~{target_strike:.2f}")

                liquid_contract, limit_price = await find_liquid_option(
                    ib, config, strategy, target_strike, option_type, chain['strikes_by_expiration'][chosen_expiration]
                )

                if liquid_contract and limit_price > 0:
                    if config['symbol'] == 'KC': liquid_contract.strike /= 100
                    order = LimitOrder('SELL', strategy['quantity'], limit_price)
                    trade = ib.placeOrder(liquid_contract, order)
                    log_with_timestamp(f"Placed new order {trade.order.orderId}. Waiting up to 60 seconds for fill...")
                    
                    fill_wait_start = time.time()
                    fill_timeout = 60
                    while not trade.isDone():
                        await asyncio.sleep(1)
                        if time.time() - fill_wait_start > fill_timeout:
                            log_with_timestamp(f"Order {trade.order.orderId} was not filled within {fill_timeout} seconds. Canceling.")
                            ib.cancelOrder(trade.order)
                            break
                    if trade.orderStatus.status == OrderStatus.Filled:
                        log_trade_to_ledger(trade)
                else:
                    log_with_timestamp("No order placed.")
            
            else:
                log_with_timestamp("Market is closed. Skipping trading logic.")

            # --- 3. WAIT AND HEARTBEAT ---
            wait_hours = config.get('trade_interval_hours', 2)
            log_with_timestamp(f"Trading logic complete. Waiting for {wait_hours} hours...")
            wait_seconds = wait_hours * 3600
            heartbeat_interval = 300
            
            end_time = time.time() + wait_seconds
            while time.time() < end_time:
                remaining_time, sleep_duration = end_time - time.time(), min(heartbeat_interval, end_time - time.time())
                if sleep_duration <= 0: break
                await asyncio.sleep(sleep_duration)
                if time.time() < end_time:
                    await ib.reqCurrentTimeAsync()
                    log_with_timestamp("Heartbeat sent. Connection alive.")

        except (ConnectionError, OSError) as e:
            log_with_timestamp(f"A connection error occurred (likely daily reset): {e}. Attempting to reconnect in 5 minutes...")
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(300)
        except Exception as e:
            log_with_timestamp(f"An unexpected critical error occurred: {e}. Restarting logic in 1 minute...")
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        util.run(main_runner())
    except (KeyboardInterrupt, SystemExit):
        log_with_timestamp("Script stopped manually.")

