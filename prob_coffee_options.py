import json
import numpy as np
from ib_insync import *
from datetime import datetime, timedelta
import asyncio
import pytz
import os
import random

def log(message: str):
    """Prints a message with a timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def is_market_open(trading_hours_str: str, exchange_tz_str: str) -> (bool, str):
    """
    Parses IBKR's trading hours string (which can span multiple days) 
    and checks if the market is currently open.
    """
    if not trading_hours_str:
        return True, "Trading hours not available, proceeding with caution."
    try:
        exchange_tz = pytz.timezone(exchange_tz_str)
        now = datetime.now(exchange_tz)
        sessions = trading_hours_str.split(';')
        for session in sessions:
            if 'CLOSED' in session: continue
            if '-' not in session: continue
            start_str, end_str = session.split('-')
            start_date_str, start_time_str = start_str.split(':')
            start_time = exchange_tz.localize(datetime.strptime(f"{start_date_str}{start_time_str}", '%Y%m%d%H%M'))
            end_date_str = start_date_str
            if ':' in end_str:
                end_date_str, end_time_str = end_str.split(':')
            else:
                end_time_str = end_str
            end_time = exchange_tz.localize(datetime.strptime(f"{end_date_str}{end_time_str}", '%Y%m%d%H%M'))
            if start_time <= now < end_time:
                return True, f"Market is open. Current session: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%M')} {exchange_tz_str}"
        return False, "Market is closed. No active session for today."
    except Exception as e:
        log(f"Warning: Could not parse trading hours: {trading_hours_str} due to {e}")
        return True, "Could not parse trading hours, proceeding with caution."

async def close_all_option_positions(ib: IB, config: dict):
    """Finds and closes all open option positions for a given underlying symbol."""
    log("--- Closing All Open Positions ---")
    symbol = config['symbol']
    positions = await ib.reqPositionsAsync()
    closed_count = 0
    for position in positions:
        contract = position.contract
        if isinstance(contract, FuturesOption) and contract.symbol == symbol:
            quantity = abs(position.position)
            if quantity == 0: continue
            action = 'BUY' if position.position < 0 else 'SELL'
            if not contract.exchange: contract.exchange = config['exchange']
            log(f"Closing position: {action} {quantity} of {contract.localSymbol}")
            order = MarketOrder(action, quantity)
            trade = ib.placeOrder(contract, order)
            await asyncio.sleep(2)
            log(f"Close order for {contract.localSymbol} submitted. Status: {trade.orderStatus.status}")
            closed_count += 1
    if closed_count == 0: log("No open positions to close.")
    log("--------------------------------\n")

async def find_liquid_option(ib: IB, config: dict, strategy: dict, target_strike: float, option_type: str, available_strikes: list):
    """Finds the nearest liquid option to a target strike that has an active bid/ask spread."""
    log(f"--- Searching for a liquid option near strike {target_strike:.2f} ---")
    sorted_strikes = sorted(available_strikes, key=lambda x: abs(x - target_strike))
    option_exchange = strategy.get('option_exchange', config['exchange'])
    for strike in sorted_strikes[:10]:
        log(f"Checking liquidity for strike: {strike}...")
        contract = FuturesOption(symbol=config['symbol'], lastTradeDateOrContractMonth=strategy['expiration_date'], strike=strike, right=option_type, exchange=option_exchange)
        await ib.qualifyContractsAsync(contract)
        ticker = ib.reqMktData(contract, '', False, False)
        await asyncio.sleep(2)
        log(f"Market data for strike {strike}: Bid={ticker.bid}, Ask={ticker.ask}, Last={ticker.last}")
        if ticker and not util.isNan(ticker.bid) and ticker.bid > 0 and not util.isNan(ticker.ask) and ticker.ask > 0:
            limit_price = ticker.bid
            log(f"Found liquid option! Strike: {strike}, Using reliable BID price: {limit_price:.2f}")
            ib.cancelMktData(contract)
            return contract, round(limit_price, 2)
        else:
            log(f"Strike {strike} is illiquid (no valid bid/ask spread). Trying next closest.")
            ib.cancelMktData(contract)
    log("--- Could not find any liquid options with an active bid/ask spread. ---")
    return None, 0

async def find_front_month_future_details(ib: IB, symbol: str, exchange: str):
    """Finds the ContractDetails for the nearest-expiring (front-month) future contract."""
    log(f"--- Finding front-month future for {symbol} ---")
    contracts_details = await ib.reqContractDetailsAsync(Future(symbol=symbol, exchange=exchange))
    if not contracts_details: return None
    today = datetime.now().date()
    valid_contracts_details = sorted([cd for cd in contracts_details if datetime.strptime(cd.contract.lastTradeDateOrContractMonth, '%Y%m%d').date() >= today], key=lambda cd: cd.contract.lastTradeDateOrContractMonth)
    if not valid_contracts_details: return None
    front_month_detail = valid_contracts_details[0]
    log(f"Found front-month future: {front_month_detail.contract.localSymbol}")
    return front_month_detail

async def get_option_chain_details(ib: IB, symbol: str, exchange: str):
    """A faster way to get option chain details by fetching contract details and building the chain manually."""
    log(f"Fetching contract details for {symbol} on {exchange}...")
    try:
        details = await asyncio.wait_for(ib.reqContractDetailsAsync(FuturesOption(symbol=symbol, exchange=exchange)), timeout=180.0)
    except asyncio.TimeoutError:
        log("Request for contract details timed out after 3 minutes."); return None, None
    if not details: return None, None
    expirations = sorted(list(set(d.contract.lastTradeDateOrContractMonth for d in details)))
    strikes = sorted(list(set(d.contract.strike for d in details)))
    return expirations, strikes

async def run_trading_logic(ib: IB, config: dict):
    """Contains the main logic for a single trading cycle."""
    strategy = config['strategy']
    await close_all_option_positions(ib, config)
    underlying_future_details = await find_front_month_future_details(ib, config['symbol'], config['exchange'])
    if not underlying_future_details:
        log("Could not find a valid front-month future."); return
    is_open, message = is_market_open(underlying_future_details.tradingHours, config.get('exchange_timezone', 'America/New_York'))
    log(message)
    if not is_open: return
    
    ticker = ib.reqMktData(underlying_future_details.contract, '', False, False)
    await asyncio.sleep(3)
    if ticker is None or util.isNan(ticker.last):
        log("Could not get market price for the underlying future."); return
    current_underlying_price = ticker.last
    log(f"Current underlying price is: {current_underlying_price}")

    option_exchange = config['exchange']
    expirations, strikes = await get_option_chain_details(ib, config['symbol'], option_exchange)
    if not expirations:
        fallback_exchange = 'ICE'
        log(f"No chain found on {config['exchange']}. Trying fallback: {fallback_exchange}...")
        expirations, strikes = await get_option_chain_details(ib, config['symbol'], fallback_exchange)
        if expirations: option_exchange = fallback_exchange
    if not expirations:
        log("Could not build option chain on primary or fallback exchange."); return

    strategy['option_exchange'] = option_exchange
    today_str = datetime.now().strftime('%Y%m%d')
    upcoming_expirations = [exp for exp in sorted(expirations) if exp > today_str]
    if not upcoming_expirations:
        log("Could not find any valid upcoming option expirations."); return
    
    expiration_choice = strategy.get("expiration_to_use", "nearest")
    expiration_index = 1 if expiration_choice == "next" else 0
    if len(upcoming_expirations) <= expiration_index:
        expiration_index = -1
    strategy['expiration_date'] = upcoming_expirations[expiration_index]
    log(f"Using the '{expiration_choice}' option expiration: {strategy['expiration_date']}")

    prob = np.random.uniform(0, 1)
    option_type = 'P' if prob > 0.5 else 'C'
    offset = strategy['strike_offset']
    target_strike = current_underlying_price + offset if option_type == 'P' else current_underlying_price - offset
    
    liquid_contract, limit_price = await find_liquid_option(ib, config, strategy, target_strike, option_type, strikes)
    if liquid_contract and limit_price > 0:
        if config['symbol'] == 'KC': liquid_contract.strike = liquid_contract.strike / 100
        order = LimitOrder('SELL', strategy['quantity'], limit_price)
        trade = ib.placeOrder(liquid_contract, order)
        await asyncio.sleep(3)
        log(f"Order status: {trade.orderStatus.status}")

async def main():
    """Main function to run the bot continuously."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    try:
        with open(config_path, 'r') as f: config = json.load(f)
    except FileNotFoundError:
        log(f"CRITICAL: 'config.json' not found at {config_path}. Bot cannot start."); return

    ib = IB()

    while True:
        try:
            if not ib.isConnected():
                client_id = random.randint(1, 1000)
                log(f"Connecting to IBKR with Client ID: {client_id}...")
                await ib.connectAsync('127.0.0.1', 7497, clientId=client_id, timeout=10)
                log("Successfully connected.")

            await run_trading_logic(ib, config)

            log(f"Trading logic complete. Waiting for 2 hours...")
            for i in range(24): # 24 * 5 minutes = 120 minutes (2 hours)
                await asyncio.sleep(300)
                await ib.reqCurrentTimeAsync()
                log(f"Heartbeat sent. Connection alive.")

        except ConnectionRefusedError:
            log("Connection refused. Is IB Gateway running? Retrying in 5 minutes...")
            await asyncio.sleep(300)
        except asyncio.TimeoutError:
            log("Connection timed out. Retrying in 1 minute...")
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)
        except Exception as e:
            log(f"An unexpected error occurred: {e}. Attempting to reconnect in 1 minute...")
            if ib.isConnected():
                try:
                    ib.disconnect()
                except Exception as disconnect_e:
                    log(f"Error during disconnect: {disconnect_e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        util.run(main())
    except KeyboardInterrupt:
        log("Bot stopped manually.")
    except Exception as e:
        log(f"A critical error occurred in the main runner: {e}")

