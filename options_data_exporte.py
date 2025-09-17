import asyncio
import csv
import json
import os
import random
from datetime import datetime

import pytz
from ib_insync import *


def log(message: str):
    """Prints a message with a timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def is_market_open(contract_details, exchange_timezone_str: str):
    """
    Checks if the market for a given contract is currently open.
    """
    if not contract_details or not contract_details.liquidHours:
        log("Warning: Could not get liquid hours for contract.")
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
                log(f"Market is open for {contract_details.contract.symbol}.")
                return True
        except ValueError as e:
            log(f"Warning: Could not parse trading session '{session_str}'. Error: {e}")
    
    log(f"Market is closed for {contract_details.contract.symbol}.")
    return False


async def find_front_month_contract(ib: IB, symbol: str, secType: str, exchange: str):
    """Finds the most liquid, nearest-expiring (front-month) contract."""
    log(f"--- Finding front-month contract for {symbol} ---")
    
    contract_class = Future if secType == 'FUT' else Index
    contracts = await ib.reqContractDetailsAsync(contract_class(symbol=symbol, exchange=exchange))
    
    if not contracts:
        return None
        
    today = datetime.now().date()
    
    # Filter for contracts with valid expiration dates in the future
    valid_contracts = []
    for c in contracts:
        try:
            # Some contracts might not have a standard lastTradeDateOrContractMonth
            if c.contract.lastTradeDateOrContractMonth and datetime.strptime(c.contract.lastTradeDateOrContractMonth, '%Y%m%d').date() >= today:
                valid_contracts.append(c)
        except (ValueError, TypeError):
            continue # Ignore contracts with unparseable dates

    if not valid_contracts:
        return None
        
    # Sort by expiration date to find the nearest one
    valid_contracts.sort(key=lambda c: c.contract.lastTradeDateOrContractMonth)
    
    front_month_contract_details = valid_contracts[0]
    log(f"Found front-month contract: {front_month_contract_details.contract.localSymbol}")
    return front_month_contract_details


async def build_option_chain(ib: IB, symbol: str, exchange: str):
    """
    Builds a precise option chain by mapping strikes to their valid expirations.
    """
    log(f"Fetching option chain details for {symbol} on {exchange}...")
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
        log(f"Successfully built option chain from {len(contracts)} contracts.")
        return {'exchange': exchange, 'expirations': expirations, 'strikes_by_expiration': strikes_by_expiration}
    except Exception as e:
        log(f"Failed to build option chain on {exchange}: {e}")
        return None


async def process_target(ib: IB, target: dict):
    """Processes a single target from the config file."""
    log(f"\n================ PROCESSING TARGET: {target['name']} ================")
    
    underlying_details = await find_front_month_contract(ib, target['symbol'], target['secType'], target['exchange'])
    if not underlying_details:
        log(f"Could not find underlying contract for {target['name']}. Skipping.")
        return

    if not is_market_open(underlying_details, target['exchange_timezone']):
        log(f"Skipping {target['name']} as its market is closed.")
        return

    underlying_contract = underlying_details.contract
    ib.reqMarketDataType(1) # Ensure live data
    ticker_list = await ib.reqTickersAsync(underlying_contract)
    underlying_price = ticker_list[0].marketPrice() if ticker_list and not util.isNan(ticker_list[0].marketPrice()) else ticker_list[0].close
    log(f"Current underlying price for {underlying_contract.localSymbol} is: {underlying_price}")

    chain = await build_option_chain(ib, underlying_contract.symbol, target['option_exchange'])
    if not chain:
         log(f"Could not build option chain for {target['name']}. Skipping.")
         return
    
    today_str = datetime.now().strftime('%Y%m%d')
    upcoming_expirations = [exp for exp in sorted(chain['expirations']) if exp > today_str]
    
    if len(upcoming_expirations) < 2:
        log(f"Fewer than two upcoming expirations found for {target['name']}. Skipping.")
        return
        
    next_expiration = upcoming_expirations[1]
    log(f"Selected 'next' expiration date: {next_expiration}")

    exp_date = datetime.strptime(next_expiration, '%Y%m%d').date()
    days_to_expiration = (exp_date - datetime.now().date()).days

    all_strikes = chain['strikes_by_expiration'].get(next_expiration, [])
    if not all_strikes:
        log(f"No strikes found for expiration {next_expiration}. Skipping.")
        return
    log(f"Found {len(all_strikes)} strikes. Fetching data...")

    contracts_to_fetch = [FuturesOption(target['symbol'], next_expiration, strike, right, chain['exchange']) for strike in all_strikes for right in ['C', 'P']]
    
    batch_size = 50
    all_tickers = []
    for i in range(0, len(contracts_to_fetch), batch_size):
        batch = contracts_to_fetch[i:i + batch_size]
        await ib.qualifyContractsAsync(*batch)
        tickers = await ib.reqTickersAsync(*batch)
        all_tickers.extend(tickers)
        await asyncio.sleep(1) 

    data_rows = []
    call_data = {t.contract.strike: t for t in all_tickers if t.contract.right == 'C'}
    put_data = {t.contract.strike: t for t in all_tickers if t.contract.right == 'P'}
    pull_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for strike in all_strikes:
        call, put = call_data.get(strike), put_data.get(strike)
        def clean(val): return '' if val is None or val < 0 or not isinstance(val, (int, float)) else val
        def get_greek(ticker, greek_name):
            greeks = ticker.modelGreeks if ticker and ticker.modelGreeks else None
            return clean(getattr(greeks, greek_name, None)) if greeks else ''
        
        data_rows.append({
            'timestamp': pull_time, 'underlying_price': underlying_price, 'expiration_date': next_expiration, 'days_to_expiration': days_to_expiration, 'strike': strike,
            'call_bid': clean(call.bid if call else None), 'call_ask': clean(call.ask if call else None), 'call_last': clean(call.last if call else None), 'call_close': clean(call.close if call else None),
            'call_iv': get_greek(call, 'impliedVol'), 'call_delta': get_greek(call, 'delta'), 'call_gamma': get_greek(call, 'gamma'), 'call_vega': get_greek(call, 'vega'), 'call_theta': get_greek(call, 'theta'),
            'put_bid': clean(put.bid if put else None), 'put_ask': clean(put.ask if put else None), 'put_last': clean(put.last if put else None), 'put_close': clean(put.close if put else None),
            'put_iv': get_greek(put, 'impliedVol'), 'put_delta': get_greek(put, 'delta'), 'put_gamma': get_greek(put, 'gamma'), 'put_vega': get_greek(put, 'vega'), 'put_theta': get_greek(put, 'theta'),
        })
    
    if not data_rows:
        log("No data processed. Aborting file write.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"{target['name']}_options_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = data_rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)
    log(f"Successfully saved data to {filepath}")


async def main():
    """Main function to connect and loop through targets."""
    ib = IB()
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exporter_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        client_id = random.randint(1, 1000)
        log(f"Connecting to IBKR with Client ID: {client_id}...")
        await ib.connectAsync('127.0.0.1', 7497, clientId=client_id, timeout=10)
        log("Successfully connected.")

        for target in config['targets']:
            await process_target(ib, target)

    except Exception as e:
        log(f"A critical error occurred: {e}")
    finally:
        if ib.isConnected():
            log("Disconnecting from IBKR.")
            ib.disconnect()

if __name__ == "__main__":
    try:
        util.run(main())
    except (KeyboardInterrupt, SystemExit):
        log("Script stopped manually.")

