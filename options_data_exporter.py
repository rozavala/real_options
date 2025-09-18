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
    
    # FIX: This is the robust way to handle timezones.
    # First, get the current time in a neutral format (UTC).
    now_utc = datetime.now(pytz.utc)
    # Then, convert that UTC time to the target timezone.
    now_tz = now_utc.astimezone(tz)
    
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


async def find_front_month_contract(ib: IB, symbol: str, secType: str, exchange: str, currency: str = None, tradingClass: str = None):
    """Finds the most liquid, nearest-expiring (front-month) contract."""
    log(f"--- Finding front-month contract for {symbol} ---")
    
    contract_class = Future if secType == 'FUT' else Index
    # Be more specific for underlying contract by including tradingClass if available
    underlying = contract_class(
        symbol=symbol, 
        exchange=exchange, 
        currency=currency, 
        tradingClass=tradingClass or ''
    )
    contracts = await ib.reqContractDetailsAsync(underlying)
    
    if not contracts:
        return None
        
    today = datetime.now().date()
    
    valid_contracts = []
    for c in contracts:
        try:
            # Ensure the contract has a valid expiration date in the future
            if c.contract.lastTradeDateOrContractMonth and datetime.strptime(c.contract.lastTradeDateOrContractMonth, '%Y%m%d').date() >= today:
                valid_contracts.append(c)
        except (ValueError, TypeError):
            continue

    if not valid_contracts:
        return None
        
    # Sort by expiration to find the nearest contract
    valid_contracts.sort(key=lambda c: c.contract.lastTradeDateOrContractMonth)
    
    front_month_contract_details = valid_contracts[0]
    log(f"Found front-month contract: {front_month_contract_details.contract.localSymbol}")
    return front_month_contract_details


async def process_target(ib: IB, target: dict):
    """Processes a single target from the config file."""
    log(f"\n================ PROCESSING TARGET: {target['name']} ================")
    
    # Pass tradingClass to be more specific in the contract search
    underlying_details = await find_front_month_contract(
        ib, target['symbol'], target['secType'], target['exchange'], 
        target.get('currency'), target.get('tradingClass')
    )
    if not underlying_details:
        log(f"Could not find underlying contract for {target['name']}. Skipping.")
        return

    if not is_market_open(underlying_details, target['exchange_timezone']):
        log(f"Skipping {target['name']} as its market is closed.")
        return

    underlying_contract = underlying_details.contract
    ib.reqMarketDataType(3) # Use delayed data as a fallback for non-subscribed products
    
    # Use the more robust reqTickersAsync for a clean snapshot
    tickers = await ib.reqTickersAsync(underlying_contract)
    if not tickers:
        log(f"Could not get market data for {underlying_contract.localSymbol}. Skipping.")
        return
    ticker = tickers[0]

    underlying_price = ticker.marketPrice() if not util.isNan(ticker.marketPrice()) else ticker.close

    log(f"Current underlying price for {underlying_contract.localSymbol} is: {underlying_price}")
    if util.isNan(underlying_price):
        log(f"Could not determine underlying price for {target['name']}. Skipping.")
        return

    # Fetch all available option chain definitions for the underlying
    chains = await ib.reqSecDefOptParamsAsync(
        underlyingSymbol=underlying_contract.symbol,
        futFopExchange=underlying_contract.exchange,
        underlyingSecType=underlying_contract.secType,
        underlyingConId=underlying_contract.conId
    )
    if not chains:
        log(f"Could not get option chain definitions for {target['name']}. Skipping.")
        return

    # Consolidate all unique expirations from all chains
    all_expirations = sorted(list(set(
        exp for chain in chains for exp in chain.expirations)))
    
    today_str = datetime.now().strftime('%Y%m%d')
    upcoming_expirations = [exp for exp in all_expirations if exp >= today_str]
    
    if not upcoming_expirations:
        log(f"No upcoming expirations found for {target['name']}. Skipping.")
        return
        
    # Select the nearest upcoming expiration
    next_expiration = upcoming_expirations[0]
    log(f"Selected 'next' expiration date: {next_expiration}")

    exp_date = datetime.strptime(next_expiration, '%Y%m%d').date()
    days_to_expiration = (exp_date - datetime.now().date()).days
    
    contracts_to_fetch = []
    all_strikes = set()
    
    target_option_tc = target.get('option_tradingClass')
    
    # Get all chains that match the selected expiration
    chains_for_expiration = [c for c in chains if next_expiration in c.expirations]
    
    # Try to find a chain that matches the trading class from the config
    matching_chain = None
    if target_option_tc:
        for c in chains_for_expiration:
            if c.tradingClass == target_option_tc:
                matching_chain = c
                break
    
    # If no specific match is found, but chains for the expiration exist, use the first one as a fallback
    if not matching_chain and chains_for_expiration:
        matching_chain = chains_for_expiration[0]
        log(f"Warning: Could not find chain with tradingClass '{target_option_tc}'. Using fallback: '{matching_chain.tradingClass}'")

    if matching_chain:
        log(f"Found matching chain for expiration {next_expiration} on exchange {matching_chain.exchange} with tradingClass {matching_chain.tradingClass}")
        all_strikes.update(matching_chain.strikes)
        contracts_to_fetch.extend([
            FuturesOption(
                symbol=target['symbol'],
                lastTradeDateOrContractMonth=next_expiration,
                strike=strike,
                right=right,
                exchange=matching_chain.exchange,
                tradingClass=matching_chain.tradingClass,
                currency=target.get('currency')
            ) for strike in matching_chain.strikes for right in ['C', 'P']
        ])

    if not contracts_to_fetch:
        log(f"Could not build any contracts for expiration {next_expiration}. Skipping.")
        return
    
    log(f"Found {len(all_strikes)} unique strikes. Fetching data for {len(contracts_to_fetch)} contracts...")
    
    batch_size = 100
    all_tickers = []
    for i in range(0, len(contracts_to_fetch), batch_size):
        batch = contracts_to_fetch[i:i + batch_size]
        await ib.qualifyContractsAsync(*batch)
        tickers = await ib.reqTickersAsync(*batch)
        all_tickers.extend(tickers)
        await asyncio.sleep(0.5) 

    data_rows = []
    call_data = {t.contract.strike: t for t in all_tickers if t.contract.right == 'C'}
    put_data = {t.contract.strike: t for t in all_tickers if t.contract.right == 'P'}
    pull_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for strike in sorted(list(all_strikes)):
        call, put = call_data.get(strike), put_data.get(strike)
        def clean(val): return '' if val is None or util.isNan(val) or val < 0 else val
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

