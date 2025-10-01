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


async def check_market_data_liveness(ib: IB, contract: Contract) -> bool:
    """
    Requests a market data snapshot to determine if an asset is actively trading.
    This serves as a reliable fallback when liquidHours data is unavailable.
    Returns True if live data is present, False otherwise.
    """
    log(f"DEBUG: No liquid hours. Checking market data liveness for {contract.localSymbol}...")
    # Request delayed data if available. This avoids using real-time data subscriptions
    # for a simple liveness check and prevents pacing violations.
    ib.reqMarketDataType(3)
    tickers = await ib.reqTickersAsync(contract)
    
    if not tickers:
        log(f"DEBUG: Liveness check failed (no ticker data received). Assuming market is closed.")
        return False
    
    ticker = tickers[0]
    # A market is considered "live" if there's a recent trade price or an active quote (bid/ask).
    # We check if the values are valid (not NaN) and greater than zero.
    is_live = (not util.isNan(ticker.last) and ticker.last > 0) or \
              (not util.isNan(ticker.bid) and ticker.bid > 0 and not util.isNan(ticker.ask) and ticker.ask > 0)

    if is_live:
        log(f"DEBUG: Liveness check PASSED for {contract.localSymbol} (Last: {ticker.last}, Bid: {ticker.bid}, Ask: {ticker.ask}). Market appears to be active.")
    else:
        log(f"DEBUG: Liveness check FAILED for {contract.localSymbol}. Data is stale (Last: {ticker.last}, Bid: {ticker.bid}, Ask: {ticker.ask}). Assuming market is closed.")
        
    return is_live


async def is_market_open(ib: IB, contract_details, exchange_timezone_str: str) -> bool:
    """
    Checks if the market for a given contract is currently open.
    If schedule information (liquidHours) is missing, it falls back to checking for live data.
    """
    # --- Fallback Mechanism ---
    # If the API doesn't provide liquid hours, we check for live market data directly.
    if not contract_details or not contract_details.liquidHours:
        log(f"Warning: Liquid hours data not available for {contract_details.contract.symbol}. Using market data liveness check as a fallback.")
        return await check_market_data_liveness(ib, contract_details.contract)

    # --- Primary Method: Parse Liquid Hours Schedule ---
    log(f"Liquid hours for {contract_details.contract.symbol}: {contract_details.liquidHours}")

    tz = pytz.timezone(exchange_timezone_str)
    now_local = datetime.now(pytz.utc).astimezone(tz)
    now_local_naive = now_local.replace(tzinfo=None)
    log(f"DEBUG: Current exchange time (naive): {now_local_naive}")

    sessions_str = contract_details.liquidHours.split(';')

    for session_str in sessions_str:
        if not session_str: continue
        try:
            # FIX: Gracefully skip entries like "20250920:CLOSED"
            if 'CLOSED' in session_str.upper():
                continue
            
            start_str, end_str = session_str.split('-')
            start_dt_str, start_hm_str = start_str.split(':')
            end_dt_str, end_hm_str = end_str.split(':')
            
            start_naive = datetime.strptime(f"{start_dt_str}{start_hm_str}", '%Y%m%d%H%M')
            end_naive = datetime.strptime(f"{end_dt_str}{end_hm_str}", '%Y%m%d%H%M')

            is_open = start_naive <= now_local_naive < end_naive
            log(f"DEBUG: Checking session start={start_naive}, end={end_naive}. Is open? {is_open}")

            if is_open:
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
    
    params = {
        'symbol': symbol,
        'exchange': exchange,
        'currency': currency
    }
    if tradingClass:
        params['tradingClass'] = tradingClass

    underlying = contract_class(**params)
    contracts = await ib.reqContractDetailsAsync(underlying)
    
    if not contracts:
        return None
        
    today = datetime.now().date()
    
    valid_contracts = []
    for c in contracts:
        try:
            if c.contract.lastTradeDateOrContractMonth and datetime.strptime(c.contract.lastTradeDateOrContractMonth, '%Y%m%d').date() >= today:
                valid_contracts.append(c)
        except (ValueError, TypeError):
            continue

    if not valid_contracts:
        return None
        
    valid_contracts.sort(key=lambda c: c.contract.lastTradeDateOrContractMonth)
    
    front_month_contract_details = valid_contracts[0]
    log(f"Found front-month contract: {front_month_contract_details.contract.localSymbol}")
    return front_month_contract_details


async def get_option_chain_details(ib: IB, target: dict, underlying_details: ContractDetails):
    """
    A robust method to get all available expirations and strikes for an underlying.
    """
    log(f"Building option chain for {target['name']} manually...")
    
    # Define a generic option contract to find all available chains.
    # For some contracts (like Coffee), a tradingClass and multiplier are needed for this initial search.
    params = {
        'symbol': target['symbol'],
        'exchange': target['option_exchange'],
        'currency': target.get('currency'),
        'multiplier': underlying_details.contract.multiplier
    }
    # If the config provides a specific option trading class, use it as a hint.
    if 'option_tradingClass' in target:
        params['tradingClass'] = target['option_tradingClass']
        log(f"Using tradingClass hint '{target['option_tradingClass']}' for initial option search.")

    option_contract = FuturesOption(**params)
    
    # Fetch details for all options matching the parameters.
    all_option_details = await ib.reqContractDetailsAsync(option_contract)

    if not all_option_details:
        log(f"Manual chain build failed: No option contracts found for {target['symbol']} on {target['option_exchange']}.")
        return None, None

    # Use a dictionary to store strikes and trading classes per expiration.
    chain_details_by_expiry = {}
    for detail in all_option_details:
        contract = detail.contract
        expiry = contract.lastTradeDateOrContractMonth
        strike = contract.strike
        trading_class = contract.tradingClass
        
        if expiry not in chain_details_by_expiry:
            chain_details_by_expiry[expiry] = set()
        # Store a tuple of (strike, tradingClass) to handle cases where different
        # strikes might have different trading classes for the same expiration.
        chain_details_by_expiry[expiry].add((strike, trading_class))
    
    all_expirations = sorted(list(chain_details_by_expiry.keys()))
    
    log(f"Manual chain build successful. Found {len(all_expirations)} expirations.")
    return all_expirations, chain_details_by_expiry


async def process_target(ib: IB, target: dict):
    """Processes a single target from the config file."""
    log(f"\n================ PROCESSING TARGET: {target['name']} ================")
    
    underlying_details = await find_front_month_contract(
        ib, target['symbol'], target['secType'], target['exchange'], 
        target.get('currency'), target.get('tradingClass')
    )
    if not underlying_details:
        log(f"Could not find underlying contract for {target['name']}. Skipping.")
        return

    if not await is_market_open(ib, underlying_details, target['exchange_timezone']):
        log(f"Skipping {target['name']} as its market is closed.")
        return

    underlying_contract = underlying_details.contract
    ib.reqMarketDataType(3) 
    
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

    # --- REVISED LOGIC: Manually build option chain ---
    all_expirations, chain_details_by_expiry = await get_option_chain_details(ib, target, underlying_details)
    
    if not all_expirations:
        log(f"Could not get option chain definitions for {target['name']}. Skipping.")
        return

    today_str = datetime.now().strftime('%Y%m%d')
    upcoming_expirations = [exp for exp in all_expirations if exp >= today_str]
    
    if not upcoming_expirations:
        log(f"No upcoming expirations found for {target['name']}. Skipping.")
        return
        
    next_expiration = upcoming_expirations[0]
    log(f"Selected 'next' expiration date: {next_expiration}")

    exp_date = datetime.strptime(next_expiration, '%Y%m%d').date()
    days_to_expiration = (exp_date - datetime.now().date()).days
    
    chain_details = chain_details_by_expiry.get(next_expiration, set())
    if not chain_details:
        log(f"No strikes found for expiration {next_expiration}. Skipping.")
        return

    # Create the list of specific option contracts to request data for.
    contracts_to_fetch = []
    unique_strikes = set()
    for strike, trading_class in chain_details:
        unique_strikes.add(strike)
        for right in ['C', 'P']:
            contracts_to_fetch.append(
                FuturesOption(
                    symbol=target['symbol'],
                    lastTradeDateOrContractMonth=next_expiration,
                    strike=strike,
                    right=right,
                    exchange=target['option_exchange'],
                    tradingClass=trading_class, # Use the correct trading class found
                    currency=target.get('currency'),
                    multiplier=underlying_details.contract.multiplier
                )
            )

    log(f"Found {len(unique_strikes)} unique strikes. Fetching data for {len(contracts_to_fetch)} contracts...")
    
    batch_size = 100
    all_tickers = []
    for i in range(0, len(contracts_to_fetch), batch_size):
        batch = contracts_to_fetch[i:i + batch_size]
        await ib.qualifyContractsAsync(*batch)
        tickers = await ib.reqTickersAsync(*batch)
        all_tickers.extend(tickers)
        if len(contracts_to_fetch) > batch_size:
             await asyncio.sleep(0.5) 

    data_rows = []
    call_data = {t.contract.strike: t for t in all_tickers if t.contract.right == 'C'}
    put_data = {t.contract.strike: t for t in all_tickers if t.contract.right == 'P'}
    pull_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for strike in sorted(list(unique_strikes)):
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

