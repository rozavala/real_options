import asyncio
import csv
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
        return False  # Fail safe

    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    
    sessions_str = contract_details.liquidHours.split(';')

    for session_str in sessions_str:
        if not session_str:
            continue
        
        try:
            start_str, end_str = session_str.split('-')
            start_dt_str, start_hm_str = start_str.split(':')
            end_dt_str, end_hm_str = end_str.split(':')

            start_datetime = tz.localize(datetime.strptime(f"{start_dt_str}{start_hm_str}", '%Y%m%d%H%M'))
            end_datetime = tz.localize(datetime.strptime(f"{end_dt_str}{end_hm_str}", '%Y%m%d%H%M'))

            if start_datetime <= now_tz < end_datetime:
                log(f"Market is open. Current session: {start_datetime.strftime('%Y-%m-%d %H:%M')} to {end_datetime.strftime('%Y-%m-%d %H:%M')} {exchange_timezone_str}")
                return True
        except ValueError as e:
            log(f"Warning: Could not parse trading session '{session_str}'. Error: {e}")
            continue

    log("Market is closed. No active session found for the current time.")
    return False


async def find_front_month_future(ib: IB, symbol: str, exchange: str):
    """Finds the most liquid, nearest-expiring (front-month) future contract."""
    log(f"--- Finding front-month future for {symbol} ---")
    contracts = await ib.reqContractDetailsAsync(Future(symbol=symbol, exchange=exchange))
    if not contracts:
        return None
    today = datetime.now().date()
    valid_contracts = sorted(
        [c for c in contracts if datetime.strptime(c.contract.lastTradeDateOrContractMonth, '%Y%m%d').date() >= today],
        key=lambda c: c.contract.lastTradeDateOrContractMonth
    )
    if not valid_contracts:
        return None
    front_month_contract_details = valid_contracts[0]
    log(f"Found front-month future: {front_month_contract_details.contract.localSymbol}")
    return front_month_contract_details


async def build_option_chain(ib: IB, symbol: str, exchange: str):
    """
    Builds a precise option chain by mapping strikes to their valid expirations.
    """
    log(f"Fetching contract details for {symbol} on {exchange}...")
    try:
        contracts = await ib.reqContractDetailsAsync(FuturesOption(symbol, exchange=exchange))
        if not contracts:
            return None

        strikes_by_expiration = {}
        for c in contracts:
            exp = c.contract.lastTradeDateOrContractMonth
            strike = c.contract.strike
            if exp not in strikes_by_expiration:
                strikes_by_expiration[exp] = set()
            strikes_by_expiration[exp].add(strike)

        for exp in strikes_by_expiration:
            strikes_by_expiration[exp] = sorted(list(strikes_by_expiration[exp]))
            
        expirations = sorted(strikes_by_expiration.keys())
        
        log(f"Successfully built option chain from {len(contracts)} contracts.")
        return {'exchange': exchange, 'expirations': expirations, 'strikes_by_expiration': strikes_by_expiration}
    except Exception as e:
        log(f"Failed to build option chain on {exchange}: {e}")
        return None


async def main():
    """Main function to connect, fetch data, and write to CSV."""
    ib = IB()
    try:
        client_id = random.randint(1, 1000)
        log(f"Connecting to IBKR with Client ID: {client_id}...")
        await ib.connectAsync('127.0.0.1', 7497, clientId=client_id, timeout=10)
        log("Successfully connected.")

        underlying_future_details = await find_front_month_future(ib, 'KC', 'NYBOT')
        if not underlying_future_details:
            log("Could not find underlying future. Aborting.")
            return

        if not is_market_open(underlying_future_details, 'America/New_York'):
            log("Aborting data export.")
            return

        underlying_future = underlying_future_details.contract
        # Request model greeks by default
        ib.reqMarketDataType(4)
        ticker = await ib.reqTickersAsync(underlying_future)
        underlying_price = ticker[0].marketPrice()
        log(f"Current underlying price for {underlying_future.localSymbol} is: {underlying_price}")

        primary_exchange = underlying_future.exchange
        chain = await build_option_chain(ib, underlying_future.symbol, primary_exchange)
        
        if not chain:
            fallback_exchange = 'ICE'
            log(f"No chain found on {primary_exchange}. Trying fallback: {fallback_exchange}...")
            chain = await build_option_chain(ib, underlying_future.symbol, fallback_exchange)

        if not chain:
             log("Could not build option chain on primary or fallback exchange. Aborting.")
             return
        
        log(f"Using option chain from exchange: {chain['exchange']}")

        today_str = datetime.now().strftime('%Y%m%d')
        upcoming_expirations = [exp for exp in sorted(chain['expirations']) if exp > today_str]
        
        if len(upcoming_expirations) < 2:
            log("Fewer than two upcoming expirations found. Cannot select the 'next' one. Aborting.")
            return
            
        next_expiration = upcoming_expirations[1]
        log(f"Selected 'next' expiration date: {next_expiration}")

        exp_date = datetime.strptime(next_expiration, '%Y%m%d').date()
        today_date = datetime.now().date()
        days_to_expiration = (exp_date - today_date).days
        log(f"Days to expiration: {days_to_expiration}")

        all_strikes = chain['strikes_by_expiration'].get(next_expiration, [])
        log(f"Found {len(all_strikes)} strikes for {next_expiration}. Fetching data for all...")

        if not all_strikes:
            log(f"No strikes found for expiration {next_expiration}. Aborting.")
            return

        contracts_to_fetch = []
        for strike in all_strikes:
            contracts_to_fetch.append(FuturesOption('KC', next_expiration, strike, 'C', chain['exchange']))
            contracts_to_fetch.append(FuturesOption('KC', next_expiration, strike, 'P', chain['exchange']))
        
        batch_size = 50
        all_tickers = []
        for i in range(0, len(contracts_to_fetch), batch_size):
            batch = contracts_to_fetch[i:i + batch_size]
            log(f"Fetching batch {i//batch_size + 1} of {len(contracts_to_fetch)//batch_size + 1}...")
            await ib.qualifyContractsAsync(*batch)
            tickers = await ib.reqTickersAsync(*batch)
            all_tickers.extend(tickers)
            await asyncio.sleep(1) 

        data_rows = []
        call_data = {t.contract.strike: t for t in all_tickers if t.contract.right == 'C'}
        put_data = {t.contract.strike: t for t in all_tickers if t.contract.right == 'P'}

        pull_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for strike in all_strikes:
            call = call_data.get(strike)
            put = put_data.get(strike)

            def clean_value(val):
                return '' if val is None or val < 0 or not isinstance(val, (int, float)) else val

            # Use modelGreeks as they are often available even when bid/ask greeks are not
            call_greeks = call.modelGreeks if call and call.modelGreeks else None
            put_greeks = put.modelGreeks if put and put.modelGreeks else None

            data_rows.append({
                'timestamp': pull_time,
                'underlying_price': underlying_price,
                'expiration_date': next_expiration,
                'days_to_expiration': days_to_expiration,
                'strike': strike,
                'call_bid': clean_value(call.bid if call else None),
                'call_ask': clean_value(call.ask if call else None),
                'call_last': clean_value(call.last if call else None),
                'call_close': clean_value(call.close if call else None),
                'call_iv': clean_value(call_greeks.impliedVol if call_greeks else None),
                'call_delta': clean_value(call_greeks.delta if call_greeks else None),
                'call_gamma': clean_value(call_greeks.gamma if call_greeks else None),
                'call_vega': clean_value(call_greeks.vega if call_greeks else None),
                'call_theta': clean_value(call_greeks.theta if call_greeks else None),
                'put_bid': clean_value(put.bid if put else None),
                'put_ask': clean_value(put.ask if put else None),
                'put_last': clean_value(put.last if put else None),
                'put_close': clean_value(put.close if put else None),
                'put_iv': clean_value(put_greeks.impliedVol if put_greeks else None),
                'put_delta': clean_value(put_greeks.delta if put_greeks else None),
                'put_gamma': clean_value(put_greeks.gamma if put_greeks else None),
                'put_vega': clean_value(put_greeks.vega if put_greeks else None),
                'put_theta': clean_value(put_greeks.theta if put_greeks else None),
            })
        
        log(f"Successfully processed data for {len(data_rows)} strikes.")

        if not data_rows:
            log("No data to write. Aborting.")
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"kc_options_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(script_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'underlying_price', 'expiration_date', 'days_to_expiration', 'strike', 
                'call_bid', 'call_ask', 'call_last', 'call_close', 'call_iv', 'call_delta', 'call_gamma', 'call_vega', 'call_theta',
                'put_bid', 'put_ask', 'put_last', 'put_close', 'put_iv', 'put_delta', 'put_gamma', 'put_vega', 'put_theta'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_rows)
        
        log(f"Successfully saved data to {filepath}")

    except Exception as e:
        log(f"An error occurred: {e}")
    finally:
        if ib.isConnected():
            log("Disconnecting from IBKR.")
            ib.disconnect()

if __name__ == "__main__":
    try:
        util.run(main())
    except (KeyboardInterrupt, SystemExit):
        log("Script stopped manually.")

