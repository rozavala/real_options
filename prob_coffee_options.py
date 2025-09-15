import asyncio
import csv
import json
import os
import random
import time
from datetime import datetime

import pytz
from ib_insync import *

# --- 1. Placeholder for Your Predictive Models ---
# You will replace the logic inside these functions with your own models.

def get_price_direction_prediction() -> str:
    """
    Placeholder for your model that predicts the future's price direction.
    Replace this with your actual model logic.
    It should return either 'UP' or 'DOWN'.
    """
    # TODO: Insert your directional model logic here.
    # For now, we will simulate a random prediction.
    log_with_timestamp("Model 1: Predicting future price direction...")
    prediction = 'UP' if random.uniform(0, 1) > 0.5 else 'DOWN'
    log_with_timestamp(f"Model 1 Prediction: {prediction}")
    return prediction

def get_theoretical_option_price(underlying_price: float, strike_price: float, days_to_expiration: int, option_type: str) -> float:
    """
    Placeholder for your model that calculates the "fair value" of an option.
    Replace this with your Black-Scholes or other pricing model.
    """
    # TODO: Insert your option pricing model logic here (e.g., Black-Scholes).
    # This requires inputs like volatility and the risk-free rate, which you
    # would need to source or estimate.
    # For now, this is a highly simplified dummy model for demonstration.
    
    # A very basic pricing model for demonstration purposes:
    # It calculates the intrinsic value and adds a small time premium.
    price = 0.0
    time_premium_per_day = 0.05 # 5 cents per day
    
    if option_type == 'C': # Call option
        intrinsic_value = max(0, underlying_price - strike_price)
        price = intrinsic_value + (days_to_expiration * time_premium_per_day)
    elif option_type == 'P': # Put option
        intrinsic_value = max(0, strike_price - underlying_price)
        price = intrinsic_value + (days_to_expiration * time_premium_per_day)
        
    return price if price > 0 else 0.01


# --- The rest of the script is the engine that uses your models ---

def log_with_timestamp(message: str):
    """Prints a message with a timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def log_trade_to_ledger(trade: Trade):
    """Appends a record of a filled trade to the trade_ledger.csv file."""
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv')
    file_exists = os.path.isfile(ledger_path)
    if trade.orderStatus.status != OrderStatus.Filled: return
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
            if not file_exists: writer.writeheader()
            writer.writerow(row)
        log_with_timestamp(f"Logged trade to ledger: {row['action']} {row['quantity']} {row['local_symbol']} @ {row['avg_fill_price']}")
    except Exception as e:
        log_with_timestamp(f"Error writing to trade ledger: {e}")

def is_market_open(contract_details, exchange_timezone_str: str):
    if not contract_details or not contract_details.liquidHours: return False
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
                log_with_timestamp(f"Market is open.")
                return True
        except ValueError: continue
    log_with_timestamp("Market is closed.")
    return False

async def get_full_contract_details(ib: IB, contract):
    details = await ib.reqContractDetailsAsync(contract)
    return details[0] if details else None

async def build_option_chain(ib: IB, symbol: str, exchange: str):
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

async def find_best_mispriced_option(ib: IB, config: dict, strategy: dict, option_type: str, all_strikes: list, days_to_exp: int, underlying_price: float):
    log_with_timestamp(f"\n--- Model 2: Searching for best overpriced {option_type} to sell ---")
    
    best_option = None
    highest_overpricing = -1

    # Create contracts for all strikes of the chosen type (Call or Put)
    contracts_to_check = [
        FuturesOption(
            config['symbol'],
            lastTradeDateOrContractMonth=strategy['expiration_date'],
            strike=s,
            right=option_type,
            exchange=strategy['option_exchange']
        ) for s in all_strikes
    ]
    
    await ib.qualifyContractsAsync(*contracts_to_check)
    tickers = await ib.reqTickersAsync(*contracts_to_check)

    for ticker in tickers:
        # Check if the market is liquid for this specific option
        if util.isNan(ticker.bid) or ticker.bid <= 0:
            continue
            
        market_price = ticker.bid
        strike_price = ticker.contract.strike

        # Get the theoretical price from your model
        theoretical_price = get_theoretical_option_price(underlying_price, strike_price, days_to_exp, option_type)
        
        # Calculate how much the market price is above your model's price
        if theoretical_price > 0:
            overpricing_pct = (market_price / theoretical_price) - 1
            if overpricing_pct > highest_overpricing:
                highest_overpricing = overpricing_pct
                best_option = {
                    'contract': ticker.contract,
                    'market_price': market_price,
                    'theoretical_price': theoretical_price,
                    'overpricing_pct': overpricing_pct
                }

    if best_option and best_option['overpricing_pct'] >= strategy['min_overpricing_pct']:
        log_with_timestamp(
            f"Found best option to sell: {best_option['contract'].localSymbol} "
            f"is {best_option['overpricing_pct']:.2%} overpriced. "
            f"(Market: {best_option['market_price']:.2f}, Model: {best_option['theoretical_price']:.2f})"
        )
        return best_option['contract'], round(best_option['market_price'], 2)
    
    log_with_timestamp("No sufficiently overpriced options found.")
    return None, 0


async def main_runner():
    ib = IB()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

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

            log_with_timestamp("--- Issuing Global Cancel for All Open Orders ---")
            ib.reqGlobalCancel()
            await asyncio.sleep(2)
            log_with_timestamp("-------------------------------------------------\n")

            log_with_timestamp("--- Closing All Open Positions ---")
            positions = await ib.reqPositionsAsync()
            close_trades = []
            for pos in positions:
                if isinstance(pos.contract, FuturesOption) and pos.contract.symbol == config['symbol'] and pos.position != 0:
                    contract = pos.contract
                    if not contract.exchange: contract.exchange = config.get('option_exchange', config['exchange'])
                    action = 'BUY' if pos.position < 0 else 'SELL'
                    quantity = abs(pos.position)
                    order = MarketOrder(action, quantity)
                    trade = ib.placeOrder(contract, order)
                    close_trades.append(trade)
            
            if not close_trades:
                log_with_timestamp("No open positions to close.")
            else:
                log_with_timestamp(f"Waiting for {len(close_trades)} closing trade(s) to fill...")
                for trade in close_trades:
                    while not trade.isDone(): await asyncio.sleep(1)
                    if trade.orderStatus.status == OrderStatus.Filled: log_trade_to_ledger(trade)
            log_with_timestamp("--------------------------------\n")

            log_with_timestamp("--- Finding front-month future for KC ---")
            future_details = await get_full_contract_details(ib, Future(config['symbol'], exchange=config['exchange']))
            if not future_details: raise ConnectionError("Could not find a valid front-month future.")

            if is_market_open(future_details, config['exchange_timezone']):
                strategy = config['strategy']
                ticker_list = await ib.reqTickersAsync(future_details.contract)
                current_underlying_price = ticker_list[0].marketPrice()

                primary_exchange, fallback_exchange = config['exchange'], 'ICE'
                chain = await build_option_chain(ib, config['symbol'], primary_exchange)
                option_exchange = primary_exchange
                if not chain:
                    chain = await build_option_chain(ib, config['symbol'], fallback_exchange)
                    if chain: option_exchange = fallback_exchange
                if not chain: raise ConnectionError("Could not build option chain.")
                
                strategy['option_exchange'] = option_exchange
                today_str = datetime.now().strftime('%Y%m%d')
                upcoming_expirations = [exp for exp in sorted(chain['expirations']) if exp > today_str]
                exp_choice = config.get('expiration_to_use', 'next')
                exp_index = 1 if exp_choice == 'next' and len(upcoming_expirations) > 1 else 0
                if not upcoming_expirations or len(upcoming_expirations) <= exp_index:
                    raise ValueError(f"Not enough upcoming expirations for '{exp_choice}'.")
                
                chosen_expiration = upcoming_expirations[exp_index]
                strategy['expiration_date'] = chosen_expiration
                days_to_expiration = (datetime.strptime(chosen_expiration, '%Y%m%d').date() - datetime.now().date()).days
                
                # --- NEW: Use Model 1 for direction ---
                direction = get_price_direction_prediction()
                option_type = 'P' if direction == 'UP' else 'C'

                # --- NEW: Use Model 2 to find best option ---
                all_strikes = chain['strikes_by_expiration'][chosen_expiration]
                contract_to_sell, limit_price = await find_best_mispriced_option(
                    ib, config, strategy, option_type, all_strikes, days_to_expiration, current_underlying_price
                )

                if contract_to_sell and limit_price > 0:
                    if config['symbol'] == 'KC': contract_to_sell.strike /= 100
                    order = LimitOrder('SELL', strategy['quantity'], limit_price)
                    trade = ib.placeOrder(contract_to_sell, order)
                    log_with_timestamp(f"Placed new order {trade.order.orderId}. Waiting up to 60s for fill...")
                    fill_wait_start = time.time()
                    while not trade.isDone():
                        await asyncio.sleep(1)
                        if time.time() - fill_wait_start > 60:
                            log_with_timestamp(f"Order {trade.order.orderId} not filled. Canceling.")
                            ib.cancelOrder(trade.order)
                            break
                    if trade.orderStatus.status == OrderStatus.Filled: log_trade_to_ledger(trade)
                else:
                    log_with_timestamp("No profitable trades found by the model.")

            wait_hours = config.get('trade_interval_hours', 2)
            log_with_timestamp(f"Trading logic complete. Waiting for {wait_hours} hours...")
            wait_seconds = wait_hours * 3600
            heartbeat_interval = 300
            end_time = time.time() + wait_seconds
            while time.time() < end_time:
                sleep_duration = min(heartbeat_interval, end_time - time.time())
                if sleep_duration <= 0: break
                await asyncio.sleep(sleep_duration)
                if time.time() < end_time:
                    await ib.reqCurrentTimeAsync()
                    log_with_timestamp("Heartbeat sent. Connection alive.")

        except (ConnectionError, OSError) as e:
            log_with_timestamp(f"Connection error: {e}. Reconnecting in 5 mins...")
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(300)
        except Exception as e:
            log_with_timestamp(f"Unexpected critical error: {e}. Restarting in 1 min...")
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        util.run(main_runner())
    except (KeyboardInterrupt, SystemExit):
        log_with_timestamp("Script stopped manually.")

