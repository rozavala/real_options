import asyncio
import csv
import json
import os
import random
import time
from datetime import datetime, time as dt_time, timedelta
import numpy as np
from scipy.stats import norm
import requests
import pytz
from ib_insync import *
import logging
import traceback
import yfinance as yf
import pandas as pd
from fredapi import Fred
import nasdaqdatalink as ndl
from pandas.tseries.offsets import BDay

# --- Logging Setup ---
# This ensures all messages, including from ib_insync, are captured.
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_bot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ])
# Capture ib_insync's internal logs
ib_logger = logging.getLogger('ib_insync')
ib_logger.setLevel(logging.INFO)


# --- 1. API Integration & Notifications ---

def send_notification(config: dict, title: str, message: str):
    """Sends a push notification via Pushover if enabled in the config."""
    notif_config = config.get('notifications', {})
    if not notif_config.get('enabled'):
        return

    user_key = notif_config.get('pushover_user_key')
    api_token = notif_config.get('pushover_api_token')

    if not user_key or not api_token or 'YOUR_KEY' in user_key or 'YOUR_TOKEN' in api_token:
        logging.warning("Notification sending skipped: Pushover keys not configured.")
        return

    try:
        response = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": api_token,
            "user": user_key,
            "title": f"Coffee Trader Alert: {title}",
            "message": message,
            "sound": "persistent",
            "html": 1
        }, timeout=10)
        response.raise_for_status()
        logging.info(f"Sent notification: '{title}'")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send notification: {e}")


# --- Data Aggregation for API (from coffee_factors_data_pull_new.py) ---

def _get_kc_expiration_date(year, month_code):
    """Calculates the expiration date for a given Coffee 'C' futures contract."""
    month_map = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
    month = month_map.get(month_code)
    if not month:
        raise ValueError(f"Invalid month code: {month_code}")
    last_day_of_month = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)
    last_business_day = last_day_of_month
    while last_business_day.weekday() > 4:
        last_business_day -= pd.Timedelta(days=1)
    return last_business_day - BDay(1)

def _get_yfinance_coffee_tickers(num_contracts=5):
    """Determines the yfinance tickers for the next N active coffee futures contracts."""
    now = datetime.now()
    potential_tickers = []
    for year_offset in range(3):
        year = now.year + year_offset
        for code in ['H', 'K', 'N', 'U', 'Z']:
            try:
                exp_date = _get_kc_expiration_date(year, code)
                if exp_date.date() >= now.date():
                    potential_tickers.append((f"KC{code}{str(year)[-2:]}.NYB", exp_date))
            except ValueError:
                continue
    potential_tickers.sort(key=lambda x: x[1])
    return [ticker for ticker, exp in potential_tickers[:num_contracts]]

def _fetch_and_prepare_data_for_api(config: dict) -> pd.DataFrame | None:
    """
    Fetches, combines, and processes all market, weather, and economic data
    required by the prediction model. Returns a pandas DataFrame.
    """
    logging.info("--- Starting Data Aggregation for API ---")
    try:
        fred = Fred(api_key=config['fred_api_key'])
        ndl.api_key = config['nasdaq_api_key']
        start_date, end_date = datetime(1980, 1, 1), datetime.now()
        all_data = {}

        # 1. Fetch Coffee Futures Data from yfinance
        coffee_tickers = _get_yfinance_coffee_tickers()
        if coffee_tickers:
            df_price = yf.download(coffee_tickers, start=start_date, end=end_date, progress=False)['Close']
            df_price.columns = [f"coffee_price_{t.replace('.NYB', '')}" for t in df_price.columns]
            all_data['coffee_prices'] = df_price
        else:
            logging.error("Could not determine active coffee tickers for yfinance.")
            return None

        # (In a real scenario, weather and other economic data would be fetched here)

        # Consolidate and process data
        if 'coffee_prices' not in all_data or all_data['coffee_prices'].empty:
            logging.error("Failed to fetch primary coffee price data.")
            return None

        final_df = all_data['coffee_prices'].copy()
        # (Join other data sources here)
        final_df.ffill(inplace=True)
        final_df.dropna(inplace=True)

        # Reformat columns as the model expects
        for col in list(final_df.columns):
            if col.startswith('coffee_price_KC'):
                ticker_part = col.replace('coffee_price_', '')
                month_code, year_str = ticker_part[2], ticker_part[3:]
                year = int(f"20{year_str}")
                exp = _get_kc_expiration_date(year, month_code)
                final_df[f'{month_code}_dte'] = (exp.tz_localize(None) - final_df.index).days
                final_df.rename(columns={col: f'{month_code}_price'}, inplace=True)

        # Ensure all required columns exist, even if null
        for col in config.get('final_column_order', []):
            if col not in final_df.columns:
                final_df[col] = None

        logging.info("Data aggregation for API successful.")
        return final_df[config['final_column_order']]

    except Exception as e:
        logging.error(f"An error occurred during data aggregation: {e}", exc_info=True)
        return None


def get_prediction_from_api(config: dict, sorted_futures: list[Contract]) -> list | None:
    """
    Orchestrates the entire signal generation pipeline: fetches data,
    calls the prediction API, and translates the response into signals.
    """
    logging.info("===== Starting Full Signal Generation Pipeline =====")
    api_url = config.get('api_base_url')
    if not api_url:
        logging.error("API URL is not configured in config.json.")
        return None

    # 1. Fetch and prepare the data
    data_df = _fetch_and_prepare_data_for_api(config)
    if data_df is None:
        logging.error("Signal generation failed: Data aggregation step was unsuccessful.")
        return None

    # 2. Send data to API and poll for results
    try:
        # Reverted to sending data as a CSV string, as this was the original working format.
        data_payload = data_df.tail(600).to_csv(index=False)
        request_body = {"data": data_payload}

        prediction_endpoint = f"{api_url}/predictions"
        logging.info(f"Sending data to {prediction_endpoint} to start prediction job...")
        response = requests.post(prediction_endpoint, json=request_body, timeout=20)
        response.raise_for_status()

        job_id = response.json().get('id')
        if not job_id:
            logging.error("API did not return a valid job ID.")
            return None

        result_url = f"{prediction_endpoint}/{job_id}"
        timeout_seconds = 180
        start_time = time.time()

        raw_predictions = None
        while time.time() - start_time < timeout_seconds:
            try:
                logging.info(f"Polling for result (Job ID: {job_id})...")
                result_response = requests.get(result_url, timeout=10)
                result_response.raise_for_status()
                status_info = result_response.json()

                if status_info.get('status') == 'completed':
                    raw_predictions = status_info.get('result')
                    break
                elif status_info.get('status') == 'failed':
                    logging.error(f"Prediction job failed on server. Reason: {status_info.get('error')}")
                    return None
                time.sleep(10)
            except requests.exceptions.HTTPError as e:
                logging.error(f"Prediction server returned an HTTP error while polling job {job_id}: {e}")
                return None

        if not raw_predictions:
            logging.error("Polling timed out. The prediction job took too long.")
            return None

        # 3. Translate raw predictions into structured signals
        logging.info(f"Received raw prediction data: {raw_predictions}")
        price_changes = raw_predictions.get('price_changes', [])

        if len(price_changes) != len(sorted_futures):
            logging.error(f"Signal mismatch: Received {len(price_changes)} predictions but have {len(sorted_futures)} contracts.")
            return None

        signals = []
        for i, price_change in enumerate(price_changes):
            direction = "BULLISH" if price_change > 7 else "BEARISH" if price_change < 0 else None
            if direction:
                future = sorted_futures[i]
                signals.append({
                    "contract_month": future.lastTradeDateOrContractMonth[:6],
                    "prediction_type": "DIRECTIONAL", "direction": direction, "confidence": 1.0
                })

        logging.info(f"Successfully generated {len(signals)} actionable trading signals.")
        return signals

    except Exception as e:
        logging.error(f"An error occurred during API communication or signal translation: {e}", exc_info=True)
        return None

# --- 2. Core Trading Logic & Strategy Execution ---

def price_option_black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict | None:
    """Calculates the theoretical price and Greeks of an option using the Black-Scholes model."""
    if T <= 0 or sigma <= 0:
        logging.warning(f"Invalid input for B-S model: T={T}, sigma={sigma}. Cannot price option.")
        return None

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = 0.0
    if option_type.upper() == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type.upper() == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    else:
        return None

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type.upper() == 'C' else -d2)) / 365

    results = {"price": round(price, 4), "delta": round(delta, 4), "gamma": round(gamma, 4), "vega": round(vega, 4), "theta": round(theta, 4)}
    logging.info(f"Theoretical price calculated for {option_type} @ {K}: {results}")
    return results

async def get_option_market_data(ib: IB, contract: Contract) -> dict | None:
    """Fetches live market data for a single option contract, including implied volatility."""
    logging.info(f"Fetching market data for option: {contract.localSymbol}")
    ticker = ib.reqMktData(contract, '106', False, False)
    await asyncio.sleep(2)
    iv = ticker.modelGreeks.impliedVol if ticker.modelGreeks and not util.isNan(ticker.modelGreeks.impliedVol) else 0.25
    ib.cancelMktData(contract)
    return {'implied_volatility': iv, 'risk_free_rate': 0.04}

def get_position_details(position: Position) -> dict:
    """Determines the strategy type from a position object."""
    contract = position.contract
    details = {'type': 'UNKNOWN', 'key_strikes': []}
    if isinstance(contract, FuturesOption):
        details['type'] = 'SINGLE_LEG'
        details['key_strikes'].append(contract.strike)
    elif isinstance(contract, Bag):
        legs = contract.comboLegs
        actions = ''.join(sorted([leg.action for leg in legs]))
        rights = ''.join(sorted([leg.right for leg in legs]))
        strikes = sorted([leg.strike for leg in legs])
        if len(legs) == 2:
            details['key_strikes'] = strikes
            if rights == 'CC' and actions == 'BS': details['type'] = 'BULL_CALL_SPREAD'
            elif rights == 'PP' and actions == 'BS': details['type'] = 'BEAR_PUT_SPREAD'
            elif rights == 'CP' and actions == 'BB': details['type'] = 'LONG_STRADDLE'
        if len(legs) == 4 and rights == 'CCPP' and actions == 'BBSS':
            details['type'] = 'IRON_CONDOR'
            details['key_strikes'] = [strikes[1], strikes[2]]
    return details

async def manage_existing_positions(ib: IB, config: dict, signal: dict, underlying_price: float, future_contract: Contract) -> bool:
    """Checks for existing positions for a given future and closes them if misaligned."""
    target_future_conId = future_contract.conId
    logging.info(f"--- Managing Positions for Future Contract: {future_contract.localSymbol} (conId: {target_future_conId}) ---")

    all_positions = await ib.reqPositionsAsync()
    positions_for_this_future = []

    for p in all_positions:
        if p.contract.symbol != config['symbol'] or p.position == 0: continue
        try:
            details_list = await ib.reqContractDetailsAsync(Contract(conId=p.contract.conId))
            if details_list and hasattr(details_list[0], 'underConId') and details_list[0].underConId == target_future_conId:
                positions_for_this_future.append(p)
        except Exception as e:
            logging.warning(f"Could not resolve underlying for position {p.contract.localSymbol}: {e}")

    if not positions_for_this_future:
        logging.info(f"No existing positions found for future {future_contract.localSymbol}.")
        return True

    target_strategy = ''
    if signal['prediction_type'] == 'DIRECTIONAL':
        target_strategy = 'BULL_CALL_SPREAD' if signal['direction'] == 'BULLISH' else 'BEAR_PUT_SPREAD'
    elif signal['prediction_type'] == 'VOLATILITY':
        target_strategy = 'LONG_STRADDLE' if signal['level'] == 'HIGH' else 'IRON_CONDOR'

    trades_to_close, position_is_aligned = [], False
    for pos in positions_for_this_future:
        pos_details = get_position_details(pos)
        current_strategy = pos_details['type']
        logging.info(f"Found open position for {future_contract.localSymbol}: {pos.position} of {current_strategy}")
        
        if current_strategy != target_strategy:
            logging.info(f"Position MISALIGNED. Current: {current_strategy}, Target: {target_strategy}. Closing.")
            trades_to_close.append(pos)
        else:
            logging.info("Position is ALIGNED. Holding.")
            position_is_aligned = True

    if trades_to_close:
        for pos_to_close in trades_to_close:
            try:
                contract_to_close = Contract(conId=pos_to_close.contract.conId)
                await ib.qualifyContractsAsync(contract_to_close)
                if contract_to_close.strike > 100: contract_to_close.strike /= 100.0
                order = MarketOrder('BUY' if pos_to_close.position < 0 else 'SELL', abs(pos_to_close.position))
                trade = ib.placeOrder(contract_to_close, order)
                await wait_for_fill(ib, trade, config, reason="Position Misaligned")
            except Exception as e:
                logging.error(f"Failed to close position for conId {pos_to_close.contract.conId}: {e}\n{traceback.format_exc()}")
        return True
    
    if position_is_aligned:
        logging.info(f"An aligned position for {future_contract.localSymbol} already exists.")
        return False
    return True

async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    logging.info(f"--- Executing {signal['direction']} Spread for {future_contract.localSymbol} ---")
    strategy_params, tuning = config['strategy'], config.get('strategy_tuning', {})
    spread_width = tuning.get('vertical_spread_strikes_apart', 2)
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None
    strikes = exp_details['strikes']
    try:
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
    except (ValueError, IndexError):
        logging.error("Could not find ATM strike in the chain."); return None

    if signal['direction'] == 'BULLISH':
        if atm_idx + spread_width >= len(strikes): logging.warning("Not enough strikes for Bull Call Spread."); return None
        legs_def = [('C', 'BUY', strikes[atm_idx]), ('C', 'SELL', strikes[atm_idx + spread_width])]
    else: # BEARISH
        if atm_idx - spread_width < 0: logging.warning("Not enough strikes for Bear Put Spread."); return None
        legs_def = [('P', 'BUY', strikes[atm_idx]), ('P', 'SELL', strikes[atm_idx - spread_width])]

    return await place_combo_order(ib, config, 'BUY', legs_def, exp_details, chain, underlying_price)

async def execute_volatility_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    logging.info(f"--- Executing {signal['level']} Volatility Strategy for {future_contract.localSymbol} ---")
    tuning = config.get('strategy_tuning', {})
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None
    strikes = exp_details['strikes']
    try:
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
    except (ValueError, IndexError):
        logging.error("Could not find ATM strike in the chain."); return None

    legs_def, order_action = [], ''
    if signal['level'] == 'HIGH': # Long Straddle
        legs_def, order_action = [('C', 'BUY', strikes[atm_idx]), ('P', 'BUY', strikes[atm_idx])], 'BUY'
    elif signal['level'] == 'LOW': # Iron Condor
        short_dist, wing_width = tuning.get('iron_condor_short_strikes_from_atm', 2), tuning.get('iron_condor_wing_strikes_apart', 2)
        if not (atm_idx - short_dist - wing_width >= 0 and atm_idx + short_dist + wing_width < len(strikes)):
            logging.warning("Not enough strikes for Iron Condor."); return None
        s = { 'lp': strikes[atm_idx - short_dist - wing_width], 'sp': strikes[atm_idx - short_dist],
              'sc': strikes[atm_idx + short_dist], 'lc': strikes[atm_idx + short_dist + wing_width] }
        legs_def, order_action = [('P', 'BUY', s['lp']), ('P', 'SELL', s['sp']), ('C', 'SELL', s['sc']), ('C', 'BUY', s['lc'])], 'SELL'

    return await place_combo_order(ib, config, order_action, legs_def, exp_details, chain, underlying_price)

async def place_combo_order(ib: IB, config: dict, action: str, legs_def: list, exp_details: dict, chain: dict, underlying_price: float) -> Trade | None:
    logging.info("--- Pricing individual legs before execution ---")

    net_theoretical_price = 0.0
    leg_prices = []

    for right, leg_action, strike in legs_def:
        leg_contract = FuturesOption(config['symbol'], exp_details['exp_date'], strike, right, chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(leg_contract)
        market_data = await get_option_market_data(ib, leg_contract)
        if market_data:
            pricing_result = price_option_black_scholes(S=underlying_price, K=strike, T=exp_details['days_to_exp'] / 365, r=market_data['risk_free_rate'], sigma=market_data['implied_volatility'], option_type=right)
            if pricing_result:
                price = pricing_result['price']
                leg_prices.append(price)
                if leg_action == 'BUY':
                    net_theoretical_price += price
                else: # SELL
                    net_theoretical_price -= price

    if len(leg_prices) != len(legs_def):
        logging.error("Failed to price all legs. Aborting combo order.")
        return None

    logging.info("--- End of pricing section ---")

    # Add a slippage buffer to the theoretical price to create a workable limit price
    slippage_allowance = config.get('strategy_tuning', {}).get('slippage_usd_per_contract', 5.0)

    # For a debit (BUY), we are willing to pay a bit more. For a credit (SELL), we are willing to accept a bit less.
    if action == 'BUY': # Debit
        limit_price = round(net_theoretical_price + slippage_allowance, 2)
    else: # Credit
        limit_price = round(net_theoretical_price - slippage_allowance, 2)

    logging.info(f"Net theoretical price: {net_theoretical_price:.2f}, Slippage allowance: {slippage_allowance:.2f}, Final Limit Price: {limit_price:.2f}")

    combo = Bag(symbol=config['symbol'], exchange=chain['exchange'], currency='USD')
    for right, leg_action, strike in legs_def:
        contract = FuturesOption(config['symbol'], exp_details['exp_date'], strike, right, chain['exchange'], tradingClass=chain['tradingClass'])
        await ib.qualifyContractsAsync(contract)
        combo.comboLegs.append(ComboLeg(conId=contract.conId, ratio=1, action=leg_action, exchange=chain['exchange']))

    order = LimitOrder(action, config['strategy']['quantity'], limit_price)
    trade = ib.placeOrder(combo, order)
    await wait_for_fill(ib, trade, config, timeout=180)
    return trade

# --- 3. Helper & Utility Functions ---

def log_trade_to_ledger(trade: Trade, reason: str = "Strategy Execution"):
    if trade.orderStatus.status != OrderStatus.Filled: return
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'local_symbol': trade.contract.localSymbol,
        'action': trade.order.action, 'quantity': trade.orderStatus.filled, 'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * (100 if isinstance(trade.contract, Bag) else 37500),
        'order_id': trade.order.orderId, 'reason': reason
    }
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv'), 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not os.path.isfile(f.name): writer.writeheader()
            writer.writerow(row)
        logging.info(f"Logged trade to ledger ({reason}): {row['action']} {row['quantity']} @ {row['avg_fill_price']}")
    except Exception as e:
        logging.error(f"Error writing to trade ledger: {e}")

async def wait_for_fill(ib: IB, trade: Trade, config: dict, timeout: int = 180, reason: str = "Strategy Execution"):
    logging.info(f"Waiting for order {trade.order.orderId} to fill...")
    start_time = time.time()
    while not trade.isDone():
        await asyncio.sleep(1)
        if (time.time() - start_time) > timeout:
            logging.warning(f"Order {trade.order.orderId} not filled within {timeout}s. Canceling.")
            send_notification(config, "Order Canceled", f"Order {trade.order.orderId} for {trade.contract.localSymbol} was canceled due to timeout.")
            ib.cancelOrder(trade.order)
            break
    if trade.orderStatus.status == OrderStatus.Filled:
        log_trade_to_ledger(trade, reason)


def is_market_open(contract_details, exchange_timezone_str: str):
    if not contract_details or not contract_details.liquidHours: return False
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str: continue
        try:
            start_str, end_str = session_str.split('-')
            start_dt, start_hm, end_hm = datetime.strptime(start_str.split(':')[0], '%Y%m%d').date(), datetime.strptime(start_str.split(':')[1], '%H%M').time(), datetime.strptime(end_str.split(':')[1], '%H%M').time()
            if now_tz.date() == start_dt and start_hm <= now_tz.time() < end_hm:
                logging.info("Market is open."); return True
        except (ValueError, IndexError): continue
    logging.info("Market is closed."); return False

def calculate_wait_until_market_open(contract_details, exchange_timezone_str: str) -> float:
    if not contract_details or not contract_details.liquidHours: return 3600
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    next_open_dt = None
    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str: continue
        try:
            session_start_dt = tz.localize(datetime.strptime(session_str.split('-')[0], '%Y%m%d:%H%M'))
            if session_start_dt > now_tz and (next_open_dt is None or session_start_dt < next_open_dt):
                next_open_dt = session_start_dt
        except (ValueError, IndexError): continue
    if next_open_dt:
        wait_seconds = (next_open_dt - now_tz).total_seconds() + 60
        logging.info(f"Next market open: {next_open_dt:%Y-%m-%d %H:%M:%S}. Waiting {wait_seconds / 3600:.2f} hours.")
        return wait_seconds
    return 3600


async def get_active_futures(ib: IB, symbol: str, exchange: str, count: int = 5) -> list[Contract]:
    """
    Fetches the next 'count' active futures contracts, sorted alphabetically
    by month code to match the prediction API's output.
    """
    logging.info(f"Fetching {count} active futures contracts for {symbol}, sorted alphabetically...")
    try:
        cds = await ib.reqContractDetailsAsync(Future(symbol, exchange=exchange))

        active_contracts = [
            cd.contract for cd in cds
            if cd.contract.lastTradeDateOrContractMonth > datetime.now().strftime('%Y%m')
        ]

        def get_month_code(contract):
            try:
                return contract.localSymbol.split(' ')[0][2]
            except (IndexError, AttributeError):
                return ''

        month_code_order = ['H', 'K', 'N', 'U', 'Z']

        active_contracts.sort(key=lambda c: month_code_order.index(get_month_code(c)) if get_month_code(c) in month_code_order else 99)

        logging.info(f"Found and sorted {len(active_contracts)} active contracts. Returning the first {count}.")
        return active_contracts[:count]

    except Exception as e:
        logging.error(f"Error fetching active futures: {e}"); return []

async def build_option_chain(ib: IB, future_contract: Contract):
    logging.info(f"Fetching option chain for future {future_contract.localSymbol}...")
    try:
        chains = await ib.reqSecDefOptParamsAsync(future_contract.symbol, future_contract.exchange, 'FUT', future_contract.conId)
        if not chains: return None
        chain = next((c for c in chains if c.exchange == future_contract.exchange), chains[0])
        return {'exchange': chain.exchange, 'tradingClass': chain.tradingClass, 'expirations': sorted(chain.expirations), 'strikes_by_expiration': {exp: sorted(chain.strikes) for exp in chain.expirations}}
    except Exception as e:
        logging.error(f"Failed to build option chain for {future_contract.localSymbol}: {e}"); return None

def get_expiration_details(chain: dict, future_exp: str) -> dict | None:
    valid_exp = [exp for exp in sorted(chain['expirations']) if exp <= future_exp]
    if not valid_exp:
        logging.warning(f"No option expiration before future {future_exp}. Using nearest available.")
        valid_exp = [exp for exp in sorted(chain['expirations']) if exp > datetime.now().strftime('%Y%m%d')]
        if not valid_exp: logging.error(f"No suitable option expirations for future {future_exp}."); return None
    chosen_exp = valid_exp[-1] if future_exp in [exp[:6] for exp in valid_exp] else valid_exp[0]
    days = (datetime.strptime(chosen_exp, '%Y%m%d').date() - datetime.now().date()).days
    return {'exp_date': chosen_exp, 'days_to_exp': days, 'strikes': chain['strikes_by_expiration'][chosen_exp]}


# --- 3.5 Position Management ---

async def close_orphaned_single_legs(ib: IB, config: dict):
    """
    Scans all open positions and closes any that are single-leg options,
    which are considered "orphans" in a spread-based strategy.
    """
    logging.info("--- Scanning for Orphaned Single-Leg Option Positions ---")

    try:
        all_positions = await ib.reqPositionsAsync()
        if not all_positions:
            logging.info("No open positions found to scan.")
            return

        orphans_found = []
        for p in all_positions:
            # An orphan is a position in a single FuturesOption contract with a non-zero position size.
            if isinstance(p.contract, FuturesOption) and p.position != 0:
                logging.warning(
                    f"ORPHAN DETECTED: Found a single-leg option position: "
                    f"{p.position} of {p.contract.localSymbol}."
                )
                orphans_found.append(p)

        if not orphans_found:
            logging.info("Scan complete. No orphaned single-leg positions found.")
            return

        logging.warning(f"Attempting to close {len(orphans_found)} orphaned position(s)...")
        send_notification(
            config,
            "Orphaned Position Alert",
            f"Closing {len(orphans_found)} single-leg option position(s). Please verify."
        )

        for orphan in orphans_found:
            try:
                # Create a new, clean contract object for the order to avoid conflicts.
                c = orphan.contract
                normalized_strike = c.strike / 100.0 if c.strike > 100 else c.strike

                contract_to_close = FuturesOption(
                    symbol=c.symbol,
                    lastTradeDateOrContractMonth=c.lastTradeDateOrContractMonth,
                    strike=normalized_strike,
                    right=c.right,
                    exchange=c.exchange,
                    tradingClass=c.tradingClass
                )

                action = 'BUY' if orphan.position < 0 else 'SELL'
                quantity = abs(orphan.position)

                logging.info(f"Placing market order to {action} {quantity} of {c.localSymbol} with normalized strike {normalized_strike}...")
                order = MarketOrder(action, quantity)
                trade = ib.placeOrder(contract_to_close, order)

                # We can run the fill check in the background
                asyncio.create_task(wait_for_fill(ib, trade, config, reason="Orphaned Leg Cleanup"))

            except Exception as e:
                logging.error(
                    f"Failed to place closing order for orphan {orphan.contract.localSymbol}. "
                    f"Manual intervention may be required. Error: {e}"
                )
    except Exception as e:
        logging.error(f"An error occurred during the orphan scan: {e}", exc_info=True)


# --- 4. Risk, Sanity Checks & Connection Heartbeat ---

def is_trade_sane(signal: dict, config: dict) -> bool:
    min_confidence = config.get('risk_management', {}).get('min_confidence_threshold', 0.0)
    if signal['confidence'] < min_confidence:
        logging.warning(f"Sanity Check FAILED: Signal confidence {signal['confidence']} is below threshold {min_confidence}.")
        return False
    logging.info("Sanity Check PASSED."); return True

async def monitor_positions_for_risk(ib: IB, config: dict):
    """
    A background task that continuously monitors open positions for stop-loss
    or take-profit triggers. This function is stateful and manages its own P&L subscriptions.
    """
    risk_params = config.get('risk_management', {})
    stop_loss_pct = risk_params.get('stop_loss_percentage')
    take_profit_pct = risk_params.get('take_profit_percentage')
    check_interval = risk_params.get('check_interval_seconds', 300)

    if not stop_loss_pct and not take_profit_pct:
        logging.info("Risk monitor disabled: No stop-loss or take-profit percentage configured.")
        return

    logging.info(f"Starting intraday risk monitor. Stop: {stop_loss_pct}%, Profit: {take_profit_pct}%. Interval: {check_interval}s.")

    # Stateful dictionaries to manage subscriptions and closing orders
    pnl_subscriptions = {}  # {conId: PnLSingle object}
    closed_order_ids = set()

    while True:
        try:
            await asyncio.sleep(check_interval)
            if not ib.isConnected():
                logging.warning("Risk monitor paused: IBKR not connected.")
                continue

            positions = await ib.reqPositionsAsync()
            account = ib.managedAccounts()[0]

            active_con_ids = {p.contract.conId for p in positions if p.position != 0}

            # --- Clean up subscriptions for positions that are now closed ---
            for con_id in list(pnl_subscriptions.keys()):
                if con_id not in active_con_ids:
                    logging.info(f"Position with conId {con_id} is closed. Canceling P&L subscription.")
                    ib.cancelPnLSingle(pnl_subscriptions[con_id])
                    del pnl_subscriptions[con_id]
                    if con_id in closed_order_ids:
                        closed_order_ids.remove(con_id)

            if not positions:
                continue

            # --- Check all active positions ---
            for p in positions:
                con_id = p.contract.conId
                if p.position == 0 or con_id in closed_order_ids:
                    continue

                # --- Subscribe to P&L for new positions ---
                if con_id not in pnl_subscriptions:
                    logging.info(f"New position detected ({p.contract.localSymbol}). Subscribing to P&L for conId {con_id}.")
                    pnl_subscriptions[con_id] = ib.reqPnLSingle(account, '', con_id)
                    await asyncio.sleep(2)  # Give a moment for the first P&L update to arrive

                # --- Check P&L on existing subscription ---
                pnl_data = pnl_subscriptions[con_id]

                if util.isNan(pnl_data.unrealizedPnL):
                    logging.warning(f"Could not get PnL for {p.contract.localSymbol} (conId {con_id}). Will retry on next cycle.")
                    continue

                cost_basis_per_contract = p.avgCost
                pnl_per_contract = pnl_data.unrealizedPnL / abs(p.position)

                if cost_basis_per_contract <= 0:
                    logging.warning(f"Cost basis for {p.contract.localSymbol} is zero or negative. Cannot calculate percentage PnL.")
                    continue

                pnl_percentage = (pnl_per_contract / cost_basis_per_contract) * 100
                logging.info(f"Position {p.contract.localSymbol}: Cost/Contract=${cost_basis_per_contract:.2f}, PnL/Contract=${pnl_per_contract:.2f} ({pnl_percentage:.2f}%)")

                # --- Trigger stop-loss or take-profit based on percentage ---
                reason = None
                if stop_loss_pct and pnl_percentage <= -abs(stop_loss_pct):
                    reason = "Stop-Loss"
                elif take_profit_pct and pnl_percentage >= abs(take_profit_pct):
                    reason = "Take-Profit"

                if reason:
                    logging.warning(f"{reason.upper()} TRIGGERED for {p.contract.localSymbol} at {pnl_percentage:.2f}%. Closing position.")
                    send_notification(config, reason, f"{p.contract.localSymbol} closed. PnL: {pnl_percentage:.2f}%")

                    contract_to_close = p.contract
                    order = MarketOrder('BUY' if p.position < 0 else 'SELL', abs(p.position))
                    trade = ib.placeOrder(contract_to_close, order)

                    closed_order_ids.add(con_id)
                    asyncio.create_task(wait_for_fill(ib, trade, config, reason=reason))
                    break # Re-evaluate positions after closing one

        except (asyncio.CancelledError, ConnectionError):
            logging.info("Risk monitor stopping. Cleaning up all P&L subscriptions.")
            for sub in pnl_subscriptions.values():
                ib.cancelPnLSingle(sub)
            pnl_subscriptions.clear()
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the risk monitor loop: {e}", exc_info=True)
            await asyncio.sleep(60)

async def send_heartbeat(ib: IB):
    logging.info("Heartbeat task started.")
    while True:
        try:
            await asyncio.sleep(300)
            if ib.isConnected():
                server_time = await ib.reqCurrentTimeAsync()
                logging.info(f"Heartbeat: Connection is alive. Server time: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logging.error(f"Error in heartbeat loop: {e}")

# --- 5. Main Execution Loop ---

async def main_runner():
    ib = IB()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    monitor_task, heartbeat_task = None, None

    while True:
        try:
            with open(config_path, 'r') as f: config = json.load(f)
            conn_settings = config.get('connection', {})
            host, port = conn_settings.get('host', '127.0.0.1'), conn_settings.get('port', 7497)

            if not ib.isConnected():
                logging.info(f"Connecting to {host}:{port}...")
                await ib.connectAsync(host, port, clientId=conn_settings.get('clientId', random.randint(1, 1000)))
                send_notification(config, "Script Started", "Trading script has successfully connected to IBKR.")
                if monitor_task is None or monitor_task.done(): monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
                if heartbeat_task is None or heartbeat_task.done(): heartbeat_task = asyncio.create_task(send_heartbeat(ib))

            # Fetch the 5 active futures, sorted alphabetically for the API
            active_futures_sorted = await get_active_futures(ib, config['symbol'], config['exchange'])
            if not active_futures_sorted:
                raise ConnectionError("Could not find any active futures contracts.")

            # Use the first contract (chronologically, not necessarily the first in the alpha list) for market hours check
            # This assumes the market hours are the same for all contracts, which is a safe assumption for KC futures.
            first_future_details_list = await ib.reqContractDetailsAsync(Future(config['symbol'], exchange=config['exchange']))
            if not first_future_details_list:
                raise ConnectionError("Could not get contract details to check market hours.")

            if not is_market_open(first_future_details_list[0], config['exchange_timezone']):
                await asyncio.sleep(calculate_wait_until_market_open(first_future_details_list[0], config['exchange_timezone']))
                continue

            # --- Orphaned Position Cleanup ---
            # Before processing new trades, run the safety check to close any single-leg options.
            await close_orphaned_single_legs(ib, config)
            # Add a small delay to allow closing orders to be processed before continuing.
            await asyncio.sleep(5)

            # Get signals based on the alphabetically sorted futures list
            signals = get_prediction_from_api(config, active_futures_sorted)
            if not signals:
                logging.warning("Could not get a valid signal list from the API. Waiting for next cycle.")
                await asyncio.sleep(300)
                continue

            trades_summary = {'filled': [], 'cancelled': []}
            for signal in signals:
                # The signal now directly corresponds to a future contract from our sorted list.
                future = next((f for f in active_futures_sorted if f.lastTradeDateOrContractMonth.startswith(signal.get("contract_month", ""))), None)
                if not future:
                    logging.warning(f"Could not find a matching active future for signal month {signal.get('contract_month')}. Inconsistency detected.");
                    continue

                logging.info(f"\n===== Processing Signal for {future.localSymbol} ({signal['direction']}) =====")
                ticker = ib.reqMktData(future, '', False, False); await asyncio.sleep(1)
                price = ticker.marketPrice()
                if util.isNan(price): logging.error(f"Failed to get market price for {future.localSymbol}."); continue
                
                logging.info(f"Current price for {future.localSymbol}: {price}")
                
                if await manage_existing_positions(ib, config, signal, price, future) and is_trade_sane(signal, config):
                    chain = await build_option_chain(ib, future)
                    if not chain: continue

                    trade = await (execute_directional_strategy if signal['prediction_type'] == 'DIRECTIONAL' else execute_volatility_strategy)(ib, config, signal, chain, price, future)
                    if trade: trades_summary['filled' if trade.orderStatus.status == OrderStatus.Filled else 'cancelled'].append(trade)

            # End of cycle summary notification
            summary_msg = "<b>-- Trading Cycle Summary --</b>"
            if trades_summary['filled']: summary_msg += "\n<b>Positions Opened:</b>\n" + "".join([f"- {t.contract.localSymbol}: {t.order.action} {t.orderStatus.filled} @ ${t.orderStatus.avgFillPrice:.2f}\n" for t in trades_summary['filled']])
            if trades_summary['cancelled']: summary_msg += "\n<b>Orders Cancelled:</b>\n" + "".join([f"- {t.contract.localSymbol}: {t.order.action} {t.order.totalQuantity}\n" for t in trades_summary['cancelled']])
            if not trades_summary['filled'] and not trades_summary['cancelled']: summary_msg += "\nNo new positions were opened or attempted."
            send_notification(config, "Trading Cycle Complete", summary_msg)

            ny_tz = pytz.timezone(config['exchange_timezone'])
            now_ny = datetime.now(ny_tz)
            h, m = map(int, config.get('trade_execution_time_ny', '04:26').split(':'))
            next_trade = now_ny.replace(hour=h, minute=m, second=0, microsecond=0)
            if now_ny >= next_trade: next_trade += timedelta(days=1)
            wait = (next_trade - now_ny).total_seconds()
            logging.info(f"\nTrading cycle complete. Waiting {wait / 3600:.2f} hours until {next_trade:%Y-%m-%d %H:%M:%S} NY time.")
            await asyncio.sleep(wait)

        except (ConnectionError, OSError, asyncio.TimeoutError, asyncio.CancelledError) as e:
            msg = f"Connection error: {e}. Reconnecting..."
            logging.error(msg)
            send_notification(config, "Connection Error", msg)
            if monitor_task and not monitor_task.done(): monitor_task.cancel()
            if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
            monitor_task, heartbeat_task = None, None
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)
        except Exception as e:
            msg = f"An unexpected critical error occurred: {e}. Restarting logic in 1 min..."
            logging.critical(f"{msg}\n{traceback.format_exc()}")
            send_notification(config, "Critical Error", msg)
            if monitor_task and not monitor_task.done(): monitor_task.cancel()
            if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
            monitor_task, heartbeat_task = None, None
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        util.run(main_runner())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Script stopped manually.")
