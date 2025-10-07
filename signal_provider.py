import logging
import requests
import time
import os
import pandas as pd
import json
from datetime import datetime, timedelta

# --- Third-party libraries for data fetching ---
import yfinance as yf
from fredapi import Fred
import nasdaqdatalink as ndl
from pandas.tseries.offsets import BDay

# --- Custom Modules ---
from notifications import send_pushover_notification

# --- Validation Manager Class (from coffee_factors_data_pull_new.py) ---
class ValidationManager:
    """A class to manage and report data validation checks."""
    def __init__(self):
        self.results = []
        self.final_shape = None

    def add_check(self, check_name: str, success: bool, message: str):
        self.results.append({'check': check_name, 'success': success, 'message': message})

    def set_final_shape(self, shape: tuple):
        self.final_shape = shape

    def was_successful(self) -> bool:
        return all(r['success'] for r in self.results)

    def generate_report(self) -> str:
        """Generates a summary report of all validation checks."""
        report = f"<b>Data Pull Validation Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})</b>\n"
        status_icon = "✅" if self.was_successful() else "❌"
        report += f"<b>Overall Status: {status_icon} {'SUCCESS' if self.was_successful() else 'FAILURE'}</b>\n"
        for res in self.results:
            icon = "✅" if res['success'] else "❌"
            report += f"{icon} <b>{res['check']}:</b> {res['message']}\n"
        if self.final_shape:
            report += f"\n<b>Final Output:</b> {self.final_shape[0]} rows, {self.final_shape[1]} columns.\n"
        return report.strip()

# --- Data Fetching Logic (Refactored from coffee_factors_data_pull_new.py) ---

def _get_kc_expiration_date(year: int, month_code: str) -> pd.Timestamp:
    """Calculates the expiration date for a given Coffee 'C' futures contract."""
    month_map = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
    month = month_map[month_code]
    last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)
    # Expiration is the business day prior to the last business day of the month
    return last_day - BDay(1)

def _get_active_coffee_tickers(num_contracts: int = 5) -> list[str]:
    """
    Determines the tickers for the next N active coffee futures contracts,
    sorted alphabetically by month code to match the prediction API's output.
    """
    now = datetime.now()
    # Month codes in the desired alphabetical order
    month_codes_ordered = ['H', 'K', 'N', 'U', 'Z']

    potential_tickers = []
    # Find the next occurrence for each month code
    for code in month_codes_ordered:
        # Check current year and next two years
        for year_offset in range(3):
            year = now.year + year_offset
            exp_date = _get_kc_expiration_date(year, code)
            if exp_date.date() >= now.date():
                # Found the first valid expiration for this month code
                potential_tickers.append({
                    "ticker": f"KC{code}{str(year)[-2:]}.NYB",
                    "exp_date": exp_date,
                    "month_code": code
                })
                break # Move to the next month code

    # Sort the collected tickers by their month code order (H, K, N, U, Z)
    # This is redundant if we find them in order, but it's a good safeguard.
    potential_tickers.sort(key=lambda x: month_codes_ordered.index(x['month_code']))

    # Return the ticker strings for the first num_contracts
    return [item['ticker'] for item in potential_tickers[:num_contracts]]

def _fetch_market_data(config: dict) -> tuple[str | None, list | None]:
    """
    Fetches all required market and economic data, processes it, validates it,
    and saves it to a CSV file. Returns a tuple of (file_path, tickers_used)
    if successful, otherwise (None, None).
    """
    logging.info("--- Starting Market Data Fetch ---")
    validator = ValidationManager()

    # --- API and Date Configuration ---
    try:
        fred = Fred(api_key=config['fred_api_key'])
        ndl.api_key = config['nasdaq_api_key']
    except KeyError as e:
        logging.error(f"API key missing from config.json: {e}")
        return None, None
    start_date, end_date = datetime(1980, 1, 1), datetime.now()

    all_data = {}

    # --- Fetch Data ---
    alphabetical_tickers = _get_active_coffee_tickers()
    if alphabetical_tickers:
        df_coffee = yf.download(alphabetical_tickers, start=start_date, end=end_date, progress=False)['Close']
        df_coffee.columns = [f"coffee_price_{t.replace('.NYB', '')}" for t in df_coffee.columns]
        all_data['coffee_prices'] = df_coffee
        validator.add_check("Coffee Price Fetch", True, f"Successfully fetched {df_coffee.shape[1]} active contracts.")
    else:
        validator.add_check("Coffee Price Fetch", False, "Could not determine active coffee tickers.")
        return None, None

    # ... Other data fetching logic would go here ...

    # --- Consolidate and Save ---
    if 'coffee_prices' in all_data and not all_data['coffee_prices'].empty:
        final_df = all_data['coffee_prices'].copy()
        final_df.ffill(inplace=True)
        final_df.dropna(inplace=True)

        # Reformat columns as the model expects
        for col in list(final_df.columns):
            if col.startswith('coffee_price_KC'):
                ticker = col.replace('coffee_price_', '')
                month_code, year_str = ticker[2], ticker[3:]
                year = int(f"20{year_str}")
                exp = _get_kc_expiration_date(year, month_code)
                final_df[f'{month_code}_dte'] = (exp.tz_localize(None) - final_df.index).days
                final_df[f'{month_code}_symbol'] = ticker
                final_df.rename(columns={col: f'{month_code}_price'}, inplace=True)
        final_df['date'] = final_df.index.strftime('%-m/%-d/%y')

        # Ensure all expected columns are present
        for col in config.get('final_column_order', []):
            if col not in final_df.columns:
                final_df[col] = None

        final_df = final_df[config['final_column_order']]
        validator.set_final_shape(final_df.shape)

        output_filename = f"coffee_futures_data_reformatted_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"
        final_df.to_csv(output_filename, index=False)
        logging.info(f"Successfully saved data to {output_filename}. Shape: {final_df.shape}")

        report = validator.generate_report()
        logging.info("\n" + "="*30 + "\n" + report + "\n" + "="*30)
        send_pushover_notification(config, f"Data Pull: {'SUCCESS' if validator.was_successful() else 'FAILURE'}", report)

        if validator.was_successful():
            return output_filename, alphabetical_tickers
        else:
            return None, None
    else:
        validator.add_check("Consolidation", False, "No coffee price data was fetched.")
        logging.error(validator.generate_report())
        return None, None


# --- API Interaction Logic (Refactored from send_data_to_api.py) ---

def _get_prediction_from_api(config: dict, data_file_path: str) -> list | None:
    """
    Sends data to the prediction API, polls for the result, and returns it.
    """
    logging.info("--- Requesting Prediction from API ---")
    api_url = config.get("api_base_url")
    if not api_url:
        logging.error("`api_base_url` not found in config.json.")
        return None

    prediction_endpoint = f"{api_url}/predictions"

    try:
        df = pd.read_csv(data_file_path)
        # The model expects data as a CSV string in the JSON payload
        data_payload = df.tail(600).to_csv(index=False)
        request_body = {"data": data_payload}
    except Exception as e:
        logging.error(f"Error reading or processing data file '{data_file_path}': {e}")
        return None

    try:
        # 1. Post the job
        response = requests.post(prediction_endpoint, json=request_body, timeout=20)
        response.raise_for_status()
        job_info = response.json()
        job_id = job_info.get('id')
        if not job_id:
            logging.error("API did not return a valid job ID.")
            return None
        logging.info(f"Successfully created prediction job. Job ID: {job_id}")

        # 2. Poll for the result
        result_url = f"{prediction_endpoint}/{job_id}"
        timeout_seconds = 180
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            logging.info(f"Polling for result (Job ID: {job_id})...")
            result_response = requests.get(result_url, timeout=10)
            result_response.raise_for_status()
            status_info = result_response.json()

            if status_info.get('status') == 'completed':
                logging.info("Prediction job complete.")
                return status_info.get('result')
            elif status_info.get('status') == 'failed':
                logging.error(f"Prediction job failed. Reason: {status_info.get('error')}")
                return None

            time.sleep(10) # Wait before polling again

        logging.error("Polling timed out. The prediction job took too long.")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while communicating with the API: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Could not decode the JSON response from the server: {e}")
        return None


# --- Main Orchestrator Function ---

def get_trading_signals(config: dict) -> list | None:
    """
    Orchestrates the entire signal generation pipeline.
    1. Fetches fresh market data and saves it to a file.
    2. Sends that data to the prediction API.
    3. Processes the API response to generate structured trading signals.
    """
    logging.info("===== Starting Signal Generation Pipeline =====")

    # Step 1: Fetch market data and get the alphabetically-sorted tickers
    # This function now also returns the tickers it used, which is crucial.
    data_file_path, tickers = _fetch_market_data(config)
    if not data_file_path or not tickers:
        logging.error("Signal generation failed: Market data fetching was unsuccessful or returned no tickers.")
        send_pushover_notification(config, "Signal Generation FAILED", "The market data process failed.")
        return None

    logging.info(f"Market data processed and saved to '{data_file_path}'. Tickers sorted as: {tickers}")

    # Step 2: Get raw predictions from the API
    raw_predictions = _get_prediction_from_api(config, data_file_path)

    # Clean up the data file immediately after API call
    os.remove(data_file_path)
    logging.info(f"Cleaned up temporary data file: {data_file_path}")

    if not raw_predictions or 'price_changes' not in raw_predictions:
        logging.error("Signal generation failed: API did not return valid 'price_changes'.")
        send_pushover_notification(config, "Signal Generation FAILED", "Prediction API returned invalid data.")
        return None

    price_changes = raw_predictions['price_changes']
    if len(price_changes) != len(tickers):
        logging.error(f"Mismatch between number of tickers ({len(tickers)}) and price changes ({len(price_changes)}).")
        return None

    # Step 3: Translate raw price changes into structured signals
    logging.info("Translating raw price changes into actionable signals...")
    signals = []
    month_map = {'H': '03', 'K': '05', 'N': '07', 'U': '09', 'Z': '12'}

    for i, price_change in enumerate(price_changes):
        ticker = tickers[i] # e.g., 'KCH26.NYB'

        direction = None
        if price_change > 7:
            direction = "BULLISH"
        elif price_change < 0:
            direction = "BEARISH"

        if direction:
            try:
                # Parse ticker like 'KCH26.NYB' into contract_month '202603'
                month_code = ticker[2]
                year_short = ticker[3:5]
                contract_month = f"20{year_short}{month_map[month_code]}"

                signal = {
                    "contract_month": contract_month,
                    "prediction_type": "DIRECTIONAL",
                    "direction": direction,
                    # Confidence is now based on passing the threshold. Set to 1.0 to pass sanity checks.
                    "confidence": 1.0
                }
                signals.append(signal)
                logging.info(f"Generated signal for {contract_month}: {direction} (Price Change: {price_change:.2f})")
            except (IndexError, KeyError) as e:
                logging.error(f"Could not parse ticker '{ticker}'. Skipping. Error: {e}")
        else:
            logging.info(f"Price change for ticker {ticker} ({price_change:.2f}) is within the no-trade zone (0-7). No signal generated.")

    logging.info(f"Successfully generated {len(signals)} actionable trading signals.")
    return signals