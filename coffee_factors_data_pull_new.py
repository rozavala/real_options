"""Fetches, consolidates, and validates data for coffee futures prediction.

This script pulls data from multiple sources to create a comprehensive dataset
for analysis and model training. The sources include:
- Yahoo Finance (yfinance): For coffee futures prices and other market data.
- Federal Reserve Economic Data (FRED): For key economic indicators.
- Open-Meteo: For historical weather data from relevant locations.
- Nasdaq Data Link: For other financial data.

The script performs a series of validation checks, consolidates the data into
a single time-series DataFrame, and saves the result as a CSV file. It also
generates a report summarizing the validation results and sends a notification.
"""
# --- Install all required libraries ---
# You should run this command in your terminal or command prompt, not inside the script:
# pip install yfinance fredapi pandas requests openpyxl

# --- Import Libraries ---
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from fredapi import Fred
import requests
import zipfile
import io
from pandas.tseries.offsets import BDay

# --- Custom Modules ---
from notifications import send_pushover_notification

# --- Validation Manager Class ---
class ValidationManager:
    """A class to manage and report data validation checks."""
    def __init__(self):
        self.results = []
        self.final_shape = None

    def add_check(self, check_name: str, success: bool, message: str):
        self.results.append({'check': check_name, 'success': success, 'message': message})

    def set_final_shape(self, shape: tuple[int, int]):
        self.final_shape = shape

    def was_successful(self) -> bool:
        return all(r['success'] for r in self.results)

    def generate_report(self) -> str:
        report = "<b>Data Pull Validation Report</b>\n"
        report += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        success_count = sum(1 for r in self.results if r['success'])
        failure_count = len(self.results) - success_count
        
        status_icon = "✅" if self.was_successful() else "❌"
        report += f"<b>Overall Status: {status_icon} {'SUCCESS' if self.was_successful() else 'FAILURE'}</b>\n"
        report += f"({success_count} checks passed, {failure_count} checks failed)\n\n"

        for res in self.results:
            icon = "✅" if res['success'] else "❌"
            report += f"{icon} <b>{res['check']}:</b> {res['message']}\n"
        
        if self.final_shape:
            report += f"\n<b>Final Output File Stats:</b>\n"
            report += f"- Rows: {self.final_shape[0]}\n"
            report += f"- Columns: {self.final_shape[1]}\n"

        return report.strip()

# --- Main Script ---

def get_kc_expiration_date(year: int, month_code: str) -> pd.Timestamp:
    """Calculates the expiration date for a Coffee 'C' futures contract."""
    month_map = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
    month = month_map.get(month_code)
    if not month:
        raise ValueError(f"Invalid month code: {month_code}")
    last_day_of_month = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)
    last_business_day = last_day_of_month
    while last_business_day.weekday() > 4:
        last_business_day -= pd.Timedelta(days=1)
    return last_business_day - BDay(1)

def get_active_coffee_tickers(num_contracts: int = 5) -> list[str]:
    """Determines the tickers for the next N active coffee futures contracts."""
    print(f"Dynamically determining the next {num_contracts} coffee tickers...")
    now = datetime.now()
    current_year = now.year
    contract_months = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
    potential_tickers = []
    for year_offset in range(3):
        year = current_year + year_offset
        for code in contract_months.keys():
            try:
                expiration_date = get_kc_expiration_date(year, code)
                # Filter out contracts expiring in less than 45 days (to account for early Option expiration)
                if expiration_date.date() >= (now.date() + timedelta(days=45)):
                    potential_tickers.append((f"KC{code}{str(year)[-2:]}.NYB", expiration_date))
            except ValueError:
                continue
    potential_tickers.sort(key=lambda x: x[1])
    active_tickers = [ticker for ticker, exp in potential_tickers[:num_contracts]]
    print(f"... chronological tickers found: {active_tickers}")
    return active_tickers

def main(config: dict) -> pd.DataFrame | None:
    """
    Runs the main data pull, validation, and processing pipeline.
    Returns a DataFrame on success, None on failure.
    """
    if not config:
        return None

    validator = ValidationManager()

    # --- API and Date Configuration ---
    fred = Fred(api_key=config['fred_api_key'])
    start_date = datetime(1980, 1, 1)
    end_date = datetime.now()

    print("Setup complete.")
    all_data = {}
    ordinal_names = ["front_month", "second_month", "third_month", "fourth_month", "fifth_month"]

    # --- 1. Fetch Coffee Futures Data ---
    chronological_tickers = get_active_coffee_tickers(num_contracts=5)
    df_coffee = pd.DataFrame()
    if not chronological_tickers:
        validator.add_check("Coffee Ticker Generation", False, "Could not determine active coffee tickers.")
    else:
        print("\nFetching Coffee Prices...")
        try:
            df_coffee = yf.download(chronological_tickers, start=start_date, end=end_date, progress=False)
            if not df_coffee.empty and 'Close' in df_coffee and not df_coffee['Close'].dropna(how='all').empty:
                df_price = df_coffee['Close'].copy()
                df_price.dropna(axis=1, how='all', inplace=True)

                valid_tickers = [t for t in chronological_tickers if t in df_price.columns]
                price_rename_map = {
                    ticker: f"{ordinal_names[i]}_price"
                    for i, ticker in enumerate(valid_tickers) if i < len(ordinal_names)
                }
                df_price.rename(columns=price_rename_map, inplace=True)

                all_data['coffee_prices'] = df_price
                validator.add_check("Coffee Price Fetch", True, f"Successfully fetched {df_price.shape[1]} active contracts.")
            else:
                validator.add_check("Coffee Price Fetch", False, "yfinance returned no data for coffee tickers.")
        except Exception as e:
            validator.add_check("Coffee Price Fetch", False, f"An error occurred: {e}")

    # --- 2. Fetch Weather Data ---
    print("\nFetching weather data...")
    weather_stations = config['weather_stations']
    try:
        lats = ",".join(map(str, [v[0] for v in weather_stations.values()]))
        lons = ",".join(map(str, [v[1] for v in weather_stations.values()]))
        api_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lats}&longitude={lons}&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&daily=temperature_2m_mean,precipitation_sum&timezone=GMT"
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        batch_data = response.json()

        for i, location in enumerate(weather_stations.keys()):
            location_data = batch_data[i]['daily']
            weather_df = pd.DataFrame(location_data)
            weather_df['time'] = pd.to_datetime(weather_df['time'])
            weather_df.set_index('time', inplace=True)
            weather_df.rename(columns={'temperature_2m_mean': f'{location}_avg_temp', 'precipitation_sum': f'{location}_precipitation'}, inplace=True)
            all_data[location] = weather_df
        validator.add_check("Weather Data Fetch", True, f"Successfully fetched data for {len(weather_stations)} locations.")
    except Exception as e:
        validator.add_check("Weather Data Fetch", False, f"Could not fetch weather data: {e}")

    # --- 3. Fetch Economic and Market Data ---
    print("\nFetching economic data from FRED...")
    for name, code in config['fred_series'].items():
        try:
            data = fred.get_series(code, start_date, end_date).to_frame(name=name)
            all_data[name] = data if not data.empty else pd.DataFrame()
        except Exception as e:
            validator.add_check(f"FRED Fetch ({name})", False, f"Could not fetch series '{code}'. Error: {e}")
            all_data[name] = pd.DataFrame()
    validator.add_check("FRED Data Fetch", True, "Completed all FRED series requests.")

    print("\nFetching other market data from yfinance...")
    try:
        tickers_list = list(config['yf_series_map'].keys())
        yf_multi_data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)

        if not yf_multi_data.empty:
            if len(tickers_list) == 1:
                ticker = tickers_list[0]
                name = config['yf_series_map'][ticker]
                all_data[name] = yf_multi_data[['Close']].copy().rename(columns={'Close': name})
            else:
                for ticker, name in config['yf_series_map'].items():
                    if 'Close' in yf_multi_data.columns and ticker in yf_multi_data['Close'].columns:
                        all_data[name] = yf_multi_data['Close'][[ticker]].copy().rename(columns={ticker: name})
        validator.add_check("Other Market Data Fetch", True, "Completed all other yfinance requests.")
    except Exception as e:
         validator.add_check("Other Market Data Fetch", False, f"An error occurred: {e}")

    # --- 3a. Process OHLV and Fetch COT Data ---
    print("\nProcessing OHLV and fetching COT data...")

    if not df_coffee.empty:
        try:
            df_ohlv = df_coffee[['Open', 'High', 'Low', 'Volume']].copy()
            df_ohlv.columns = ['_'.join(col).strip() for col in df_ohlv.columns.values]

            ticker_to_ordinal = {
                ticker: ordinal_names[i]
                for i, ticker in enumerate(chronological_tickers)
                if i < len(ordinal_names)
            }

            rename_map = {}
            for col in df_ohlv.columns:
                measure, ticker = col.split('_', 1)
                if ticker in ticker_to_ordinal:
                    ordinal_name = ticker_to_ordinal[ticker]
                    rename_map[col] = f"{ordinal_name}_{measure.lower()}"

            df_ohlv.rename(columns=rename_map, inplace=True)

            all_data['coffee_ohlv'] = df_ohlv
            validator.add_check("Coffee OHLV Processing", True, "Successfully processed OHLV data.")
        except Exception as e:
            validator.add_check("Coffee OHLV Processing", False, f"Error processing OHLV data: {e}")

    try:
        print("Fetching historical COT data directly from CFTC...")
        all_cot_dfs = []
        cftc_ticker = 'COFFEE C'
        
        for year in range(start_date.year, end_date.year + 1):
            url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        # IMPROVEMENT: Dynamically find the text file
                        file_name = next((f for f in z.namelist() if f.endswith('.txt') or 'annual' in f.lower()), None)

                        if file_name:
                            with z.open(file_name) as f:
                                yearly_df = pd.read_csv(f, low_memory=False)

                                # OPTIMIZATION: Filter immediately to reduce memory usage
                                # Strip whitespace from column names to be safe
                                yearly_df.columns = yearly_df.columns.str.strip()

                                # Filter for Coffee
                                if 'Market and Exchange Names' in yearly_df.columns:
                                    yearly_df = yearly_df[yearly_df['Market and Exchange Names'].str.contains(cftc_ticker, na=False, case=False)]

                                    # Keep only necessary columns
                                    required_cols = [
                                        'As of Date in Form YYYY-MM-DD',
                                        'Noncommercial Positions-Long (All)',
                                        'Noncommercial Positions-Short (All)'
                                    ]
                                    # Check if columns exist (naming might vary slightly over 40 years)
                                    existing_cols = [c for c in required_cols if c in yearly_df.columns]

                                    if len(existing_cols) == 3:
                                        all_cot_dfs.append(yearly_df[existing_cols])
                                        print(f"... Successfully processed and filtered year {year} (Rows: {len(yearly_df)})")
                                    else:
                                         print(f"... Warning: Missing columns in year {year}. Found: {existing_cols}")
                                else:
                                     print(f"... Warning: 'Market and Exchange Names' column not found in year {year}")

                        else:
                            print(f"... Warning: No valid text file found in zip for year {year}")
            except Exception as e:
                print(f"... Warning: Could not process COT data for year {year}. Error: {e}")

        print("Concatenating COT data...")
        if all_cot_dfs:
            cot_df = pd.concat(all_cot_dfs, ignore_index=True)
            cot_df['Date'] = pd.to_datetime(cot_df['As of Date in Form YYYY-MM-DD'], format='%Y-%m-%d')
            cot_df.set_index('Date', inplace=True)
            cot_cols = {'Noncommercial Positions-Long (All)': 'cot_noncomm_long', 'Noncommercial Positions-Short (All)': 'cot_noncomm_short'}
            cot_df.rename(columns=cot_cols, inplace=True)
            cot_df['cot_noncomm_long'] = pd.to_numeric(cot_df['cot_noncomm_long'], errors='coerce')
            cot_df['cot_noncomm_short'] = pd.to_numeric(cot_df['cot_noncomm_short'], errors='coerce')
            cot_df['cot_noncomm_net'] = cot_df['cot_noncomm_long'] - cot_df['cot_noncomm_short']
            all_data['cot_report'] = cot_df[['cot_noncomm_net']]
            validator.add_check("COT Report Fetch", True, f"Successfully fetched CFTC data. Last entry: {cot_df.index.max().strftime('%Y-%m-%d')}")
        else:
             validator.add_check("COT Report Fetch", False, "No COT data could be aggregated.")

    except Exception as e:
        validator.add_check("COT Report Fetch", False, f"Could not fetch CFTC data directly: {e}")

    print("\nAll data fetching complete.")

    # --- 4. Consolidate, Validate, and Save ---
    if 'coffee_prices' in all_data and not all_data['coffee_prices'].empty:
        final_df = all_data['coffee_prices'].copy()
        for name, df in all_data.items():
            if name != 'coffee_prices':
                final_df = final_df.join(df, how='left')
        
        total_cells = final_df.size
        missing_before = final_df.isnull().sum().sum()
        missing_pct = (missing_before / total_cells) * 100 if total_cells > 0 else 0
        validator.add_check("Pre-Fill Missing Data", missing_pct < 50, f"Total missing data before fill is {missing_pct:.2f}%.")
        final_df.ffill(inplace=True)

        front_month_col = "front_month_price"
        last_date = final_df.index.max()
        is_recent = (datetime.now() - last_date) < timedelta(days=5) if pd.notna(last_date) else False
        validator.add_check("Data Recency", is_recent, f"Most recent data point is from {last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'N/A'}.")
        
        if front_month_col in final_df.columns:
            final_df['price_pct_change'] = final_df[front_month_col].pct_change().abs()
            max_spike = final_df['price_pct_change'].max()
            spike_threshold = config['validation_thresholds']['price_spike_pct']
            no_spikes = max_spike < spike_threshold
            validator.add_check("Price Spike Detection", no_spikes, f"Max daily change was {max_spike:.2%}. Threshold is {spike_threshold:.0%}.")
            final_df.drop(columns=['price_pct_change'], inplace=True)
        else:
            validator.add_check("Price Spike Detection", False, "Front-month price column not found.")

        if validator.was_successful():
            if front_month_col in final_df.columns:
                final_df.dropna(subset=[front_month_col], inplace=True)

            for i, ticker in enumerate(chronological_tickers):
                if i < len(ordinal_names):
                    try:
                        month_code, year_str = ticker[2], ticker[3:5]
                        year = int(f"20{year_str}")
                        exp = get_kc_expiration_date(year, month_code)
                        dte_col_name = f"{ordinal_names[i]}_dte"
                        final_df[dte_col_name] = (exp.tz_localize(None) - final_df.index).days
                        final_df.loc[final_df[dte_col_name] < 0, dte_col_name] = None
                    except (ValueError, TypeError, IndexError):
                        print(f"Warning: Could not parse ticker details for DTE from '{ticker}'.")
            
            final_df['date'] = final_df.index.strftime('%-m/%-d/%y')
            
            final_cols_from_config = config.get('final_column_order', [])
            final_df = final_df.reindex(columns=final_cols_from_config)
            
            for col in [c for c in final_cols_from_config if '_dte' in c and c in final_df.columns]:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').astype('Int64')
            
            final_df.dropna(inplace=True)
            print(f"\nSuccessfully processed data. Shape: {final_df.shape}")
            validator.set_final_shape(final_df.shape)
        else:
            print("\nValidation failures occurred. Skipping final processing.")
            final_df = None

    else:
        validator.add_check("Consolidation", False, "No coffee price data was fetched.")
        final_df = None
    
    report = validator.generate_report()
    print("\n" + "="*30 + "\n" + report + "\n" + "="*30)
    
    notification_title = f"Coffee Data Pull: {'SUCCESS' if validator.was_successful() else 'FAILURE'}"
    send_pushover_notification(config.get('notifications', {}), notification_title, report)

    return final_df
