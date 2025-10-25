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
import re
from pandas.tseries.offsets import BDay
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# --- Custom Modules ---
from notifications import send_pushover_notification
from config_loader import load_config

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

def get_historical_coffee_tickers(years: int = 10) -> list[str]:
    """Generates a list of historical coffee futures tickers for the last N years."""
    print(f"Generating historical coffee tickers for the last {years} years...")
    now = datetime.now()
    end_year = now.year
    start_year = end_year - years

    contract_months = ['H', 'K', 'N', 'U', 'Z']
    tickers = []

    for year in range(start_year, end_year + 1):
        for month_code in contract_months:
            # Construct the ticker, e.g., KCH24.NYB for March 2024
            ticker = f"KC{month_code}{str(year)[-2:]}.NYB"
            tickers.append(ticker)

    print(f"...generated {len(tickers)} tickers.")
    return tickers


def main(config: dict) -> bool:
    """Runs the main data pull, validation, and processing pipeline."""
    if not config:
        return False

    validator = ValidationManager()
    
    # --- API and Date Configuration ---
    fred = Fred(api_key=config['fred_api_key'])
    start_date = datetime.now() - timedelta(days=10*365) # Approx 10 years
    end_date = datetime.now()
    
    print("Setup complete.")
    all_data = {}
    
    # --- 1. Fetch Coffee Futures Data ---
    historical_tickers = get_historical_coffee_tickers(years=10)
    df_coffee = pd.DataFrame()
    if not historical_tickers:
        validator.add_check("Coffee Ticker Generation", False, "Could not determine historical coffee tickers.")
    else:
        print(f"\nFetching {len(historical_tickers)} historical coffee contracts for full OHLCV data...")
        try:
            # Download all OHLCV data. The default yfinance format will have multi-level columns.
            df_coffee = yf.download(historical_tickers, start=start_date, end=datetime.now(), progress=False)

            if not df_coffee.empty:
                # The raw data is now in df_coffee. It will be processed in the next step.
                validator.add_check("Coffee Full Data Fetch", True, f"Successfully fetched OHLCV data for {len(df_coffee.columns.levels[1])} contracts.")
            else:
                validator.add_check("Coffee Full Data Fetch", False, "yfinance returned no data for historical coffee tickers.")
        except Exception as e:
            validator.add_check("Coffee Full Data Fetch", False, f"An error occurred: {e}")

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
            # If only one ticker is requested, yfinance returns a DataFrame with simple columns
            if len(tickers_list) == 1:
                ticker = tickers_list[0]
                name = config['yf_series_map'][ticker]
                all_data[name] = yf_multi_data[['Close']].copy().rename(columns={'Close': name})
            # If multiple tickers are requested, yfinance returns a DataFrame with multi-level columns
            else:
                for ticker, name in config['yf_series_map'].items():
                    if 'Close' in yf_multi_data.columns and ticker in yf_multi_data['Close'].columns:
                        all_data[name] = yf_multi_data['Close'][[ticker]].copy().rename(columns={ticker: name})
        validator.add_check("Other Market Data Fetch", True, "Completed all other yfinance requests.")
    except Exception as e:
         validator.add_check("Other Market Data Fetch", False, f"An error occurred: {e}")

    # --- 3a. Fetch ICE Coffee Stock Data ---
    print("\nFetching ICE Certified Coffee Stocks...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto("https://en.macromicro.me/charts/23838/ice-coffee-stock")
            # Wait for the chart to be visible
            page.wait_for_selector('.highcharts-series-group')
            html_content = page.content()
            browser.close()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the script tag containing the chart data
        chart_data = None
        for script_tag in soup.find_all('script'):
            if script_tag.string:
                # Use a regex to find a JSON object that looks like chart data
                match = re.search(r'{\s*"series":\s*\[.*\]\s*}', script_tag.string)
                if match:
                    try:
                        chart_data = json.loads(match.group(0))
                        break
                    except json.JSONDecodeError:
                        continue # Ignore scripts that look like data but aren't valid JSON

        if chart_data:
            # The data is in the 'series' list, first element
            data_points = chart_data['series'][0]['data']

            dates = [datetime.fromtimestamp(d[0] / 1000) for d in data_points]
            values = [d[1] for d in data_points]

            df_ice_stocks = pd.DataFrame({'ice_coffee_stocks': values}, index=dates)
            all_data['ice_coffee_stocks'] = df_ice_stocks
            validator.add_check("ICE Coffee Stocks Fetch", True, "Successfully scraped data from macromicro.me.")
        else:
            validator.add_check("ICE Coffee Stocks Fetch", False, "Could not find chart data script on the page.")

    except Exception as e:
        validator.add_check("ICE Coffee Stocks Fetch", False, f"Could not scrape ICE coffee stocks: {e}")

    # --- 3b. Fetch COT Data ---
    print("\nFetching COT data...")
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
                        with z.open('annual.txt') as f:
                            yearly_df = pd.read_csv(f, low_memory=False)
                            all_cot_dfs.append(yearly_df)
                    print(f"... Successfully processed year {year}")
            except Exception as e:
                print(f"... Warning: Could not process COT data for year {year}. Error: {e}")
                
        cot_df = pd.concat(all_cot_dfs, ignore_index=True)

        cot_df = cot_df[cot_df['Market and Exchange Names'].str.contains(cftc_ticker, na=False)].copy()
        cot_df['Date'] = pd.to_datetime(cot_df['As of Date in Form YYYY-MM-DD'], format='%Y-%m-%d')
        cot_df.set_index('Date', inplace=True)
        
        cot_cols = {
            'Noncommercial Positions-Long (All)': 'cot_noncomm_long',
            'Noncommercial Positions-Short (All)': 'cot_noncomm_short',
        }
        cot_df = cot_df.rename(columns=cot_cols)
        
        cot_df['cot_noncomm_long'] = pd.to_numeric(cot_df['cot_noncomm_long'])
        cot_df['cot_noncomm_short'] = pd.to_numeric(cot_df['cot_noncomm_short'])
        
        cot_df['cot_noncomm_net'] = cot_df['cot_noncomm_long'] - cot_df['cot_noncomm_short']
        
        all_data['cot_report'] = cot_df[['cot_noncomm_net']]
        validator.add_check("COT Report Fetch", True, f"Successfully fetched CFTC data. Last entry: {cot_df.index.max().strftime('%Y-%m-%d')}")

    except Exception as e:
        validator.add_check("COT Report Fetch", False, f"Could not fetch CFTC data directly: {e}")

    print("\nAll data fetching complete.")

    # --- 4. Process Historical Contracts and Consolidate ---
    if 'df_coffee' in locals() and not df_coffee.empty:
        final_df = process_historical_contracts(df_coffee)
        validator.add_check("Process Historical Contracts", True, "Successfully processed and structured historical coffee data.")

        # Join other data sources
        for name, df in all_data.items():
            final_df = final_df.join(df, how='left')
        
        total_cells = final_df.size
        missing_before = final_df.isnull().sum().sum()
        missing_pct = (missing_before / total_cells) * 100 if total_cells > 0 else 0
        validator.add_check("Pre-Fill Missing Data", missing_pct < 50, f"Total missing data before fill is {missing_pct:.2f}%.")

        final_df.ffill(inplace=True)
        final_df.bfill(inplace=True)

        # Validation (Recency and Spikes)
        # Using the front month price for validation checks
        last_date = final_df.index.max()
        is_recent = (datetime.now() - last_date) < timedelta(days=5)
        validator.add_check("Data Recency", is_recent, f"Most recent data point is from {last_date.strftime('%Y-%m-%d')}.")

        # Find the first available price column to use for spike detection
        price_column = next((col for col in final_df.columns if '_price' in col), None)

        if price_column:
            final_df['price_pct_change'] = final_df[price_column].pct_change().abs()
            max_spike = final_df['price_pct_change'].max()
            spike_threshold = config['validation_thresholds']['price_spike_pct']
            no_spikes = max_spike < spike_threshold
            validator.add_check("Price Spike Detection", no_spikes, f"Max daily change on '{price_column}' was {max_spike:.2%}. Threshold is {spike_threshold:.0%}.")
            final_df.drop(columns=['price_pct_change'], inplace=True)
        else:
            validator.add_check("Price Spike Detection", False, "Could not find a price column for spike detection.")

        if validator.was_successful():
            final_df.index = pd.to_datetime(final_df.index)
            final_df['date'] = final_df.index.strftime('%-m/%-d/%y')
            
            final_cols_from_config = config['final_column_order']
            all_available_cols = list(final_df.columns)
            final_cols = [c for c in final_cols_from_config if c in all_available_cols]

            for col in all_available_cols:
                if col not in final_cols:
                    final_cols.append(col)

            final_df = final_df[final_cols]
            final_df.dropna(inplace=True)
            output_filename = f"coffee_futures_data_reformatted_{datetime.now().strftime('%Y-%m-%d')}.csv"
            final_df.to_csv(output_filename, index=False)
            
            print(f"\nSuccessfully saved data to {output_filename}. Shape: {final_df.shape}")
            validator.set_final_shape(final_df.shape)
        else:
            print("\nValidation failures occurred. Skipping final processing and file save.")
    else:
        validator.add_check("Consolidation", False, "No coffee price data was fetched.")
    
    report = validator.generate_report()
    print("\n" + "="*30 + "\n" + report + "\n" + "="*30)
    
    notification_title = f"Coffee Data Pull: {'SUCCESS' if validator.was_successful() else 'FAILURE'}"
    send_pushover_notification(config.get('notifications', {}), notification_title, report)

    return validator.was_successful()

def process_historical_contracts(df_coffee: pd.DataFrame) -> pd.DataFrame:
    """
    Processes historical OHLCV data to create a structured DataFrame with the
    5 closest-to-expiration contracts for each day.
    """
    print("Processing historical coffee contract data...")

    # Clean up the raw yfinance data
    df_coffee.dropna(how='all', inplace=True)
    df_coffee.index = pd.to_datetime(df_coffee.index)

    # Cache expiration dates
    expirations = {}
    contract_tickers = [col for col in df_coffee.columns.levels[1] if 'KC' in col]
    for ticker in contract_tickers:
        try:
            month_code = ticker[2]
            year_suffix = ticker[3:5]
            year = int(f"20{year_suffix}")
            expirations[ticker] = get_kc_expiration_date(year, month_code)
        except (ValueError, IndexError, TypeError):
            print(f"Warning: Could not parse expiration for ticker '{ticker}'. Skipping.")
            continue

    all_rows = []

    # Iterate through each business day in the data's range
    for date in df_coffee.index:

        # Find active contracts for this date
        active_contracts = []
        for ticker, exp_date in expirations.items():
            if date <= exp_date:
                dte = (exp_date - date).days
                active_contracts.append({'ticker': ticker, 'dte': dte})

        # Sort by DTE and take the 5 closest
        active_contracts.sort(key=lambda x: x['dte'])
        closest_contracts = active_contracts[:5]

        row_data = {'date': date}

        # For each of the 5 contracts, extract OHLCV and DTE
        for contract_info in closest_contracts:
            ticker = contract_info['ticker']
            month_code = ticker[2]

            # Check if we have data for this ticker on this date
            if ('Close', ticker) in df_coffee.columns and not pd.isna(df_coffee.loc[date, ('Close', ticker)]):
                row_data[f'{month_code}_dte'] = contract_info['dte']
                row_data[f'{month_code}_price'] = df_coffee.loc[date, ('Close', ticker)]
                row_data[f'{month_code}_open'] = df_coffee.loc[date, ('Open', ticker)]
                row_data[f'{month_code}_high'] = df_coffee.loc[date, ('High', ticker)]
                row_data[f'{month_code}_low'] = df_coffee.loc[date, ('Low', ticker)]
                row_data[f'{month_code}_volume'] = df_coffee.loc[date, ('Volume', ticker)]

        if len(row_data) > 1: # Only add row if it has more than just the date
            all_rows.append(row_data)

    processed_df = pd.DataFrame(all_rows).set_index('date')
    print("...historical data processed.")
    return processed_df

if __name__ == "__main__":
    print("Running data pull script directly...")
    config = load_config()
    if config:
        main(config)
    else:
        print("CRITICAL: Could not load configuration. Exiting.")
