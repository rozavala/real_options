# --- Install all required libraries ---
# You should run this command in your terminal or command prompt, not inside the script:
# pip install yfinance fredapi pandas nasdaq-data-link xlrd openpyxl requests

# --- Import Libraries ---
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from fredapi import Fred
import nasdaqdatalink as ndl
import requests
from pandas.tseries.offsets import BDay

# --- Custom Modules ---
from notifications import send_pushover_notification

# --- Validation Manager Class ---
class ValidationManager:
    """A class to manage and report data validation checks."""
    def __init__(self):
        self.results = []

    def add_check(self, check_name, success, message):
        """Adds a validation check result."""
        self.results.append({'check': check_name, 'success': success, 'message': message})

    def was_successful(self):
        """Returns True if all checks passed, False otherwise."""
        return all(r['success'] for r in self.results)

    def generate_report(self):
        """Generates a summary report of all validation checks."""
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
        
        return report.strip()

# --- Main Script ---

def load_config():
    """Loads the configuration from config.json."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("FATAL: config.json not found. Please create it.")
        return None
    except json.JSONDecodeError:
        print("FATAL: config.json is not valid JSON.")
        return None

def get_kc_expiration_date(year, month_code):
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

def get_active_coffee_tickers(num_contracts=5):
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
                if expiration_date.date() >= now.date():
                    potential_tickers.append((f"KC{code}{str(year)[-2:]}.NYB", expiration_date))
            except ValueError:
                continue
    potential_tickers.sort(key=lambda x: x[1])
    active_tickers = [ticker for ticker, exp in potential_tickers[:num_contracts]]
    print(f"... chronological tickers found: {active_tickers}")
    return active_tickers

def main():
    """Main execution function."""
    config = load_config()
    if not config:
        return

    validator = ValidationManager()
    
    # --- API and Date Configuration ---
    fred = Fred(api_key=config['fred_api_key'])
    ndl.api_key = config['nasdaq_api_key']
    start_date = datetime(1980, 1, 1)
    end_date = datetime.now()
    
    print("Setup complete.")
    all_data = {}
    
    # --- 1. Fetch Coffee Futures Data ---
    chronological_tickers = get_active_coffee_tickers(num_contracts=5)
    if not chronological_tickers:
        validator.add_check("Coffee Ticker Generation", False, "Could not determine active coffee tickers.")
    else:
        print("\nFetching Coffee Prices...")
        try:
            df_coffee = yf.download(chronological_tickers, start=start_date, end=end_date, progress=False)
            if not df_coffee.empty and not df_coffee['Close'].dropna(how='all').empty:
                df_price = df_coffee['Close'].copy()
                df_price.dropna(axis=1, how='all', inplace=True)
                df_price.columns = [f"coffee_price_{t.replace('.NYB', '')}" for t in df_price.columns]
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
        lats = ",".join([lat for lat, lon in weather_stations.values()])
        lons = ",".join([lon for lat, lon in weather_stations.values()])
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
            if not data.empty:
                all_data[name] = data
            else:
                 all_data[name] = pd.DataFrame() # Add empty frame to avoid key errors
        except Exception as e:
            validator.add_check(f"FRED Fetch ({name})", False, f"Could not fetch series '{code}'. Error: {e}")
            all_data[name] = pd.DataFrame()
    validator.add_check("FRED Data Fetch", True, "Completed all FRED series requests.")

    print("\nFetching other market data from yfinance...")
    try:
        yf_multi_data = yf.download(list(config['yf_series_map'].keys()), start=start_date, end=end_date, progress=False)
        for ticker, name in config['yf_series_map'].items():
            if not yf_multi_data.empty and ticker in yf_multi_data['Close'].columns:
                all_data[name] = yf_multi_data['Close'][[ticker]].copy().rename(columns={ticker: name})
    except Exception as e:
         validator.add_check("Other Market Data Fetch", False, f"An error occurred: {e}")
    validator.add_check("Other Market Data Fetch", True, "Completed all other yfinance requests.")

    print("\nAll data fetching complete.")

    # --- 4. Consolidate, Validate, and Save ---
    if 'coffee_prices' in all_data and not all_data['coffee_prices'].empty:
        final_df = all_data['coffee_prices'].copy()
        for name, df in all_data.items():
            if name != 'coffee_prices':
                final_df = final_df.join(df, how='left')
        
        # --- Pre-Fill Validation ---
        total_cells = final_df.size
        missing_before = final_df.isnull().sum().sum()
        missing_pct = (missing_before / total_cells) * 100 if total_cells > 0 else 0
        validator.add_check("Pre-Fill Missing Data", missing_pct < 50, f"Total missing data before fill is {missing_pct:.2f}%.")

        final_df.ffill(inplace=True)

        # --- Post-Fill Validation ---
        front_month_col = f"coffee_price_{chronological_tickers[0].replace('.NYB', '')}"
        
        # Recency Check
        last_date = final_df.index.max()
        is_recent = (datetime.now() - last_date) < timedelta(days=5)
        validator.add_check("Data Recency", is_recent, f"Most recent data point is from {last_date.strftime('%Y-%m-%d')}.")
        
        # Spike Check
        if front_month_col in final_df.columns:
            final_df['price_pct_change'] = final_df[front_month_col].pct_change().abs()
            max_spike = final_df['price_pct_change'].max()
            spike_threshold = config['validation_thresholds']['price_spike_pct']
            no_spikes = max_spike < spike_threshold
            validator.add_check("Price Spike Detection", no_spikes, f"Max daily change for front-month was {max_spike:.2%}. Threshold is {spike_threshold:.0%}.")
            final_df.drop(columns=['price_pct_change'], inplace=True)
        else:
            validator.add_check("Price Spike Detection", False, "Front-month column not found for spike check.")

        # --- Reformat and Save ---
        if validator.was_successful():
            final_df.dropna(subset=[front_month_col], inplace=True)

            for col in list(final_df.columns):
                if 'coffee_price_KC' in col:
                    ticker, month_code, year_str = col.replace('coffee_price_', ''), col[16], col[17:]
                    try:
                        exp = get_kc_expiration_date(int(f"20{year_str}"), month_code)
                        final_df[f'{month_code}_dte'] = (exp - final_df.index).days
                        final_df.loc[final_df[f'{month_code}_dte'] < 0, f'{month_code}_dte'] = None
                        final_df[f'{month_code}_symbol'] = ticker
                        final_df.rename(columns={col: f'{month_code}_price'}, inplace=True)
                    except ValueError:
                        continue
            
            final_df['date'] = final_df.index.strftime('%-m/%-d/%y')
            final_cols = [c for c in config['final_column_order'] if c in final_df.columns]
            final_df = final_df[final_cols]
            
            for col in [c for c in final_cols if '_dte' in c]:
                final_df.loc[:, col] = pd.to_numeric(final_df[col], errors='coerce').astype('Int64')
            
            final_df.dropna(inplace=True)
            output_filename = f"coffee_futures_data_reformatted_{datetime.now().strftime('%Y-%m-%d')}.csv"
            final_df.to_csv(output_filename, index=False)
            print(f"\nSuccessfully saved data to {output_filename}. Shape: {final_df.shape}")
        else:
            print("\nValidation failures occurred. Skipping final processing and file save.")

    else:
        validator.add_check("Consolidation", False, "No coffee price data was fetched. Cannot create final data file.")
    
    # --- Send Final Report ---
    report = validator.generate_report()
    print("\n" + "="*30 + "\n" + report + "\n" + "="*30)
    
    notification_title = f"Coffee Data Pull: {'SUCCESS' if validator.was_successful() else 'FAILURE'}"
    send_pushover_notification(config, notification_title, report)

if __name__ == "__main__":
    main()
