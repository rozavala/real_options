# --- Install all required libraries ---
# You should run this command in your terminal or command prompt, not inside the script:
# pip install yfinance fredapi pandas nasdaq-data-link xlrd openpyxl requests matplotlib

# --- Import Libraries ---
import yfinance as yf
import pandas as pd
from datetime import datetime
from fredapi import Fred
import nasdaqdatalink as ndl
import requests
import time
from pandas.tseries.offsets import BDay

# --- API and Date Configuration ---
FRED_API_KEY = '1f8f53f43170a81a69c72fe49a1f1fc3'
fred = Fred(api_key=FRED_API_KEY)
NASDAQ_API_KEY = "TXZJiisWnW1iJe8Vrjti" # As per original script
ndl.api_key = NASDAQ_API_KEY
start_date = datetime(1980, 1, 1)
end_date = datetime.now()

print("Setup complete.")

# --- Dynamic Ticker and Expiration Calculation ---

def get_kc_expiration_date(year, month_code):
    """
    Calculates the expiration date for a given Coffee 'C' futures contract.
    Expiration is the business day prior to the last business day of the delivery month.
    """
    month_map = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
    month = month_map.get(month_code)
    if not month:
        raise ValueError(f"Invalid month code: {month_code}")

    last_day_of_month = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)
    last_business_day = last_day_of_month
    while last_business_day.weekday() > 4: # 5=Saturday, 6=Sunday
        last_business_day -= pd.Timedelta(days=1)
    expiration_date = last_business_day - BDay(1)
    return expiration_date

def get_active_coffee_tickers(num_contracts=5):
    """
    Determines the tickers for the next N active coffee futures contracts using the .NYB format.
    Returns a list of tickers sorted chronologically by expiration.
    """
    print(f"Dynamically determining the next {num_contracts} coffee tickers...")
    now = datetime.now()
    current_year = now.year
    contract_months = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
    
    potential_tickers = []
    # Check current year and next two years for potential contracts
    for year_offset in range(3):
        year = current_year + year_offset
        for code in contract_months.keys():
            try:
                # Calculate the exact expiration date
                expiration_date = get_kc_expiration_date(year, code)
                # Only include the contract if it has not expired yet
                if expiration_date.date() >= now.date():
                    ticker = f"KC{code}{str(year)[-2:]}.NYB"
                    potential_tickers.append(ticker)
            except ValueError:
                # This handles invalid month codes, though it shouldn't happen here.
                continue
    
    def sort_key(ticker):
        code = ticker[2]
        year_str = ticker[3:5]
        year = int(f"20{year_str}")
        month = contract_months[code]
        return year, month

    sorted_tickers = sorted(potential_tickers, key=sort_key)
    active_tickers = sorted_tickers[:num_contracts]
    print(f"... chronological tickers found: {active_tickers}")
    return active_tickers

# --- 1. Fetch Coffee Futures Data (Original Logic) ---
all_data = {}
chronological_tickers = get_active_coffee_tickers(num_contracts=5)

if chronological_tickers:
    print("\nFetching Coffee Prices...")
    try:
        df_coffee = yf.download(chronological_tickers, start=start_date, end=end_date)
        if not df_coffee.empty:
            df_price = df_coffee['Close'].copy()
            df_price.dropna(axis=1, how='all', inplace=True)
            df_price.columns = [f"coffee_price_{t.replace('.NYB', '')}" for t in df_price.columns]
            all_data['coffee_prices'] = df_price
            print("...fetched all coffee prices.")
        else:
             print("...yf.download for coffee returned no data.")
    except Exception as e:
        print(f"...an error occurred while fetching coffee prices. Error: {e}")

# --- 2. Fetch Weather Data (Original Logic) ---
print("\nFetching all weather data from Open-Meteo in one batch...")
weather_stations = {
    'brazil_minas_gerais': ('-19.9245', '-43.9353'),
    'vietnam_ho_chi_minh': ('10.8231', '106.6297'),
    'colombia_antioquia': ('6.2442', '-75.5812'),
    'indonesia_sumatra': ('3.5952', '98.6722')
}
try:
    lats = ",".join([lat for lat, lon in weather_stations.values()])
    lons = ",".join([lon for lat, lon in weather_stations.values()])
    api_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lats}&longitude={lons}&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&daily=temperature_2m_mean,precipitation_sum&timezone=GMT"
    response = requests.get(api_url)
    response.raise_for_status()
    batch_data = response.json()

    for i, location in enumerate(weather_stations.keys()):
        location_data = batch_data[i]['daily']
        weather_df = pd.DataFrame(location_data)
        weather_df['time'] = pd.to_datetime(weather_df['time'])
        weather_df.set_index('time', inplace=True)
        weather_df.rename(columns={'temperature_2m_mean': f'{location}_avg_temp', 'precipitation_sum': f'{location}_precipitation'}, inplace=True)
        all_data[location] = weather_df
    print("...fetched all weather data successfully.")
except Exception as e:
    print(f"...could not fetch or process weather data. Error: {e}")

# --- 3. Fetch Economic and Market Data (Original Logic) ---
print("\nFetching economic data from FRED...")
fred_series = {
    'brl_usd_exchange_rate': 'DEXBZUS', 'oil_price_wti': 'DCOILWTICO',
    'indonesia_idr_usd': 'DEXINUS', 'mexico_mxn_usd': 'DEXMXUS'
}
for name, code in fred_series.items():
    try:
        data = fred.get_series(code, start_date, end_date).to_frame(name=name)
        all_data[name] = data
        print(f"...fetched {name}")
    except Exception as e:
        print(f"...could not fetch FRED series '{name}'. Error: {e}")

print("\nFetching other market data from yfinance...")
yf_series_map = {
    'SB=F': 'sugar_price', '^GSPC': 'sp500_price', 'NSRGY': 'nestle_stock',
    'DX-Y.NYB': 'us_dollar_index', '^VIX': 'volatility_index',
    'MAERSK-B.CO': 'shipping_proxy'
}
try:
    yf_multi_data = yf.download(list(yf_series_map.keys()), start=start_date, end=end_date)
    for ticker, name in yf_series_map.items():
        if not yf_multi_data.empty and ticker in yf_multi_data['Close'].columns:
            all_data[name] = yf_multi_data['Close'][[ticker]].copy().rename(columns={ticker: name})
            print(f"...processed {name}")
except Exception as e:
    print(f"...an error occurred while fetching other yfinance data. Error: {e}")

print("\nAll data fetching complete.")

# --- 4. Consolidate, Reformat, and Save Data (New Logic) ---
print("\nConsolidating and reformatting all data...")

if 'coffee_prices' in all_data and not all_data['coffee_prices'].empty:
    final_df = all_data['coffee_prices'].copy()
    
    for name, df in all_data.items():
        if name != 'coffee_prices':
            final_df = final_df.join(df, how='left')
            
    print(f"Initial merged shape: {final_df.shape}")

    # --- Enhanced Logging for FFILL ---
    print("\nAnalyzing data before forward-fill...")
    missing_before = final_df.isnull().sum()
    cols_with_missing_before = missing_before[missing_before > 0]
    total_missing_before = cols_with_missing_before.sum()
    print(f"Found {total_missing_before} missing values across {len(cols_with_missing_before)} columns.")
    if not cols_with_missing_before.empty:
        print("Columns with missing data (and count):")
        with pd.option_context('display.max_rows', None): # Ensure all columns are printed
             print(cols_with_missing_before)
    
    final_df.ffill(inplace=True)
    
    missing_after = final_df.isnull().sum().sum()
    print(f"\nForward-fill complete. Filled {total_missing_before - missing_after} values.")
    if missing_after > 0:
        print(f"There are still {missing_after} missing values remaining, likely at the start of the series.")
    # --- End Enhanced Logging ---


    # --- Corrected Final Cleaning (Using original robust logic) ---
    # It drops rows only if the most important data (front-month contract) is missing.
    original_rows = len(final_df)
    if chronological_tickers:
        front_month_column = f"coffee_price_{chronological_tickers[0].replace('.NYB', '')}"
        if front_month_column in final_df.columns:
            final_df.dropna(subset=[front_month_column], inplace=True)
            print(f"Dropped {original_rows - len(final_df)} rows based on missing front-month contract data.")
        else:
            final_df.dropna(inplace=True)
            print(f"Warning: Front-month column not found. Dropped {original_rows - len(final_df)} rows with any missing data.")
    else:
        final_df.dropna(inplace=True)
        print(f"Warning: No chronological tickers found. Dropped {original_rows - len(final_df)} rows with any missing data.")

    # --- Reformat Coffee Data and Calculate DTE ---
    contract_months = []
    for col in list(final_df.columns):
        if 'coffee_price_KC' in col:
            ticker = col.replace('coffee_price_', '')
            month_code = ticker[2]
            year_str = ticker[3:]
            year = int(f"20{year_str}")
            contract_months.append(month_code)

            try:
                expiration_date = get_kc_expiration_date(year, month_code)
                final_df[f'{month_code}_dte'] = (expiration_date - final_df.index).days
                final_df.loc[final_df[f'{month_code}_dte'] < 0, f'{month_code}_dte'] = None
            except ValueError as e:
                print(e)
                final_df[f'{month_code}_dte'] = None

            final_df[f'{month_code}_symbol'] = ticker
            final_df.rename(columns={col: f'{month_code}_price'}, inplace=True)
    
    # --- Final Formatting and Column Ordering ---
    if not final_df.empty:
        final_df['date'] = final_df.index.strftime('%-m/%-d/%y')
        
        # User-specified static column order
        final_column_order = [
            'H_dte','K_dte','N_dte','U_dte','Z_dte',
            'H_price','K_price','N_price','U_price','Z_price',
            'H_symbol','K_symbol','N_symbol','U_symbol','Z_symbol',
            'date',
            'brazil_minas_gerais_avg_temp','brazil_minas_gerais_precipitation',
            'vietnam_ho_chi_minh_avg_temp','vietnam_ho_chi_minh_precipitation',
            'colombia_antioquia_avg_temp','colombia_antioquia_precipitation',
            'indonesia_sumatra_avg_temp','indonesia_sumatra_precipitation',
            'brl_usd_exchange_rate','oil_price_wti','indonesia_idr_usd',
            'mexico_mxn_usd','sugar_price','sp500_price','nestle_stock',
            'us_dollar_index','shipping_proxy','volatility_index'
        ]
        
        # Filter the list to only include columns that are actually present in the DataFrame
        # This handles cases where a contract month isn't active
        final_column_order_existing = [c for c in final_column_order if c in final_df.columns]
        
        final_df = final_df[final_column_order_existing]

        # Convert DTE columns to integer type after reordering
        dte_cols = [col for col in final_column_order_existing if '_dte' in col]
        for col in dte_cols:
            # Use .loc to avoid SettingWithCopyWarning
            final_df.loc[:, col] = pd.to_numeric(final_df[col], errors='coerce').astype('Int64')

        # Final drop to ensure every cell in the output has data.
        original_rows = len(final_df)
        final_df.dropna(inplace=True)
        print(f"\nDropped {original_rows - len(final_df)} rows with any remaining missing data.")

        output_filename = f"coffee_futures_data_reformatted_{datetime.now().strftime('%Y-%m-%d')}.csv"
        final_df.to_csv(output_filename, index=False)
        
        print(f"\nSuccessfully cleaned, reformatted, and saved data to {output_filename}")
        print(f"Final data shape: {final_df.shape}")
    else:
        print("\nDataframe is empty after cleaning. No file saved.")
else:
    print("\nNo coffee price data was fetched. Cannot create the final data file.")

