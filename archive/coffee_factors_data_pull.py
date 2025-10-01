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
from matplotlib import pyplot as plt
import re
from collections import deque

# --- API and Date Configuration ---
# Get your free API key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = '1f8f53f43170a81a69c72fe49a1f1fc3'
fred = Fred(api_key=FRED_API_KEY)

# Replace with your Nasdaq Data Link API key
NASDAQ_API_KEY = "TXZJiisWnW1iJe8Vrjti"
ndl.api_key = NASDAQ_API_KEY

# Define the date range for all data pulls
start_date = datetime(1980, 1, 1)
end_date = datetime.now()

print("Setup complete.")


# --- CORRECTED FUNCTION ---
def get_active_coffee_tickers(num_contracts=5):
    """
    Determines the tickers for the next N active coffee futures contracts.
    Returns two lists: one sorted chronologically (for cleaning) and one
    sorted by month/year (for final column order).
    """
    print(f"Dynamically determining the next {num_contracts} coffee tickers...")
    now = datetime.now()
    start_year = now.year

    expiration_months = {3: 'H', 5: 'K', 7: 'N', 9: 'U', 12: 'Z'}
    sorted_months = sorted(expiration_months.keys())
    
    # --- Step 1: Find the next N contracts chronologically ---
    target_month = next((m for m in sorted_months if m > now.month), sorted_months[0])
    if target_month == sorted_months[0] and now.month == 12:
        start_year += 1
        
    month_deque = deque(sorted_months)
    while month_deque[0] != target_month:
        month_deque.rotate(-1)
        
    current_year = start_year
    generated_contracts = []
    
    for i in range(num_contracts):
        contract_month = month_deque[0]
        if i > 0 and contract_month < generated_contracts[-1]['month']:
            current_year += 1

        month_code = expiration_months[contract_month]
        year_code = str(current_year)[-2:]
        ticker = f"KC{month_code}{year_code}.NYB"
        generated_contracts.append({'ticker': ticker, 'month': contract_month, 'year': current_year})
        month_deque.rotate(-1)

    # --- Step 2: Create the two required lists ---
    # List 1: Chronological (for identifying the true front-month)
    chronological_tickers = [c['ticker'] for c in generated_contracts]
    print(f"... chronological tickers found: {chronological_tickers}")

    # List 2: Sorted by month, then year (for CSV column order)
    generated_contracts.sort(key=lambda x: (x['month'], x['year']))
    month_sorted_tickers = [c['ticker'] for c in generated_contracts]
    print(f"... month-sorted tickers found: {month_sorted_tickers}")
    
    return chronological_tickers, month_sorted_tickers


# --- 1. Define All Data Points to Fetch ---
chronological_tickers, coffee_tickers_for_download = get_active_coffee_tickers(num_contracts=5)

weather_locations = {
    'brazil_minas_gerais': (-19.9245, -43.9353),
    'vietnam_ho_chi_minh': (10.8231, 106.6297),
    'colombia_antioquia': (6.2442, -75.5812),
    'indonesia_sumatra': (3.5952, 98.6722)
}
fred_tickers = {
    'brl_usd_exchange_rate': 'DEXBZUS',
    'oil_price_wti': 'DCOILWTICO',
    'indonesia_idr_usd': 'DEXINUS',
    'mexico_mxn_usd': 'DEXMXUS'
}
yf_tickers = {
    'sugar_price': 'SB=F',
    'sp500_price': '^GSPC',
    'nestle_stock': 'NSRGY',
    'us_dollar_index': 'DX-Y.NYB',
    'shipping_proxy': 'MAERSK-B.CO',
    'volatility_index': '^VIX',
}


# --- Function to fetch all weather data in a single batch request ---
def fetch_all_weather_data(locations, start_date, end_date):
    """
    Fetches daily weather data for multiple locations in a single API call
    using the Open-Meteo API.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    latitudes = [loc[0] for loc in locations.values()]
    longitudes = [loc[1] for loc in locations.values()]

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data from Open-Meteo: {e}")
        return {}
    
    results = response.json()
    weather_dataframes = {}

    for i, name in enumerate(locations.keys()):
        res = results[i]
        df = pd.DataFrame(data=res['daily'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.rename(columns={
            'temperature_2m_mean': f'{name}_avg_temp',
            'precipitation_sum': f'{name}_precipitation'
        }, inplace=True)
        weather_dataframes[name] = df
    return weather_dataframes

# --- 3. Execute Data Fetching ---
all_data = {}

# --- Fetch Coffee Prices using the month-sorted tickers for correct column order ---
print("\nFetching Coffee Prices...")
try:
    df_coffee = yf.download(coffee_tickers_for_download, start=start_date, end=end_date)
    df_coffee_close = df_coffee['Close']
    df_coffee_close.columns = [f'coffee_price_{ticker}' for ticker in df_coffee_close.columns]
    all_data['coffee_prices'] = df_coffee_close
    print("...fetched all coffee prices.")
except Exception as e:
    print(f"...error fetching data for coffee prices. Error: {e}. Skipping.")

# Fetch all Weather Data in a single, efficient call
print("\nFetching all weather data from Open-Meteo in one batch...")
try:
    weather_dfs = fetch_all_weather_data(weather_locations, start_date, end_date)
    all_data.update(weather_dfs)
    print("...fetched all weather data successfully.")
except Exception as e:
    print(f"...error fetching weather data. Error: {e}. Skipping.")

# Fetch FRED Data
print("\nFetching economic data from FRED...")
for name, ticker in fred_tickers.items():
    try:
        all_data[name] = fred.get_series(ticker, start_date=start_date, end_date=end_date).to_frame(name=name)
        print(f"...fetched {name}")
    except Exception as e:
        print(f"...error fetching {name} ({ticker}). Error: {e}. Skipping.")

# --- Fetch other yfinance Data in a single batch ---
print("\nFetching other market data from yfinance...")
try:
    tickers_list = list(yf_tickers.values())
    df_multi = yf.download(tickers_list, start=start_date, end=end_date)
    df_close = df_multi['Close']
    for name, ticker in yf_tickers.items():
        if ticker in df_close.columns:
            all_data[name] = pd.DataFrame(df_close[ticker].values, index=df_close.index, columns=[name])
            print(f"...processed {name}")
        else:
            print(f"...could not find data for {name} ({ticker}) in the batch result.")
except Exception as e:
    print(f"...an error occurred while fetching other market data in batch. Error: {e}.")

print("\nAll data fetching complete.")

# --- 4. Consolidate, Clean, and Save Data ---
print("\nConsolidating all data into a single DataFrame...")

if 'coffee_prices' in all_data and not all_data['coffee_prices'].empty:
    final_df = all_data['coffee_prices'].copy()
    
    for name, df in all_data.items():
        if name != 'coffee_prices':
            final_df = final_df.join(df, how='left')
            
    print(f"Initial merged shape: {final_df.shape}")

    final_df.ffill(inplace=True)
    print("Forward-filled missing values.")

    # --- CORRECTED CLEANING LOGIC ---
    # Use the *chronologically first* contract to determine which rows to drop.
    original_rows = len(final_df)
    front_month_contract_column = f'coffee_price_{chronological_tickers[0]}'
    final_df.dropna(subset=[front_month_contract_column], inplace=True)
    print(f"Dropped {original_rows - len(final_df)} rows with missing front-month contract data.")
    
    # --- Save to CSV ---
    output_filename = f"coffee_futures_data_{datetime.now().strftime('%Y-%m-%d')}.csv"
    final_df.to_csv(output_filename)
    
    print(f"\nSuccessfully cleaned and saved data to {output_filename}")
    print(f"Final dataset shape: {final_df.shape}")
    print("\nScript finished.")
else:
    print("\nNo coffee price data was fetched. Cannot proceed with data consolidation. Exiting.")
