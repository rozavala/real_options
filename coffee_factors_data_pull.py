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


# --- UPDATED: Function to dynamically get the next N active coffee futures tickers ---
def get_active_coffee_tickers(num_contracts=5):
    """
    Determines the tickers for the next N active coffee futures contracts.
    Coffee C futures contracts expire in March, May, July, September, and December.
    This function calculates the next expiration dates from today and builds the
    correct ticker symbols for yfinance.
    """
    print(f"Dynamically determining the next {num_contracts} coffee tickers...")
    tickers = []
    now = datetime.now()
    start_year = now.year
    start_month = now.month

    # Expiration months and their corresponding yfinance letter codes
    expiration_months = {
        3: 'H',  # March
        5: 'K',  # May
        7: 'N',  # July
        9: 'U',  # September
        12: 'Z'  # December
    }
    
    # Create a circular list of expiration months for easy iteration
    sorted_months = sorted(expiration_months.keys())
    
    # Find the starting point for our search
    next_month_gen = (m for m in sorted_months if m > start_month)
    target_month = next(next_month_gen, None)
    
    if target_month is None:
        start_year += 1
        target_month = sorted_months[0]

    # Use a deque to easily get the next months in order, handling year rollovers
    month_deque = deque(sorted_months)
    while month_deque[0] != target_month:
        month_deque.rotate(-1)
        
    current_year = start_year
    for i in range(num_contracts):
        # Get the next contract month from our deque
        contract_month = month_deque[0]
        
        # If the next contract month is less than the previous one, it's a new year
        if i > 0 and contract_month < tickers[-1]['month']:
            current_year += 1

        month_code = expiration_months[contract_month]
        year_code = str(current_year)[-2:]
        
        ticker = f"KC{month_code}{year_code}.NYB"
        tickers.append({'ticker': ticker, 'month': contract_month})
        
        # Move to the next month for the next iteration
        month_deque.rotate(-1)

    final_tickers = [t['ticker'] for t in tickers]
    print(f"... tickers found: {final_tickers}")
    return final_tickers


# --- 1. Define All Data Points to Fetch ---
# The coffee tickers are now determined dynamically by the function above
coffee_tickers = get_active_coffee_tickers(num_contracts=5)

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
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data from Open-Meteo: {e}")
        return {}


    results = response.json()
    weather_dataframes = {}

    for i, name in enumerate(locations.keys()):
        # The API returns a list of dictionaries, one for each location
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

# --- UPDATED: Fetch all Coffee Prices using the dynamic tickers ---
print("\nFetching Coffee Prices...")
try:
    df_coffee = yf.download(coffee_tickers, start=start_date, end=end_date)
    df_coffee_close = df_coffee['Close']
    # Rename columns to be more descriptive for the final CSV
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


# --- REFACTORED: Fetch other yfinance Data in a single batch ---
print("\nFetching other market data from yfinance...")
try:
    # Get a list of all tickers to fetch
    tickers_list = list(yf_tickers.values())
    
    # Download all tickers in one efficient API call
    df_multi = yf.download(tickers_list, start=start_date, end=end_date)
    
    # The result for multiple tickers has multi-level columns (e.g., 'Close', 'Open').
    # We are interested only in the 'Close' price for each ticker.
    df_close = df_multi['Close']

    # Iterate through our original dictionary to process the results
    for name, ticker in yf_tickers.items():
        if ticker in df_close.columns:
            # Create a new DataFrame for each ticker to maintain the original data structure
            all_data[name] = pd.DataFrame(df_close[ticker].values, index=df_close.index, columns=[name])
            print(f"...processed {name}")
        else:
            print(f"...could not find data for {name} ({ticker}) in the batch result.")
except Exception as e:
    print(f"...an error occurred while fetching other market data in batch. Error: {e}.")


print("\nAll data fetching complete.")

# --- 4. Consolidate, Clean, and Save Data ---
print("\nConsolidating all data into a single DataFrame...")

# Check if the primary data (coffee_prices) was fetched
if 'coffee_prices' in all_data and not all_data['coffee_prices'].empty:
    # Start with coffee prices as the base DataFrame
    final_df = all_data['coffee_prices'].copy()
    
    # Join all other DataFrames using a left join
    for name, df in all_data.items():
        if name != 'coffee_prices':
            final_df = final_df.join(df, how='left')
            
    print(f"Initial merged shape: {final_df.shape}")

    # --- Data Cleaning ---
    # Forward-fill missing values. This is a common method for time-series
    # data, assuming a value remains the same until a new one is reported (e.g., on weekends).
    final_df.ffill(inplace=True)
    print("Forward-filled missing values.")

    # --- UPDATED CLEANING LOGIC ---
    # Only drop rows where the *front-month* coffee contract price is missing.
    # This preserves historical data for other variables, even if later contracts didn't exist yet.
    original_rows = len(final_df)
    front_month_contract_column = f'coffee_price_{coffee_tickers[0]}'
    final_df.dropna(subset=[front_month_contract_column], inplace=True)
    print(f"Dropped {original_rows - len(final_df)} rows with missing front-month contract data.")
    
    # --- Save to CSV ---
    # Create a dynamic filename with the current date
    output_filename = f"coffee_futures_data_{datetime.now().strftime('%Y-%m-%d')}.csv"
    final_df.to_csv(output_filename)
    
    print(f"\nSuccessfully cleaned and saved data to {output_filename}")
    print(f"Final dataset shape: {final_df.shape}")
    print("\nScript finished.")
else:
    print("\nNo coffee price data was fetched. Cannot proceed with data consolidation. Exiting.")

