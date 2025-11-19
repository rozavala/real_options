import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import os
import glob
from datetime import datetime, timedelta
import sys
import random
import math
import yfinance as yf  # <--- NEW LIBRARY

# --- Asyncio Event Loop Fix ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- Pre-computation ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from performance_analyzer import get_trade_ledger_df
from config_loader import load_config
from trading_bot.ib_interface import get_active_futures
from ib_insync import IB, util

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Coffee Bot HQ", initial_sidebar_state="expanded")

# --- Caching Data Loading ---
@st.cache_data(ttl=60)
def load_trade_data():
    try:
        df = get_trade_ledger_df()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Failed to load trade_ledger.csv: {e}")
        return pd.DataFrame()

@st.cache_data
def get_config():
    return load_config() or {}

@st.cache_data(ttl=15)
def load_log_data():
    try:
        list_of_logs = glob.glob('logs/*.log')
        if not list_of_logs:
            return None, "No logs found."
        latest_log = max(list_of_logs, key=os.path.getctime)
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        return latest_log, lines[-50:]
    except Exception as e:
        return None, f"Error: {e}"

# --- NEW: Benchmark Fetcher ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_benchmark_data(start_date, end_date):
    """Fetches S&P 500 (SPY) and Coffee (KC=F) data from Yahoo Finance."""
    try:
        # KC=F is the ticker for Coffee Futures on Yahoo
        tickers = ['SPY', 'KC=F']
        data = yf.download(tickers, start=start_date, end=end_date + timedelta(days=1), progress=False)['Close']
        
        # Normalize to % Return (0.0 to 1.0 scale)
        # We divide the entire column by the first valid price found
        normalized = data.apply(lambda x: (x / x.dropna().iloc[0]) - 1) * 100
        return normalized
    except Exception as e:
        st.warning(f"Could not fetch benchmarks: {e}")
        return pd.DataFrame()

# --- Data Fetching (IB) ---
def fetch_market_data(config):
    # ... (Keep your existing fetch_market_data logic here) ...
    # (For brevity, I'm assuming you keep the exact same function from previous turn)
    ib = IB()
    try:
        ib.connect(config['connection']['host'], config['connection']['port'], clientId=random.randint(100, 1000))
        active_futures = ib.run(get_active_futures(ib, config['symbol'], config['exchange']))
        if not active_futures: return pd.DataFrame()
        tickers = [ib.reqMktData(c, '', False, False) for c in active_futures]
        ib.sleep(2) # Short wait
        data = []
        for c, t in zip(active_futures, tickers):
            price = t.last if not util.isNan(t.last) else (t.close if not util.isNan(t.close) else 0.0)
            data.append({"Contract": c.localSymbol, "Price": price, "Bid": t.bid, "Ask": t.ask})
        ib.disconnect()
        return pd.DataFrame(data)
    except: return pd.DataFrame()

def fetch_portfolio_data(config, trade_ledger_df):
    # ... (Keep your existing fetch_portfolio_data logic here) ...
    # (Same as previous turn)
    return pd.DataFrame() # Placeholder for brevity

# --- Main Layout ---
st.title("Coffee Bot Mission Control ☕")

trade_df = load_trade_data()
config = get_config()
log_file, log_lines = load_log_data()

# --- Sidebar (Keep your existing sidebar) ---
with st.sidebar:
    st.header("Manual Controls")
    if st.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear(); st.rerun()
    # ... (Keep Cancel/Close buttons) ...

# --- KPI Section (Enhanced) ---
st.header("Performance vs Benchmarks")

if not trade_df.empty:
    # 1. Bot Performance
    total_pnl = trade_df['total_value_usd'].sum()
    
    # Estimate Bot Return % (Requires an assumption of starting capital, e.g., $50k)
    # You can make this a config setting later.
    STARTING_CAPITAL = 50000 
    bot_return_pct = (total_pnl / STARTING_CAPITAL) * 100

    # 2. Benchmark Performance
    start_date = trade_df['timestamp'].min().date()
    end_date = datetime.now().date()
    benchmarks = fetch_benchmark_data(start_date, end_date)
    
    spy_return = benchmarks['SPY'].iloc[-1] if not benchmarks.empty else 0.0
    kc_return = benchmarks['KC=F'].iloc[-1] if not benchmarks.empty else 0.0

    # 3. Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Bot P&L (USD)", f"${total_pnl:,.2f}", delta_color="normal")
    
    col2.metric(f"Bot Return %", f"{bot_return_pct:.2f}%", 
                help=f"Based on assumed capital of ${STARTING_CAPITAL:,}")
    
    col3.metric("S&P 500 Return", f"{spy_return:.2f}%", 
                delta=f"{bot_return_pct - spy_return:.2f}% vs SPY")
    
    col4.metric("Coffee (Buy & Hold)", f"{kc_return:.2f}%", 
                delta=f"{bot_return_pct - kc_return:.2f}% vs KC")
else:
    st.info("No trade data available.")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💼 Portfolio", "📊 Comparison Charts", "📈 Ledger", "📋 Logs", "💹 Market"])

with tab1:
    # ... (Keep your Portfolio Monitor logic) ...
    st.write("Portfolio View (Same as before)")

with tab2:
    st.subheader("Strategy vs. Market (Cumulative Return %)")
    
    if not trade_df.empty and not benchmarks.empty:
        # 1. Prepare Bot Data
        trade_df_sorted = trade_df.sort_values('timestamp')
        # Create a daily equity curve
        daily_bot = trade_df_sorted.set_index('timestamp').resample('D')['total_value_usd'].sum().cumsum().fillna(method='ffill')
        
        # Normalize Bot to % Return
        bot_series = (daily_bot / STARTING_CAPITAL) * 100
        bot_series.name = "Bot Strategy"

        # 2. Merge with Benchmarks
        # Reindex bot data to match benchmark dates (handling weekends/holidays)
        comparison_df = pd.DataFrame(index=benchmarks.index)
        comparison_df = comparison_df.join(benchmarks)
        comparison_df = comparison_df.join(bot_series, how='outer').fillna(method='ffill').fillna(0)

        # 3. Plot
        fig = px.line(comparison_df, title="Performance Comparison (Life-to-Date %)")
        
        # Customize Line Colors
        new_colors = {"Bot Strategy": "blue", "SPY": "gray", "KC=F": "brown"}
        for d in fig.data:
            if d.name in new_colors:
                d.line.color = new_colors[d.name]
                if d.name == "Bot Strategy":
                    d.line.width = 3 # Make our bot line thicker

        st.plotly_chart(fig, width='stretch')
        
        st.caption(f"*Bot return calculation assumes a starting account size of ${STARTING_CAPITAL:,.0f}.")
    else:
        st.info("Insufficient data for comparison charts.")

with tab3:
    # ... (Keep Ledger) ...
    if not trade_df.empty: st.dataframe(trade_df, width='stretch')

with tab4:
    # ... (Keep Logs) ...
    if log_file: st.code(''.join(log_lines))

with tab5:
    # ... (Keep Market Data) ...
    if st.button("Fetch Live Data"):
        st.write("Fetching...")
