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
import yfinance as yf

# --- 1. Asyncio Event Loop Fix for Streamlit ---
# Essential to prevent crashes when using asyncio libraries in Streamlit
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- 2. Path Setup & Local Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from performance_analyzer import get_trade_ledger_df
from config_loader import load_config
from trading_bot.ib_interface import get_active_futures
from ib_insync import IB, util

# --- 3. Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Coffee Bot HQ",
    initial_sidebar_state="expanded"
)

# --- Configuration Constants ---
STARTING_CAPITAL = 50000  # Adjust this to your actual starting account balance

# --- 4. Caching Data Loading Functions ---

@st.cache_data(ttl=60)
def load_trade_data():
    """Loads and caches the consolidated trade ledger."""
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
    """Loads and caches the application configuration."""
    config = load_config()
    if config is None:
        st.error("Fatal: Could not load config.json. Manual controls are disabled.")
        return {}
    return config

@st.cache_data(ttl=15)
def load_log_data():
    """Finds all relevant log files (excluding archived ones) and caches their last 50 lines."""
    try:
        list_of_logs = glob.glob('logs/*.log')
        if not list_of_logs:
            return {}

        logs_content = {}
        for log_path in list_of_logs:
            filename = os.path.basename(log_path)
            # Skip archived logs (those with numbers in their filename)
            if any(char.isdigit() for char in filename):
                continue

            # Create a readable name from filename (e.g. 'dashboard.log' -> 'Dashboard')
            name = filename.split('.')[0].capitalize()

            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                logs_content[name] = lines[-50:]
            except Exception as e:
                logs_content[name] = [f"Error reading file: {e}"]

        return logs_content
    except Exception as e:
        st.error(f"Error accessing logs directory: {e}")
        return {}

@st.cache_data(ttl=3600)  # Cache benchmarks for 1 hour
def fetch_benchmark_data(start_date, end_date):
    """Fetches S&P 500 (SPY) and Coffee Futures (KC=F) from Yahoo Finance."""
    try:
        tickers = ['SPY', 'KC=F']
        # Fetch slightly more data to ensure we have a valid starting point
        data = yf.download(tickers, start=start_date, end=end_date + timedelta(days=1), progress=False, auto_adjust=True)['Close']
        
        if data.empty:
            return pd.DataFrame()

        # Normalize to Percentage Return (0.0 = 0%, 1.0 = 100%)
        # We divide the entire column by the first valid price found
        normalized = data.apply(lambda x: (x / x.dropna().iloc[0]) - 1) * 100
        return normalized
    except Exception as e:
        st.warning(f"Could not fetch benchmarks: {e}")
        return pd.DataFrame()

# --- 5. Live Data Functions (IB Connection) ---

@st.cache_data(ttl=60)
def fetch_account_summary_data(_config):
    """Fetches account summary (Net Liquidation Value) from IB."""
    ib = IB()
    summary_data = {"NetLiquidation": 0.0, "UnrealizedPnL": 0.0, "RealizedPnL": 0.0}
    try:
        ib.connect(
            _config['connection']['host'],
            _config['connection']['port'],
            clientId=random.randint(1000, 9999)
        )

        # accountSummary() returns a list of AccountValue objects
        # We fetch all tags to be safe as tags arg can be problematic in some versions
        summary = ib.accountSummary()

        for item in summary:
            if item.tag in summary_data:
                try:
                    summary_data[item.tag] = float(item.value)
                except ValueError:
                    pass

        ib.disconnect()
        return summary_data

    except Exception:
        # Fail silently to avoid disrupting dashboard if IB is offline
        if ib.isConnected():
            ib.disconnect()
        return {}

def fetch_market_data(config):
    """Connects to IB, fetches future prices, handles closed markets, and disconnects."""
    ib = IB()
    try:
        ib.connect(
            config['connection']['host'],
            config['connection']['port'],
            clientId=random.randint(100, 1000)
        )
        st.toast("Connected to IB Gateway...", icon="🔗")

        # Use ib.run() to execute async function on the correct loop
        active_futures = ib.run(get_active_futures(ib, config['symbol'], config['exchange']))
        
        if not active_futures:
            st.warning("No active futures contracts found.")
            ib.disconnect()
            return pd.DataFrame()

        tickers = [ib.reqMktData(c, '', False, False) for c in active_futures]

        # Wait up to 4 seconds for data to populate
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < 4:
            ib.sleep(0.2)
            if all(not util.isNan(t.last) or not util.isNan(t.close) for t in tickers):
                break
        
        market_data = []
        for contract, ticker in zip(active_futures, tickers):
            # Fallback Logic: If Last is NaN (Market Closed), use Close
            if not util.isNan(ticker.last):
                price = ticker.last
                source = "Live"
            elif not util.isNan(ticker.close):
                price = ticker.close
                source = "Close (Prev)"
            else:
                price = 0.0
                source = "N/A"

            market_data.append({
                "Contract": contract.localSymbol,
                "Price": price,
                "Source": source,
                "Bid": ticker.bid if not util.isNan(ticker.bid) else 0.0,
                "Ask": ticker.ask if not util.isNan(ticker.ask) else 0.0,
                "Time": ticker.time.strftime('%H:%M:%S') if ticker.time else "N/A"
            })

        ib.disconnect()
        return pd.DataFrame(market_data)

    except Exception as e:
        st.error(f"Failed to fetch market data: {e}")
        if ib.isConnected():
            ib.disconnect()
        return pd.DataFrame()

def fetch_portfolio_data(config, trade_ledger_df):
    """Fetches live portfolio from IB and merges with ledger to find 'Days Held'."""
    ib = IB()
    try:
        ib.connect(
            config['connection']['host'],
            config['connection']['port'],
            clientId=random.randint(100, 1000)
        )
        st.toast("Connected to IB Gateway...", icon="🔗")

        portfolio = ib.portfolio()
        
        # Prepare ledger for lookup
        if not trade_ledger_df.empty:
            # Sort by timestamp to find the FIRST trade easily
            sorted_ledger = trade_ledger_df.sort_values('timestamp')
        else:
            sorted_ledger = pd.DataFrame()

        position_data = []
        for item in portfolio:
            symbol = item.contract.localSymbol
            
            # Logic to find "Days Held"
            days_held = 0
            open_date_str = 'N/A'
            
            if not sorted_ledger.empty:
                # Find 'Strategy Execution' trades for this symbol
                relevant_trades = sorted_ledger[
                    (sorted_ledger['local_symbol'] == symbol) & 
                    (sorted_ledger['reason'] == 'Strategy Execution')
                ]
                if not relevant_trades.empty:
                    first_trade_date = relevant_trades.iloc[0]['timestamp']
                    days_held = (datetime.now() - first_trade_date).days
                    open_date_str = first_trade_date.strftime('%Y-%m-%d')

            position_data.append({
                "Symbol": symbol,
                "Quantity": item.position,
                "Mkt Price": item.marketPrice,
                "Mkt Value": item.marketValue,
                "Avg Cost": item.averageCost,
                "Unrealized P&L": item.unrealizedPNL,
                "Days Held": days_held,
                "Open Date": open_date_str
            })

        ib.disconnect()
        return pd.DataFrame(position_data)

    except Exception as e:
        st.error(f"Failed to fetch portfolio data: {e}")
        if ib.isConnected():
            ib.disconnect()
        return pd.DataFrame()

# --- 6. Main Application Logic ---

st.title("Coffee Bot Mission Control ☕")

# Load Data
trade_df = load_trade_data()
config = get_config()
logs_data = load_log_data()

# --- Sidebar: Manual Controls ---
with st.sidebar:
    st.header("Manual Controls")
    st.warning("Actions below interact with the live account.")

    if st.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    if st.button("⛔ Cancel All Open Orders", type="primary", width='stretch'):
        if config:
            with st.spinner("Executing cancellation..."):
                try:
                    from trading_bot.order_manager import cancel_all_open_orders
                    asyncio.run(cancel_all_open_orders(config))
                    st.success("Orders cancelled.")
                    st.toast("Open orders cancelled!", icon="✅")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Config not loaded.")

    if st.button("📉 Force Close 5+ Day Positions", width='stretch'):
        if config:
            with st.spinner("Executing force close..."):
                try:
                    from trading_bot.order_manager import close_positions_after_5_days
                    asyncio.run(close_positions_after_5_days(config))
                    st.success("Force close complete.")
                    st.toast("Aged positions closed!", icon="✅")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Config not loaded.")

# --- Header: KPIs & Benchmarks ---
st.header("Performance Dashboard")

if not trade_df.empty:
    # 1. Fetch Account Summary for Accurate P&L
    account_data = fetch_account_summary_data(config)
    net_liq = account_data.get("NetLiquidation", 0.0)

    # Calculate P&L: If we have live data, use it. Else fallback to ledger (which may show dip due to open positions)
    if net_liq > 0:
        total_pnl = net_liq - STARTING_CAPITAL
        pnl_label = "Bot P&L (Total Equity)"
        pnl_help = f"Net Liquidation Value (${net_liq:,.2f}) - Starting Capital (${STARTING_CAPITAL:,.0f})"
    else:
        total_pnl = trade_df['total_value_usd'].sum()
        pnl_label = "Bot Cash Flow (Realized + Cost)"
        pnl_help = "Note: This reflects Net Cash Flow. Open positions are counted as cost/loss until closed. Connect IB for Equity P&L."

    bot_return_pct = (total_pnl / STARTING_CAPITAL) * 100
    
    # Profit Factor Logic
    if 'combo_id' in trade_df.columns:
        trade_groups = trade_df.groupby('combo_id')['total_value_usd'].sum()
        gross_profit = trade_groups[trade_groups > 0].sum()
        gross_loss = abs(trade_groups[trade_groups < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0
    else:
        profit_factor = 0.0

    # 2. Benchmark Metrics
    start_date = trade_df['timestamp'].min().date()
    end_date = datetime.now().date()
    benchmarks = fetch_benchmark_data(start_date, end_date)
    
    spy_return = benchmarks['SPY'].iloc[-1] if not benchmarks.empty and 'SPY' in benchmarks else 0.0
    kc_return = benchmarks['KC=F'].iloc[-1] if not benchmarks.empty and 'KC=F' in benchmarks else 0.0

    # 3. Display
    kpi_cols = st.columns(4)
    
    kpi_cols[0].metric(pnl_label, f"${total_pnl:,.2f}",
                       help=pnl_help,
                       delta_color="normal")
    
    kpi_cols[1].metric("Bot Return %", f"{bot_return_pct:.2f}%", 
                       help=f"Based on ${STARTING_CAPITAL:,.0f} starting capital")
    
    kpi_cols[2].metric("S&P 500 Benchmark", f"{spy_return:.2f}%", 
                       delta=f"{bot_return_pct - spy_return:.2f}% vs SPY")
    
    kpi_cols[3].metric("Coffee Benchmark", f"{kc_return:.2f}%", 
                       delta=f"{bot_return_pct - kc_return:.2f}% vs Coffee")

else:
    st.info("No trade history found. KPIs unavailable.")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💼 Portfolio", "📊 Analytics", "📈 Trade Ledger", "📋 System Health", "💹 Market"])

with tab1:
    st.subheader("Active Portfolio")
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame()

    if st.button("🔄 Fetch Active Positions", width='stretch'):
        with st.spinner("Fetching portfolio from IB..."):
            st.session_state.portfolio_data = fetch_portfolio_data(config, trade_df)

    if not st.session_state.portfolio_data.empty:
        # Highlight aged positions
        def highlight_rows(row):
            if isinstance(row['Days Held'], int) and row['Days Held'] >= 4:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)

        st.dataframe(
            st.session_state.portfolio_data.style.apply(highlight_rows, axis=1).format({
                "Mkt Price": "${:.2f}", 
                "Avg Cost": "${:.2f}",
                "Unrealized P&L": "${:.2f}"
            }),
            width='stretch'
        )
        
        # Unreaized Summary
        unrealized = st.session_state.portfolio_data['Unrealized P&L'].sum()
        st.metric("Total Unrealized P&L", f"${unrealized:,.2f}")
    else:
        st.info("Portfolio empty or not fetched.")

with tab2:
    st.subheader("Strategy vs. Benchmarks")
    if not trade_df.empty and not benchmarks.empty:
        # Prepare Bot Data (Daily Equity)
        trade_df_sorted = trade_df.sort_values('timestamp')
        daily_bot = trade_df_sorted.set_index('timestamp').resample('D')['total_value_usd'].sum().cumsum().ffill()
        
        # Normalize Bot to %
        bot_series = (daily_bot / STARTING_CAPITAL) * 100
        bot_series.name = "Bot Strategy"

        # Merge
        comparison_df = pd.DataFrame(index=benchmarks.index)
        comparison_df = comparison_df.join(benchmarks)
        comparison_df = comparison_df.join(bot_series, how='outer').ffill().fillna(0)

        # Plot Comparison
        fig_comp = px.line(comparison_df, title="Cumulative Return % (Life-to-Date)")
        
        # Style Lines
        colors = {"Bot Strategy": "blue", "SPY": "gray", "KC=F": "brown"}
        for d in fig_comp.data:
            if d.name in colors:
                d.line.color = colors[d.name]
                if d.name == "Bot Strategy": d.line.width = 3

        st.plotly_chart(fig_comp, width='stretch')

        # Drawdown Chart
        st.subheader("Drawdown Analysis")
        trade_df_sorted['equity'] = trade_df_sorted['total_value_usd'].cumsum()
        trade_df_sorted['peak'] = trade_df_sorted['equity'].cummax()
        trade_df_sorted['drawdown'] = trade_df_sorted['equity'] - trade_df_sorted['peak']
        
        fig_dd = px.area(trade_df_sorted, x='timestamp', y='drawdown', title="Drawdown from Peak Equity ($)", color_discrete_sequence=['red'])
        st.plotly_chart(fig_dd, width='stretch')
    else:
        st.info("Insufficient data for comparison charts.")

with tab3:
    st.subheader("Trade Ledger")
    if not trade_df.empty:
        st.dataframe(trade_df.sort_values('timestamp', ascending=False), width='stretch', height=500)

with tab4:
    st.subheader("System Health Logs")
    if logs_data:
        # Display health summary for each log
        cols = st.columns(len(logs_data))
        sorted_logs = sorted(logs_data.items()) # Sort by name

        for idx, (name, lines) in enumerate(sorted_logs):
            status_msg = "Unknown"
            is_healthy = False
            mins_ago = -1
            
            if lines:
                try:
                    last_line = lines[-1]
                    # Attempt to parse timestamp from log line
                    # Format usually: "YYYY-MM-DD HH:MM:SS - ..." or "YYYY-MM-DD HH:MM:SS,ms - ..."
                    last_ts_str = last_line.split(' - ')[0]
                    if ',' in last_ts_str: # Handle milliseconds if present
                        last_ts_str = last_ts_str.split(',')[0]

                    last_ts = datetime.strptime(last_ts_str.strip(), '%Y-%m-%d %H:%M:%S')
                    diff = datetime.now() - last_ts
                    mins_ago = int(diff.total_seconds() // 60)

                    if diff < timedelta(minutes=15):
                        is_healthy = True
                        status_msg = f"Active ({mins_ago}m ago)"
                    else:
                        status_msg = f"Stalled ({mins_ago}m ago)"
                except Exception:
                    status_msg = "Parse Error"

            with cols[idx]:
                if is_healthy:
                    st.success(f"✅ **{name}**: {status_msg}")
                else:
                    st.error(f"⚠️ **{name}**: {status_msg}")

        st.divider()

        # Display log content
        for name, lines in sorted_logs:
            with st.expander(f"{name} Log", expanded=False):
                st.code("".join(lines), language='text')
    else:
        st.info("No active log files found (logs with digits in name are skipped).")

with tab5:
    st.subheader("Live Market Data")
    if 'market_data' not in st.session_state:
        st.session_state.market_data = pd.DataFrame()

    if st.button("📈 Fetch Market Snapshot", width='stretch'):
        with st.spinner("Connecting to IB..."):
            st.session_state.market_data = fetch_market_data(config)

    if not st.session_state.market_data.empty:
        st.dataframe(
            st.session_state.market_data.style.format({
                "Price": "${:.2f}", "Bid": "${:.2f}", "Ask": "${:.2f}"
            }),
            width='stretch'
        )
    else:
        st.info("Click button to fetch data.")
