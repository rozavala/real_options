import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio
import os
import glob
from datetime import datetime, timedelta
import sys
import random
import math

# --- Asyncio Event Loop Fix for Streamlit ---
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

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Coffee Bot HQ",
    initial_sidebar_state="expanded"
)

# --- Caching Data Loading Functions ---
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
    """Finds the latest log file and caches the last 50 lines."""
    try:
        list_of_logs = glob.glob('logs/*.log')
        if not list_of_logs:
            return None, "No .log files found in the 'logs' directory."
        latest_log = max(list_of_logs, key=os.path.getctime)
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        return latest_log, lines[-50:]
    except Exception as e:
        return None, f"Error reading log files: {e}"

# --- Data Fetching Functions ---
def fetch_market_data(config):
    """Connects to IB, fetches future prices, and disconnects."""
    ib = IB()
    try:
        # Connect with a random clientId to prevent orchestrator conflicts
        ib.connect(
            config['connection']['host'],
            config['connection']['port'],
            clientId=random.randint(100, 1000)
        )
        st.toast("Connected to IB Gateway...", icon="🔗")

        # FIX: Use ib.run() instead of asyncio.run() to use the correct event loop
        active_futures = ib.run(get_active_futures(ib, config['symbol'], config['exchange']))
        
        if not active_futures:
            st.warning("No active futures contracts found.")
            return pd.DataFrame()

        # Request market data (Snapshot=False allows streaming updates during sleep)
        tickers = [ib.reqMktData(c, '', False, False) for c in active_futures]

        # --- Robust Polling for Market Data ---
        # Wait up to 10 seconds for data to populate
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < 10:
            ib.sleep(0.5) # This pumps the IB event loop
            # If we have Last OR Close for all tickers, we are good
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
        st.toast("Disconnected from IB Gateway.", icon="🔌")
        return pd.DataFrame(market_data)

    except Exception as e:
        st.error(f"Failed to fetch market data: {e}")
        if ib.isConnected():
            ib.disconnect()
        return pd.DataFrame()

def fetch_portfolio_data(config, trade_ledger_df):
    """Fetches live portfolio data from IB and merges it with historical data."""
    ib = IB()
    try:
        ib.connect(
            config['connection']['host'],
            config['connection']['port'],
            clientId=random.randint(100, 1000)
        )
        st.toast("Connected to IB Gateway...", icon="🔗")

        portfolio = ib.portfolio()
        if not portfolio:
            st.info("No active positions found in the portfolio.")
            ib.disconnect()
            return pd.DataFrame()

        # Prepare ledger for lookup
        if not trade_ledger_df.empty:
            trade_ledger_df['timestamp'] = pd.to_datetime(trade_ledger_df['timestamp'])
            # Sort by timestamp ascending to get the first trade easily
            sorted_ledger = trade_ledger_df.sort_values('timestamp')
        else:
            sorted_ledger = pd.DataFrame()

        position_data = []
        for item in portfolio:
            symbol = item.contract.localSymbol
            
            # Logic to find "Days Held"
            days_held = 'N/A'
            open_date_str = 'N/A'
            
            if not sorted_ledger.empty:
                # Find trades for this symbol that were opening trades (Strategy Execution)
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
        st.toast("Disconnected from IB Gateway.", icon="🔌")
        return pd.DataFrame(position_data)

    except Exception as e:
        st.error(f"Failed to fetch portfolio data: {e}")
        if ib.isConnected():
            ib.disconnect()
        return pd.DataFrame()

# --- Main Application Layout ---

st.title("Coffee Bot Mission Control ☕")

trade_df = load_trade_data()
config = get_config()
log_file, log_lines = load_log_data()

# --- Sidebar ---
with st.sidebar:
    st.header("Manual Controls")
    st.warning("These actions interact directly with the live trading account.")

    if st.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    if st.button("⛔ Cancel All Open Orders", type="primary", width='stretch'):
        if config:
            with st.spinner("Executing cancellation..."):
                try:
                    from trading_bot.order_manager import cancel_all_open_orders
                    asyncio.run(cancel_all_open_orders(config))
                    st.success("Cancellation command sent successfully.")
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
                    st.success("Force close command sent successfully.")
                    st.toast("Aged positions closed!", icon="✅")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Config not loaded.")

# --- KPI Section ---
st.header("Key Performance Indicators")
if not trade_df.empty:
    total_pnl = trade_df['total_value_usd'].sum()
    
    # Check if combo_id exists, otherwise fallback
    if 'combo_id' in trade_df.columns:
        trade_results = trade_df.groupby('combo_id')['total_value_usd'].sum()
        win_rate = (trade_results > 0).mean() * 100 if not trade_results.empty else 0
        avg_win = trade_results[trade_results > 0].mean()
        avg_loss = trade_results[trade_results <= 0].mean()
        gross_profit = trade_results[trade_results > 0].sum()
        gross_loss = trade_results[trade_results <= 0].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

    last_trade_date = trade_df['timestamp'].max()
    pnl_color = "green" if total_pnl >= 0 else "red"

    # Custom Styling for metrics
    st.markdown(f"""
    <style>
    div[data-testid="stMetricValue"] {{
        color: {pnl_color};
    }}
    </style>
    """, unsafe_allow_html=True)

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Total Realized P&L", f"${total_pnl:,.2f}")
    kpi_cols[1].metric("Win Rate", f"{win_rate:.1f}%")
    kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
    kpi_cols[3].metric("Avg Win", f"${avg_win:,.2f}" if not math.isnan(avg_win) else "$0.00")
    kpi_cols[4].metric("Avg Loss", f"${avg_loss:,.2f}" if not math.isnan(avg_loss) else "$0.00")
    kpi_cols[5].metric("Last Trade", last_trade_date.strftime('%Y-%m-%d'))

else:
    st.info("No trade data found to calculate KPIs.")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💼 Portfolio Monitor", "📊 Analytics", "📈 Trade Ledger", "📋 System Health & Logs", "💹 Market Data"])

with tab1:
    st.subheader("Live Portfolio Monitor")
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame()

    if st.button("🔄 Fetch Active Positions", width='stretch'):
        with st.spinner("Connecting to IB... Fetching portfolio..."):
            st.session_state.portfolio_data = fetch_portfolio_data(config, trade_df)

    if not st.session_state.portfolio_data.empty:
        # Highlight logic
        def highlight_aged_positions(row):
            style = [''] * len(row)
            if isinstance(row['Days Held'], int) and row['Days Held'] >= 4:
                style = ['background-color: #ffcccc'] * len(row)
            return style

        st.dataframe(
            st.session_state.portfolio_data.style.apply(highlight_aged_positions, axis=1).format({
                "Mkt Price": "${:.2f}", 
                "Mkt Value": "${:.2f}",
                "Avg Cost": "${:.2f}",
                "Unrealized P&L": "${:.2f}"
            }),
            width='stretch'
        )
    else:
        st.info("Click the button to fetch live portfolio data.")

with tab2:
    st.subheader("Risk-Adjusted Performance")
    if not trade_df.empty:
        trade_df_sorted = trade_df.sort_values('timestamp')
        trade_df_sorted['cumulative_pnl'] = trade_df_sorted['total_value_usd'].cumsum()
        
        fig_equity = px.line(trade_df_sorted, x='timestamp', y='cumulative_pnl', title="Equity Curve")
        st.plotly_chart(fig_equity, width='stretch')

        # Drawdown
        trade_df_sorted['running_max'] = trade_df_sorted['cumulative_pnl'].cummax()
        trade_df_sorted['drawdown'] = trade_df_sorted['cumulative_pnl'] - trade_df_sorted['running_max']
        
        fig_drawdown = px.area(trade_df_sorted, x='timestamp', y='drawdown', title="Drawdown ($)", color_discrete_sequence=['red'])
        st.plotly_chart(fig_drawdown, width='stretch')
    else:
        st.info("No trade data to display charts.")

with tab3:
    st.subheader("Full Trade Ledger")
    if not trade_df.empty:
        st.dataframe(trade_df.sort_values('timestamp', ascending=False), width='stretch', height=600)
    else:
        st.info("Trade ledger is empty.")

with tab4:
    st.subheader("System Health")
    if log_file and log_lines:
        # Heartbeat Check
        try:
            last_line = log_lines[-1]
            # Extract timestamp (Assuming format YYYY-MM-DD HH:MM:SS...)
            last_log_time_str = last_line.split(',')[0] 
            # Sometimes logs have different prefixes, robust split needed
            if " - " in last_log_time_str: 
                last_log_time_str = last_log_time_str.split(' - ')[0]

            last_log_time = datetime.strptime(last_log_time_str.strip(), '%Y-%m-%d %H:%M:%S')
            
            time_diff = datetime.now() - last_log_time
            if time_diff > timedelta(minutes=15):
                st.warning(f"**Heartbeat:** STALLED (Last log: {last_log_time.strftime('%H:%M:%S')} - {time_diff.seconds//60} min ago)")
            else:
                st.success(f"**Heartbeat:** OK (Last log: {last_log_time.strftime('%H:%M:%S')})")
        except Exception as e:
            st.warning(f"**Heartbeat:** Unknown (Could not parse log timestamp: {e})")

        st.text_area("Log Output", ''.join(log_lines), height=300)
    else:
        st.error("No logs found.")

with tab5:
    st.subheader("Live Coffee Futures Prices")
    if 'market_data' not in st.session_state:
        st.session_state.market_data = pd.DataFrame()

    if st.button("📈 Fetch Snapshot", width='stretch'):
        with st.spinner("Connecting to IB... Fetching data..."):
            st.session_state.market_data = fetch_market_data(config)

    if not st.session_state.market_data.empty:
        st.dataframe(
            st.session_state.market_data.style.format({
                "Price": "${:.2f}", 
                "Bid": "${:.2f}", 
                "Ask": "${:.2f}"
            }), 
            width='stretch'
        )
    else:
        st.info("Click the button to fetch live market data.")
