
import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio
import os
import glob
from datetime import datetime, timedelta
import sys

# --- Pre-computation ---
# Add the project root to the Python path to allow importing local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from performance_analyzer import get_trade_ledger_df
from config_loader import load_config
# DEFERRED: from trading_bot.order_manager import close_positions_after_5_days, cancel_all_open_orders

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
        list_of_logs = glob.glob('*.log')
        if not list_of_logs:
            return None, "No .log files found in the project root."
        latest_log = max(list_of_logs, key=os.path.getctime)
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        return latest_log, lines[-50:]
    except Exception as e:
        return None, f"Error reading log files: {e}"

# --- Main Application ---

# --- Title ---
st.title("Coffee Bot Mission Control ☕")

# --- Load all data sources ---
trade_df = load_trade_data()
config = get_config()
log_file, log_lines = load_log_data()

# --- Sidebar ---
with st.sidebar:
    st.header("Manual Controls")
    st.warning("These actions interact directly with the live trading account.")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("⛔ Cancel All Open Orders", type="primary", use_container_width=True):
        if config:
            with st.spinner("Executing cancellation..."):
                try:
                    from trading_bot.order_manager import cancel_all_open_orders
                    asyncio.run(cancel_all_open_orders(config))
                    st.success("Cancellation command sent successfully.")
                    st.toast("Open orders cancelled!", icon="✅")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.toast(f"Error cancelling orders: {e}", icon="❌")
        else:
            st.error("Config not loaded.")

    if st.button("📉 Force Close 5+ Day Positions", use_container_width=True):
        if config:
            with st.spinner("Executing force close..."):
                try:
                    from trading_bot.order_manager import close_positions_after_5_days
                    asyncio.run(close_positions_after_5_days(config))
                    st.success("Force close command sent successfully.")
                    st.toast("Aged positions closed!", icon="✅")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.toast(f"Error closing positions: {e}", icon="❌")
        else:
            st.error("Config not loaded.")

# --- Main Content ---

# Section A: KPI Ticker
st.header("Key Performance Indicators")
if not trade_df.empty:
    total_pnl = trade_df['total_value_usd'].sum()
    if 'combo_id' in trade_df.columns:
        trade_results = trade_df.groupby('combo_id')['total_value_usd'].sum()
        win_rate = (trade_results > 0).mean() * 100 if not trade_results.empty else 0
    else:
        win_rate = "N/A"
    last_trade_date = trade_df['timestamp'].max()

    pnl_color = "green" if total_pnl >= 0 else "red"
    st.markdown(f"""
    <style>
    .metric-container {{
        border: 1px solid #2e2e2e;
        border-radius: 8px;
        padding: 10px;
    }}
    .pnl-metric .stMetric-value {{
        color: {pnl_color};
    }}
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.metric("Total Realized P&L", f"${total_pnl:,.2f}")
    with col2:
        with st.container(border=True):
            st.metric("Win Rate", f"{win_rate:.2f}%" if isinstance(win_rate, float) else win_rate)
    with col3:
        with st.container(border=True):
            st.metric("Total Trades (Rows)", f"{len(trade_df)}")
    with col4:
        with st.container(border=True):
            st.metric("Last Trade Date", last_trade_date.strftime('%Y-%m-%d %H:%M'))

else:
    st.info("No trade data found to calculate KPIs.")

# Section B, C, D: Tabs
tab1, tab2, tab3 = st.tabs(["📊 Performance Charts", "📈 Trade Ledger", "📋 System Health & Logs"])

with tab1:
    st.subheader("Charts")
    if not trade_df.empty:
        # Equity Curve
        trade_df_sorted = trade_df.sort_values('timestamp')
        trade_df_sorted['cumulative_pnl'] = trade_df_sorted['total_value_usd'].cumsum()
        fig_equity = px.line(trade_df_sorted, x='timestamp', y='cumulative_pnl', title="Equity Curve")
        st.plotly_chart(fig_equity, use_container_width=True)

        # Periodical P&L
        period = 'D' if (trade_df['timestamp'].max() - trade_df['timestamp'].min()).days > 1 else 'H'
        pnl_by_period = trade_df.set_index('timestamp').resample(period)['total_value_usd'].sum().reset_index()
        pnl_by_period['color'] = pnl_by_period['total_value_usd'].apply(lambda x: 'green' if x >= 0 else 'red')
        fig_bar = px.bar(pnl_by_period, x='timestamp', y='total_value_usd', title=f"{'Daily' if period == 'D' else 'Hourly'} P&L", color='color', color_discrete_map={'green':'#2ca02c', 'red':'#d62728'})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No trade data to display charts.")

with tab2:
    st.subheader("Full Trade Ledger")
    if not trade_df.empty:
        st.dataframe(trade_df, use_container_width=True, height=600)
    else:
        st.info("Trade ledger is empty.")

with tab3:
    st.subheader("System Health")
    if log_file and log_lines:
        # Heartbeat
        try:
            last_log_time = datetime.strptime(log_lines[-1].split(',')[0], '%Y-%m-%d %H:%M:%S')
            if datetime.now() - last_log_time > timedelta(minutes=15):
                st.warning(f"**Heartbeat:** STALLED (Last log: {last_log_time.strftime('%H:%M:%S')})")
            else:
                st.success(f"**Heartbeat:** OK (Last log: {last_log_time.strftime('%H:%M:%S')})")
        except:
            st.error("**Heartbeat:** Could not parse timestamp from logs.")

        # Log Viewer
        st.code('\n'.join(log_lines), language='log')
    else:
        st.error(log_lines) # Display error message
