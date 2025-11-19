
import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio
import os
import glob
from datetime import datetime, timedelta
import sys
import random

# --- Asyncio Event Loop Fix for Streamlit ---
# Streamlit runs in a different thread than the main thread, and asyncio needs
# an event loop to be explicitly created and set in that thread.
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- Pre-computation ---
# Add the project root to the Python path to allow importing local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from performance_analyzer import get_trade_ledger_df
from config_loader import load_config
from trading_bot.ib_interface import get_active_futures
from ib_insync import IB, Ticker
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
        list_of_logs = glob.glob('logs/*.log')
        if not list_of_logs:
            return None, "No .log files found in the 'logs' directory."
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
                    st.toast(f"Error cancelling orders: {e}", icon="❌")
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

    # --- Advanced Metrics ---
    if 'combo_id' in trade_df.columns:
        gross_profit = trade_results[trade_results > 0].sum()
        gross_loss = trade_results[trade_results <= 0].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
        avg_win = trade_results[trade_results > 0].mean()
        avg_loss = trade_results[trade_results <= 0].mean()
    else:
        profit_factor = "N/A"
        avg_win = "N/A"
        avg_loss = "N/A"

    kpi_cols = st.columns(6)
    with kpi_cols[0]:
        with st.container(border=True):
            st.metric("Total Realized P&L", f"${total_pnl:,.2f}")
    with kpi_cols[1]:
        with st.container(border=True):
            st.metric("Win Rate", f"{win_rate:.2f}%" if isinstance(win_rate, float) else win_rate)
    with kpi_cols[2]:
        with st.container(border=True):
            st.metric("Profit Factor", f"{profit_factor:.2f}" if isinstance(profit_factor, float) else profit_factor)
    with kpi_cols[3]:
        with st.container(border=True):
            st.metric("Avg Win", f"${avg_win:,.2f}" if isinstance(avg_win, float) else avg_win)
    with kpi_cols[4]:
        with st.container(border=True):
            st.metric("Avg Loss", f"${avg_loss:,.2f}" if isinstance(avg_loss, float) else avg_loss)
    with kpi_cols[5]:
        with st.container(border=True):
            st.metric("Last Trade", last_trade_date.strftime('%Y-%m-%d'))

else:
    st.info("No trade data found to calculate KPIs.")

# --- Data Fetching Functions ---
def fetch_market_data(config):
    """Connects to IB, fetches future prices, and disconnects."""
    ib = IB()
    try:
        ib.connect(
            config['connection']['host'],
            config['connection']['port'],
            clientId=random.randint(100, 1000) # Use a random client ID to avoid collisions
        )
        st.toast("Connected to IB Gateway...", icon="🔗")

        active_futures = asyncio.run(get_active_futures(ib, config['symbol'], config['exchange']))
        if not active_futures:
            st.warning("No active futures contracts found.")
            return pd.DataFrame()

        tickers = [ib.reqMktData(c, '', False, False) for c in active_futures]

        # --- Robust Polling for Market Data ---
        max_wait = 15
        for _ in range(max_wait * 2):
            if all(t.last for t in tickers):
                break
            ib.sleep(0.5)

        if not all(t.last for t in tickers):
            st.warning("Timed out waiting for some market data to arrive.")

        market_data = []
        for contract, ticker in zip(active_futures, tickers):
            market_data.append({
                "Contract": contract.localSymbol,
                "Last Price": ticker.last,
                "Bid": ticker.bid,
                "Ask": ticker.ask,
                "Close": ticker.close
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
            clientId=random.randint(100, 1000) # Use a random client ID to avoid collisions
        )
        st.toast("Connected to IB Gateway...", icon="🔗")

        portfolio = ib.portfolio()
        if not portfolio:
            st.info("No active positions found in the portfolio.")
            return pd.DataFrame()

        # Get the earliest timestamp for each position from the local ledger
        trade_ledger_df['timestamp'] = pd.to_datetime(trade_ledger_df['timestamp'])
        open_dates = trade_ledger_df.loc[trade_ledger_df.groupby('symbol')['timestamp'].idxmin()]

        position_data = []
        for item in portfolio:
            open_date_info = open_dates[open_dates['symbol'] == item.contract.symbol]
            open_date = open_date_info['timestamp'].iloc[0] if not open_date_info.empty else None
            days_held = (datetime.now() - open_date).days if open_date else 'N/A'

            position_data.append({
                "Symbol": item.contract.localSymbol,
                "Quantity": item.position,
                "Market Price": item.marketPrice,
                "Market Value": item.marketValue,
                "Average Cost": item.averageCost,
                "Unrealized P&L": item.unrealizedPNL,
                "Days Held": days_held,
                "Open Date": open_date.strftime('%Y-%m-%d') if open_date else 'N/A'
            })

        ib.disconnect()
        st.toast("Disconnected from IB Gateway.", icon="🔌")
        return pd.DataFrame(position_data)

    except Exception as e:
        st.error(f"Failed to fetch portfolio data: {e}")
        if ib.isConnected():
            ib.disconnect()
        return pd.DataFrame()

# Section B, C, D: Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💼 Portfolio Monitor", "📊 Analytics", "📈 Trade Ledger", "📋 System Health & Logs", "💹 Market Data"])

with tab1:
    st.subheader("Live Portfolio Monitor")
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame()

    if st.button("🔄 Fetch Active Positions", width='stretch'):
        with st.spinner("Connecting to IB... Fetching portfolio..."):
            st.session_state.portfolio_data = fetch_portfolio_data(config, trade_df)

    if not st.session_state.portfolio_data.empty:
        # --- Apply conditional formatting ---
        def highlight_aged_positions(row):
            if row['Days Held'] >= 4:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)

        st.dataframe(
            st.session_state.portfolio_data.style.apply(highlight_aged_positions, axis=1),
            width='stretch'
        )
    else:
        st.info("Click the button to fetch live portfolio data.")

with tab2:
    st.subheader("Risk-Adjusted Performance")
    if not trade_df.empty:
        # Equity Curve
        trade_df_sorted = trade_df.sort_values('timestamp')
        trade_df_sorted['cumulative_pnl'] = trade_df_sorted['total_value_usd'].cumsum()
        fig_equity = px.line(trade_df_sorted, x='timestamp', y='cumulative_pnl', title="Equity Curve")
        st.plotly_chart(fig_equity, width='stretch')

        # Periodical P&L
        period = 'D' if (trade_df['timestamp'].max() - trade_df['timestamp'].min()).days > 1 else 'H'
        pnl_by_period = trade_df.set_index('timestamp').resample(period)['total_value_usd'].sum().reset_index()
        pnl_by_period['color'] = pnl_by_period['total_value_usd'].apply(lambda x: 'green' if x >= 0 else 'red')
        fig_bar = px.bar(pnl_by_period, x='timestamp', y='total_value_usd', title=f"{'Daily' if period == 'D' else 'Hourly'} P&L", color='color', color_discrete_map={'green':'#2ca02c', 'red':'#d62728'})
        st.plotly_chart(fig_bar, width='stretch')

        # Drawdown Chart
        trade_df_sorted['running_max'] = trade_df_sorted['cumulative_pnl'].cummax()
        trade_df_sorted['drawdown'] = trade_df_sorted['running_max'] - trade_df_sorted['cumulative_pnl']
        fig_drawdown = px.area(trade_df_sorted, x='timestamp', y='drawdown', title="Drawdown from Peak Equity")
        st.plotly_chart(fig_drawdown, width='stretch')

        # P&L by Strategy/Combo
        if 'combo_id' in trade_df.columns:
            pnl_by_strategy = trade_df.groupby('combo_id')['total_value_usd'].sum().sort_values()
            pnl_by_strategy = pnl_by_strategy.reset_index()
            pnl_by_strategy['color'] = pnl_by_strategy['total_value_usd'].apply(lambda x: 'green' if x >= 0 else 'red')
            fig_strategy = px.bar(pnl_by_strategy, y='combo_id', x='total_value_usd', orientation='h', title="P&L by Strategy", color='color', color_discrete_map={'green':'#2ca02c', 'red':'#d62728'})
            fig_strategy.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_strategy, width='stretch')
    else:
        st.info("No trade data to display charts.")

with tab3:
    st.subheader("Full Trade Ledger")
    if not trade_df.empty:
        st.dataframe(trade_df, width='stretch', height=600)
    else:
        st.info("Trade ledger is empty.")

with tab4:
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

with tab5:
    st.subheader("Live Coffee Futures Prices")

    if 'market_data' not in st.session_state:
        st.session_state.market_data = pd.DataFrame()

    if st.button("📈 Fetch Snapshot", width='stretch'):
        with st.spinner("Connecting to IB... Fetching data..."):
            st.session_state.market_data = fetch_market_data(config)

    if not st.session_state.market_data.empty:
        st.dataframe(st.session_state.market_data, width='stretch')
    else:
        st.info("Click the button to fetch live market data.")
