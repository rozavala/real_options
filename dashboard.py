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
from collections import deque

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
from trading_bot.model_signals import get_model_signals_df
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
DEFAULT_STARTING_CAPITAL = 250000

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

@st.cache_data(ttl=300)
def fetch_todays_benchmark_data():
    """Fetches today's performance for SPY and KC=F."""
    try:
        tickers = ['SPY', 'KC=F']
        # Fetch 5 days to ensure we have previous close
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)['Close']

        if data.empty:
            return {}

        changes = {}
        for ticker in tickers:
            try:
                # Get the series for the ticker
                if isinstance(data, pd.DataFrame) and ticker in data.columns:
                    series = data[ticker].dropna()
                elif isinstance(data, pd.Series) and data.name == ticker:
                    series = data.dropna()
                else:
                    # Handle multi-index or single column edge cases
                    # If data is single column, yf might return Series or DataFrame with that col
                    if len(tickers) == 1:
                        series = data.dropna()
                    else:
                        continue

                if len(series) >= 2:
                    prev_close = series.iloc[-2]
                    current = series.iloc[-1]
                    if prev_close != 0:
                        pct_change = ((current - prev_close) / prev_close) * 100
                        changes[ticker] = pct_change
                    else:
                        changes[ticker] = 0.0
                else:
                    changes[ticker] = 0.0
            except Exception:
                changes[ticker] = 0.0

        return changes
    except Exception as e:
        # Fail silently
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
    # Initialize with all required keys
    required_tags = [
        "NetLiquidation", "UnrealizedPnL", "RealizedPnL",
        "MaintMarginReq", "EquityWithLoanValue", "PreviousDayEquityWithLoanValue"
    ]
    summary_data = {tag: 0.0 for tag in required_tags}

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
    """
    Fetches live portfolio from IB and breaks it down into individual spread legs
    using the local ledger to recover Strategy/Combo IDs and Entry Prices.
    """
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
            sorted_ledger = trade_ledger_df.sort_values('timestamp')
        else:
            sorted_ledger = pd.DataFrame()

        detailed_position_data = []

        for item in portfolio:
            symbol = item.contract.localSymbol
            real_qty = item.position
            
            # --- Inventory Reconstruction with Detailed Attributes ---
            # We want to know exactly WHICH ledger entries correspond to the current open position
            # to recover the Combo ID and Original Entry Price.
            
            active_constituents = [] # List of dicts: {qty, price, combo_id, timestamp}

            if not sorted_ledger.empty:
                symbol_trades = sorted_ledger[sorted_ledger['local_symbol'] == symbol]

                if not symbol_trades.empty:
                    # Inventory Stack: List of dicts
                    inventory = deque()

                    for _, trade in symbol_trades.iterrows():
                        t_qty = trade['quantity']
                        if trade['action'] == 'SELL':
                            t_qty = -t_qty

                        entry_price = trade['avg_fill_price'] if 'avg_fill_price' in trade else 0.0
                        combo_id = trade['combo_id'] if 'combo_id' in trade else 'Unknown'

                        trade_details = {
                            'qty': t_qty,
                            'price': entry_price,
                            'combo_id': combo_id,
                            'timestamp': trade['timestamp']
                        }

                        if not inventory:
                            inventory.append(trade_details)
                            continue

                        head = inventory[0]

                        # Check Direction Match
                        head_qty = head['qty']

                        # Same Direction -> Add to stack
                        if (t_qty > 0 and head_qty > 0) or (t_qty < 0 and head_qty < 0):
                            inventory.append(trade_details)
                        else:
                            # Opposite Direction -> Reduce (FIFO)
                            remaining_close = abs(t_qty)

                            while remaining_close > 0 and inventory:
                                lot = inventory[0]
                                lot_qty = lot['qty']

                                # Safety: Break if direction flipped in stack (unlikely)
                                if (lot_qty > 0 and t_qty > 0) or (lot_qty < 0 and t_qty < 0):
                                     break

                                matched = min(abs(lot_qty), remaining_close)
                                remaining_close -= matched

                                # Update lot remaining quantity
                                new_lot_abs = abs(lot_qty) - matched

                                if new_lot_abs < 1e-9:
                                    inventory.popleft() # Fully consumed
                                else:
                                    # Update head with reduced quantity
                                    new_signed_qty = new_lot_abs if lot_qty > 0 else -new_lot_abs
                                    lot['qty'] = new_signed_qty
                                    inventory[0] = lot # Update deque

                            # If we flipped direction (Net Position Change)
                            if remaining_close > 1e-9:
                                remaining_signed = remaining_close if t_qty > 0 else -remaining_close
                                trade_details['qty'] = remaining_signed
                                inventory.append(trade_details)

                    # --- Reconciliation ---
                    # Match Inventory to Actual IB Position (Newest -> Oldest)
                    target_abs = abs(real_qty)
                    collected = 0

                    # Iterate backwards to find the specific lots that make up the current position
                    for lot in reversed(list(inventory)):
                        if abs(collected - target_abs) < 1e-9:
                            break

                        lot_qty = lot['qty']
                        needed = target_abs - collected
                        take = min(abs(lot_qty), needed)

                        # Add a record for this "slice" of the position
                        # Determine sign based on lot direction
                        take_signed = take if lot_qty > 0 else -take

                        active_constituents.append({
                            'qty': take_signed,
                            'price': lot['price'],
                            'combo_id': lot['combo_id'],
                            'timestamp': lot['timestamp']
                        })

                        collected += take

            # If we couldn't match with ledger (or ledger empty), create a dummy constituent
            # derived from IB data.
            if not active_constituents and abs(real_qty) > 0:
                # Fallback Price Logic:
                # If KC/KO, item.averageCost from IB is typically Total Cost ($) per contract.
                # We need to convert this to Price per Unit in Cents to match our system convention.
                fallback_price = item.averageCost
                try:
                    f_mult = float(item.contract.multiplier)
                except (ValueError, TypeError):
                    f_mult = 37500.0 if 'KC' in symbol or 'KO' in symbol else 1.0

                if f_mult == 37500.0:
                     # Convert Total Cost ($) to Price in Cents: (Cost / 37500) * 100
                     fallback_price = (item.averageCost / f_mult) * 100.0

                active_constituents.append({
                    'qty': real_qty,
                    'price': fallback_price,
                    'combo_id': 'Unmatched',
                    'timestamp': datetime.now() # Unknown age
                })

            # --- Build DataFrame Rows ---
            for constituent in active_constituents:
                # Calculate P&L for this specific constituent
                qty = constituent['qty']
                entry_price = constituent['price']
                mkt_price = item.marketPrice

                # Multiplier
                try:
                    mult = float(item.contract.multiplier)
                except (ValueError, TypeError):
                    mult = 37500.0 if 'KC' in symbol or 'KO' in symbol else 1.0

                # Adjustment for Cents (KC quotes in cents)
                # If multiplier is 37500, we assume it's Coffee.
                if mult == 37500.0:
                    mult = mult / 100.0
                    # IB returns marketPrice in Dollars (e.g. 0.1911), but Entry Price (from Ledger) is in Cents (e.g. 19.11).
                    # We must convert marketPrice to Cents to calculate P&L correctly.
                    mkt_price = mkt_price * 100.0

                # P&L: (Mark - Entry) * Qty * Multiplier
                unrealized_pnl = (mkt_price - entry_price) * qty * mult
                mkt_value = mkt_price * qty * mult

                days_held = (datetime.now() - constituent['timestamp']).days

                detailed_position_data.append({
                    "Combo ID": str(constituent['combo_id']),
                    "Symbol": symbol,
                    "Quantity": qty,
                    "Entry Price": entry_price,
                    "Mark Price": mkt_price,
                    "Mkt Value": mkt_value,
                    "Unrealized P&L": unrealized_pnl,
                    "Days Held": days_held,
                    "Open Date": constituent['timestamp'].strftime('%Y-%m-%d')
                })

        ib.disconnect()
        return pd.DataFrame(detailed_position_data)

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

# Determine Starting Capital
starting_capital = DEFAULT_STARTING_CAPITAL
equity_file = os.path.join("data", "daily_equity.csv")
if os.path.exists(equity_file):
    try:
        equity_df_start = pd.read_csv(equity_file)
        if not equity_df_start.empty:
             equity_df_start['timestamp'] = pd.to_datetime(equity_df_start['timestamp'])
             equity_df_start = equity_df_start.sort_values('timestamp')
             starting_capital = equity_df_start.iloc[0]['total_value_usd']
    except Exception as e:
        st.warning(f"Failed to load daily_equity.csv for starting capital: {e}")

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

# --- Live Today's Section ---
st.header("Live Today's Market")

# Fetch Account Data
account_data = fetch_account_summary_data(config)
net_liq = account_data.get("NetLiquidation", 0.0)
maint_margin = account_data.get("MaintMarginReq", 0.0)
equity_with_loan = account_data.get("EquityWithLoanValue", 0.0)
prev_day_equity = account_data.get("PreviousDayEquityWithLoanValue", 0.0)

# Fetch Benchmarks
todays_benchmarks = fetch_todays_benchmark_data()
spy_daily_change = todays_benchmarks.get('SPY', 0.0)
kc_daily_change = todays_benchmarks.get('KC=F', 0.0)

# Calculate Daily P&L
if equity_with_loan > 0 and prev_day_equity > 0:
    daily_pnl = equity_with_loan - prev_day_equity
    daily_pnl_pct = (daily_pnl / prev_day_equity) * 100
else:
    daily_pnl = 0.0
    daily_pnl_pct = 0.0

# Calculate Margin Utilization
if net_liq > 0:
    margin_util_pct = (maint_margin / net_liq) * 100
    margin_cushion_pct = 100 - margin_util_pct
else:
    margin_util_pct = 0.0
    margin_cushion_pct = 100.0

# Display Metrics
live_cols = st.columns(4)

# 1. Net Liquidation & Daily P&L
live_cols[0].metric(
    "Net Liquidation",
    f"${net_liq:,.2f}",
    delta=f"${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%) Today",
    help="Net Liquidation Value. Delta shows Daily P&L (EquityWithLoan - PrevDayEquity)."
)

# 2. Margin Utilization
margin_label = "Margin Utilization"
if margin_cushion_pct < 10:
    margin_label += " ⚠️ (CRITICAL)"

live_cols[1].metric(
    margin_label,
    f"{margin_util_pct:.1f}%",
    delta=f"{margin_cushion_pct:.1f}% Cushion",
    delta_color="normal",
    help=f"Maintenance Margin: ${maint_margin:,.2f}"
)

# 3. Benchmarks
live_cols[2].metric("S&P 500 (Daily)", f"{spy_daily_change:+.2f}%")
live_cols[3].metric("Coffee (Daily)", f"{kc_daily_change:+.2f}%")

st.divider()

# --- Header: Performance Dashboard ---
st.header("Performance Dashboard")

if not trade_df.empty:
    # 1. Fetch Account Summary for Accurate P&L
    # Reuse account_data fetched above

    # Calculate P&L: If we have live data, use it. Else fallback to ledger (which may show dip due to open positions)
    if net_liq > 0:
        total_pnl = net_liq - starting_capital
        pnl_label = "Bot P&L (Total Equity)"
        pnl_help = f"Net Liquidation Value (${net_liq:,.2f}) - Starting Capital (${starting_capital:,.0f})"
    else:
        total_pnl = trade_df['total_value_usd'].sum()
        pnl_label = "Net Cash Flow (Not P&L)"
        pnl_help = "Warning: This is purely cash flow. It ignores the current value of open positions."
        st.warning("⚠️ Disconnected from IB. This metric shows Net Cash Flow only. It treats open Long positions as 100% loss and open Short positions as 100% profit until closed.")

    bot_return_pct = (total_pnl / starting_capital) * 100
    
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
                       help=f"Based on ${starting_capital:,.0f} starting capital")
    
    kpi_cols[2].metric("S&P 500 Benchmark", f"{spy_return:.2f}%", 
                       delta=f"{bot_return_pct - spy_return:.2f}% vs SPY")
    
    kpi_cols[3].metric("Coffee Benchmark", f"{kc_return:.2f}%", 
                       delta=f"{bot_return_pct - kc_return:.2f}% vs Coffee")

else:
    st.info("No trade history found. KPIs unavailable.")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💼 Portfolio", "📊 Analytics", "📈 Trade Ledger", "📋 System Health", "💹 Market", "🤖 Model Signals"])

with tab1:
    st.subheader("Active Portfolio")
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame()

    if st.button("🔄 Fetch Active Positions", width='stretch'):
        with st.spinner("Fetching portfolio from IB..."):
            st.session_state.portfolio_data = fetch_portfolio_data(config, trade_df)

    if not st.session_state.portfolio_data.empty:
        df = st.session_state.portfolio_data

        # Group by Combo ID
        combo_groups = df.groupby("Combo ID")

        total_pnl_all = df["Unrealized P&L"].sum()
        st.metric("Total Portfolio Unrealized P&L", f"${total_pnl_all:,.2f}")
        
        st.divider()

        for combo_id, group in combo_groups:
            # Calculate Spread Stats
            spread_pnl = group["Unrealized P&L"].sum()
            max_days = group["Days Held"].max()

            # Create a label for the expander
            label = f"Spread: {combo_id} | P&L: ${spread_pnl:,.2f} | Max Days Held: {max_days}"

            # Determine color/icon based on P&L or Age
            icon = "🟢" if spread_pnl >= 0 else "🔴"
            if max_days >= 4:
                icon = "⚠️" # Risk Warning
                label += " (Review Age)"

            with st.expander(f"{icon} {label}", expanded=True):
                 # Format and Display the Legs
                 display_df = group[[
                     "Symbol", "Quantity", "Entry Price", "Mark Price",
                     "Unrealized P&L", "Days Held", "Open Date"
                 ]].copy()

                 def highlight_age(row):
                     if row['Days Held'] >= 4:
                         return ['background-color: #ffcccc'] * len(row)
                     return [''] * len(row)

                 st.dataframe(
                     display_df.style.apply(highlight_age, axis=1).format({
                         "Entry Price": "${:.4f}",
                         "Mark Price": "${:.4f}",
                         "Unrealized P&L": "${:.2f}"
                     }),
                     width='stretch',
                     hide_index=True
                 )

    else:
        st.info("Portfolio empty or not fetched.")

with tab2:
    st.subheader("Strategy vs. Benchmarks")
    if not trade_df.empty and not benchmarks.empty:
        trade_df_sorted = trade_df.sort_values('timestamp')

        # Prepare Bot Data (Daily Equity)
        # Check for daily_equity.csv first (for accurate Equity Curve)
        if os.path.exists(equity_file):
            # Equity DF already loaded during startup, just reload to be safe if file changed
            equity_df = pd.read_csv(equity_file)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            daily_vals = equity_df.set_index('timestamp')['total_value_usd']
            # Calculate Return %: (NetLiq / Start) - 1
            bot_series = ((daily_vals / starting_capital) - 1) * 100
        else:
            # Fallback to Cash Flow (The "Dip" / "Spike" chart)
            st.warning("⚠️ daily_equity.csv not found. Showing Cash Flow instead of Equity.")
            daily_bot = trade_df_sorted.set_index('timestamp').resample('D')['total_value_usd'].sum().cumsum().ffill()
            bot_series = (daily_bot / starting_capital) * 100

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

with tab6:
    st.subheader("Model Signals Log")
    signals_df = get_model_signals_df()

    if not signals_df.empty:
        # Display the raw table
        st.dataframe(
            signals_df.sort_values('timestamp', ascending=False),
            width='stretch',
            height=300,
            column_config={
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "sma_200": st.column_config.NumberColumn("SMA 200", format="$%.2f"),
                "expected_price": st.column_config.NumberColumn("Exp Price", format="$%.2f"),
                "confidence": st.column_config.NumberColumn("Conf", format="%.2%"),
            }
        )

        st.divider()
        st.subheader("Price vs. Prediction Analysis")

        # Visualization
        # Filter for only rows that have price data (backward compatibility)
        viz_df = signals_df.dropna(subset=['price', 'sma_200']).copy()

        if not viz_df.empty:
            # Dropdown to select contract
            contracts = viz_df['contract'].unique()
            selected_contract = st.selectbox("Select Contract to Visualize", contracts)

            contract_df = viz_df[viz_df['contract'] == selected_contract].sort_values('timestamp')

            if not contract_df.empty:
                fig = go.Figure()

                # Actual Price
                fig.add_trace(go.Scatter(
                    x=contract_df['timestamp'],
                    y=contract_df['price'],
                    mode='lines+markers',
                    name='Price'
                ))

                # SMA 200
                fig.add_trace(go.Scatter(
                    x=contract_df['timestamp'],
                    y=contract_df['sma_200'],
                    mode='lines',
                    name='SMA 200',
                    line=dict(dash='dash', color='orange')
                ))

                # Expected Price
                if 'expected_price' in contract_df.columns:
                     fig.add_trace(go.Scatter(
                        x=contract_df['timestamp'],
                        y=contract_df['expected_price'],
                        mode='markers',
                        name='Expected Price',
                        marker=dict(symbol='x', size=10, color='green')
                    ))

                fig.update_layout(title=f"Model Signals: {selected_contract}", xaxis_title="Timestamp", yaxis_title="Price")
                st.plotly_chart(fig, width='stretch')
            else:
                st.info(f"No data for {selected_contract}")
        else:
             st.info("No signal data with price information available yet.")

    else:
        st.info("No model signals logged yet.")
