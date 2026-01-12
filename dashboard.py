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
import textwrap
import yfinance as yf
from collections import deque

# --- 1. Asyncio Event Loop Fix for Streamlit ---
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
# Get initial capital from .env, default to 250000 if missing
DEFAULT_STARTING_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "250000"))

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
    """Finds all relevant log files and caches their last 50 lines."""
    try:
        list_of_logs = glob.glob('logs/*.log')
        if not list_of_logs:
            return {}

        logs_content = {}
        for log_path in list_of_logs:
            filename = os.path.basename(log_path)
            if any(char.isdigit() for char in filename): continue
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
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)['Close']
        if data.empty: return {}
        changes = {}
        for ticker in tickers:
            try:
                if isinstance(data, pd.DataFrame) and ticker in data.columns:
                    series = data[ticker].dropna()
                else:
                    if len(tickers) == 1: series = data.dropna()
                    else: continue

                if len(series) >= 2:
                    changes[ticker] = ((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100
                else:
                    changes[ticker] = 0.0
            except:
                changes[ticker] = 0.0
        return changes
    except:
        return {}

@st.cache_data(ttl=3600)
def fetch_benchmark_data(start_date, end_date):
    """Fetches S&P 500 (SPY) and Coffee Futures (KC=F) from Yahoo Finance."""
    try:
        tickers = ['SPY', 'KC=F']
        data = yf.download(tickers, start=start_date, end=end_date + timedelta(days=1), progress=False, auto_adjust=True)['Close']
        if data.empty: return pd.DataFrame()
        normalized = data.apply(lambda x: (x / x.dropna().iloc[0]) - 1) * 100
        return normalized
    except Exception as e:
        st.warning(f"Could not fetch benchmarks: {e}")
        return pd.DataFrame()

# --- 5. Live Data Functions (IB Connection) ---

@st.cache_data(ttl=60)
def fetch_live_dashboard_data(_config):
    """Consolidated fetcher for Account Summary, Daily P&L, and Active Futures Data."""
    ib = IB()
    data = {"NetLiquidation": 0.0, "MaintMarginReq": 0.0, "DailyPnL": 0.0, "DailyPnLPct": 0.0, "KC_DailyChange": 0.0, "KC_Price": 0.0, "MarketData": pd.DataFrame()}

    try:
        ib.connect(_config['connection']['host'], _config['connection']['port'], clientId=random.randint(1000, 9999))

        # 1. Account Summary
        summary = ib.accountSummary()
        for item in summary:
            if item.tag == "NetLiquidation": data["NetLiquidation"] = float(item.value)
            elif item.tag == "MaintMarginReq": data["MaintMarginReq"] = float(item.value)

        # 2. Daily P&L
        accounts = ib.managedAccounts()
        if accounts:
            acct = accounts[0]
            pnl_stream = ib.reqPnL(acct)
            start_pnl = datetime.now()
            while (datetime.now() - start_pnl).seconds < 2:
                ib.sleep(0.1)
                if pnl_stream.dailyPnL != 0.0 and not util.isNan(pnl_stream.dailyPnL): break
            if not util.isNan(pnl_stream.dailyPnL):
                data["DailyPnL"] = pnl_stream.dailyPnL
                start_equity = data["NetLiquidation"] - data["DailyPnL"]
                if start_equity > 0: data["DailyPnLPct"] = (data["DailyPnL"] / start_equity) * 100
            ib.cancelPnL(acct)

        # 3. Active Futures
        active_futures = ib.run(get_active_futures(ib, _config['symbol'], _config['exchange']))
        if active_futures:
            tickers = [ib.reqMktData(c, '', False, False) for c in active_futures]
            start_mkt = datetime.now()
            while (datetime.now() - start_mkt).seconds < 4:
                ib.sleep(0.2)
                if all(not util.isNan(t.last) or not util.isNan(t.close) for t in tickers): break

            market_rows = []
            for contract, ticker in zip(active_futures, tickers):
                price = ticker.last if not util.isNan(ticker.last) else (ticker.close if not util.isNan(ticker.close) else 0.0)
                source = "Live" if not util.isNan(ticker.last) else "Close (Prev)"
                market_rows.append({
                    "Contract": contract.localSymbol, "Price": price, "Source": source,
                    "Bid": ticker.bid if not util.isNan(ticker.bid) else 0.0,
                    "Ask": ticker.ask if not util.isNan(ticker.ask) else 0.0,
                    "Time": ticker.time.strftime('%H:%M:%S') if ticker.time else "N/A"
                })
            data["MarketData"] = pd.DataFrame(market_rows)

            if len(tickers) > 0:
                front = tickers[0]
                last = front.last if not util.isNan(front.last) else front.close
                prev = front.close if not util.isNan(front.close) else 0.0
                data["KC_Price"] = last
                if prev > 0 and last > 0: data["KC_DailyChange"] = ((last - prev) / prev) * 100

        ib.disconnect()
        return data
    except Exception as e:
        if ib.isConnected(): ib.disconnect()
        return data

def fetch_portfolio_data(config, trade_ledger_df):
    """
    Fetches live portfolio from IB and breaks it down into individual spread legs
    using the local ledger to recover Strategy/Combo IDs and Entry Prices.
    (ROBUST FIFO LOGIC PRESERVED)
    """
    ib = IB()
    try:
        ib.connect(
            config['connection']['host'],
            config['connection']['port'],
            clientId=random.randint(100, 1000)
        )
        # st.toast("Connected to IB Gateway...", icon="🔗") # Optional toast

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
            active_constituents = []

            if not sorted_ledger.empty:
                symbol_trades = sorted_ledger[sorted_ledger['local_symbol'] == symbol]

                if not symbol_trades.empty:
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
                        head_qty = head['qty']

                        if (t_qty > 0 and head_qty > 0) or (t_qty < 0 and head_qty < 0):
                            inventory.append(trade_details)
                        else:
                            remaining_close = abs(t_qty)
                            while remaining_close > 0 and inventory:
                                lot = inventory[0]
                                lot_qty = lot['qty']
                                if (lot_qty > 0 and t_qty > 0) or (lot_qty < 0 and t_qty < 0): break

                                matched = min(abs(lot_qty), remaining_close)
                                remaining_close -= matched
                                new_lot_abs = abs(lot_qty) - matched

                                if new_lot_abs < 1e-9:
                                    inventory.popleft()
                                else:
                                    new_signed_qty = new_lot_abs if lot_qty > 0 else -new_lot_abs
                                    lot['qty'] = new_signed_qty
                                    inventory[0] = lot

                            if remaining_close > 1e-9:
                                remaining_signed = remaining_close if t_qty > 0 else -remaining_close
                                trade_details['qty'] = remaining_signed
                                inventory.append(trade_details)

                    # --- Reconciliation ---
                    target_abs = abs(real_qty)
                    collected = 0

                    for lot in reversed(list(inventory)):
                        if abs(collected - target_abs) < 1e-9: break
                        lot_qty = lot['qty']
                        needed = target_abs - collected
                        take = min(abs(lot_qty), needed)
                        take_signed = take if lot_qty > 0 else -take

                        active_constituents.append({
                            'qty': take_signed,
                            'price': lot['price'],
                            'combo_id': lot['combo_id'],
                            'timestamp': lot['timestamp']
                        })
                        collected += take

            if not active_constituents and abs(real_qty) > 0:
                fallback_price = item.averageCost
                try:
                    f_mult = float(item.contract.multiplier)
                except:
                    f_mult = 37500.0 if 'KC' in symbol or 'KO' in symbol else 1.0

                if f_mult == 37500.0:
                     fallback_price = (item.averageCost / f_mult) * 100.0

                active_constituents.append({
                    'qty': real_qty,
                    'price': fallback_price,
                    'combo_id': 'Unmatched',
                    'timestamp': datetime.now()
                })

            for constituent in active_constituents:
                qty = constituent['qty']
                entry_price = constituent['price']
                mkt_price = item.marketPrice

                try:
                    mult = float(item.contract.multiplier)
                except:
                    mult = 37500.0 if 'KC' in symbol or 'KO' in symbol else 1.0

                if mult == 37500.0:
                    mult = mult / 100.0
                    if mkt_price < 50.0: mkt_price = mkt_price * 100.0

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
        if ib.isConnected(): ib.disconnect()
        return pd.DataFrame()

# --- 6. Main Application Logic ---

st.title("Coffee Bot Mission Control ☕")

trade_df = load_trade_data()
config = get_config()
logs_data = load_log_data()

# Starting Capital Logic
starting_capital = DEFAULT_STARTING_CAPITAL
equity_file = os.path.join("data", "daily_equity.csv")
# Load equity data for charts
equity_df = pd.DataFrame()
if os.path.exists(equity_file):
    try:
        e_df = pd.read_csv(equity_file)
        if not e_df.empty:
            starting_capital = e_df.sort_values('timestamp').iloc[0]['total_value_usd']
            equity_df = e_df
    except: pass

# Sidebar
with st.sidebar:
    st.header("Manual Controls")
    if st.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    if st.button("⛔ Cancel All Orders", type="primary", width='stretch'):
        if config:
            with st.spinner("Cancelling..."):
                try:
                    from trading_bot.order_manager import cancel_all_open_orders
                    asyncio.run(cancel_all_open_orders(config))
                    st.success("Orders cancelled.")
                except Exception as e: st.error(f"Error: {e}")
    if st.button("📉 Force Close Aged (Stale)", width='stretch'):
        if config:
            with st.spinner("Closing Stale Positions..."):
                try:
                    from trading_bot.order_manager import close_stale_positions
                    asyncio.run(close_stale_positions(config))
                    st.success("Stale positions closed.")
                except Exception as e: st.error(f"Error: {e}")

# Live Header
live_data = fetch_live_dashboard_data(config)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Net Liquidation", f"${live_data['NetLiquidation']:,.2f}")
col2.metric("Daily P&L", f"${live_data['DailyPnL']:,.2f}", f"{live_data['DailyPnLPct']:+.2f}%")
col3.metric("Margin Util", f"{(live_data['MaintMarginReq']/live_data['NetLiquidation'])*100:.1f}%" if live_data['NetLiquidation'] > 0 else "0%")
benchs = fetch_todays_benchmark_data()
col4.metric("S&P 500", f"{benchs.get('SPY',0):+.2f}%")
col5.metric("Coffee", f"{live_data.get('KC_DailyChange',0):+.2f}%")

st.divider()

# --- Performance Dashboard (Restored) ---
# Display Life-to-Date (LTD) Return vs Benchmarks
st.subheader("📈 Performance Dashboard (Life-to-Date)")

# Calculate Bot Return
bot_return_pct = 0.0
if live_data["NetLiquidation"] > 0 and starting_capital > 0:
    bot_return_pct = ((live_data["NetLiquidation"] - starting_capital) / starting_capital) * 100

# Fetch Benchmark Data (Using default range for LTD)
# Assuming a start date of approx 2024-01-01 for context or using dynamic range if equity_df exists
start_date = datetime(2024, 1, 1)
if not equity_df.empty:
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    start_date = equity_df['timestamp'].min()

benchmark_df = fetch_benchmark_data(start_date, datetime.now())

# Calculate Benchmark Returns
spy_return_pct = 0.0
kc_return_pct = 0.0
if not benchmark_df.empty:
    if 'SPY' in benchmark_df.columns:
        spy_return_pct = benchmark_df['SPY'].iloc[-1]
    if 'KC=F' in benchmark_df.columns:
        kc_return_pct = benchmark_df['KC=F'].iloc[-1]

perf_cols = st.columns(3)
perf_cols[0].metric("🤖 Bot Return (LTD)", f"{bot_return_pct:+.2f}%")
perf_cols[1].metric("🇺🇸 S&P 500 (LTD)", f"{spy_return_pct:+.2f}%", delta_color="off")
perf_cols[2].metric("☕ Coffee Futures (LTD)", f"{kc_return_pct:+.2f}%", delta_color="off")

st.divider()

# Tabs
tabs = st.tabs(["💼 Portfolio", "📊 Analytics", "📈 Trade Ledger", "📋 System Health", "💹 Market", "🤖 Model Signals", "🧠 Council Scorecard"])

with tabs[0]: # Portfolio
    if st.button("Fetch Portfolio"):
        st.session_state.port_df = fetch_portfolio_data(config, trade_df)
    if 'port_df' in st.session_state and not st.session_state.port_df.empty:
        # Group by Combo ID for cleaner view (Matches logic from previous version)
        combo_groups = st.session_state.port_df.groupby("Combo ID")
        for combo_id, group in combo_groups:
             spread_pnl = group["Unrealized P&L"].sum()
             max_days = group["Days Held"].max()
             icon = "🟢" if spread_pnl >= 0 else "🔴"
             with st.expander(f"{icon} Spread: {combo_id} | P&L: ${spread_pnl:,.2f} | Days: {max_days}"):
                 st.dataframe(group)
    else: st.info("Portfolio empty or not fetched.")

with tabs[1]: # Analytics
    if not trade_df.empty:
        # Metric Row
        if live_data.get('NetLiquidation', 0) > 0:
            total_pnl = live_data['NetLiquidation'] - starting_capital
            st.metric("Total P&L (Realized + Unrealized)", f"${total_pnl:,.2f}")
        else:
            st.metric("Total P&L (Realized + Unrealized)", "Offline")
        
        # --- Restore Charts ---
        # We need to construct a basic Equity Curve if possible
        if not equity_df.empty:
             equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
             equity_df = equity_df.sort_values('timestamp')

             # Chart 1: Equity Curve
             fig_equity = px.line(equity_df, x='timestamp', y='total_value_usd', title='Equity Curve (Net Liquidation)')
             st.plotly_chart(fig_equity, width="stretch")

             # Chart 2: Daily Drawdown
             # Calculate drawdown
             equity_df['peak'] = equity_df['total_value_usd'].cummax()
             equity_df['drawdown'] = (equity_df['total_value_usd'] - equity_df['peak']) / equity_df['peak']

             fig_dd = px.area(equity_df, x='timestamp', y='drawdown', title='Drawdown (%)', color_discrete_sequence=['red'])
             st.plotly_chart(fig_dd, width="stretch")

        else:
             st.warning("Daily Equity data not found. Showing Cash Flow only.")
             # Fallback: Cumulative Cash Flow
             trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
             cf = trade_df.set_index('timestamp').sort_index()['total_value_usd'].cumsum()
             st.line_chart(cf)

    else: st.info("No trade history.")

with tabs[2]: # Ledger
    if not trade_df.empty: st.dataframe(trade_df.sort_values('timestamp', ascending=False))

with tabs[3]: # Health
    if logs_data:
        for name, lines in logs_data.items():
            with st.expander(f"{name} Log"): st.code("".join(lines))

with tabs[4]: # Market
    if st.button("🔄 Refresh Market Data", key="refresh_market_tab"):
        st.cache_data.clear()
        st.rerun()

    if 'MarketData' in live_data and not live_data['MarketData'].empty:
        st.dataframe(live_data['MarketData'])
    else: st.info("No market data active.")

with tabs[5]: # Signals
    sigs = get_model_signals_df()
    if not sigs.empty: st.dataframe(sigs.sort_values('timestamp', ascending=False))

# =========================================================
# TAB 7: COUNCIL SCORECARD (FINAL MERGED VERSION)
# =========================================================
with tabs[6]:
    st.header("🧠 Council Scorecard & Brain Health")
    st.caption("Detailed breakdown of Multi-Agent decision making, accuracy, and consensus.")

    council_file = os.path.join("data", "council_history.csv")
    if os.path.exists(council_file):
        try:
            # Load and Prep Data
            council_df = pd.read_csv(council_file)
            council_df['timestamp'] = pd.to_datetime(council_df['timestamp'])
            council_df = council_df.sort_values('timestamp', ascending=False) # Latest top

            # --- NORMALIZE ML SIGNAL FOR COMPARISON ---
            if 'ml_signal' in council_df.columns:
                council_df['ml_sentiment'] = council_df['ml_signal'].map({
                    'LONG': 'BULLISH',
                    'SHORT': 'BEARISH',
                    'NEUTRAL': 'NEUTRAL'
                }).fillna('NEUTRAL')
            else:
                council_df['ml_sentiment'] = 'NEUTRAL'

            # --- 1. HEADLINE DEFENSIVE STATS (Restored) ---
            # Saved by the Bell: ML != NEUTRAL but Master == NEUTRAL
            saved_mask = (council_df['ml_signal'] != 'NEUTRAL') & (council_df['master_decision'] == 'NEUTRAL')
            saved_count = saved_mask.sum()
            # Hallucinations: Compliance Blocked
            hallucination_mask = council_df['compliance_approved'] == False
            hallucination_count = hallucination_mask.sum()

            score_cols = st.columns(3)
            score_cols[0].metric("🛡️ Trades Blocked", f"{saved_count}", help="ML Signal was Active, but Council Vetoed.")
            score_cols[1].metric("🚨 Compliance Blocks", f"{hallucination_count}", help="Master Decision rejected by Compliance Officer.")

            # --- 2. AGENT LEADERBOARD (Who is the smartest?) ---
            market_snapshot = live_data.get("MarketData", pd.DataFrame())

            def get_current_price_for_scoring(contract):
                if market_snapshot.empty: return None
                clean = contract.split('(')[0].strip().replace(' ', '')
                match = market_snapshot[market_snapshot['Contract'].str.contains(clean, case=False)]
                if not match.empty: return match.iloc[0]['Price']
                return None

            # Calculate Scores
            agents = ['master_decision', 'ml_sentiment', 'meteorologist_sentiment', 'macro_sentiment',
                      'geopolitical_sentiment', 'fundamentalist_sentiment', 'sentiment_sentiment', 'technical_sentiment', 'volatility_sentiment'] # <--- NEW
            scores = {a: {'correct': 0, 'total': 0} for a in agents}

            for _, row in council_df.iterrows():
                # Priority 1: Use Reconciled Result
                if 'actual_trend_direction' in row and pd.notna(row['actual_trend_direction']) and row['actual_trend_direction'] in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                    actual_trend = row['actual_trend_direction']
                else:
                    # Priority 2: Use Live Price estimate (Fallback)
                    curr_price = get_current_price_for_scoring(row['contract'])
                    entry_price = row.get('entry_price', 0.0)
                    actual_trend = 'NEUTRAL'
                    if curr_price and entry_price > 0:
                        if curr_price > entry_price: actual_trend = 'BULLISH'
                        elif curr_price < entry_price: actual_trend = 'BEARISH'

                # Grading
                for agent_col in agents:
                    vote = row.get(agent_col, 'N/A')
                    is_correct = False

                    if vote == actual_trend:
                        is_correct = True

                    if vote in ['BULLISH', 'BEARISH']:
                        scores[agent_col]['total'] += 1
                        if is_correct: scores[agent_col]['correct'] += 1

            # Display Leaderboard
            pretty_names = {
                'master_decision': '👑 Master',
                'ml_sentiment': '🤖 ML Model',
                'meteorologist_sentiment': '🌦️ Meteo',
                'macro_sentiment': '💵 Macro',
                'geopolitical_sentiment': '🌍 Geo',
                'fundamentalist_sentiment': '📦 Stocks',
                'sentiment_sentiment': '🧠 Sentiment',
                'technical_sentiment': '📉 Techs',
                'volatility_sentiment': 'Vol Agent' # <--- NEW
            }

            # Add Master Win Rate to Top Row
            m_stats = scores['master_decision']
            m_rate = (m_stats['correct'] / m_stats['total'] * 100) if m_stats['total'] > 0 else 0.0
            score_cols[2].metric("👑 Master Win Rate", f"{m_rate:.1f}%", f"{m_stats['total']} Calls")

            # --- OVERALL WIN RATE (RECONCILED) ---
            if 'pnl_realized' in council_df.columns and 'actual_trend_direction' in council_df.columns:
                reconciled_mask = pd.notna(council_df['pnl_realized'])
                if reconciled_mask.any():
                    reconciled_df = council_df[reconciled_mask]
                    total_rec = len(reconciled_df)
                    wins = (reconciled_df['pnl_realized'] > 0).sum()
                    win_rate_rec = (wins / total_rec * 100) if total_rec > 0 else 0.0
                    st.caption(f"Actual Realized Win Rate (Reconciled): **{win_rate_rec:.1f}%** over {total_rec} graded trades.")

            st.divider()
            st.subheader("🏆 Sub-Agent Accuracy")

            # Use 'actual_trend_direction' for grading if available, otherwise fallback to current price
            # We already computed scores above based on live price. Let's stick to that for simplicity
            # unless we want to strictly use reconciled history for the leaderboard.
            # Ideally, we should use reconciled data for past trades and live data for open ones.
            # For now, keeping the hybrid approach (live price based) for the leaderboard to be responsive.

            sub_agents = [a for a in agents if a != 'master_decision']
            l_cols = st.columns(len(sub_agents))
            for idx, agent_key in enumerate(sub_agents):
                stats = scores[agent_key]
                rate = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
                l_cols[idx].metric(pretty_names[agent_key], f"{rate:.1f}%", f"{stats['total']} Calls")

            st.divider()

            # --- 3. CONSENSUS MATRIX (The "Table" View) ---
            st.subheader("📊 Agent Consensus Matrix")
            st.caption("Compare agent votes side-by-side. Green=Bullish, Red=Bearish.")

            display_cols = ['timestamp', 'contract', 'master_decision', 'ml_sentiment',
                            'meteorologist_sentiment', 'macro_sentiment',
                            'geopolitical_sentiment', 'fundamentalist_sentiment',
                            'sentiment_sentiment', 'technical_sentiment', 'volatility_sentiment'] # <--- NEW

            # Add Result columns if they exist
            if 'actual_trend_direction' in council_df.columns:
                display_cols.extend(['actual_trend_direction', 'pnl_realized'])

            matrix_df = council_df[display_cols].copy()

            # Rename for display
            col_map = {
                'timestamp': 'Time', 'contract': 'Contract', 'master_decision': 'MASTER',
                'ml_sentiment': 'ML Model', 'meteorologist_sentiment': 'Meteo',
                'macro_sentiment': 'Macro', 'geopolitical_sentiment': 'Geo',
                'fundamentalist_sentiment': 'Stocks', 'sentiment_sentiment': 'Sentiment',
                'technical_sentiment': 'Technicals', 'volatility_sentiment': 'Vol Agent',
                'actual_trend_direction': 'Actual Trend', 'pnl_realized': 'P&L (Theo)'
            }
            matrix_df = matrix_df.rename(columns=col_map)

            def color_sentiment(val):
                if val == 'BULLISH': return 'color: #00CC96; font-weight: bold'
                if val == 'BEARISH': return 'color: #EF553B; font-weight: bold'
                if val == 'NEUTRAL': return 'color: gray'
                return ''

            # Define columns to apply coloring to (handle missing columns dynamically)
            subset_cols = [c for c in ['MASTER', 'ML Model', 'Meteo', 'Macro', 'Geo', 'Stocks', 'Sentiment', 'Technicals', 'Vol Agent', 'Actual Trend'] if c in matrix_df.columns]

            st.dataframe(
                matrix_df.style.map(color_sentiment, subset=subset_cols)
                         .format({"Time": lambda t: t.strftime("%m-%d %H:%M")}),
                width="stretch",
                height=400
            )

            # --- 4. DEEP DIVE READER (The "Text" View) ---
            st.divider()
            st.subheader("📝 Decision Deep Dive")

            options = council_df.index
            def format_option(idx):
                r = council_df.loc[idx]
                return f"{r['timestamp'].strftime('%Y-%m-%d %H:%M')} | {r['contract']} | {r['master_decision']}"

            selected_idx = st.selectbox("Select Decision to Inspect:", options, format_func=format_option)

            if selected_idx is not None:
                row = council_df.loc[selected_idx]

                st.info(f"**👑 Master Strategist Decision:** {row['master_decision']} (Confidence: {row.get('master_confidence',0):.2f})")
                st.markdown(f"**Reasoning:** {row['master_reasoning']}")
                if 'entry_price' in row:
                    st.caption(f"Price at Decision: {row['entry_price']}")

                st.markdown("---")

                # 4-Column Layout (2 cards per column)
                c1, c2, c3, c4 = st.columns(4)

                def render_agent(col, title, sentiment, summary):
                    color = "gray"
                    if sentiment == 'BULLISH': color = "green"
                    elif sentiment == 'BEARISH': color = "red"
                    with col:
                        with st.container(border=True):
                            st.markdown(f"#### :{color}[{title}]")
                            st.markdown(f"**Vote:** {sentiment}")
                            with st.expander("Full Report", expanded=True):
                                st.write(summary if isinstance(summary, str) else "No report.")

                with c1:
                    st.caption("🤖 Models & Technicals")
                    # Keep "ML Model" name as requested
                    render_agent(c1, "🤖 ML Model", row.get('ml_sentiment'), f"**Raw Signal:** {row.get('ml_signal')}\n\n**Confidence:** {row.get('ml_confidence', 0):.2%}")
                    render_agent(c1, "📉 Technical Analyst", row.get('technical_sentiment'), row.get('technical_summary'))

                with c2:
                    st.caption("🌱 Physical Market")
                    render_agent(c2, "🌦️ Meteorologist", row.get('meteorologist_sentiment'), row.get('meteorologist_summary'))
                    render_agent(c2, "📦 Inventory/Stocks", row.get('fundamentalist_sentiment'), row.get('fundamentalist_summary'))

                with c3:
                    st.caption("🌍 Macro & Geo")
                    render_agent(c3, "💵 Macro Economist", row.get('macro_sentiment'), row.get('macro_summary'))
                    render_agent(c3, "🌍 Geopolitical", row.get('geopolitical_sentiment'), row.get('geopolitical_summary'))

                with c4:
                    st.caption("🧠 Sentiment & Risk")
                    render_agent(c4, "🧠 Sentiment/COT", row.get('sentiment_sentiment'), row.get('sentiment_summary'))
                    # [NEW] Volatility Agent
                    render_agent(c4, "⚡ Volatility Agent", row.get('volatility_sentiment'), row.get('volatility_summary'))

            # --- 5. HALLUCINATION TABLE (Restored) ---
            if hallucination_count > 0:
                st.divider()
                st.subheader("🚨 Recent Hallucinations (Compliance Blocks)")
                st.dataframe(
                    council_df[hallucination_mask][['timestamp', 'contract', 'master_reasoning', 'master_decision']],
                    width="stretch"
                )

        except Exception as e:
            st.error(f"Error loading Council Data: {e}")
    else:
        st.info("No Council history found yet. Run the bot to generate signals.")
