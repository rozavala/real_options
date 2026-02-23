"""
Real Options Portfolio — Main Entry Point

Cross-commodity portfolio home page. Shows health, financials, and recent
activity across all active commodities. Individual commodity drill-down
lives on pages 1-7 (each page has its own commodity selector).

Streamlit's native multi-page support handles routing via the pages/ directory.
"""

import streamlit as st
from trading_bot.logging_config import setup_logging
import os
import json

# Single consolidated dashboard log (not per-commodity)
setup_logging(log_file="logs/dashboard.log")

st.set_page_config(
    layout="wide",
    page_title="Real Options Portfolio",
    page_icon="\U0001f4ca",
    initial_sidebar_state="expanded"
)

# === IMPORTS ===
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import calendar
from datetime import datetime, timezone
from dashboard_utils import (
    discover_active_commodities,
    get_system_heartbeat_for_commodity,
    load_council_history_for_commodity,
    grade_decision_quality,
    fetch_all_live_data,
    get_config,
    load_equity_data,
    fetch_benchmark_data,
    get_starting_capital,
)

config = get_config()

_COMMODITY_META = {
    "KC": {"name": "Coffee", "emoji": "\u2615"},
    "CC": {"name": "Cocoa", "emoji": "\U0001f36b"},
    "NG": {"name": "Natural Gas", "emoji": "\U0001f525"},
    "SB": {"name": "Sugar", "emoji": "\U0001f36c"},
}

st.title("\U0001f4ca Real Options Portfolio")

# =====================================================================
# SECTION 1: Portfolio Health — per-commodity orchestrator status
# =====================================================================
active_commodities = discover_active_commodities()

st.markdown("### Portfolio Health")
health_cols = st.columns(max(len(active_commodities), 1))

for idx, ticker in enumerate(active_commodities):
    meta = _COMMODITY_META.get(ticker, {"name": ticker, "emoji": "\U0001f4ca"})
    hb = get_system_heartbeat_for_commodity(ticker)
    status = hb["orchestrator_status"]

    with health_cols[idx]:
        if status == "ONLINE":
            st.metric(
                f"{meta['emoji']} {meta['name']} ({ticker})",
                "Online",
                delta="Healthy",
                delta_color="normal",
            )
        elif status == "STALE":
            st.metric(
                f"{meta['emoji']} {meta['name']} ({ticker})",
                "Stale",
                delta="Check logs",
                delta_color="off",
            )
        else:
            st.metric(
                f"{meta['emoji']} {meta['name']} ({ticker})",
                "Offline",
                delta="Down",
                delta_color="inverse",
            )

# =====================================================================
# SECTION 2: Financial Summary — NLV, Daily P&L, Portfolio VaR
# =====================================================================
st.markdown("---")
st.markdown("### Financial Summary")

fin_col1, fin_col2, fin_col3 = st.columns(3)

# IB account data (single account, shared across commodities)
try:
    live = fetch_all_live_data(config) if config else {}
except Exception:
    live = {}

with fin_col1:
    try:
        if live.get("connection_status") == "CONNECTED":
            nlv = live.get("net_liquidation", 0.0)
            st.metric("Net Liquidation", f"${nlv:,.2f}")
        else:
            st.metric("Net Liquidation", "IB Offline")
    except Exception:
        st.metric("Net Liquidation", "IB Offline")

with fin_col2:
    try:
        if live.get("connection_status") == "CONNECTED":
            daily_pnl = live.get("daily_pnl", 0.0)
            nlv = live.get("net_liquidation", 0.0)
            pnl_pct = (daily_pnl / nlv * 100) if nlv > 0 else 0.0
            st.metric("Daily P&L", f"${daily_pnl:,.0f}", delta=f"{pnl_pct:+.2f}%")
        else:
            st.metric("Daily P&L", "IB Offline")
    except Exception:
        st.metric("Daily P&L", "IB Offline")

with fin_col3:
    try:
        var_path = os.path.join("data", "var_state.json")
        if os.path.exists(var_path):
            with open(var_path, "r") as f:
                var_data = json.load(f)
            var_95 = var_data.get("var_95", 0)
            limit = var_data.get("var_limit", 0)
            utilization = (var_95 / limit * 100) if limit > 0 else 0.0
            st.metric("Portfolio VaR (95%)", f"${var_95:,.0f}", delta=f"{utilization:.0f}% utilized")
        else:
            st.metric("Portfolio VaR (95%)", "No data")
    except Exception:
        st.metric("Portfolio VaR (95%)", "No data")

# =====================================================================
# SECTION 2b: Equity Curve + Drawdown (account-wide from daily_equity.csv)
# =====================================================================
st.markdown("---")
st.markdown("### Equity Curve")

# Always load from KC — equity_logger only runs for primary commodity but
# records account-wide NLV.
equity_df = load_equity_data(ticker="KC")

if not equity_df.empty:
    equity_df = equity_df.sort_values('timestamp')

    # Merge trade markers from ALL commodities
    all_trade_markers = []
    for _tk in active_commodities:
        _cdf = load_council_history_for_commodity(_tk)
        if not _cdf.empty and 'timestamp' in _cdf.columns:
            _markers = _cdf[['timestamp', 'master_decision']].copy()
            _markers['timestamp'] = pd.to_datetime(_markers['timestamp'])
            all_trade_markers.append(_markers)
    trade_markers = pd.concat(all_trade_markers, ignore_index=True) if all_trade_markers else pd.DataFrame()

    fig_eq = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Equity Curve', 'Drawdown')
    )

    fig_eq.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['total_value_usd'],
            mode='lines',
            name='Equity',
            line=dict(color='#636EFA', width=2)
        ),
        row=1, col=1
    )

    if not trade_markers.empty:
        buys = trade_markers[trade_markers['master_decision'] == 'BULLISH']
        sells = trade_markers[trade_markers['master_decision'] == 'BEARISH']

        if not buys.empty:
            buy_eq = pd.merge_asof(
                buys.sort_values('timestamp'),
                equity_df[['timestamp', 'total_value_usd']].sort_values('timestamp'),
                on='timestamp'
            )
            fig_eq.add_trace(
                go.Scatter(
                    x=buy_eq['timestamp'],
                    y=buy_eq['total_value_usd'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=12, color='#00CC96')
                ),
                row=1, col=1
            )

        if not sells.empty:
            sell_eq = pd.merge_asof(
                sells.sort_values('timestamp'),
                equity_df[['timestamp', 'total_value_usd']].sort_values('timestamp'),
                on='timestamp'
            )
            fig_eq.add_trace(
                go.Scatter(
                    x=sell_eq['timestamp'],
                    y=sell_eq['total_value_usd'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=12, color='#EF553B')
                ),
                row=1, col=1
            )

    # Drawdown
    equity_df['peak'] = equity_df['total_value_usd'].cummax()
    equity_df['drawdown'] = (equity_df['total_value_usd'] - equity_df['peak']) / equity_df['peak'] * 100

    fig_eq.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['drawdown'],
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#EF553B', width=1)
        ),
        row=2, col=1
    )

    fig_eq.update_layout(height=600, showlegend=True, hovermode='x unified')
    fig_eq.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig_eq.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    st.plotly_chart(fig_eq, use_container_width=True)

    max_dd = equity_df['drawdown'].min()
    st.caption(f"Maximum Drawdown: {max_dd:.2f}%")
else:
    st.info("No equity data available. Ensure equity_logger is running.")

# =====================================================================
# SECTION 2c: Risk Metrics — Sharpe Ratio, Max Drawdown
# =====================================================================
st.markdown("---")
st.markdown("### Risk Metrics")

if not equity_df.empty and len(equity_df) >= 10:
    eq_sorted = equity_df.sort_values('timestamp').copy()
    eq_sorted['daily_return'] = eq_sorted['total_value_usd'].pct_change()
    _daily_returns = eq_sorted['daily_return'].dropna()

    # Max drawdown + recovery
    eq_sorted['_peak'] = eq_sorted['total_value_usd'].cummax()
    eq_sorted['_dd'] = (eq_sorted['total_value_usd'] - eq_sorted['_peak']) / eq_sorted['_peak']
    _max_dd_pct = eq_sorted['_dd'].min() * 100

    trough_idx = eq_sorted['_dd'].idxmin()
    post_trough = eq_sorted.loc[trough_idx:]
    recovered = post_trough[post_trough['total_value_usd'] >= post_trough.iloc[0]['_peak']]
    _recovery_days = (recovered.iloc[0]['timestamp'] - eq_sorted.loc[trough_idx, 'timestamp']).days if not recovered.empty else None

    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        if len(_daily_returns) >= 10:
            daily_std = _daily_returns.std()
            daily_mean = _daily_returns.mean()
            sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
            st.metric(
                "Sharpe Ratio", f"{sharpe:.2f}",
                help="Risk-adjusted return. >1.0 is good, >2.0 is excellent, <0.5 is poor"
            )
        else:
            st.metric("Sharpe Ratio", "N/A", help="Needs daily equity data")

    with risk_col2:
        recovery_text = f" ({_recovery_days}d recovery)" if _recovery_days is not None else " (ongoing)"
        st.metric(
            "Max Drawdown", f"{_max_dd_pct:.1f}%",
            help=f"Deepest peak-to-trough decline{recovery_text}"
        )
else:
    st.info("Insufficient equity data for risk metrics (need 10+ daily snapshots).")

# =====================================================================
# SECTION 2d: Monthly Returns Heatmap
# =====================================================================
st.markdown("---")
st.markdown("### Monthly Returns")
st.caption("Calendar view of monthly performance")

if not equity_df.empty:
    _eq_hm = equity_df.copy()
    _eq_hm['month'] = _eq_hm['timestamp'].dt.tz_localize(None).dt.to_period('M')

    monthly = _eq_hm.groupby('month').agg({'total_value_usd': ['first', 'last']})
    monthly.columns = ['start', 'end']
    monthly['return'] = ((monthly['end'] - monthly['start']) / monthly['start']) * 100

    monthly = monthly.reset_index()
    monthly['year'] = monthly['month'].dt.year
    monthly['month_num'] = monthly['month'].dt.month
    monthly['month_name'] = monthly['month_num'].apply(lambda x: calendar.month_abbr[x])

    if len(monthly) > 1:
        pivot = monthly.pivot(index='year', columns='month_name', values='return')
        month_order = [calendar.month_abbr[i] for i in range(1, 13)]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

        fig_hm = px.imshow(
            pivot,
            labels=dict(x="Month", y="Year", color="Return %"),
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            aspect='auto'
        )
        fig_hm.update_layout(height=300)
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Not enough monthly data for heatmap.")
else:
    st.info("Equity data not available for monthly analysis.")

# =====================================================================
# SECTION 2e: Benchmark Comparison
# =====================================================================
st.markdown("---")
st.markdown("### Benchmark Comparison")

if not equity_df.empty:
    starting_capital = get_starting_capital(config) if config else 50000.0
    if not equity_df.empty:
        starting_capital = equity_df.iloc[0]['total_value_usd']

    start_date = equity_df['timestamp'].min()
    end_date = equity_df['timestamp'].max()

    benchmark_df = fetch_benchmark_data(start_date, end_date)

    if not benchmark_df.empty:
        bot_returns = (equity_df.set_index('timestamp')['total_value_usd'] / starting_capital - 1) * 100
        bot_returns = bot_returns.resample('D').last().dropna()

        fig_bm = go.Figure()

        fig_bm.add_trace(go.Scatter(
            x=bot_returns.index,
            y=bot_returns.values,
            name='Real Options',
            line=dict(color='#636EFA', width=2)
        ))

        if 'SPY' in benchmark_df.columns:
            fig_bm.add_trace(go.Scatter(
                x=benchmark_df.index,
                y=benchmark_df['SPY'],
                name='S&P 500',
                line=dict(color='#FFA15A', width=1, dash='dot')
            ))

        from config import get_active_profile
        profile = get_active_profile(config) if config else None
        if profile:
            benchmark_col = f"{profile.ticker}=F"
            if benchmark_col in benchmark_df.columns:
                fig_bm.add_trace(go.Scatter(
                    x=benchmark_df.index,
                    y=benchmark_df[benchmark_col],
                    name=f'{profile.name} Futures',
                    line=dict(color='#00CC96', width=1, dash='dot')
                ))

        fig_bm.update_layout(
            title='Returns vs Benchmarks',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig_bm, use_container_width=True)
    else:
        st.info("Could not fetch benchmark data.")
else:
    st.info("Equity data required for benchmark comparison.")

# =====================================================================
# SECTION 3: Per-Commodity Cards — trade count, last decision, win rate
# =====================================================================
st.markdown("---")
st.markdown("### Commodity Summary")

card_cols = st.columns(max(len(active_commodities), 1))

for idx, ticker in enumerate(active_commodities):
    meta = _COMMODITY_META.get(ticker, {"name": ticker, "emoji": "\U0001f4ca"})
    council_df = load_council_history_for_commodity(ticker)

    with card_cols[idx]:
        st.markdown(f"#### {meta['emoji']} {meta['name']}")

        if council_df.empty:
            st.caption("No trading history yet.")
            continue

        total_trades = len(council_df)

        # Win rate from graded decisions
        graded = grade_decision_quality(council_df)
        resolved = graded[graded["outcome"].isin(["WIN", "LOSS"])] if not graded.empty else pd.DataFrame()
        wins = len(resolved[resolved["outcome"] == "WIN"]) if not resolved.empty else 0
        win_rate = (wins / len(resolved) * 100) if len(resolved) > 0 else 0.0

        # Last decision
        last_row = council_df.iloc[0]
        last_decision = last_row.get("master_decision", "---")
        last_strategy = last_row.get("strategy_type", "---")

        st.metric("Trades", str(total_trades))
        st.metric("Win Rate", f"{win_rate:.0f}%", delta=f"{wins}W / {len(resolved) - wins}L")
        st.caption(f"Last: **{last_decision}** / {last_strategy}")

# =====================================================================
# SECTION 4: Recent Activity Feed — merged council history (top 10)
# =====================================================================
st.markdown("---")
st.markdown("### Recent Activity")

try:
    all_dfs = []
    for ticker in active_commodities:
        df = load_council_history_for_commodity(ticker)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged = merged.sort_values("timestamp", ascending=False).head(10).copy()

        graded_merged = grade_decision_quality(merged)

        display_cols = []
        if "timestamp" in graded_merged.columns:
            now = datetime.now(timezone.utc)

            def _relative_time(ts):
                try:
                    if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                        import pytz
                        ts = pytz.utc.localize(ts)
                    delta = now - ts
                    hours = delta.total_seconds() / 3600
                    if hours < 1:
                        return f"{int(delta.total_seconds() / 60)}m ago"
                    elif hours < 24:
                        return f"{int(hours)}h ago"
                    else:
                        return f"{int(hours / 24)}d ago"
                except Exception:
                    return "?"

            graded_merged["Time"] = graded_merged["timestamp"].apply(_relative_time)
            display_cols.append("Time")

        if "commodity" in graded_merged.columns:
            graded_merged["Commodity"] = graded_merged["commodity"]
            display_cols.append("Commodity")

        col_map = {
            "contract": "Contract",
            "master_decision": "Decision",
            "master_confidence": "Confidence",
            "strategy_type": "Strategy",
            "thesis_strength": "Thesis",
            "trigger_type": "Trigger",
            "outcome": "Outcome",
            "pnl_realized": "P&L",
        }
        for src, dst in col_map.items():
            if src in graded_merged.columns:
                graded_merged[dst] = graded_merged[src]
                display_cols.append(dst)

        # Format confidence as percentage
        if "Confidence" in graded_merged.columns:
            graded_merged["Confidence"] = graded_merged["Confidence"].apply(
                lambda x: f"{float(x)*100:.0f}%" if x is not None else "?"
            )

        # Format outcome with visual indicators
        if "Outcome" in graded_merged.columns:
            graded_merged["Outcome"] = graded_merged["Outcome"].apply(
                lambda x: "\u2705 WIN" if x == "WIN" else "\u274c LOSS" if x == "LOSS" else "\u2014"
            )

        # Format P&L as currency
        if "P&L" in graded_merged.columns:
            graded_merged["P&L"] = graded_merged["P&L"].apply(
                lambda x: f"${float(x):+,.2f}" if pd.notna(x) and x != 0 else "\u2014"
            )

        if display_cols:
            st.dataframe(
                graded_merged[display_cols],
                width='stretch',
                hide_index=True,
            )
        else:
            st.info("No decision columns available.")
    else:
        st.info("No council decisions yet.")
except Exception:
    st.info("No decision data available.")

# =====================================================================
# SECTION 5: Navigation
# =====================================================================
st.markdown("---")
st.markdown("### Navigate")

if hasattr(st, "page_link"):
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/1_Cockpit.py", label="Cockpit", icon="\U0001f985",
                      help="Is the system running? Check positions, health, emergencies", width="stretch")
        st.page_link("pages/3_The_Council.py", label="The Council", icon="\U0001f9e0",
                      help="Why did we decide that? Agent debate, voting, forensics", width="stretch")
        st.page_link("pages/5_Utilities.py", label="Utilities", icon="\U0001f527",
                      help="Debug and control: logs, manual trading, reconciliation", width="stretch")
        st.page_link("pages/7_Brier_Analysis.py", label="Brier Analysis", icon="\U0001f3af",
                      help="Which agents need tuning? Accuracy, calibration, learning", width="stretch")
    with col2:
        st.page_link("pages/2_The_Scorecard.py", label="The Scorecard", icon="\u2696\ufe0f",
                      help="How are we performing? Win rates, decision quality, learning curves", width="stretch")
        st.page_link("pages/4_Financials.py", label="Trade Analytics", icon="\U0001f4c8",
                      help="Per-commodity strategy performance, trade breakdown, execution ledger", width="stretch")
        st.page_link("pages/6_Signal_Overlay.py", label="Signal Overlay", icon="\U0001f3af",
                      help="How do signals align with price? Visual forensics", width="stretch")
        st.page_link("pages/8_LLM_Monitor.py", label="LLM Monitor", icon="\U0001f4b0",
                      help="API costs, budget utilization, provider health, latency", width="stretch")
        st.page_link("pages/9_Portfolio.py", label="Portfolio", icon="\U0001f4ca",
                      help="Account-wide risk status, cross-commodity positions, engine health", width="stretch")
else:
    st.markdown("""
    Use the sidebar to navigate between pages:

    | Page | Purpose |
    |------|---------|
    | **Cockpit** | Is the system running? Check positions, health, emergencies |
    | **Scorecard** | How are we performing? Win rates, decision quality, learning curves |
    | **Council** | Why did we decide that? Agent debate, voting, forensics |
    | **Trade Analytics** | Per-commodity strategy performance, trade breakdown, execution ledger |
    | **Utilities** | Debug and control: logs, manual trading, reconciliation |
    | **Signal Overlay** | How do signals align with price? Visual forensics |
    | **Brier Analysis** | Which agents need tuning? Accuracy, calibration, learning |
    | **LLM Monitor** | API costs, budget utilization, provider health, latency |
    | **Portfolio** | Account-wide risk status, cross-commodity positions, engine health |
    """)

active_str = ", ".join(
    f"{_COMMODITY_META.get(t, {}).get('emoji', '')} {t}" for t in active_commodities
)
st.caption(f"Active commodities: {active_str}")
