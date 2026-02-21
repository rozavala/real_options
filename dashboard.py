"""
Coffee Bot Real Options - Main Entry Point

This file serves as the entry point and redirects to the multi-page app.
Streamlit's native multi-page support handles routing via the pages/ directory.
"""

import streamlit as st
from config_loader import load_config, deep_merge
from trading_bot.logging_config import setup_logging

# Set up dashboard-specific logging (per-commodity to avoid log collision)
import os
_dashboard_ticker = os.environ.get("COMMODITY_TICKER", "KC").lower()
setup_logging(log_file=f"logs/dashboard_{_dashboard_ticker}.log")

# Dynamic configuration for commodity-aware branding
_cfg = load_config()
_dashboard_ticker_upper = os.environ.get("COMMODITY_TICKER", "KC").upper()

# Mirror orchestrator's commodity override logic
if _cfg:
    _overrides = _cfg.get('commodity_overrides', {}).get(_dashboard_ticker_upper, {})
    if _overrides:
        _cfg = deep_merge(_cfg, _overrides)
    _cfg.setdefault('commodity', {})['ticker'] = _dashboard_ticker_upper

_EMOJI_MAP = {'Coffee': '\u2615', 'Cocoa': '\U0001f36b', 'Sugar': '\U0001f36c'}
_commodity_name = _cfg.get('commodity', {}).get('name', 'Coffee') if _cfg else 'Coffee'
_commodity_emoji = _EMOJI_MAP.get(_commodity_name.split()[0], '\U0001f4ca')

st.set_page_config(
    layout="wide",
    page_title=f"{_commodity_name} Real Options",
    page_icon=_commodity_emoji,
    initial_sidebar_state="expanded"
)

# === COMMODITY SELECTOR (sidebar) ===
_COMMODITIES = ["KC", "CC", "SB"]
_COMMODITY_LABELS = {
    "KC": "\u2615 KC (Coffee)",
    "CC": "\U0001f36b CC (Cocoa)",
    "SB": "\U0001f36c SB (Sugar)",
}

if 'commodity_ticker' not in st.session_state:
    st.session_state['commodity_ticker'] = os.environ.get('COMMODITY_TICKER', 'KC')

selected = st.sidebar.selectbox(
    "Commodity",
    _COMMODITIES,
    index=_COMMODITIES.index(st.session_state['commodity_ticker']),
    format_func=lambda x: _COMMODITY_LABELS.get(x, x),
)
if selected != st.session_state['commodity_ticker']:
    st.session_state['commodity_ticker'] = selected
    os.environ['COMMODITY_TICKER'] = selected
    st.cache_data.clear()
    st.rerun()

# === IMPORTS ===
from dashboard_utils import (
    get_system_heartbeat,
    fetch_live_dashboard_data,
    fetch_all_live_data,
    get_config,
    fetch_todays_benchmark_data,
    load_council_history,
    get_sentinel_status,
    _resolve_data_path,
)

config = get_config()

# The presence of files in pages/ directory enables multi-page mode automatically.
# This file becomes the "home" page.

st.title(f"{_commodity_emoji} {_commodity_name} Real Options")

# === SYSTEM HEALTH SUMMARY (4 columns) ===
heartbeat = get_system_heartbeat()

col1, col2, col3, col4 = st.columns(4)

with col1:
    orch_status = heartbeat['orchestrator_status']
    if orch_status == "ONLINE":
        st.metric("Orchestrator", "Online", delta="Healthy", delta_color="normal")
    elif orch_status == "STALE":
        st.metric("Orchestrator", "Stale", delta="Check logs", delta_color="off")
    else:
        st.metric("Orchestrator", "Offline", delta="Down", delta_color="inverse")

with col2:
    try:
        live = fetch_all_live_data(config) if config else {}
        conn = live.get('connection_status', 'DISCONNECTED')
        if conn == 'CONNECTED':
            st.metric("IB Gateway", "Connected", delta="Healthy", delta_color="normal")
        elif conn == 'ERROR':
            st.metric("IB Gateway", "Error", delta=live.get('error', '')[:30], delta_color="inverse")
        else:
            st.metric("IB Gateway", "Offline", delta="No connection", delta_color="inverse")
    except Exception:
        st.metric("IB Gateway", "Offline", delta="No connection", delta_color="inverse")
        live = {}

with col3:
    try:
        sentinels = get_sentinel_status()
        stale_count = sum(1 for s in sentinels.values() if s.get('is_stale'))
        total_sentinels = len(sentinels)
        active_count = total_sentinels - stale_count
        if stale_count == 0:
            st.metric("Sentinels", f"{active_count}/{total_sentinels}", delta="All active", delta_color="normal")
        else:
            st.metric("Sentinels", f"{active_count}/{total_sentinels}", delta=f"{stale_count} stale", delta_color="inverse")
    except Exception:
        st.metric("Sentinels", "Unknown", delta="No data", delta_color="off")

with col4:
    try:
        from trading_bot.utils import is_market_open
        market_open = is_market_open(config) if config else False
        if market_open:
            st.metric("Market", "Open", delta="Trading", delta_color="normal")
        else:
            st.metric("Market", "Closed", delta="After hours", delta_color="off")
    except Exception:
        st.metric("Market", "Unknown", delta="No data", delta_color="off")

# === DAILY P&L ===
st.markdown("---")

pnl_col, nlv_col, bench_col = st.columns(3)

with pnl_col:
    try:
        if live.get('connection_status') == 'CONNECTED':
            daily_pnl = live.get('daily_pnl', 0.0)
            nlv = live.get('net_liquidation', 0.0)
            pnl_pct = (daily_pnl / nlv * 100) if nlv > 0 else 0.0
            st.metric("Daily P&L", f"${daily_pnl:,.0f}", delta=f"{pnl_pct:+.2f}%")
        else:
            st.metric("Daily P&L", "IB Offline")
    except Exception:
        st.metric("Daily P&L", "IB Offline")

with nlv_col:
    try:
        if live.get('connection_status') == 'CONNECTED':
            nlv = live.get('net_liquidation', 0.0)
            st.metric("Net Liquidation", f"${nlv:,.2f}")
        else:
            st.metric("Net Liquidation", "IB Offline")
    except Exception:
        st.metric("Net Liquidation", "IB Offline")

with bench_col:
    try:
        benchmarks = fetch_todays_benchmark_data()
        ticker = _cfg.get('commodity', {}).get('ticker', 'KC') if _cfg else 'KC'
        benchmark_symbol = f"{ticker}=F"
        pct = benchmarks.get(benchmark_symbol, 0)
        st.metric(f"{ticker} Benchmark", f"{pct:+.2f}%")
    except Exception:
        st.metric("Benchmark", "No data")

# === RECENT DECISIONS FEED ===
st.markdown("---")
st.markdown("### Recent Decisions")

try:
    council_df = load_council_history()
    if not council_df.empty:
        from datetime import datetime, timezone
        recent = council_df.head(5).copy()

        display_cols = []
        if 'timestamp' in recent.columns:
            now = datetime.now(timezone.utc)
            def _relative_time(ts):
                try:
                    if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
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
            recent['Time'] = recent['timestamp'].apply(_relative_time)
            display_cols.append('Time')

        col_map = {
            'contract': 'Contract',
            'master_decision': 'Decision',
            'master_confidence': 'Confidence',
            'strategy_type': 'Strategy',
            'thesis_strength': 'Thesis',
            'trigger_type': 'Trigger',
            'outcome': 'Outcome',
            'pnl_realized': 'P&L',
        }
        for src, dst in col_map.items():
            if src in recent.columns:
                recent[dst] = recent[src]
                display_cols.append(dst)

        # Format confidence as percentage
        if 'Confidence' in recent.columns:
            recent['Confidence'] = recent['Confidence'].apply(
                lambda x: f"{float(x)*100:.0f}%" if x is not None else "?"
            )

        # Format outcome with visual indicators
        if 'Outcome' in recent.columns:
            recent['Outcome'] = recent['Outcome'].apply(
                lambda x: '\u2705 WIN' if x == 'WIN' else '\u274c LOSS' if x == 'LOSS' else '\u2014'
            )

        # Format P&L as currency
        if 'P&L' in recent.columns:
            recent['P&L'] = recent['P&L'].apply(
                lambda x: f"${float(x):+,.2f}" if pd.notna(x) and x != 0 else "\u2014"
            )

        if display_cols:
            st.dataframe(
                recent[display_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No decision columns available.")
    else:
        st.info("No council decisions yet.")
except Exception as e:
    st.info(f"No decision data available.")

# === NAVIGATION ===
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
        st.page_link("pages/4_Financials.py", label="Financials", icon="\U0001f4c8",
                      help="Are we making money? P&L, equity curve, risk metrics", width="stretch")
        st.page_link("pages/6_Signal_Overlay.py", label="Signal Overlay", icon="\U0001f3af",
                      help="How do signals align with price? Visual forensics", width="stretch")
else:
    st.markdown("""
    Use the sidebar to navigate between pages:

    | Page | Purpose |
    |------|---------|
    | **Cockpit** | Is the system running? Check positions, health, emergencies |
    | **Scorecard** | How are we performing? Win rates, decision quality, learning curves |
    | **Council** | Why did we decide that? Agent debate, voting, forensics |
    | **Financials** | Are we making money? P&L, equity curve, risk metrics |
    | **Utilities** | Debug and control: logs, manual trading, reconciliation |
    | **Signal Overlay** | How do signals align with price? Visual forensics |
    | **Brier Analysis** | Which agents need tuning? Accuracy, calibration, learning |
    """)

st.caption(f"Select a page from the sidebar or above to begin. Viewing: {_commodity_emoji} {_dashboard_ticker_upper}")
