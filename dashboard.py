"""
Coffee Bot Real Options - Main Entry Point

This file serves as the entry point and redirects to the multi-page app.
Streamlit's native multi-page support handles routing via the pages/ directory.
"""

import streamlit as st
from config_loader import load_config
from trading_bot.logging_config import setup_logging

# Set up dashboard-specific logging (per-commodity to avoid log collision)
import os
_dashboard_ticker = os.environ.get("COMMODITY_TICKER", "KC").lower()
setup_logging(log_file=f"logs/dashboard_{_dashboard_ticker}.log")

# Dynamic configuration for commodity-aware branding
_cfg = load_config()
_commodity_name = _cfg.get('commodity', {}).get('name', 'Coffee') if _cfg else 'Coffee'
_commodity_emoji = {'Coffee': '☕', 'Cocoa': '🍫', 'Sugar': '🍬'}.get(_commodity_name.split()[0], '📊')

st.set_page_config(
    layout="wide",
    page_title=f"{_commodity_name} Real Options",
    page_icon=_commodity_emoji,
    initial_sidebar_state="expanded"
)

# The presence of files in pages/ directory enables multi-page mode automatically.
# This file becomes the "home" page or can redirect.

st.title(f"{_commodity_emoji} {_commodity_name} Real Options")
st.markdown("---")

st.markdown("### Navigation")

# Interactive Navigation with Progressive Enhancement
if hasattr(st, "page_link"):
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/1_Cockpit.py", label="Cockpit", icon="🦅", help="Live operations, system health, emergency controls", width="stretch")
        st.page_link("pages/3_The_Council.py", label="The Council", icon="🧠", help="Agent explainability, consensus visualization", width="stretch")
        st.page_link("pages/5_Utilities.py", label="Utilities", icon="🔧", help="Log collection, equity sync, system maintenance", width="stretch")
    with col2:
        st.page_link("pages/2_The_Scorecard.py", label="The Scorecard", icon="⚖️", help="Decision quality analysis, win rates, confusion matrix", width="stretch")
        st.page_link("pages/4_Financials.py", label="Financials", icon="📈", help="ROI, equity curve, strategy performance", width="stretch")
        st.page_link("pages/6_Signal_Overlay.py", label="Signal Overlay", icon="🎯", help="Decision forensics against price action", width="stretch")
else:
    st.markdown("""
    Use the sidebar to navigate between pages:

    | Page | Purpose |
    |------|---------|
    | **🦅 Cockpit** | Live operations, system health, emergency controls |
    | **⚖️ Scorecard** | Decision quality analysis, win rates, confusion matrix |
    | **🧠 Council** | Agent explainability, consensus visualization |
    | **📈 Financials** | ROI, equity curve, strategy performance |
    | **🔧 Utilities** | Log collection, equity sync, system maintenance |
    """)

st.markdown("---")
st.markdown("### Quick Status")

# Import utils for quick status display
from dashboard_utils import (
    get_system_heartbeat,
    fetch_live_dashboard_data,
    get_config,
    fetch_todays_benchmark_data
)

config = get_config()
heartbeat = get_system_heartbeat()

col1, col2, col3 = st.columns(3)

with col1:
    status = heartbeat['orchestrator_status']
    color = "🟢" if status == "ONLINE" else "🔴" if status == "OFFLINE" else "🟡"
    st.metric(
        "Orchestrator",
        f"{color} {status}",
        help="System health based on recent activity logs. Green indicates active polling within the last 10 minutes."
    )

with col2:
    if config:
        live_data = fetch_live_dashboard_data(config)
        st.metric(
            "Net Liquidation",
            f"${live_data['NetLiquidation']:,.2f}",
            help="Total account value (Cash + Market Value of Positions) from live Interactive Brokers data."
        )
    else:
        st.metric("Net Liquidation", "Offline")

with col3:
    benchmarks = fetch_todays_benchmark_data()
    # E2 FIX: Commodity-agnostic benchmark
    profile = _cfg.get('commodity', {})
    ticker = profile.get('ticker', 'KC')
    benchmark_symbol = f"{ticker}=F"
    st.metric(
        f"{ticker} Benchmark",
        f"{benchmarks.get(benchmark_symbol, 0):+.2f}%",
        help=f"Today's percentage change for {ticker} futures, sourced from Yahoo Finance."
    )

st.markdown("---")
st.caption("Select a page from the sidebar to begin.")
