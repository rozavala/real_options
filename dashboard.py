"""
Coffee Bot Mission Control - Main Entry Point

This file serves as the entry point and redirects to the multi-page app.
Streamlit's native multi-page support handles routing via the pages/ directory.
"""

import streamlit as st
from config_loader import load_config
from trading_bot.logging_config import setup_logging

# Set up dashboard-specific logging
setup_logging(log_file="logs/dashboard.log")

# Dynamic configuration for commodity-aware branding
_cfg = load_config()
_commodity_name = _cfg.get('commodity', {}).get('name', 'Coffee') if _cfg else 'Coffee'
_commodity_emoji = {'Coffee': '☕', 'Cocoa': '🍫', 'Sugar': '🍬'}.get(_commodity_name.split()[0], '📊')

st.set_page_config(
    layout="wide",
    page_title=f"{_commodity_name} Mission Control",
    page_icon=_commodity_emoji,
    initial_sidebar_state="expanded"
)

# The presence of files in pages/ directory enables multi-page mode automatically.
# This file becomes the "home" page or can redirect.

st.title(f"{_commodity_emoji} {_commodity_name} Mission Control")
st.markdown("---")

st.markdown("""
### Navigation

Use the sidebar to navigate between pages:

| Page | Purpose |
|------|---------|
| **🦅 Cockpit** | Live operations, system health, emergency controls |
| **⚖️ Scorecard** | Decision quality analysis, win rates, confusion matrix |
| **🧠 Council** | Agent explainability, consensus visualization |
| **📈 Financials** | ROI, equity curve, strategy performance |
| **🔧 Utilities** | Log collection, equity sync, system maintenance |

---

### Quick Status
""")

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
    st.metric("Orchestrator", f"{color} {status}")

with col2:
    if config:
        live_data = fetch_live_dashboard_data(config)
        st.metric("Net Liquidation", f"${live_data['NetLiquidation']:,.2f}")
    else:
        st.metric("Net Liquidation", "Offline")

with col3:
    benchmarks = fetch_todays_benchmark_data()
    # E2 FIX: Commodity-agnostic benchmark
    profile = _cfg.get('commodity', {})
    ticker = profile.get('ticker', 'KC')
    benchmark_symbol = f"{ticker}=F"
    st.metric(f"{ticker} Benchmark", f"{benchmarks.get(benchmark_symbol, 0):+.2f}%")

st.markdown("---")
st.caption("Select a page from the sidebar to begin.")
