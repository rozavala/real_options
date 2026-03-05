"""
Page 9: Portfolio Overview (Account-Wide)

Cross-commodity risk status, position breakdown, engine health, and VaR utilization.
This page is NOT per-commodity — it shows the unified account view from
PortfolioRiskGuard and VaR calculator.
"""

import streamlit as st
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import _relative_time, discover_active_commodities, get_active_theses

st.set_page_config(layout="wide", page_title="Portfolio | Real Options")
st.title("Portfolio Overview")
st.caption("Account-wide risk status, position breakdown, and engine health")

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# === SECTION 1: Portfolio Risk Status ===
st.markdown("---")
st.subheader("Portfolio Risk Status")

prg_state = _load_json(os.path.join(DATA_ROOT, "portfolio_risk_state.json"))

if prg_state:
    status = prg_state.get("status", "UNKNOWN")
    if status == "NORMAL":
        st.success(f"**Portfolio Status: {status}**")
    elif status == "WARNING":
        st.warning(f"**Portfolio Status: {status}**")
    elif status in ["HALT", "PANIC"]:
        st.error(f"**Portfolio Status: {status}**")
    else:
        st.info(f"**Portfolio Status: {status}**")

    cols = st.columns(4)
    with cols[0]:
        equity = prg_state.get("current_equity", 0)
        st.metric(
            "Net Liquidation", f"${equity:,.0f}" if equity else "N/A",
            help="Total unified account value including cash and market value of all positions."
        )
    with cols[1]:
        peak = prg_state.get("peak_equity", 0)
        st.metric(
            "Peak Equity", f"${peak:,.0f}" if peak else "N/A",
            help="The highest net liquidation value observed for the account since inception."
        )
    with cols[2]:
        daily_pnl = prg_state.get("daily_pnl", 0)
        st.metric(
            "Daily P&L", f"${daily_pnl:+,.0f}" if daily_pnl else "$0",
            help="Account-wide P&L for the current trading day."
        )
    with cols[3]:
        if peak > 0 and equity > 0:
            dd_pct = max(0.0, ((peak - equity) / peak) * 100)
            st.metric(
                "Drawdown", f"{dd_pct:.2f}%",
                help="Peak-to-trough drawdown: percentage decline from all-time peak equity."
            )
        else:
            st.metric("Drawdown", "N/A", help="Cannot calculate drawdown without peak equity.")

    last_upd = prg_state.get('last_updated', 'unknown')
    st.caption(f"Last updated: {last_upd} ({_relative_time(last_upd)})")
else:
    st.info("No portfolio risk state found. PortfolioRiskGuard has not run yet.")


# === SECTION 2: Per-Commodity Position Breakdown ===
st.markdown("---")
st.subheader("Position Breakdown by Commodity")

# Live thesis count from TMS (not stale portfolio_risk_state.json)
active_tickers = discover_active_commodities()
_live_positions = {}
for _t in active_tickers:
    _theses = get_active_theses(_t)
    if _theses:
        _live_positions[_t] = len(_theses)

if _live_positions:
    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame([
        {"Commodity": k, "Positions": v}
        for k, v in sorted(_live_positions.items())
    ])
    fig = px.bar(
        df, x="Commodity", y="Positions",
        color="Commodity",
        text_auto=True,
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, width='stretch')

    total = sum(_live_positions.values())
    st.caption(f"Total open positions: {total} (live from TMS)")
else:
    st.info("No active positions across any commodity.")


# === SECTION 3: Engine Health ===
st.markdown("---")
st.subheader("Engine Health")

if active_tickers:
    cols = st.columns(min(len(active_tickers), 4))
    for i, ticker in enumerate(active_tickers):
        state_path = os.path.join(DATA_ROOT, ticker, "state.json")
        state = _load_json(state_path)
        with cols[i % len(cols)]:
            if state:
                thesis_count = _live_positions.get(ticker, 0)

                # Last IB activity from sensors (this field actually exists)
                sensors = state.get("sensors", {})
                last_ib = sensors.get("last_ib_success", {})
                last_activity = last_ib.get("data", "unknown") if isinstance(last_ib, dict) else "unknown"

                st.metric(
                    ticker,
                    f"{thesis_count} Theses",
                    delta=f"IB: {_relative_time(last_activity)}",
                    delta_color="off",
                    help=(
                        f"**Active Theses:** {thesis_count} (live from TMS)\n"
                        f"**Last IB Activity:** {last_activity}"
                    )
                )
            else:
                st.metric(ticker, "No data", delta="OFFLINE", delta_color="inverse")
else:
    st.info("No engine state files found.")


# === SECTION 4: VaR Utilization ===
st.markdown("---")
st.subheader("VaR Utilization")

var_state = _load_json(os.path.join(DATA_ROOT, "var_state.json"))
if var_state:
    cols = st.columns(3)
    with cols[0]:
        var_95 = var_state.get("var_95", 0)
        st.metric(
            "VaR (95%)", f"${var_95:,.0f}" if var_95 else "N/A",
            help="Value at Risk (95% confidence): Estimated maximum loss over one day based on current portfolio correlations."
        )
    with cols[1]:
        var_limit = var_state.get("var_limit", 0)
        st.metric(
            "VaR Limit", f"${var_limit:,.0f}" if var_limit else "N/A",
            help="Maximum daily VaR allowed by the compliance system."
        )
    with cols[2]:
        if var_95 and var_limit and var_limit > 0:
            utilization = (var_95 / var_limit) * 100
            st.metric(
                "Utilization", f"{utilization:.1f}%",
                help="Percentage of the VaR limit currently being used."
            )
        else:
            st.metric("Utilization", "N/A", help="VaR utilization not available.")

    enforcement = var_state.get("enforcement_mode", "unknown")
    last_comp = var_state.get('last_computed', 'unknown')
    st.caption(f"Enforcement mode: **{enforcement}** | Last computed: {last_comp} ({_relative_time(last_comp)})")
else:
    st.info("No VaR state found. VaR calculator has not run yet.")
