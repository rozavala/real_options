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
    status_colors = {
        "NORMAL": "green", "WARNING": "orange", "HALT": "red", "PANIC": "red",
    }
    color = status_colors.get(status, "grey")
    st.markdown(
        f"### Status: :{color}[{status}]"
    )

    cols = st.columns(4)
    with cols[0]:
        equity = prg_state.get("current_equity", 0)
        st.metric("Net Liquidation", f"${equity:,.0f}" if equity else "N/A")
    with cols[1]:
        peak = prg_state.get("peak_equity", 0)
        st.metric("Peak Equity", f"${peak:,.0f}" if peak else "N/A")
    with cols[2]:
        daily_pnl = prg_state.get("daily_pnl", 0)
        st.metric("Daily P&L", f"${daily_pnl:+,.0f}" if daily_pnl else "$0")
    with cols[3]:
        starting = prg_state.get("starting_equity", 0)
        if starting > 0 and equity > 0:
            dd_pct = ((starting - equity) / starting) * 100
            st.metric("Drawdown", f"{dd_pct:.2f}%")
        else:
            st.metric("Drawdown", "N/A")

    st.caption(f"Last updated: {prg_state.get('last_updated', 'unknown')}")
else:
    st.info("No portfolio risk state found. PortfolioRiskGuard has not run yet.")


# === SECTION 2: Per-Commodity Position Breakdown ===
st.markdown("---")
st.subheader("Position Breakdown by Commodity")

positions = prg_state.get("positions", {})
if positions:
    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame([
        {"Commodity": k, "Positions": v}
        for k, v in sorted(positions.items())
    ])
    fig = px.bar(
        df, x="Commodity", y="Positions",
        color="Commodity",
        text_auto=True,
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

    total = sum(positions.values())
    st.caption(f"Total open positions: {total}")
else:
    st.info("No position data available.")


# === SECTION 3: Engine Health ===
st.markdown("---")
st.subheader("Engine Health")

# Discover commodity data dirs
engine_dirs = []
if os.path.isdir(DATA_ROOT):
    for entry in sorted(os.listdir(DATA_ROOT)):
        state_path = os.path.join(DATA_ROOT, entry, "state.json")
        if os.path.exists(state_path):
            engine_dirs.append((entry, state_path))

if engine_dirs:
    cols = st.columns(min(len(engine_dirs), 4))
    for i, (ticker, state_path) in enumerate(engine_dirs):
        state = _load_json(state_path)
        with cols[i % len(cols)]:
            st.markdown(f"**{ticker}**")
            if state:
                last_cycle = state.get("last_cycle_time", "unknown")
                cycle_count = state.get("cycle_count", "?")
                active_theses = state.get("active_theses", "?")
                st.write(f"Cycles: {cycle_count}")
                st.write(f"Active theses: {active_theses}")
                st.write(f"Last cycle: {last_cycle}")
            else:
                st.write("No state data")
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
        st.metric("VaR (95%)", f"${var_95:,.0f}" if var_95 else "N/A")
    with cols[1]:
        var_limit = var_state.get("var_limit", 0)
        st.metric("VaR Limit", f"${var_limit:,.0f}" if var_limit else "N/A")
    with cols[2]:
        if var_95 and var_limit and var_limit > 0:
            utilization = (var_95 / var_limit) * 100
            st.metric("Utilization", f"{utilization:.1f}%")
        else:
            st.metric("Utilization", "N/A")

    enforcement = var_state.get("enforcement_mode", "unknown")
    st.caption(f"Enforcement mode: **{enforcement}** | Last computed: {var_state.get('last_computed', 'unknown')}")
else:
    st.info("No VaR state found. VaR calculator has not run yet.")
