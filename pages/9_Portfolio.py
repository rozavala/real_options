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
st.title("💼 Portfolio Overview")
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
st.subheader("🛡️ Portfolio Risk Status")

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
            "💰 Net Liquidation", f"${equity:,.0f}" if equity else "N/A",
            help="Total unified account value including cash and market value of all positions."
        )
    with cols[1]:
        peak = prg_state.get("peak_equity", 0)
        st.metric(
            "🏔️ Peak Equity", f"${peak:,.0f}" if peak else "N/A",
            help="The highest net liquidation value observed for the account since inception."
        )
    with cols[2]:
        daily_pnl = prg_state.get("daily_pnl", 0)
        st.metric(
            "💵 Daily P&L", f"${daily_pnl:+,.0f}" if daily_pnl else "$0",
            help="Account-wide P&L for the current trading day."
        )
    with cols[3]:
        if peak > 0 and equity > 0:
            dd_pct = max(0.0, ((peak - equity) / peak) * 100)
            st.metric(
                "📉 Drawdown", f"{dd_pct:.2f}%",
                help="Peak-to-trough drawdown: percentage decline from all-time peak equity."
            )
        else:
            st.metric("📉 Drawdown", "N/A", help="Cannot calculate drawdown without peak equity.")

    last_upd = prg_state.get('last_updated', 'unknown')
    st.caption(f"Last updated: {last_upd} ({_relative_time(last_upd)})")
else:
    st.info("No portfolio risk state found. PortfolioRiskGuard has not run yet.")


# === SECTION 2: Per-Commodity Position Breakdown ===
st.markdown("---")
st.subheader("📊 Position Breakdown by Commodity")

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
st.subheader("⚙️ Engine Health")

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
                st.metric(ticker, "No data", delta="OFFLINE", delta_color="inverse", help=f"Engine state file not found for {ticker}")
else:
    st.info("No engine state files found.")


# === SECTION 4: VaR Utilization ===
st.markdown("---")
st.subheader("⚖️ VaR Utilization")

var_state = _load_json(os.path.join(DATA_ROOT, "var_state.json"))
if var_state:
    var_95 = var_state.get("var_95", 0)
    var_95_pct = var_state.get("var_95_pct", 0)
    var_99 = var_state.get("var_99", 0)
    var_99_pct = var_state.get("var_99_pct", 0)
    pos_count = var_state.get("position_count", 0)
    commodities = var_state.get("commodities", [])

    # VaR limit comes from config, not var_state.json
    try:
        from config_loader import load_config as _load_cfg
        _cfg = _load_cfg()
        var_limit_pct = _cfg.get('compliance', {}).get('var_limit_pct', 0.03)
    except Exception:
        var_limit_pct = 0.03

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "⚖️ VaR (95%)", f"{var_95_pct:.1%}" if var_95_pct else "N/A",
            delta=f"${var_95:,.0f}" if var_95 else None,
            delta_color="off",
            help="Value at Risk (95% confidence): Estimated maximum loss over one day."
        )
    with cols[1]:
        st.metric(
            "⚖️ VaR (99%)", f"{var_99_pct:.1%}" if var_99_pct else "N/A",
            delta=f"${var_99:,.0f}" if var_99 else None,
            delta_color="off",
            help="Value at Risk (99% confidence): Estimated maximum loss in extreme conditions."
        )
    with cols[2]:
        if var_95_pct and var_limit_pct > 0:
            utilization = (var_95_pct / var_limit_pct) * 100
            st.metric(
                "🔌 Utilization", f"{utilization:.0f}%",
                help=f"VaR(95%) as percentage of the {var_limit_pct:.0%} limit."
            )
        else:
            st.metric("🔌 Utilization", "N/A", help="VaR utilization not available.")
    with cols[3]:
        commodity_str = ", ".join(commodities) if commodities else "None"
        st.metric(
            "🦵 Legs", f"{pos_count}",
            help=f"Option contract legs across: {commodity_str}"
        )

    # Staleness check
    import time as _time
    computed_epoch = var_state.get("computed_epoch", 0)
    if computed_epoch:
        age_hours = (_time.time() - computed_epoch) / 3600
        age_label = f"{age_hours * 60:.0f}m ago" if age_hours < 1 else f"{age_hours:.1f}h ago"
    else:
        age_label = "unknown"
    status = var_state.get("last_attempt_status", "OK")
    st.caption(f"Last computed: {age_label} | Status: **{status}**")
    if status == "FAILED":
        st.warning(f"Last VaR computation failed: {var_state.get('last_attempt_error', 'Unknown')}")
else:
    st.info("No VaR state found. VaR calculator has not run yet.")
