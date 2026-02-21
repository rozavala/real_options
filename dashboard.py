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
from datetime import datetime, timezone
from dashboard_utils import (
    discover_active_commodities,
    get_system_heartbeat_for_commodity,
    load_council_history_for_commodity,
    grade_decision_quality,
    fetch_all_live_data,
    get_config,
)

config = get_config()

_COMMODITY_META = {
    "KC": {"name": "Coffee", "emoji": "\u2615"},
    "CC": {"name": "Cocoa", "emoji": "\U0001f36b"},
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

active_str = ", ".join(
    f"{_COMMODITY_META.get(t, {}).get('emoji', '')} {t}" for t in active_commodities
)
st.caption(f"Active commodities: {active_str}")
