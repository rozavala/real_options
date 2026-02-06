"""
Page 7: Brier Analysis

Purpose: Deep analytics on agent prediction quality, calibration curves,
regime-specific performance, and feedback loop health.

This page makes agent learning VISIBLE — without it, we're flying blind.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timezone, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(layout="wide", page_title="Brier Analysis | Mission Control")
st.title("🎯 Brier Analysis")
st.caption("Agent prediction quality, calibration curves, and learning feedback")


# === DATA LOADING ===

@st.cache_data(ttl=120)
def load_enhanced_brier():
    """Load enhanced Brier data from JSON."""
    path = "data/enhanced_brier.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=120)
def load_structured_predictions():
    """Load structured prediction CSV."""
    path = "data/agent_accuracy_structured.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def load_legacy_accuracy():
    """Load legacy accuracy CSV."""
    path = "data/agent_accuracy.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# === SECTION 1: System Health Overview ===
st.subheader("📊 Feedback Loop Overview")

enhanced_data = load_enhanced_brier()
struct_df = load_structured_predictions()
legacy_df = load_legacy_accuracy()

# Health metrics
col1, col2, col3, col4, col5 = st.columns(5)

if not struct_df.empty:
    total = len(struct_df)
    pending = (struct_df['actual'] == 'PENDING').sum()
    orphaned = (struct_df['actual'] == 'ORPHANED').sum() if 'actual' in struct_df.columns else 0
    resolved = total - pending - orphaned
    resolvable = total - orphaned

    col1.metric("Total Predictions", total)
    col2.metric("Resolved", resolved)
    col3.metric("Pending", pending)
    col4.metric("Orphaned", orphaned)
    col5.metric(
        "Resolution Rate",
        f"{resolved / resolvable * 100:.0f}%" if resolvable > 0 else "N/A"
    )
else:
    st.info("No structured prediction data available yet.")

# Enhanced system status
if enhanced_data:
    preds = enhanced_data.get('predictions', [])
    resolved_enhanced = sum(1 for p in preds if p.get('actual_outcome'))
    unresolved_enhanced = sum(1 for p in preds if not p.get('actual_outcome'))

    st.success(
        f"✅ Enhanced Brier Active: {len(preds)} predictions tracked "
        f"({resolved_enhanced} resolved, {unresolved_enhanced} pending)"
    )
else:
    st.warning("⚠️ Enhanced Brier tracker has no data yet. Predictions will accumulate after v6.4 deployment.")

st.markdown("---")


# === SECTION 2: Per-Agent Brier Scores ===
st.subheader("🧠 Agent Performance (Brier Scores)")

if enhanced_data and enhanced_data.get('agent_scores'):
    agent_scores = enhanced_data['agent_scores']

    rows = []
    for agent, regimes in agent_scores.items():
        for regime, scores in regimes.items():
            if scores:
                recent = scores[-30:]
                rows.append({
                    'Agent': agent,
                    'Regime': regime,
                    'Avg Brier': round(np.mean(recent), 4),
                    'Best Brier': round(min(recent), 4),
                    'Worst Brier': round(max(recent), 4),
                    'Samples': len(scores),
                    'Recent (30)': len(recent),
                })

    if rows:
        scores_df = pd.DataFrame(rows)
        scores_df = scores_df.sort_values('Avg Brier')

        # Color-code: lower Brier = better (green), higher = worse (red)
        st.dataframe(
            scores_df,
            width="stretch",
            hide_index=True,
        )

        # Quick interpretation
        st.caption(
            "**Brier Score Guide:** 0.0 = perfect, 0.25 = random baseline, 0.5 = always wrong. "
            "Lower is better. Agents need 5+ resolved predictions for reliable scoring."
        )
    else:
        st.info("No Brier scores computed yet. Scores appear after predictions are resolved via reconciliation.")
else:
    st.info("Enhanced Brier scoring will populate after v6.4 deployment and first reconciliation cycle.")


st.markdown("---")


# === SECTION 3: Calibration Curves ===
st.subheader("📈 Calibration Curves")
st.caption("Perfect calibration = diagonal line. Above = overconfident, Below = underconfident.")

if enhanced_data and enhanced_data.get('calibration_buckets'):
    cal_data = enhanced_data['calibration_buckets']

    # Select agent
    agents = list(cal_data.keys())
    if agents:
        selected_agent = st.selectbox("Select Agent", agents)

        buckets = cal_data.get(selected_agent, [])
        if buckets:
            cal_rows = []
            for b in buckets:
                predictions = b.get('predictions', 0)
                correct = b.get('correct', 0)
                lower = b.get('lower', 0)
                upper = b.get('upper', 0)
                midpoint = (lower + upper) / 2

                if predictions > 0:
                    cal_rows.append({
                        'Predicted Probability': midpoint,
                        'Actual Accuracy': correct / predictions,
                        'Sample Size': predictions,
                    })

            if cal_rows:
                cal_df = pd.DataFrame(cal_rows)

                # Simple line chart (Streamlit native)
                st.line_chart(
                    cal_df.set_index('Predicted Probability')['Actual Accuracy'],
                    width="stretch",
                )

                # Show raw data
                with st.expander("Calibration Data"):
                    st.dataframe(cal_df, hide_index=True)
            else:
                st.info(f"No calibration data for {selected_agent} yet.")
    else:
        st.info("No agents with calibration data yet.")
else:
    st.info("Calibration curves will populate after predictions are resolved.")


st.markdown("---")


# === SECTION 4: Prediction Timeline ===
st.subheader("📅 Prediction Timeline")

if not struct_df.empty and 'timestamp' in struct_df.columns:
    # Daily prediction counts
    struct_df['date'] = struct_df['timestamp'].dt.date

    daily = struct_df.groupby('date').agg(
        total=('actual', 'count'),
        pending=('actual', lambda x: (x == 'PENDING').sum()),
        resolved=('actual', lambda x: ((x != 'PENDING') & (x != 'ORPHANED')).sum()),
    ).reset_index()

    st.bar_chart(
        daily.set_index('date')[['resolved', 'pending']],
        width="stretch",
    )

    # Show per-agent breakdown
    with st.expander("Per-Agent Breakdown"):
        if 'agent' in struct_df.columns:
            agent_summary = struct_df.groupby('agent').agg(
                total=('actual', 'count'),
                pending=('actual', lambda x: (x == 'PENDING').sum()),
                correct=('actual', lambda x: (
                    (x == struct_df.loc[x.index, 'direction']).sum()
                    if 'direction' in struct_df.columns else 0
                )),
            ).sort_values('total', ascending=False)
            st.dataframe(agent_summary, width="stretch")
else:
    st.info("No prediction data available yet.")


st.markdown("---")


# === SECTION 5: Reliability Multipliers (Current) ===
st.subheader("⚖️ Current Reliability Multipliers")
st.caption("These weights influence council voting. 1.0 = baseline, >1.0 = trusted, <1.0 = distrusted.")

try:
    from trading_bot.brier_bridge import get_agent_reliability
    from trading_bot.agent_names import AGENT_DISPLAY_NAMES

    agent_names = list(AGENT_DISPLAY_NAMES.keys()) if hasattr(AGENT_DISPLAY_NAMES, 'keys') else [
        'agronomist', 'inventory', 'macro', 'sentiment',
        'technical', 'volatility', 'geopolitical'
    ]

    weight_rows = []
    for agent in agent_names:
        for regime in ['NORMAL', 'HIGH_VOL', 'WEATHER_EVENT', 'MACRO_SHIFT']:
            mult = get_agent_reliability(agent, regime)
            if mult != 1.0 or regime == 'NORMAL':
                weight_rows.append({
                    'Agent': AGENT_DISPLAY_NAMES.get(agent, agent),
                    'Regime': regime,
                    'Multiplier': round(mult, 3),
                    'Status': '🟢 Trusted' if mult > 1.2 else '🔴 Distrusted' if mult < 0.8 else '⚪ Baseline',
                })

    if weight_rows:
        st.dataframe(pd.DataFrame(weight_rows), hide_index=True, width="stretch")
    else:
        st.info("All agents at baseline (1.0). Weights will differentiate after sufficient resolved predictions.")

except ImportError as e:
    st.warning(f"Brier bridge import failed: {e}. Check that all v6.4 modules are deployed.")
except Exception as e:
    st.error(f"Error loading reliability data: {e}")


st.markdown("---")
st.caption("Brier Analysis | Coffee Bot Mission Control v6.4")
