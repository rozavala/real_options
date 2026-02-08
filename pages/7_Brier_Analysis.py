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

st.set_page_config(layout="wide", page_title="Brier Analysis | Real Options")
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

# Primary metrics from Enhanced Brier (source of truth)
if enhanced_data:
    preds = enhanced_data.get('predictions', [])
    total_enhanced = len(preds)
    resolved_enhanced = sum(1 for p in preds if p.get('actual_outcome'))
    pending_enhanced = total_enhanced - resolved_enhanced

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", total_enhanced)
    col2.metric("Resolved", resolved_enhanced)
    col3.metric("Pending", pending_enhanced)
    col4.metric(
        "Resolution Rate",
        f"{resolved_enhanced / total_enhanced * 100:.0f}%" if total_enhanced > 0 else "N/A"
    )

    # Sync status with legacy CSV (compare resolved counts, not totals)
    if not struct_df.empty:
        csv_resolved = len(struct_df) - (struct_df['actual'] == 'PENDING').sum() - (
            (struct_df['actual'] == 'ORPHANED').sum() if 'actual' in struct_df.columns else 0
        )
        resolved_diff = abs(resolved_enhanced - csv_resolved)
        if resolved_diff <= 5:
            st.success(
                f"Stores in sync: {resolved_enhanced} resolved in JSON, "
                f"{csv_resolved} resolved in CSV"
            )
        else:
            st.warning(
                f"Resolution drift: Enhanced JSON has {resolved_enhanced} resolved, "
                f"CSV has {csv_resolved} resolved ({resolved_diff} difference). "
                f"Run backfill to converge."
            )

    # Legacy CSV details in expander
    if not struct_df.empty:
        with st.expander("Legacy CSV Pipeline Details"):
            csv_total = len(struct_df)
            csv_pending = (struct_df['actual'] == 'PENDING').sum()
            csv_orphaned = (struct_df['actual'] == 'ORPHANED').sum() if 'actual' in struct_df.columns else 0
            csv_resolved = csv_total - csv_pending - csv_orphaned

            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("CSV Total", csv_total)
            lc2.metric("CSV Resolved", csv_resolved)
            lc3.metric("CSV Pending", csv_pending)
            lc4.metric("CSV Orphaned", csv_orphaned)
            st.caption("The CSV pipeline is deprecated and will be removed after 2026-03-01.")

elif not struct_df.empty:
    # Fallback: no enhanced data, show CSV metrics directly
    col1, col2, col3, col4, col5 = st.columns(5)
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
    st.info("Enhanced Brier tracker has no data yet. Showing legacy CSV metrics.")
else:
    st.info("No prediction data available yet.")

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

    NON_NORMAL_REGIMES = ['HIGH_VOL', 'RANGE_BOUND', 'WEATHER_EVENT', 'MACRO_SHIFT']
    weight_rows = []
    has_regime_specific = False

    for agent in agent_names:
        # Always get NORMAL baseline
        normal_mult = get_agent_reliability(agent, 'NORMAL')
        status = '🟢 Trusted' if normal_mult > 1.2 else '🔴 Distrusted' if normal_mult < 0.8 else '⚪ Baseline'

        weight_rows.append({
            'Agent': AGENT_DISPLAY_NAMES.get(agent, agent),
            'Regime': 'ALL (baseline)',
            'Multiplier': round(normal_mult, 3),
            'Status': status,
        })

        # Only show non-NORMAL regimes if they differ from NORMAL (i.e., have regime-specific data)
        for regime in NON_NORMAL_REGIMES:
            regime_mult = get_agent_reliability(agent, regime)
            if regime_mult != 1.0 and abs(regime_mult - normal_mult) > 0.001:
                has_regime_specific = True
                regime_status = '🟢 Trusted' if regime_mult > 1.2 else '🔴 Distrusted' if regime_mult < 0.8 else '⚪ Baseline'
                weight_rows.append({
                    'Agent': AGENT_DISPLAY_NAMES.get(agent, agent),
                    'Regime': regime,
                    'Multiplier': round(regime_mult, 3),
                    'Status': regime_status,
                })

    if not has_regime_specific:
        st.caption(
            "ℹ️ Showing baseline multipliers only. Regime-specific rows will appear "
            "as Brier data accumulates under different market regimes (HIGH_VOL, RANGE_BOUND, etc.)."
        )

    if weight_rows:
        st.dataframe(pd.DataFrame(weight_rows), hide_index=True, width="stretch")
    else:
        st.info("All agents at baseline (1.0). Weights will differentiate after sufficient resolved predictions.")

except ImportError as e:
    st.warning(f"Brier bridge import failed: {e}. Check that all v6.4 modules are deployed.")
except Exception as e:
    st.error(f"Error loading reliability data: {e}")


# === WEIGHT EVOLUTION OVER TIME ===
st.markdown("---")
st.subheader("📈 Weight Evolution")

weight_csv = os.path.join('data', 'weight_evolution.csv')
if os.path.exists(weight_csv):
    try:
        weight_df = pd.read_csv(weight_csv, parse_dates=['timestamp'])

        if not weight_df.empty:
            # Let user select agents to compare
            available_agents = sorted(weight_df['agent'].unique())
            selected_agents = st.multiselect(
                "Select agents to compare:",
                available_agents,
                default=available_agents[:4]  # Show first 4 by default
            )

            if selected_agents:
                filtered = weight_df[weight_df['agent'].isin(selected_agents)]

                # Pivot for plotting
                pivot = filtered.pivot_table(
                    index='timestamp',
                    columns='agent',
                    values='reliability_mult',
                    aggfunc='last'
                )

                st.line_chart(pivot)

                # Summary table: current state
                latest = weight_df.sort_values('timestamp').groupby('agent').last()
                st.dataframe(
                    latest[['regime', 'domain_weight', 'reliability_mult', 'final_weight']],
                    use_container_width=True
                )
        else:
            st.info("Weight evolution data will appear after the next trading cycle.")
    except Exception as e:
        st.warning(f"Could not load weight evolution data: {e}")
else:
    st.info("Weight evolution tracking not yet active. Data will appear after the next trading cycle.")


# === TMS TEMPORAL DECAY VISUALIZATION ===
st.markdown("---")
st.subheader("🕐 TMS Temporal Decay Curves")
st.caption("Shows how different document types lose relevance over time")

import math
import numpy as np

# Load decay rates from commodity profile
try:
    from config.commodity_profiles import get_active_profile
    # Config is not available in this scope, try loading default/mock
    # In a real app, config is loaded from config_loader
    from config_loader import load_config
    config = load_config()
    profile = get_active_profile(config)
    decay_rates = getattr(profile, 'tms_decay_rates', {
        'weather': 0.15, 'logistics': 0.10, 'news': 0.08,
        'macro': 0.02, 'technical': 0.05, 'default': 0.05
    })
except Exception:
    decay_rates = {
        'weather': 0.15, 'logistics': 0.10, 'news': 0.08,
        'macro': 0.02, 'technical': 0.05, 'default': 0.05
    }

# Generate decay curves
days = np.arange(0, 60, 0.5)
chart_data = {}

display_types = ['weather', 'logistics', 'news', 'macro', 'technical', 'trade_journal']
for doc_type in display_types:
    lam = decay_rates.get(doc_type, 0.05)
    chart_data[f"{doc_type} (λ={lam})"] = [math.exp(-lam * d) for d in days]

import pandas as pd
decay_df = pd.DataFrame(chart_data, index=days)
decay_df.index.name = 'Age (days)'

st.line_chart(decay_df)

st.caption(
    "**Reading the chart:** A document at 50% relevance has lost half its informational value. "
    "Weather data (λ=0.15) hits 50% at ~4.6 days. Macro data (λ=0.02) takes ~34.7 days. "
    "Decay rates are configurable per commodity profile."
)


st.markdown("---")
st.caption("Brier Analysis | Coffee Bot Real Options v6.4")
