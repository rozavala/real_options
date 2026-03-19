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
import math
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import _resolve_data_path_for
from _date_filter import date_range_filter

st.set_page_config(layout="wide", page_title="Brier Analysis | Real Options")

from _commodity_selector import selected_commodity
ticker = selected_commodity()

st.title("🎯 Brier Analysis")
st.caption("Agent prediction quality, calibration curves, and learning feedback")


# === DATA LOADING ===

@st.cache_data(ttl=120)
def load_enhanced_brier(ticker: str = "KC"):
    """Load enhanced Brier data from JSON."""
    path = _resolve_data_path_for("enhanced_brier.json", ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=120)
def load_weight_evolution(ticker: str = "KC"):
    """Load weight evolution CSV."""
    path = _resolve_data_path_for('weight_evolution.csv', ticker)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=['timestamp'])
    except Exception:
        return pd.DataFrame()


# Load display names once
try:
    from trading_bot.agent_names import AGENT_DISPLAY_NAMES as _DISPLAY_NAMES
except ImportError:
    _DISPLAY_NAMES = {}


# === SECTION 1: System Health Overview ===
st.subheader("📊 Feedback Loop Overview")

enhanced_data = load_enhanced_brier(ticker)

# Primary metrics
if enhanced_data:
    preds = enhanced_data.get('predictions', [])
    total = len(preds)
    resolved = sum(1 for p in preds if p.get('actual_outcome'))
    pending = total - resolved

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Predictions", total, help="Total number of agent predictions recorded.")
    col2.metric("✅ Resolved", resolved, help="Number of predictions with a known market outcome.")
    col3.metric("⏳ Pending", pending, help="Predictions waiting for market resolution (T+1).")
    col4.metric(
        "🏁 Resolution Rate",
        f"{resolved / total * 100:.0f}%" if total > 0 else "N/A",
        help="Percentage of predictions that have been resolved"
    )
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

        st.dataframe(
            scores_df,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "**Brier Score Guide:** 0.0 = perfect, 0.25 = random baseline, 0.5 = always wrong. "
            "Lower is better. Agents need 5+ resolved predictions for reliable scoring."
        )
    else:
        st.info("No Brier scores computed yet. Scores appear after predictions are resolved via reconciliation.")
else:
    st.info("Enhanced Brier scoring will populate after v6.4 deployment and first reconciliation cycle.")


st.markdown("---")


# === SECTION 3: Calibration Curves (Enhanced with reference line + sample counts) ===
st.subheader("📈 Calibration Curves")
st.caption("Perfect calibration = diagonal line. Above = underconfident, Below = overconfident.")

if enhanced_data and enhanced_data.get('calibration_buckets'):
    cal_data = enhanced_data['calibration_buckets']

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

                fig = go.Figure()

                # Perfect calibration reference line (diagonal)
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray', width=1),
                    name='Perfect Calibration',
                    showlegend=True
                ))

                # Actual calibration data with sample count labels
                fig.add_trace(go.Scatter(
                    x=cal_df['Predicted Probability'],
                    y=cal_df['Actual Accuracy'],
                    mode='lines+markers+text',
                    marker=dict(size=10),
                    text=[f"n={n}" for n in cal_df['Sample Size']],
                    textposition='top center',
                    textfont=dict(size=10),
                    name=selected_agent,
                    showlegend=True
                ))

                fig.update_layout(
                    xaxis_title='Predicted Probability',
                    yaxis_title='Actual Accuracy',
                    height=400,
                    margin=dict(t=30, b=40),
                )

                fig.add_annotation(
                    x=0.75, y=0.55,
                    text="Above = underconfident<br>Below = overconfident",
                    showarrow=False,
                    font=dict(size=10, color='gray'),
                    bgcolor='rgba(255,255,255,0.8)',
                )

                st.plotly_chart(fig, width="stretch")

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

# Use enhanced Brier JSON as primary source (legacy CSV deprecated since Mar 1)
_timeline_built = False
if enhanced_data and enhanced_data.get('predictions'):
    try:
        _preds_df = pd.DataFrame(enhanced_data['predictions'])
        if 'timestamp' in _preds_df.columns and len(_preds_df) > 0:
            _preds_df['timestamp'] = pd.to_datetime(_preds_df['timestamp'], utc=True)
            _preds_df['date'] = _preds_df['timestamp'].dt.date
            _preds_df['is_resolved'] = _preds_df['resolved_at'].notna()

            daily = _preds_df.groupby('date').agg(
                resolved=('is_resolved', 'sum'),
                pending=('is_resolved', lambda x: (~x).sum()),
            ).reset_index()

            st.bar_chart(
                daily.set_index('date')[['resolved', 'pending']],
                width="stretch",
            )

            with st.expander("Per-Agent Breakdown"):
                if 'agent' in _preds_df.columns:
                    agent_summary = _preds_df.groupby('agent').agg(
                        total=('is_resolved', 'count'),
                        pending=('is_resolved', lambda x: (~x).sum()),
                        resolved=('is_resolved', 'sum'),
                    ).sort_values('total', ascending=False)
                    st.dataframe(agent_summary, width="stretch")

            _timeline_built = True
    except Exception:
        pass

if not _timeline_built:
    st.info("No prediction data available yet.")


st.markdown("---")


# === SECTION 5: Reliability Multipliers (Current) ===
st.subheader("⚖️ Current Reliability Multipliers")
st.caption("These weights influence council voting. 1.0 = baseline, >1.0 = trusted, <1.0 = distrusted.")

try:
    from trading_bot.brier_bridge import get_agent_reliability

    agent_names_list = list(_DISPLAY_NAMES.keys()) if _DISPLAY_NAMES else [
        'agronomist', 'inventory', 'macro', 'sentiment',
        'technical', 'volatility', 'geopolitical', 'supply_chain'
    ]

    NON_NORMAL_REGIMES = ['HIGH_VOL', 'RANGE_BOUND', 'WEATHER_EVENT', 'MACRO_SHIFT']
    weight_rows = []
    has_regime_specific = False

    for agent in agent_names_list:
        normal_mult = get_agent_reliability(agent, 'NORMAL')
        status = '🟢 Trusted' if normal_mult > 1.2 else '🔴 Distrusted' if normal_mult < 0.8 else '⚪ Baseline'

        weight_rows.append({
            'Agent': _DISPLAY_NAMES.get(agent, agent),
            'Regime': 'ALL (baseline)',
            'Multiplier': round(normal_mult, 3),
            'Status': status,
        })

        for regime in NON_NORMAL_REGIMES:
            regime_mult = get_agent_reliability(agent, regime)
            if regime_mult != 1.0 and abs(regime_mult - normal_mult) > 0.001:
                has_regime_specific = True
                regime_status = '🟢 Trusted' if regime_mult > 1.2 else '🔴 Distrusted' if regime_mult < 0.8 else '⚪ Baseline'
                weight_rows.append({
                    'Agent': _DISPLAY_NAMES.get(agent, agent),
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

# === SECTION 5B: Contribution-Based Multipliers ===
try:
    from trading_bot.contribution_bridge import is_contribution_scoring_enabled
    from trading_bot.contribution_scorer import ContributionTracker
    _is_contrib = is_contribution_scoring_enabled()

    st.markdown("---")
    st.subheader("🔄 Contribution-Based Multipliers")
    if _is_contrib:
        st.caption("**ACTIVE** — These multipliers drive council voting weights.")
    else:
        st.caption("**INACTIVE** — Contribution scores computed but Brier multipliers still in use.")

    _contrib_path = _resolve_data_path_for("contribution_scores.json", ticker)
    _contrib_tracker = ContributionTracker(data_path=_contrib_path) if os.path.exists(_contrib_path) else None
    if _contrib_tracker:
        _summary = _contrib_tracker.get_summary()
        if _summary:
            _rows = []
            for agent, regimes in _summary.items():
                for regime, info in regimes.items():
                    _mult = info["multiplier"]
                    _status = (
                        "🟢 Trusted" if _mult > 1.2
                        else "🔴 Distrusted" if _mult < 0.8
                        else "⚪ Baseline"
                    )
                    _rows.append({
                        "Agent": _DISPLAY_NAMES.get(agent, agent),
                        "Regime": regime,
                        "Multiplier": _mult,
                        "Avg Score": info["avg_score"],
                        "Samples": info["total_scores"],
                        "Status": _status,
                    })
            if _rows:
                st.dataframe(
                    pd.DataFrame(_rows).sort_values(["Agent", "Regime"]),
                    hide_index=True, width="stretch",
                )
            else:
                st.info("No contribution data yet. Run migration script or wait for reconciliation.")
        else:
            st.info("Contribution tracker has no data.")
    else:
        st.info("Contribution tracker not initialized. Check scoring config.")
except ImportError:
    pass  # Module not yet deployed
except Exception as e:
    st.warning(f"Contribution display error: {e}")


# === SECTION 6: AGENT INFLUENCE OVER TIME (Learning Trajectory) ===
st.markdown("---")
st.subheader("📈 Agent Influence Over Time")
st.caption("Agents above 1.0 have earned more influence through accurate predictions. Below 1.0 means the system trusts them less.")

weight_df = load_weight_evolution(ticker)

weight_df = date_range_filter(weight_df, key_prefix='brier')

if not weight_df.empty and len(weight_df) >= 5:
    smoothing = st.slider("Smoothing window (cycles)", 1, 30, 7,
                           help="Rolling average window. 1 = raw data.",
                           key="brier_smoothing")

    available_agents = sorted(weight_df['agent'].unique())

    _AGENT_COLORS = {
        'agronomist': '#2ca02c',
        'macro': '#1f77b4',
        'geopolitical': '#ff7f0e',
        'supply_chain': '#d62728',
        'inventory': '#9467bd',
        'sentiment': '#8c564b',
        'technical': '#e377c2',
        'volatility': '#7f7f7f',
    }

    fig = go.Figure()

    # Baseline reference at 1.0
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="gray", line_width=1,
        annotation_text="Baseline (1.0)",
        annotation_position="bottom right",
        annotation_font_color="gray",
    )

    for agent in available_agents:
        agent_data = weight_df[weight_df['agent'] == agent].sort_values('timestamp')
        y_vals = agent_data['final_weight'].ewm(span=smoothing, min_periods=1).mean()
        display_name = _DISPLAY_NAMES.get(agent, agent.title())
        color = _AGENT_COLORS.get(agent, None)

        fig.add_trace(go.Scatter(
            x=agent_data['timestamp'],
            y=y_vals,
            mode='lines',
            name=display_name,
            line=dict(color=color, width=2) if color else dict(width=2),
        ))

    fig.update_layout(
        yaxis_title='Final Weight',
        xaxis_title='Time',
        height=420,
        margin=dict(t=20, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )

    st.plotly_chart(fig, width="stretch")

    # --- Agent trajectory summary: current vs 30 cycles ago ---
    summary_rows = []
    for agent in available_agents:
        agent_data = weight_df[weight_df['agent'] == agent].sort_values('timestamp')
        current_weight = agent_data['final_weight'].iloc[-1]

        if len(agent_data) > 30:
            past_weight = agent_data['final_weight'].iloc[-31]
        elif len(agent_data) > 1:
            past_weight = agent_data['final_weight'].iloc[0]
        else:
            past_weight = current_weight

        delta = current_weight - past_weight
        display_name = _DISPLAY_NAMES.get(agent, agent.title())

        if delta > 0.005:
            trend_str = f"from {past_weight:.2f}"
            arrow = "^"
        elif delta < -0.005:
            trend_str = f"from {past_weight:.2f}"
            arrow = "v"
        else:
            trend_str = "stable"
            arrow = "="

        summary_rows.append({
            'Agent': display_name,
            'Current Weight': round(current_weight, 2),
            'Trend': arrow,
            'vs 30 Cycles Ago': trend_str,
            'Delta': round(delta, 3),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('Current Weight', ascending=False)

    def _color_delta(val):
        if val > 0.005:
            return 'color: #2ca02c'
        elif val < -0.005:
            return 'color: #d62728'
        return 'color: gray'

    styled = summary_df.style.map(_color_delta, subset=['Delta'])
    st.dataframe(styled, hide_index=True, width="stretch")

elif not weight_df.empty:
    st.info("Insufficient weight evolution data (need at least 5 rows). Data accumulates as trading cycles run.")
else:
    st.info("Weight evolution tracking not yet active. Data will appear after the next trading cycle.")


# === SECTION 7: REGIME-SPECIFIC AGENT RANKING ===
st.markdown("---")
st.subheader("🏆 Agent Accuracy by Market Regime")
st.caption("Which agents perform best in each market condition?")

if enhanced_data and enhanced_data.get('agent_scores'):
    agent_scores = enhanced_data['agent_scores']

    rows = []
    for agent, regimes in agent_scores.items():
        for regime, scores in regimes.items():
            if len(scores) >= 3:
                rows.append({
                    'agent': agent,
                    'regime': regime,
                    'avg_brier': np.mean(scores),
                    'count': len(scores),
                })

    if rows:
        regime_df = pd.DataFrame(rows)

        pivot = regime_df.pivot_table(index='agent', columns='regime', values='avg_brier')
        counts = regime_df.pivot_table(index='agent', columns='regime', values='count')

        display_data = {}
        for regime in pivot.columns:
            col_vals = []
            for agent in pivot.index:
                brier = pivot.loc[agent, regime] if not pd.isna(pivot.loc[agent, regime]) else None
                cnt = counts.loc[agent, regime] if not pd.isna(counts.loc[agent, regime]) else 0
                if brier is None or cnt < 3:
                    col_vals.append("---")
                else:
                    col_vals.append(f"{brier:.3f}")
            display_data[regime] = col_vals

        display_df = pd.DataFrame(display_data, index=[
            _DISPLAY_NAMES.get(a, a.title()) for a in pivot.index
        ])
        display_df.index.name = 'Agent'

        def _color_brier(val):
            if val == "---":
                return 'color: gray'
            try:
                score = float(val)
                if score <= 0.15:
                    return 'background-color: rgba(44, 160, 44, 0.2); color: #2ca02c'
                elif score <= 0.25:
                    return 'background-color: rgba(255, 193, 7, 0.2); color: #856404'
                else:
                    return 'background-color: rgba(214, 39, 40, 0.2); color: #d62728'
            except (ValueError, AttributeError):
                return ''

        styled_regime = display_df.style.map(_color_brier)
        st.dataframe(styled_regime, width="stretch")

        st.caption(
            "**Brier Score:** 0.0 = perfect, 0.25 = random, 0.5 = worst. "
            "Lower is better. Minimum 3 samples per cell."
        )

        best_agents = []
        for regime in pivot.columns:
            valid = pivot[regime].dropna()
            valid = valid[counts[regime].fillna(0) >= 3]
            if not valid.empty:
                best = valid.idxmin()
                best_name = _DISPLAY_NAMES.get(best, best.title())
                best_score = valid.min()
                best_agents.append(f"**{regime}**: {best_name} ({best_score:.3f})")

        if best_agents:
            st.markdown("**Top performer per regime:** " + " | ".join(best_agents))
    else:
        st.info("Not enough data for regime-specific ranking. Need at least 3 resolved predictions per agent per regime.")
else:
    st.info("Regime-specific Brier scores will appear after predictions are resolved via reconciliation.")


# === TMS TEMPORAL DECAY VISUALIZATION ===
st.markdown("---")
st.subheader("🕐 TMS Temporal Decay Curves")
st.caption("Shows how different document types lose relevance over time")

try:
    from config.commodity_profiles import get_active_profile
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

days = np.arange(0, 60, 0.5)
chart_data = {}

display_types = ['weather', 'logistics', 'news', 'macro', 'technical', 'trade_journal']
for doc_type in display_types:
    lam = decay_rates.get(doc_type, 0.05)
    chart_data[f"{doc_type} (λ={lam})"] = [math.exp(-lam * d) for d in days]

decay_df = pd.DataFrame(chart_data, index=days)
decay_df.index.name = 'Age (days)'

st.line_chart(decay_df)

st.caption(
    "**Reading the chart:** A document at 50% relevance has lost half its informational value. "
    "Weather data (λ=0.15) hits 50% at ~4.6 days. Macro data (λ=0.02) takes ~34.7 days. "
    "Decay rates are configurable per commodity profile."
)


st.markdown("---")
st.caption("Brier Analysis | Real Options v6.4")
