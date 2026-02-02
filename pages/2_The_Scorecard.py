"""
Page 2: The Scorecard (Decision Quality)

Purpose: Most critical addition - audits the intelligence of the AI.
Bridges council_history.csv with market data to evaluate signal-to-outcome.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import (
    load_council_history,
    grade_decision_quality,
    calculate_confusion_matrix,
    calculate_agent_scores,
    fetch_live_dashboard_data,
    get_config
)

def create_process_outcome_matrix(df: pd.DataFrame):
    """
    Creates a 2x2 matrix separating process quality from outcome quality.

    Process Score = master_confidence * |weighted_score| (Signal Strength)
    Outcome = pnl_realized
    """
    st.subheader("🎯 Process vs Outcome Analysis")
    st.caption("Distinguishing Skill (Good Process + Win) from Luck (Bad Process + Win)")

    # Filter to resolved trades with P&L
    if 'pnl_realized' not in df.columns:
        st.info("No P&L data available.")
        return

    resolved = df[df['pnl_realized'].notna()].copy()

    if resolved.empty:
        st.info("Need resolved trades with P&L to analyze process vs outcome.")
        return

    # Calculate Process Score (Higher = stronger conviction signal)
    # Handle missing weighted_score by defaulting to 0
    w_score = resolved['weighted_score'].abs() if 'weighted_score' in resolved.columns else 0
    resolved['process_score'] = resolved['master_confidence'].fillna(0.5) * w_score

    # Normalize Process Score to 0-1 for plotting
    max_process = resolved['process_score'].max()
    if max_process > 0:
        resolved['process_score_norm'] = resolved['process_score'] / max_process
    else:
        resolved['process_score_norm'] = 0.5

    # Classify Quadrants
    # Thresholds: Median for process, 0 for P&L
    process_threshold = resolved['process_score_norm'].median()

    def classify(row):
        good_process = row['process_score_norm'] >= process_threshold
        good_outcome = row['pnl_realized'] > 0

        if good_process and good_outcome: return 'SKILL'
        if not good_process and good_outcome: return 'LUCK'
        if good_process and not good_outcome: return 'BAD LUCK'
        return 'NO SKILL'

    resolved['quadrant'] = resolved.apply(classify, axis=1)

    # Plot
    fig = px.scatter(
        resolved,
        x='process_score_norm',
        y='pnl_realized',
        color='quadrant',
        color_discrete_map={
            'SKILL': '#00CC96',     # Green
            'LUCK': '#FFA15A',      # Orange
            'BAD LUCK': '#636EFA',  # Blue
            'NO SKILL': '#EF553B'   # Red
        },
        hover_data=['contract', 'strategy_type', 'master_decision'],
        title='Process Quality vs Trade Outcome'
    )

    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=process_threshold, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, use_container_width=True)

st.set_page_config(layout="wide", page_title="Scorecard | Mission Control")

st.title("⚖️ The Scorecard")
st.caption("Decision Quality Analysis - Is the Master Strategist generating alpha or just noise?")

# --- Load Data ---
council_df = load_council_history()
config = get_config()

if council_df.empty:
    st.warning("No council history data available. Run the bot to generate decisions.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.markdown("### Filters")
strategy_filter = st.sidebar.multiselect(
    "Strategy Types",
    options=['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD', 'LONG_STRADDLE', 'IRON_CONDOR', 'NONE'],
    default=['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD', 'LONG_STRADDLE', 'IRON_CONDOR']
)

# Get live price for grading recent decisions
live_price = None
if config:
    live_data = fetch_live_dashboard_data(config)
    live_price = live_data.get('KC_Price')

# Grade decisions
graded_df = grade_decision_quality(council_df)

# Filter dataframe based on strategy
if 'strategy_type' in graded_df.columns:
    graded_df = graded_df[graded_df['strategy_type'].isin(strategy_filter)]

st.markdown("---")

# === SECTION 1: The "Truth" Matrix (Confusion Matrix) ===
st.subheader("🎯 The Truth Matrix")
st.caption("Classification of every Master Decision against actual market outcome")

confusion = calculate_confusion_matrix(graded_df)

matrix_cols = st.columns([2, 3])

with matrix_cols[0]:
    # Visual Confusion Matrix (Using px.imshow)
    matrix_data = [
        [confusion['true_positive'], confusion['false_negative']],
        [confusion['false_positive'], confusion['true_negative']]
    ]

    # Labels for display
    x_labels = ['Market UP', 'Market DOWN']
    y_labels = ['AI: BULLISH', 'AI: BEARISH']

    fig = px.imshow(
        matrix_data,
        labels=dict(x="Actual Trend", y="AI Prediction", color="Count"),
        x=x_labels,
        y=y_labels,
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )

    # Custom text to show TP/FP/FN/TN explicitly
    annotations = []
    text_labels = [
        [f"TP: {confusion['true_positive']}", f"FN: {confusion['false_negative']}"],
        [f"FP: {confusion['false_positive']}", f"TN: {confusion['true_negative']}"]
    ]

    for i, row in enumerate(text_labels):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=x_labels[j], y=y_labels[i], text=str(val),
                showarrow=False, font=dict(color='black')
            ))

    fig.update_layout(
        title="Confusion Matrix",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        annotations=annotations
    )

    st.plotly_chart(fig, width="stretch")

with matrix_cols[1]:
    # Metrics
    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("Precision", f"{confusion['precision']:.1%}")
        st.caption("TP / (TP + FP)")

    with metric_cols[1]:
        st.metric("Recall", f"{confusion['recall']:.1%}")
        st.caption("TP / (TP + FN)")

    with metric_cols[2]:
        st.metric("Accuracy", f"{confusion['accuracy']:.1%}")
        st.caption("(TP + TN) / Total")

    with metric_cols[3]:
        st.metric("Total Graded", confusion['total'])
        st.caption("Decisions evaluated")

    # After the existing metrics, add volatility context:
    if confusion.get('vol_total', 0) > 0:
        st.caption(
            f"📊 Includes **{confusion['vol_total']}** volatility trades: "
            f"**{confusion['vol_wins']}** wins, **{confusion['vol_losses']}** losses"
        )

st.markdown("---")

# === SECTION 2: Process vs Outcome ===
create_process_outcome_matrix(graded_df)

# === NEW SECTION: Volatility Trade Performance ===
st.markdown("---")
st.subheader("⚡ Volatility Strategy Performance")

vol_df = graded_df[graded_df['prediction_type'] == 'VOLATILITY'] if 'prediction_type' in graded_df.columns else pd.DataFrame()

if not vol_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Long Straddle Performance**")
        straddle_df = vol_df[vol_df['strategy_type'] == 'LONG_STRADDLE']
        if not straddle_df.empty:
            wins = (straddle_df['outcome'] == 'WIN').sum()
            losses = (straddle_df['outcome'] == 'LOSS').sum()
            total_vol = wins + losses
            win_rate = wins/total_vol*100 if total_vol > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.caption(f"Wins when price moved >1.8%: {wins} | Losses: {losses}")
        else:
            st.info("No Long Straddle trades.")

    with col2:
        st.markdown("**Iron Condor Performance**")
        condor_df = vol_df[vol_df['strategy_type'] == 'IRON_CONDOR']
        if not condor_df.empty:
            wins = (condor_df['outcome'] == 'WIN').sum()
            losses = (condor_df['outcome'] == 'LOSS').sum()
            total_vol = wins + losses
            win_rate = wins/total_vol*100 if total_vol > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.caption(f"Wins when price stayed <2%: {wins} | Losses: {losses}")
        else:
            st.info("No Iron Condor trades.")
else:
    st.info("No volatility trades recorded yet.")

st.markdown("---")

# === SECTION 2: The "Liar's Plot" (Confidence vs. P&L) ===
st.subheader("📊 The Liar's Plot")
st.caption("High confidence should correlate with profits. Bottom-right clustering indicates overconfidence.")

# Determine which P&L column to use
pnl_col = None
if 'pnl' in graded_df.columns:
    pnl_col = 'pnl'
elif 'pnl_realized' in graded_df.columns:
    pnl_col = 'pnl_realized'

if 'master_confidence' in graded_df.columns and pnl_col is not None:
    plot_df = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])].copy()

    if not plot_df.empty and pnl_col in plot_df.columns:
        plot_df['pnl_display'] = plot_df[pnl_col].fillna(0).astype(float)

        fig = px.scatter(
            plot_df,
            x='master_confidence',
            y='pnl_display',
            color='outcome',
            color_discrete_map={'WIN': '#00CC96', 'LOSS': '#EF553B'},
            hover_data=['timestamp', 'contract', 'master_decision', 'strategy_type'],
            title="Confidence vs. Realized P&L"
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0.75, line_dash="dash", line_color="gray",
                      annotation_text="High Confidence Zone")

        fig.update_layout(
            xaxis_title="Master Confidence Score",
            yaxis_title="Realized P&L ($)",
            xaxis_range=[0.5, 1.0],
            height=400
        )

        st.plotly_chart(fig, width="stretch")

        # Quadrant Analysis
        high_conf = plot_df[plot_df['master_confidence'] >= 0.75]
        if not high_conf.empty:
            high_conf_wins = (high_conf['outcome'] == 'WIN').sum()
            high_conf_total = len(high_conf)
            st.info(f"High Confidence Zone (≥75%): **{high_conf_wins}/{high_conf_total}** wins "
                    f"({high_conf_wins/high_conf_total:.1%} accuracy)")
    else:
        st.info("Not enough graded decisions with P&L data for this chart.")
else:
    st.info("P&L data not available in graded decisions.")

st.markdown("---")

# === SECTION 3: Agent Leaderboard (Brier Score / Accuracy) ===
st.subheader("🏆 Agent Leaderboard")
st.caption("Ranking sub-agents by prediction accuracy - helps identify which agents to trust")

scores = calculate_agent_scores(council_df, live_price)

# Pretty names for display
pretty_names = {
    'meteorologist_sentiment': '🌦️ Meteorologist',
    'macro_sentiment': '💵 Macro Economist',
    'geopolitical_sentiment': '🌍 Geopolitical',
    'fundamentalist_sentiment': '📦 Fundamentalist',
    'sentiment_sentiment': '🧠 Sentiment/COT',
    'technical_sentiment': '📉 Technical',
    'volatility_sentiment': '⚡ Volatility',
    'master_decision': '👑 Master Strategist'
}

# Sort by accuracy
sorted_agents = sorted(
    [(k, v) for k, v in scores.items() if v['total'] > 0],
    key=lambda x: x[1]['accuracy'],
    reverse=True
)

if sorted_agents:
    # Create leaderboard chart
    chart_data = pd.DataFrame([
        {
            'Agent': pretty_names.get(agent, agent),
            'Accuracy': score['accuracy'] * 100,
            'Correct': score['correct'],
            'Total': score['total']
        }
        for agent, score in sorted_agents
    ])

    fig = px.bar(
        chart_data,
        x='Agent',
        y='Accuracy',
        color='Accuracy',
        color_continuous_scale='RdYlGn',
        text='Accuracy',
        hover_data=['Correct', 'Total']
    )

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        yaxis_title="Accuracy %",
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, width="stretch")

    # Detailed table
    with st.expander("📋 Detailed Scores"):
        st.dataframe(chart_data, width="stretch")
else:
    st.info("Not enough data to calculate agent scores.")

st.markdown("---")

# === FEEDBACK LOOP HEALTH ===
st.subheader("🔄 Feedback Loop Health")

# Locate data directory relative to this file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
structured_file = os.path.join(data_dir, "agent_accuracy_structured.csv")

if os.path.exists(structured_file):
    struct_df = pd.read_csv(structured_file)

    if not struct_df.empty:
        total = len(struct_df)
        pending = (struct_df['actual'] == 'PENDING').sum()
        orphaned = (struct_df['actual'] == 'ORPHANED').sum() if 'actual' in struct_df.columns else 0
        resolved = total - pending - orphaned
        resolvable = total - orphaned
        rate = resolved / resolvable * 100 if resolvable > 0 else 0

        health_cols = st.columns(5)
        health_cols[0].metric("Total Predictions", total)
        health_cols[1].metric("Resolved", resolved)
        health_cols[2].metric("Pending", pending)
        health_cols[3].metric(
            "Orphaned", orphaned,
            help="Legacy predictions (Jan 19-27) with no matching council decision"
        )
        health_cols[4].metric(
            "Resolution Rate", f"{rate:.0f}%",
            help="Excludes orphaned predictions"
        )

        # Color-coded status
        if rate >= 80:
            st.success(
                f"✅ Feedback loop healthy — {rate:.0f}% resolution rate"
                + (f" ({orphaned} legacy orphans excluded)" if orphaned > 0 else "")
            )
        elif rate >= 50:
            st.warning(f"⚠️ Feedback loop degraded — {rate:.0f}% resolution rate")
        else:
            st.error(f"🔴 Feedback loop broken — {rate:.0f}% resolution rate. Agent learning not occurring!")

        # Show cycle_id coverage
        if 'cycle_id' in struct_df.columns:
            has_cycle_id = struct_df['cycle_id'].notna() & (struct_df['cycle_id'] != '')
            cycle_id_pct = has_cycle_id.sum() / total * 100
            st.caption(f"cycle_id coverage: {cycle_id_pct:.0f}% (new system) / {100-cycle_id_pct:.0f}% (legacy)")
    else:
        st.info("No prediction data yet.")
else:
    st.info("Structured prediction file not found.")

st.markdown("---")

# === SECTION 4: Decision History Table ===
st.subheader("📜 Recent Decisions")

if not graded_df.empty:
    # Add strategy_type column to display if available
    cols_to_show = ['timestamp', 'contract', 'master_decision', 'master_confidence', 'outcome']
    if 'strategy_type' in graded_df.columns:
        cols_to_show.insert(2, 'strategy_type')

    display_df = graded_df[cols_to_show].copy()
    display_df['master_confidence'] = display_df['master_confidence'].apply(lambda x: f"{x:.1%}")

    # Color code outcomes
    def style_outcome(val):
        if val == 'WIN':
            return 'background-color: #00CC96; color: white'
        elif val == 'LOSS':
            return 'background-color: #EF553B; color: white'
        return 'background-color: gray; color: white'

    st.dataframe(
        display_df.sort_values('timestamp', ascending=False).head(20).style.map(
            style_outcome, subset=['outcome']
        ),
        width="stretch"
    )
else:
    st.info("No decisions to display.")
