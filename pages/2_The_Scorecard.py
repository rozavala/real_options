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

st.set_page_config(layout="wide", page_title="Scorecard | Coffee Bot")

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

if 'master_confidence' in graded_df.columns and 'pnl' in graded_df.columns:
    plot_df = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])].copy()

    if not plot_df.empty:
        plot_df['pnl'] = plot_df.get('pnl', 0).fillna(0)

        fig = px.scatter(
            plot_df,
            x='master_confidence',
            y='pnl',
            color='outcome',
            color_discrete_map={'WIN': '#00CC96', 'LOSS': '#EF553B'},
            hover_data=['timestamp', 'contract', 'master_decision'],
            title="Confidence vs. Realized P&L"
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0.75, line_dash="dash", line_color="gray", annotation_text="High Confidence Zone")

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
            st.info(f"High Confidence Zone (≥75%): **{high_conf_wins}/{high_conf_total}** wins ({high_conf_wins/high_conf_total:.1%} accuracy)")
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
    'master_decision': '👑 Master Strategist',
    'ml_signal': '🤖 ML Model'
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
