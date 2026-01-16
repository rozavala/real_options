"""
Page 3: The Council Chamber (Explainability)

Purpose: Understand WHY trades happened. Visualizes the internal debate from agents.py.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import load_council_history, get_status_color

st.set_page_config(layout="wide", page_title="Council | Coffee Bot")

st.title("🧠 The Council Chamber")
st.caption("Agent Explainability - Understand the reasoning behind every trade decision")

# --- Load Data ---
council_df = load_council_history()

if council_df.empty:
    st.warning("No council history data available.")
    st.stop()

st.markdown("---")

# === Decision Selector ===
st.subheader("🔍 Select Decision to Analyze")

council_df['display'] = council_df.apply(
    lambda r: f"{r['timestamp']} | {r.get('contract', 'Unknown')} | {r.get('master_decision', 'N/A')}",
    axis=1
)

selected = st.selectbox(
    "Choose a decision:",
    options=council_df['display'].tolist()[:50],  # Last 50 decisions
    index=0
)

# Get selected row
selected_idx = council_df[council_df['display'] == selected].index[0]
row = council_df.loc[selected_idx]

st.markdown("---")

# === SECTION 1: Consensus Radar Chart ===
st.subheader("🕸️ Consensus Radar")
st.caption("Perfect polygon = total alignment. Jagged shape = internal conflict")

# Map sentiments to numerical values
def sentiment_to_value(sentiment):
    if sentiment in ['BULLISH', 'Bullish', 'bullish']:
        return 1.0
    elif sentiment in ['BEARISH', 'Bearish', 'bearish']:
        return -1.0
    return 0.0

agents = ['Meteorologist', 'Macro', 'Geopolitical', 'Fundamentals', 'Sentiment', 'Technical', 'Volatility']
agent_cols = [
    'meteorologist_sentiment', 'macro_sentiment', 'geopolitical_sentiment',
    'fundamentalist_sentiment', 'sentiment_sentiment', 'technical_sentiment',
    'volatility_sentiment'
]

values = [sentiment_to_value(row.get(col, 'NEUTRAL')) for col in agent_cols]
values.append(values[0])  # Close the radar

agents_display = agents + [agents[0]]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=[abs(v) for v in values],
    theta=agents_display,
    fill='toself',
    name='Conviction Strength',
    line_color='#636EFA'
))

# Add direction indicator
colors = ['#00CC96' if v > 0 else '#EF553B' if v < 0 else '#888888' for v in values]

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    showlegend=True,
    height=400
)

radar_cols = st.columns([2, 1])

with radar_cols[0]:
    st.plotly_chart(fig, width="stretch")

with radar_cols[1]:
    st.markdown("**Agent Votes:**")
    for i, (agent, col) in enumerate(zip(agents, agent_cols)):
        sentiment = row.get(col, 'NEUTRAL')
        icon = "🟢" if sentiment in ['BULLISH', 'Bullish'] else "🔴" if sentiment in ['BEARISH', 'Bearish'] else "⚪"
        st.write(f"{icon} **{agent}**: {sentiment}")

st.markdown("---")

# === SECTION 2: The "Tug-of-War" (Bull vs Bear) ===
st.subheader("⚔️ Bull vs Bear Tug-of-War")

bullish_count = sum(1 for col in agent_cols if row.get(col, '') in ['BULLISH', 'Bullish', 'bullish'])
bearish_count = sum(1 for col in agent_cols if row.get(col, '') in ['BEARISH', 'Bearish', 'bearish'])
neutral_count = len(agent_cols) - bullish_count - bearish_count

fig = go.Figure()

fig.add_trace(go.Bar(
    y=['Consensus'],
    x=[-bearish_count],
    orientation='h',
    name='Bearish',
    marker_color='#EF553B',
    text=[f'{bearish_count} Bears'],
    textposition='inside'
))

fig.add_trace(go.Bar(
    y=['Consensus'],
    x=[bullish_count],
    orientation='h',
    name='Bullish',
    marker_color='#00CC96',
    text=[f'{bullish_count} Bulls'],
    textposition='inside'
))

fig.update_layout(
    barmode='relative',
    height=150,
    xaxis=dict(
        range=[-len(agent_cols), len(agent_cols)],
        title='← Bearish | Bullish →'
    ),
    showlegend=True,
    margin=dict(l=0, r=0, t=20, b=0)
)

st.plotly_chart(fig, width="stretch")

st.markdown("---")

# === SECTION 3: Master Decision Details ===
st.subheader("👑 Master Decision")

master_cols = st.columns(3)

with master_cols[0]:
    decision = row.get('master_decision', 'N/A')
    color = "#00CC96" if decision == 'BULLISH' else "#EF553B" if decision == 'BEARISH' else "#888888"
    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color: white; margin: 0;">{decision}</h2>
    </div>
    """, unsafe_allow_html=True)

with master_cols[1]:
    confidence = row.get('master_confidence', 0)
    st.metric("Confidence", f"{confidence:.1%}")

with master_cols[2]:
    approved = row.get('compliance_approved', True)
    status = "✅ Approved" if approved else "❌ Vetoed"
    st.metric("Compliance", status)

# Reasoning
st.markdown("**Master Reasoning:**")
reasoning = row.get('master_reasoning', 'No reasoning recorded.')
st.info(reasoning)

st.markdown("---")

# === SECTION 4: Agent Summaries ===
st.subheader("📝 Agent Analysis Summaries")

summary_cols = {
    'meteorologist_summary': ('🌦️ Meteorologist', 'meteorologist_sentiment'),
    'macro_summary': ('💵 Macro Economist', 'macro_sentiment'),
    'geopolitical_summary': ('🌍 Geopolitical', 'geopolitical_sentiment'),
    'fundamentalist_summary': ('📦 Fundamentalist', 'fundamentalist_sentiment'),
    'sentiment_summary': ('🧠 Sentiment/COT', 'sentiment_sentiment'),
    'technical_summary': ('📉 Technical', 'technical_sentiment'),
    'volatility_summary': ('⚡ Volatility', 'volatility_sentiment')
}

for summary_col, (name, sentiment_col) in summary_cols.items():
    sentiment = row.get(sentiment_col, 'N/A')
    icon = "🟢" if 'BULL' in str(sentiment).upper() else "🔴" if 'BEAR' in str(sentiment).upper() else "⚪"

    with st.expander(f"{icon} {name} ({sentiment})"):
        summary = row.get(summary_col, 'No summary available.')
        st.write(summary)

st.markdown("---")

# === SECTION 5: Hallucination Log (Compliance Vetoes) ===
st.subheader("🚨 Compliance Vetoes (Hallucination Log)")

vetoed_df = council_df[council_df['compliance_approved'] == False]

if not vetoed_df.empty:
    st.warning(f"Found {len(vetoed_df)} vetoed decisions")
    st.dataframe(
        vetoed_df[['timestamp', 'contract', 'master_decision', 'master_reasoning']].head(20),
        width="stretch"
    )
else:
    st.success("No compliance vetoes recorded - all decisions passed compliance checks.")
