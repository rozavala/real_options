"""
Page 3: The Council Chamber (Explainability)

Purpose: Understand WHY trades happened. Visualizes the internal debate from agents.py.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import html

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import load_council_history, load_trade_journal, find_journal_entry, load_prompt_traces

st.set_page_config(layout="wide", page_title="Council | Real Options")

from _commodity_selector import selected_commodity
ticker = selected_commodity()

st.title("🧠 The Council Chamber")
st.caption("Agent Explainability - Understand the reasoning behind every trade decision")

# --- Load Data ---
council_df = load_council_history(ticker=ticker)

if council_df.empty:
    st.warning("No council history data available.")
    st.stop()

# === CONTRACT CONTEXT INDICATOR ===
st.info("""
    ℹ️ **Note on Multi-Contract Analysis:** Agent reports shown below represent
    *general market context* applicable to all active futures contracts. When multiple
    contracts are analyzed, the most recent analysis is displayed. For contract-specific
    insights, refer to the individual trade thesis in the Position Cards.
""")

st.markdown("---")

# === Decision Selector ===
st.subheader("🔍 Select Decision to Analyze")

# CRITICAL FIX: Sort by timestamp descending BEFORE slicing
council_df['timestamp'] = pd.to_datetime(council_df['timestamp'])
council_df = council_df.sort_values('timestamp', ascending=False).reset_index(drop=True)

# Selection mode toggle
col_sel, col_mode = st.columns([3, 1])
with col_mode:
    mode = st.radio("Mode", ["Recent", "Calendar"], label_visibility="collapsed", horizontal=True)

if mode == "Calendar":
    date_col, decision_col = st.columns([1, 2])

    with date_col:
        min_date = council_df['timestamp'].min().date()
        max_date = council_df['timestamp'].max().date()

        selected_date = st.date_input(
            "Select Date:",
            value=max_date,  # Default to most recent
            min_value=min_date,
            max_value=max_date
        )

        # Show recent activity sparkline
        decisions_per_day = council_df.groupby(council_df['timestamp'].dt.date).size()
        recent_days = decisions_per_day.tail(7)
        st.caption("Recent: " + " | ".join([f"{d.strftime('%m/%d')}:{c}" for d, c in recent_days.items()]))

    with decision_col:
        mask = council_df['timestamp'].dt.date == selected_date
        daily_decisions = council_df[mask].copy()

        if daily_decisions.empty:
            st.warning(f"No decisions recorded for {selected_date}")
            st.stop()

        daily_decisions['display'] = daily_decisions.apply(
            lambda r: f"{r['timestamp'].strftime('%H:%M:%S')} | {r.get('contract', 'Unknown')} | {r.get('master_decision', 'N/A')}",
            axis=1
        )

        selected = st.selectbox(
            f"Decisions on {selected_date} ({len(daily_decisions)} total):",
            options=daily_decisions['display'].tolist(),
            index=0
        )
        selected_idx = daily_decisions[daily_decisions['display'] == selected].index[0]
else:
    # Traditional sorted dropdown
    council_df['display'] = council_df.apply(
        lambda r: f"{r['timestamp'].strftime('%Y-%m-%d %H:%M')} | {r.get('contract', 'Unknown')} | {r.get('master_decision', 'N/A')}",
        axis=1
    )

    selected = st.selectbox(
        "Choose a decision:",
        options=council_df['display'].tolist()[:100],
        index=0
    )
    selected_idx = council_df[council_df['display'] == selected].index[0]

row = council_df.loc[selected_idx]

# === SIDEBAR: Consensus Meter ===
with st.sidebar:
    st.markdown("### 🎯 Consensus Meter")

    weighted_score = row.get('weighted_score', None)

    # Robust check: handle empty strings, NaN, None, and valid zero scores
    ws = None
    if weighted_score is not None and weighted_score != '' and pd.notna(weighted_score):
        try:
            ws = float(weighted_score)
        except (ValueError, TypeError):
            ws = None

    if ws is not None:
        # Determine conviction level
        if ws >= 0.6:
            level, color = "STRONG BULL", "#00CC96"
        elif ws >= 0.2:
            level, color = "WEAK BULL", "#90EE90"
        elif ws > -0.2:
            level, color = "DEADLOCK", "#888888"
        elif ws > -0.6:
            level, color = "WEAK BEAR", "#FFB347"
        else:
            level, color = "STRONG BEAR", "#EF553B"

        # Display gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ws,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': level, 'font': {'size': 14}},
            gauge={
                'axis': {'range': [-1, 1], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'steps': [
                    {'range': [-1, -0.6], 'color': '#EF553B'},
                    {'range': [-0.6, -0.2], 'color': '#FFB347'},
                    {'range': [-0.2, 0.2], 'color': '#E8E8E8'},
                    {'range': [0.2, 0.6], 'color': '#90EE90'},
                    {'range': [0.6, 1], 'color': '#00CC96'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': ws
                }
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, width='stretch')
    else:
        st.info("No consensus data for this decision")

st.markdown("---")

# === SECTION 1: Consensus Radar Chart ===
st.subheader("🕸️ Consensus Radar")
st.caption("Perfect polygon = total alignment. Jagged shape = internal conflict")

# Map sentiments to numerical values
def sentiment_to_value(sentiment):
    if pd.isna(sentiment) or str(sentiment).lower() == 'nan':
        return 0.0  # Treat missing data as no signal (won't affect radar)
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
        # Handle NaN/None/empty values from stale or missing agents
        if pd.isna(sentiment) or str(sentiment).lower() == 'nan' or sentiment == '':
            icon = "🔘"
            display_sentiment = "No Data"
        elif sentiment in ['BULLISH', 'Bullish', 'bullish']:
            icon = "🟢"
            display_sentiment = sentiment
        elif sentiment in ['BEARISH', 'Bearish', 'bearish']:
            icon = "🔴"
            display_sentiment = sentiment
        else:
            icon = "⚪"
            display_sentiment = sentiment
        st.write(f"{icon} **{agent}**: {display_sentiment}")

    # Add interpretation text based on trade type
    if row.get('prediction_type') == 'VOLATILITY':
        st.caption("ℹ️ For volatility trades, agent disagreement is expected - it signals market inflection points")

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

# === SECTION 2.5: Weighted Vote Breakdown ===
st.subheader("⚖️ Weighted Vote Calculation")

import json

vote_breakdown_raw = row.get('vote_breakdown', None)

if vote_breakdown_raw and pd.notna(vote_breakdown_raw) and str(vote_breakdown_raw).strip() not in ['', '[]']:
    try:
        vote_data = json.loads(vote_breakdown_raw) if isinstance(vote_breakdown_raw, str) else vote_breakdown_raw

        if vote_data and isinstance(vote_data, list) and len(vote_data) > 0:
            vote_df = pd.DataFrame(vote_data)

            fig_vote = px.bar(
                vote_df,
                x='agent',
                y='contribution',
                color='direction',
                color_discrete_map={'BULLISH': '#00CC96', 'BEARISH': '#EF553B', 'NEUTRAL': '#888'},
                title='Agent Contribution to Final Decision',
                text_auto='.3f'
            )
            fig_vote.update_layout(height=350)
            st.plotly_chart(fig_vote, width='stretch')

            # Metrics row
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Dominant Agent", row.get('dominant_agent', 'Unknown'), help="The agent with the strongest conviction that drove the final decision.")
            with metric_cols[1]:
                ws = row.get('weighted_score', 0)
                st.metric("Weighted Score", f"{float(ws):.4f}" if ws else "N/A", help="The sum of all active agent votes multiplied by their historical reliability weights. Ranges from -1.0 (Strong Bear) to +1.0 (Strong Bull).")
            with metric_cols[2]:
                active = len([v for v in vote_data if v.get('direction') != 'NEUTRAL'])
                st.metric("Active Voters", f"{active}/{len(vote_data)}", help="Number of agents that had a clear directional opinion versus those that remained neutral.")
            with metric_cols[3]:
                trigger = row.get('trigger_type', 'scheduled')
                st.metric("Trigger", trigger.replace('_', ' ').title(), help="What initiated this decision (e.g., scheduled event, sentinel alert, emergency intervention).")
        else:
            st.info("ℹ️ Vote breakdown is empty for this decision")

    except json.JSONDecodeError as e:
        st.warning(f"⚠️ Could not parse vote breakdown: {e}")
    except Exception as e:
        st.error(f"❌ Error displaying votes: {e}")
else:
    st.warning("""
    ⚠️ **Weighted voting data not recorded for this decision.**

    Historical decisions before the schema update will show this message.
    New decisions will have full weighted vote data.
    """)

st.markdown("---")

# === SECTION 2.7: Forensic Context Panel ===

# Helper for NaN safety (needed before forensic context uses it)
def safe_display(value, fallback="N/A"):
    """Convert NaN/None/empty values to a readable fallback."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    if str(value).strip().lower() == 'nan' or str(value).strip() == '':
        return fallback
    return str(value)

st.subheader("🔎 Forensic Context")
st.caption("What triggered this decision and how strong was the conviction pipeline?")

forensic_cols = st.columns(4)

with forensic_cols[0]:
    trigger_raw = safe_display(row.get('trigger_type'), "unknown")
    trigger_lower = trigger_raw.lower().strip()
    if trigger_lower == 'scheduled':
        trigger_badge = "📅 Scheduled"
        trigger_color = "#636EFA"
    elif trigger_lower == 'emergency':
        trigger_badge = "🚨 Emergency"
        trigger_color = "#EF553B"
    else:
        # Sentinel-triggered (e.g., "PriceSentinel")
        trigger_badge = f"📡 {trigger_raw}"
        trigger_color = "#FFA15A"
    st.markdown(f"""
    <div style="background-color: {trigger_color}; padding: 8px 12px;
                border-radius: 15px; color: white; font-weight: bold; text-align: center;">
        {html.escape(trigger_badge)}
    </div>
    """, unsafe_allow_html=True)
    st.caption("Trigger Type")

with forensic_cols[1]:
    catalyst_text = safe_display(row.get('primary_catalyst'), "Not recorded")
    # Truncate to ~100 chars
    if len(catalyst_text) > 100:
        catalyst_text = catalyst_text[:97] + "..."
    st.markdown(f"**{html.escape(catalyst_text)}**")
    st.caption("Primary Catalyst")

with forensic_cols[2]:
    conv_raw = row.get('conviction_multiplier', None)
    conv_val = None
    if conv_raw is not None and conv_raw != '' and pd.notna(conv_raw):
        try:
            conv_val = float(conv_raw)
        except (ValueError, TypeError):
            pass
    if conv_val is not None:
        st.progress(conv_val, text=f"{conv_val:.2f}")
        st.caption("Conviction Pipeline")
        st.markdown(
            "<small>1.0 = full conviction. 0.70 = divergent (partial dampening). 0.50 = high disagreement</small>",
            unsafe_allow_html=True,
            help="1.0 = full conviction (FULL alignment). 0.70 = divergent (partial dampening). 0.50 = high disagreement"
        )
    else:
        st.info("N/A")
        st.caption("Conviction Pipeline")

with forensic_cols[3]:
    thesis_str = safe_display(row.get('thesis_strength'), "UNKNOWN")
    ts_colors = {'PROVEN': '#00CC96', 'PLAUSIBLE': '#FFA15A', 'SPECULATIVE': '#EF553B'}
    ts_color = ts_colors.get(thesis_str, '#888888')
    st.markdown(f"""
    <div style="background-color: {ts_color}; padding: 8px 12px;
                border-radius: 15px; color: white; font-weight: bold; text-align: center;">
        {html.escape(thesis_str)}
    </div>
    """, unsafe_allow_html=True)
    st.caption("Thesis Strength")

st.markdown("---")

# === SECTION 3: Master Decision Details ===
st.subheader("👑 Master Decision")

# Show prediction type badge
pred_type = row.get('prediction_type', 'DIRECTIONAL')
strategy = row.get('strategy_type', 'Unknown')

if pred_type == 'VOLATILITY':
    vol_level = row.get('volatility_level', 'N/A')
    st.markdown(f"#### 🎲 Trade Type: VOLATILITY ({vol_level}) → {strategy}")
else:
    direction = row.get('master_decision', 'NEUTRAL')
    st.markdown(f"#### 📊 Trade Type: DIRECTIONAL ({direction}) → {strategy}")

# === v7.1: Thesis Strength Badge ===
thesis_strength = safe_display(row.get('thesis_strength'), "UNKNOWN")
if thesis_strength != "UNKNOWN":
    thesis_colors = {
        'PROVEN': '#00CC96',      # Green
        'PLAUSIBLE': '#FFA15A',   # Orange
        'SPECULATIVE': '#EF553B', # Red
    }
    thesis_color = thesis_colors.get(thesis_strength, '#888888')
    st.markdown(f"""
    <div style="display: inline-block; background-color: {thesis_color}; padding: 5px 15px;
                border-radius: 20px; color: white; font-weight: bold; margin-bottom: 10px;">
        Thesis: {html.escape(thesis_strength)}
    </div>
    """, unsafe_allow_html=True)

# === v7.1: Primary Catalyst ===
primary_catalyst = safe_display(row.get('primary_catalyst'), "N/A")
st.markdown(f"**🎯 Primary Catalyst:** {primary_catalyst}")

master_cols = st.columns(4)

with master_cols[0]:
    decision = row.get('master_decision', 'N/A')
    color = "#00CC96" if decision == 'BULLISH' else "#EF553B" if decision == 'BEARISH' else "#888888"
    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color: white; margin: 0;">{html.escape(decision)}</h2>
    </div>
    """, unsafe_allow_html=True)

with master_cols[1]:
    confidence = row.get('master_confidence', 0)
    if pd.isna(confidence) or confidence is None:
        confidence = 0.0
    else:
        confidence = float(confidence)
    st.metric("Confidence", f"{confidence:.1%}", help="The Master Strategist's stated confidence in its decision, from 0% to 100%. Impacts position sizing.")

with master_cols[2]:
    approved = row.get('compliance_approved', True)
    status = "✅ Approved" if approved else "❌ Vetoed"
    st.metric("Compliance", status, help="Whether the decision passed all hard-coded risk rails (e.g., VaR limits, invalid contract formats, maximum exposure) before execution.")

with master_cols[3]:
    conviction = row.get('conviction_multiplier', 'N/A')
    if conviction != 'N/A' and pd.notna(conviction):
        try:
            conv_val = round(float(conviction), 2)
            conv_label = {1.0: "✅ Aligned", 0.75: "⚠️ Partial", 0.70: "🔻 Divergent", 0.65: "🔻 Divergent", 0.5: "🔻 Divergent"}.get(conv_val, f"{conv_val:.2f}")
            st.metric("Consensus", conv_label, help="Multiplier applied to trade size based on agent alignment. 1.0 = unanimous consensus; lower values indicate dissent and reduce position size.")
        except (ValueError, TypeError):
            st.metric("Consensus", "N/A", help="Multiplier applied to trade size based on agent alignment.")
    else:
        st.metric("Consensus", "N/A", help="Multiplier applied to trade size based on agent alignment.")

# Reasoning
st.markdown("**Master Reasoning:**")
reasoning = safe_display(row.get('master_reasoning'), "No reasoning recorded for this decision.")
st.info(reasoning)

# === v7.1: Dissent Acknowledged ===
dissent = safe_display(row.get('dissent_acknowledged'), "N/A")
st.markdown(f"**⚖️ Dissent Acknowledged:** {dissent}")

st.markdown("---")

# === SECTION: Decision Outcome ===
actual_trend = row.get('actual_trend_direction', None)

if pd.notna(actual_trend) and actual_trend:
    st.subheader("📊 Decision Outcome")

    outcome_cols = st.columns(4)

    with outcome_cols[0]:
        master = row.get('master_decision', 'NEUTRAL')
        is_correct = (
            (master == 'BULLISH' and actual_trend == 'UP') or
            (master == 'BEARISH' and actual_trend == 'DOWN')
        )
        if master == 'NEUTRAL':
            st.info("➖ NO POSITION")
        elif is_correct:
            st.success("✅ CORRECT")
        else:
            st.error("❌ INCORRECT")

    with outcome_cols[1]:
        trend_icon = "📈" if actual_trend == "UP" else "📉"
        st.metric("Market Move", f"{trend_icon} {actual_trend}")

    with outcome_cols[2]:
        exit_price = row.get('exit_price', None)
        if pd.notna(exit_price):
            st.metric("Exit Price", f"${float(exit_price):.2f}")
        else:
            st.metric("Exit Price", "Pending")

    with outcome_cols[3]:
        pnl = row.get('pnl_realized', None)
        if pd.notna(pnl):
            pnl_val = float(pnl)
            delta_color = "normal" if pnl_val >= 0 else "inverse"
            st.metric("P&L", f"${pnl_val:.2f}", delta_color=delta_color)
        else:
            st.metric("P&L", "Pending")
else:
    st.info("⏳ Decision pending reconciliation (T+1)")

# === Journal Reflection (post-mortem) ===
_pnl_val = row.get('pnl_realized', None)
if pd.notna(actual_trend) and actual_trend and pd.notna(_pnl_val):
    journal = load_trade_journal(ticker=ticker)
    journal_entry = find_journal_entry(journal, row.get('contract', ''), _pnl_val)

    if journal_entry:
        narrative = journal_entry.get('narrative', '')

        if isinstance(narrative, dict):
            # LLM-generated structured reflection
            st.markdown("#### 📓 Trade Reflection")

            summary = narrative.get('summary', '')
            if summary:
                st.markdown(f"**Summary:** {summary}")

            refl_cols = st.columns(2)
            with refl_cols[0]:
                went_right = narrative.get('what_went_right', [])
                if went_right:
                    st.markdown("**What went right:**")
                    for item in went_right:
                        st.markdown(f"- {item}")

            with refl_cols[1]:
                went_wrong = narrative.get('what_went_wrong', [])
                if went_wrong:
                    st.markdown("**What went wrong:**")
                    for item in went_wrong:
                        st.markdown(f"- {item}")

            lesson = narrative.get('lesson', '') or journal_entry.get('key_lesson', '')
            if lesson:
                st.info(f"**Lesson:** {lesson}")

            rule = narrative.get('rule_suggestion', '')
            if rule:
                st.caption(f"Rule suggestion: {rule}")
        elif narrative:
            # Template narrative (no LLM)
            st.markdown("#### 📓 Trade Reflection")
            st.markdown(narrative)
            key_lesson = journal_entry.get('key_lesson', '')
            if key_lesson:
                st.info(f"**Lesson:** {key_lesson}")

# === Dissent Outcome Tracking ===
try:
    if 'dissent_acknowledged' in council_df.columns and 'actual_trend_direction' in council_df.columns:
        # Find rows where dissent was present and outcome is known
        dissent_mask = council_df['dissent_acknowledged'].notna() & (council_df['dissent_acknowledged'] != '') & (council_df['dissent_acknowledged'] != 'N/A')
        outcome_mask = council_df['actual_trend_direction'].notna() & (council_df['actual_trend_direction'] != '')
        dissent_graded = council_df[dissent_mask & outcome_mask].copy()

        if len(dissent_graded) >= 5:
            # Compute win/loss when dissent was overruled
            dissent_wins = 0
            dissent_losses = 0
            for _, drow in dissent_graded.iterrows():
                d_master = drow.get('master_decision', 'NEUTRAL')
                d_trend = drow.get('actual_trend_direction', '')
                d_correct = (
                    (d_master == 'BULLISH' and d_trend == 'UP') or
                    (d_master == 'BEARISH' and d_trend == 'DOWN')
                )
                if d_master != 'NEUTRAL':
                    if d_correct:
                        dissent_wins += 1
                    else:
                        dissent_losses += 1

            total_dissent = dissent_wins + dissent_losses
            if total_dissent > 0:
                pct = (dissent_wins / total_dissent) * 100
                st.caption(f"Dissent overruled correctly: {dissent_wins}/{total_dissent} times ({pct:.0f}%)")
        elif len(dissent_graded) > 0:
            st.caption("Insufficient dissent data for analysis (need 5+ graded decisions with dissent)")
except Exception:
    pass  # Silently skip if data isn't available

# --- Cross-page Navigation ---
try:
    _nav_cols = st.columns(3)
    with _nav_cols[0]:
        st.page_link("pages/2_The_Scorecard.py", label="View in Scorecard", icon="📊")
    with _nav_cols[1]:
        st.page_link("pages/6_Signal_Overlay.py", label="View Price Action", icon="📈")
    with _nav_cols[2]:
        st.page_link("pages/7_Brier_Analysis.py", label="Agent Accuracy", icon="🎯")
except Exception:
    pass  # st.page_link may not be available in older Streamlit versions

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

        # UX: Color-coded container based on sentiment
        s_upper = str(sentiment).upper()
        if 'BULL' in s_upper:
            st.success(summary)
        elif 'BEAR' in s_upper:
            st.error(summary)
        else:
            st.info(summary)

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

# === SECTION 6: Prompt Provenance ===
st.markdown("---")
st.subheader("Prompt Provenance")
st.caption("Model routing fidelity, prompt source tracking, and persona drift detection")

traces_df = load_prompt_traces(ticker=ticker)
if traces_df.empty:
    st.info("No prompt trace data available yet. Traces are recorded during trading cycles.")
else:
    # Link prompt provenance to the SELECTED decision, not just the latest cycle
    _selected_cycle_id = row.get('cycle_id', None) if 'row' in dir() else None
    with st.expander("Agent Prompt Configuration", expanded=False):
        latest_cycle = (
            _selected_cycle_id
            if (_selected_cycle_id and 'cycle_id' in traces_df.columns
                and _selected_cycle_id in traces_df['cycle_id'].values)
            else (traces_df['cycle_id'].iloc[-1] if 'cycle_id' in traces_df.columns and len(traces_df) > 0 else None)
        )
        if latest_cycle:
            cycle_traces = traces_df[traces_df['cycle_id'] == latest_cycle]
            display_cols = ['agent', 'phase', 'prompt_source', 'model_provider', 'model_name',
                            'assigned_provider', 'assigned_model', 'persona_hash', 'demo_count',
                            'latency_ms']
            available_cols = [c for c in display_cols if c in cycle_traces.columns]
            st.dataframe(cycle_traces[available_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No cycle data found.")

    with st.expander("Model Routing Fidelity", expanded=False):
        if 'assigned_provider' in traces_df.columns and 'model_provider' in traces_df.columns:
            routed = traces_df[traces_df['assigned_provider'] != '']
            total = len(routed)
            if total > 0:
                matched = len(routed[routed['assigned_provider'] == routed['model_provider']])
                fallback_count = total - matched
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Routed Calls", total)
                col2.metric("Primary Success", matched)
                col3.metric("Fallbacks", fallback_count,
                            delta=f"-{fallback_count}" if fallback_count > 0 else None,
                            delta_color="inverse")
                if fallback_count > 0:
                    fallbacks = routed[routed['assigned_provider'] != routed['model_provider']]
                    st.dataframe(
                        fallbacks[['agent', 'assigned_provider', 'assigned_model',
                                   'model_provider', 'model_name']],
                        use_container_width=True, hide_index=True
                    )
            else:
                st.info("No routed calls with assignment data yet.")

    with st.expander("Prompt Source Trends", expanded=False):
        if 'prompt_source' in traces_df.columns:
            source_counts = traces_df['prompt_source'].value_counts()
            if not source_counts.empty:
                fig = px.pie(
                    values=source_counts.values,
                    names=source_counts.index,
                    title="Prompt Source Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("Persona Drift Detection", expanded=False):
        if 'persona_hash' in traces_df.columns and 'agent' in traces_df.columns:
            agent_traces = traces_df[traces_df['persona_hash'] != '']
            if not agent_traces.empty:
                drift = agent_traces.groupby('agent')['persona_hash'].nunique().reset_index()
                drift.columns = ['Agent', 'Unique Persona Hashes']
                drifted = drift[drift['Unique Persona Hashes'] > 1]
                if not drifted.empty:
                    st.warning(f"{len(drifted)} agent(s) have changed persona text across cycles:")
                    st.dataframe(drifted, use_container_width=True, hide_index=True)
                else:
                    st.success("All agent personas are stable (single hash per agent).")
            else:
                st.info("No persona hash data available.")
