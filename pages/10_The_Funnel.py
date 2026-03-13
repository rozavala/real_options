"""
Page 10: The Funnel (Execution Diagnostics)

Purpose: Visualize the signal-to-P&L pipeline, identify where alpha leaks,
and quantify dollar impact. Bridges the intelligence layer (Scorecard) with
the financial layer (Trade Analytics).

Data source: data/{TICKER}/execution_funnel.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import (
    load_council_history,
    get_config,
    _resolve_data_path_for,
)
from _date_filter import date_range_picker, apply_date_filter

# --- Page Config ---
st.set_page_config(page_title="The Funnel", page_icon="🔬", layout="wide")
st.title("🔬 The Funnel — Signal-to-P&L Diagnostic")
st.caption("Where does alpha leak? Track every signal from council decision through execution to P&L.")


# --- Data Loading ---
@st.cache_data(ttl=60)
def load_funnel_data(ticker: str) -> pd.DataFrame:
    """Load execution funnel data for a commodity."""
    path = _resolve_data_path_for(ticker, 'execution_funnel.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_order_events(ticker: str) -> pd.DataFrame:
    """Load order events for slippage analysis."""
    path = _resolve_data_path_for(ticker, 'order_events.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()


# --- Sidebar Filters ---
from _commodity_selector import selected_commodity
ticker = selected_commodity()

funnel_df = load_funnel_data(ticker)
council_df = load_council_history(ticker)

if funnel_df.empty and council_df.empty:
    st.warning(f"No funnel or council data for {ticker}. Run the backfill script or wait for real-time data to accumulate.")
    st.code("python scripts/backfill_execution_funnel.py --all --data-dir /home/rodrigo/real_options/data")
    st.stop()

# Date filter — one picker, applied to both DataFrames (same pattern as Financials)
_anchor_df = funnel_df if not funnel_df.empty else council_df
_funnel_dates = date_range_picker(_anchor_df, key='funnel')
if _funnel_dates:
    funnel_df = apply_date_filter(funnel_df, *_funnel_dates)
    council_df = apply_date_filter(council_df, *_funnel_dates)

# Source filter
if not funnel_df.empty and 'source' in funnel_df.columns:
    sources = ['ALL'] + sorted(funnel_df['source'].dropna().unique().tolist())
    selected_source = st.sidebar.selectbox("Data Source", sources, index=0)
    if selected_source != 'ALL':
        funnel_df = funnel_df[funnel_df['source'] == selected_source]

# Regime filter
if not funnel_df.empty and 'regime' in funnel_df.columns:
    regimes = ['ALL'] + sorted(funnel_df['regime'].dropna().unique().tolist())
    selected_regime = st.sidebar.selectbox("Regime Filter", regimes, index=0)
    if selected_regime != 'ALL':
        funnel_df = funnel_df[funnel_df['regime'] == selected_regime]


# ============================================================
# ROW 1: KPI CARDS
# ============================================================
st.subheader("Key Metrics")

# Calculate funnel metrics
def calc_funnel_kpis(df: pd.DataFrame, ch: pd.DataFrame) -> dict:
    """Calculate KPI metrics from funnel and council data."""
    kpis = {}

    if not df.empty and 'stage' in df.columns:
        decisions = df[df['stage'] == 'COUNCIL_DECISION']
        actionable = decisions[decisions['outcome'] == 'PASS']
        filled = df[df['stage'] == 'ORDER_FILLED']
        cancelled = df[df['stage'] == 'ORDER_CANCELLED']
        placed = df[df['stage'] == 'ORDER_PLACED']

        # Signal-to-Trade %
        n_actionable = len(actionable)
        n_filled = len(filled)
        kpis['signal_to_trade_pct'] = (n_filled / n_actionable * 100) if n_actionable > 0 else 0

        # Fill Rate (of placed orders)
        n_placed = len(placed)
        kpis['fill_rate_pct'] = (n_filled / n_placed * 100) if n_placed > 0 else 0

        # Avg Slippage %
        filled_with_prices = filled.dropna(subset=['fill_price', 'initial_limit'])
        if not filled_with_prices.empty:
            slippage = (filled_with_prices['fill_price'].astype(float) - filled_with_prices['initial_limit'].astype(float)).abs()
            initial = filled_with_prices['initial_limit'].astype(float).abs()
            slippage_pct = (slippage / initial.replace(0, np.nan) * 100).dropna()
            kpis['avg_slippage_pct'] = slippage_pct.mean() if not slippage_pct.empty else 0
        else:
            kpis['avg_slippage_pct'] = 0

        # Conviction gate block rate
        conviction_blocks = df[(df['stage'] == 'CONVICTION_GATE') & (df['outcome'] == 'BLOCK')]
        conviction_total = df[df['stage'] == 'CONVICTION_GATE']
        kpis['conviction_block_pct'] = (len(conviction_blocks) / len(conviction_total) * 100) if len(conviction_total) > 0 else 0

        # Walk steps (avg per order)
        walk_steps = df[df['stage'] == 'PRICE_WALK_STEP']
        kpis['avg_walk_steps'] = (len(walk_steps) * 3 / n_placed) if n_placed > 0 else 0  # *3 because we log every 3rd

    else:
        kpis['signal_to_trade_pct'] = 0
        kpis['fill_rate_pct'] = 0
        kpis['avg_slippage_pct'] = 0
        kpis['conviction_block_pct'] = 0
        kpis['avg_walk_steps'] = 0

    # Signal win rate vs Trade win rate from council_history
    if not ch.empty and 'pnl_realized' in ch.columns:
        resolved = ch[ch['pnl_realized'].notna()]
        kpis['signal_win_rate'] = (resolved['pnl_realized'] > 0).mean() * 100 if not resolved.empty else 0
        kpis['n_resolved'] = len(resolved)
    else:
        kpis['signal_win_rate'] = 0
        kpis['n_resolved'] = 0

    return kpis


kpis = calc_funnel_kpis(funnel_df, council_df)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Signal-to-Trade", f"{kpis['signal_to_trade_pct']:.0f}%",
          help="% of actionable council signals that resulted in filled orders")
k2.metric("Fill Rate", f"{kpis['fill_rate_pct']:.0f}%",
          help="% of placed orders that filled (vs timed out/cancelled)")
k3.metric("Avg Slippage", f"{kpis['avg_slippage_pct']:.1f}%",
          help="Average % difference between initial limit and fill price")
k4.metric("Conviction Blocks", f"{kpis['conviction_block_pct']:.0f}%",
          help="% of directional signals blocked by conviction gate")
k5.metric("Avg Walk Steps", f"{kpis['avg_walk_steps']:.0f}",
          help="Average adaptive walk steps per placed order (est.)")
k6.metric("Signal Win Rate", f"{kpis['signal_win_rate']:.0f}%",
          help=f"Directional accuracy of resolved signals (n={kpis['n_resolved']})")


# ============================================================
# ROW 2: FUNNEL WATERFALL
# ============================================================
st.subheader("Funnel Waterfall — Where Do Signals Die?")

if not funnel_df.empty and 'stage' in funnel_df.columns:
    # Define the funnel stages in order
    STAGE_ORDER = [
        'COUNCIL_DECISION', 'CONVICTION_GATE', 'COMPLIANCE_AUDIT',
        'CONFIDENCE_THRESHOLD', 'THESIS_COHERENCE', 'CAPITAL_CHECK',
        'ORDER_QUEUED', 'DRAWDOWN_GATE', 'LIQUIDITY_GATE',
        'ORDER_PLACED', 'ORDER_FILLED',
    ]

    waterfall_data = []
    for stage in STAGE_ORDER:
        stage_df = funnel_df[funnel_df['stage'] == stage]
        passed = len(stage_df[stage_df['outcome'] == 'PASS'])
        blocked = len(stage_df[stage_df['outcome'] == 'BLOCK'])
        total = passed + blocked
        waterfall_data.append({
            'Stage': stage.replace('_', ' ').title(),
            'Passed': passed,
            'Blocked': blocked,
            'Total': total,
        })

    wf_df = pd.DataFrame(waterfall_data)
    wf_df = wf_df[wf_df['Total'] > 0]  # Only show stages with data

    if not wf_df.empty:
        fig_waterfall = go.Figure()
        fig_waterfall.add_trace(go.Bar(
            x=wf_df['Stage'], y=wf_df['Passed'],
            name='Passed', marker_color='#2ecc71',
            text=wf_df['Passed'], textposition='auto',
        ))
        fig_waterfall.add_trace(go.Bar(
            x=wf_df['Stage'], y=wf_df['Blocked'],
            name='Blocked', marker_color='#e74c3c',
            text=wf_df['Blocked'], textposition='auto',
        ))
        fig_waterfall.update_layout(
            barmode='stack',
            xaxis_tickangle=-45,
            height=400,
            margin=dict(t=20, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Survival rate
        first_stage = wf_df.iloc[0]['Total'] if len(wf_df) > 0 else 1
        last_pass = wf_df.iloc[-1]['Passed'] if len(wf_df) > 0 else 0
        st.caption(f"End-to-end survival: {last_pass}/{first_stage} ({last_pass/max(first_stage,1)*100:.0f}%)")
    else:
        st.info("No stage data available for waterfall chart.")
else:
    # Fallback: build from council_history
    st.info("No real-time funnel data yet. Showing summary from council history.")
    if not council_df.empty:
        total_decisions = len(council_df)
        actionable = len(council_df[council_df['master_decision'].isin(['BULLISH', 'BEARISH'])])
        with_strategy = len(council_df[council_df['strategy_type'].notna() & (council_df['strategy_type'] != 'NONE')])
        compliant = len(council_df[council_df['compliance_approved'] == True])

        summary_data = {
            'Stage': ['Council Decisions', 'Actionable (Bull/Bear)', 'Strategy Selected', 'Compliance Approved'],
            'Count': [total_decisions, actionable, with_strategy, compliant],
        }
        st.bar_chart(pd.DataFrame(summary_data).set_index('Stage'))


# ============================================================
# ROW 3: SIGNAL vs OUTCOME MATRIX
# ============================================================
st.subheader("Signal vs Outcome — Skill or Luck?")

if not council_df.empty and 'pnl_realized' in council_df.columns and 'weighted_score' in council_df.columns:
    resolved = council_df[council_df['pnl_realized'].notna()].copy()
    if not resolved.empty:
        resolved['process_score'] = resolved['master_confidence'].fillna(0.5) * resolved['weighted_score'].abs().fillna(0)
        resolved['pnl'] = resolved['pnl_realized'].astype(float)

        # Classify quadrants
        process_median = resolved['process_score'].median()
        conditions = [
            (resolved['process_score'] >= process_median) & (resolved['pnl'] > 0),
            (resolved['process_score'] < process_median) & (resolved['pnl'] > 0),
            (resolved['process_score'] >= process_median) & (resolved['pnl'] <= 0),
            (resolved['process_score'] < process_median) & (resolved['pnl'] <= 0),
        ]
        labels = ['Skill', 'Lucky', 'Execution Leak', 'Bad Call']
        resolved['quadrant'] = np.select(conditions, labels, default='Unknown')

        fig_matrix = px.scatter(
            resolved,
            x='process_score',
            y='pnl',
            color='quadrant',
            size=resolved['pnl'].abs().clip(lower=0.1),
            hover_data=['cycle_id', 'contract', 'strategy_type', 'master_decision'],
            color_discrete_map={
                'Skill': '#2ecc71', 'Lucky': '#f39c12',
                'Execution Leak': '#e74c3c', 'Bad Call': '#95a5a6',
            },
            labels={'process_score': 'Process Score (confidence x |weighted_score|)', 'pnl': 'P&L (cents)'},
        )
        fig_matrix.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_matrix.add_vline(x=process_median, line_dash="dash", line_color="gray", opacity=0.5)
        fig_matrix.update_layout(height=450, margin=dict(t=20))
        st.plotly_chart(fig_matrix, use_container_width=True)

        # Quadrant summary
        quad_counts = resolved['quadrant'].value_counts()
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Skill", quad_counts.get('Skill', 0), help="Good process + positive P&L")
        qc2.metric("Lucky", quad_counts.get('Lucky', 0), help="Weak process + positive P&L")
        qc3.metric("Execution Leak", quad_counts.get('Execution Leak', 0), help="Good process + negative P&L")
        qc4.metric("Bad Call", quad_counts.get('Bad Call', 0), help="Weak process + negative P&L")
else:
    st.info("Insufficient council history data for matrix analysis.")


# ============================================================
# ROW 4: EXECUTION EFFICIENCY + POSITION LIFECYCLE
# ============================================================
tab_exec, tab_lifecycle = st.tabs(["Execution Efficiency", "Position Lifecycle"])

with tab_exec:
    if not funnel_df.empty and 'stage' in funnel_df.columns:
        # Slippage distribution
        filled = funnel_df[funnel_df['stage'] == 'ORDER_FILLED'].copy()
        if not filled.empty and 'fill_price' in filled.columns and 'initial_limit' in filled.columns:
            filled_clean = filled.dropna(subset=['fill_price', 'initial_limit'])
            if not filled_clean.empty:
                filled_clean['slippage'] = (filled_clean['fill_price'].astype(float) - filled_clean['initial_limit'].astype(float))
                filled_clean['slippage_pct'] = (filled_clean['slippage'].abs() / filled_clean['initial_limit'].astype(float).abs().replace(0, np.nan) * 100)

                st.markdown("**Slippage Distribution (fill vs initial limit)**")
                fig_slip = px.histogram(
                    filled_clean, x='slippage_pct',
                    nbins=20, color_discrete_sequence=['#3498db'],
                    labels={'slippage_pct': 'Slippage %'},
                )
                fig_slip.update_layout(height=300, margin=dict(t=20))
                st.plotly_chart(fig_slip, use_container_width=True)

        # Top unfilled orders
        cancelled = funnel_df[funnel_df['stage'] == 'ORDER_CANCELLED'].copy()
        if not cancelled.empty:
            st.markdown("**Recent Unfilled Orders**")
            display_cols = ['timestamp', 'contract', 'detail', 'walk_away_price', 'initial_limit']
            avail_cols = [c for c in display_cols if c in cancelled.columns]
            st.dataframe(
                cancelled[avail_cols].tail(10).sort_values('timestamp', ascending=False),
                hide_index=True, use_container_width=True,
            )
    else:
        st.info("No execution data yet. Funnel events will appear after the next trading cycle.")

with tab_lifecycle:
    if not funnel_df.empty and 'stage' in funnel_df.columns:
        # Exit reason distribution
        closed = funnel_df[funnel_df['stage'] == 'POSITION_CLOSED'].copy()
        if not closed.empty and 'detail' in closed.columns:
            # Extract exit reason from detail
            closed['exit_reason'] = closed['detail'].str.extract(r'exit_reason=([^,]+)', expand=False).fillna('Unknown')
            reason_counts = closed['exit_reason'].value_counts()

            if not reason_counts.empty:
                st.markdown("**Exit Reason Distribution**")
                fig_exit = px.pie(
                    values=reason_counts.values,
                    names=reason_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_exit.update_layout(height=350, margin=dict(t=20))
                st.plotly_chart(fig_exit, use_container_width=True)
        else:
            st.info("No position close events recorded yet.")

        # Risk triggers
        risk_events = funnel_df[funnel_df['stage'] == 'RISK_TRIGGER']
        if not risk_events.empty:
            st.markdown("**Recent Risk Triggers**")
            display_cols = ['timestamp', 'contract', 'detail']
            avail_cols = [c for c in display_cols if c in risk_events.columns]
            st.dataframe(
                risk_events[avail_cols].tail(10).sort_values('timestamp', ascending=False),
                hide_index=True, use_container_width=True,
            )
    else:
        st.info("No lifecycle data yet.")


# ============================================================
# ROW 5: TOP LEAKS TABLE
# ============================================================
st.subheader("Top Alpha Leaks — Ranked by Impact")

if not funnel_df.empty and 'stage' in funnel_df.columns:
    leak_data = []

    # Leak 1: Unfilled orders (opportunity cost)
    n_cancelled = len(funnel_df[funnel_df['stage'] == 'ORDER_CANCELLED'])
    n_placed = len(funnel_df[funnel_df['stage'] == 'ORDER_PLACED'])
    if n_placed > 0:
        leak_data.append({
            'Leak Category': 'Unfilled Orders',
            'Count': n_cancelled,
            '% of Pipeline': f"{n_cancelled/max(n_placed,1)*100:.0f}%",
            'Diagnosis': 'Adaptive walk exhausted without fill',
            'Config Key': 'strategy.adaptive_walk.*',
        })

    # Leak 2: Conviction gate blocks
    n_conv_block = len(funnel_df[(funnel_df['stage'] == 'CONVICTION_GATE') & (funnel_df['outcome'] == 'BLOCK')])
    n_conv_total = len(funnel_df[funnel_df['stage'] == 'CONVICTION_GATE'])
    if n_conv_total > 0:
        leak_data.append({
            'Leak Category': 'Conviction Gate',
            'Count': n_conv_block,
            '% of Pipeline': f"{n_conv_block/max(n_conv_total,1)*100:.0f}%",
            'Diagnosis': 'Weighted score below threshold',
            'Config Key': 'strategy.min_weighted_score_magnitude',
        })

    # Leak 3: Confidence threshold blocks
    n_conf_block = len(funnel_df[(funnel_df['stage'] == 'CONFIDENCE_THRESHOLD') & (funnel_df['outcome'] == 'BLOCK')])
    if n_conf_block > 0:
        leak_data.append({
            'Leak Category': 'Confidence Threshold',
            'Count': n_conf_block,
            '% of Pipeline': f"{n_conf_block/len(funnel_df)*100:.0f}%",
            'Diagnosis': 'Signal confidence below min_confidence_threshold',
            'Config Key': 'risk_management.min_confidence_threshold',
        })

    # Leak 4: Compliance blocks
    n_compl_block = len(funnel_df[(funnel_df['stage'] == 'COMPLIANCE_AUDIT') & (funnel_df['outcome'] == 'BLOCK')])
    if n_compl_block > 0:
        leak_data.append({
            'Leak Category': 'Compliance Veto',
            'Count': n_compl_block,
            '% of Pipeline': f"{n_compl_block/len(funnel_df)*100:.0f}%",
            'Diagnosis': 'Hallucination check or risk limit breach',
            'Config Key': 'compliance.*',
        })

    # Leak 5: Liquidity gate blocks
    n_liq_block = len(funnel_df[(funnel_df['stage'] == 'LIQUIDITY_GATE') & (funnel_df['outcome'] == 'BLOCK')])
    if n_liq_block > 0:
        leak_data.append({
            'Leak Category': 'Liquidity Gate',
            'Count': n_liq_block,
            '% of Pipeline': f"{n_liq_block/len(funnel_df)*100:.0f}%",
            'Diagnosis': 'Bid-ask spread too wide or volume too low',
            'Config Key': 'risk_management.max_spread_pct',
        })

    # Leak 6: Drawdown gate blocks
    n_dd_block = len(funnel_df[(funnel_df['stage'] == 'DRAWDOWN_GATE') & (funnel_df['outcome'] == 'BLOCK')])
    if n_dd_block > 0:
        leak_data.append({
            'Leak Category': 'Drawdown Circuit Breaker',
            'Count': n_dd_block,
            '% of Pipeline': f"{n_dd_block/len(funnel_df)*100:.0f}%",
            'Diagnosis': 'Cumulative drawdown exceeded threshold',
            'Config Key': 'drawdown_circuit_breaker.*',
        })

    if leak_data:
        leak_df = pd.DataFrame(leak_data).sort_values('Count', ascending=False)
        st.dataframe(leak_df, hide_index=True, use_container_width=True)
    else:
        st.success("No significant leaks detected in the current period.")
else:
    st.info("Leak analysis requires funnel event data. Will populate after trading cycles run with instrumentation.")


# ============================================================
# RAW DATA EXPLORER
# ============================================================
with st.expander("Raw Funnel Data", expanded=False):
    if not funnel_df.empty:
        # Stage filter
        stages = ['ALL'] + sorted(funnel_df['stage'].dropna().unique().tolist())
        sel_stage = st.selectbox("Filter by Stage", stages)
        display_df = funnel_df if sel_stage == 'ALL' else funnel_df[funnel_df['stage'] == sel_stage]

        st.dataframe(
            display_df.sort_values('timestamp', ascending=False).head(200),
            hide_index=True, use_container_width=True,
        )
        st.download_button(
            "Download Funnel CSV",
            display_df.to_csv(index=False).encode('utf-8'),
            f"execution_funnel_{ticker}.csv",
            "text/csv",
        )
    else:
        st.info("No funnel data available.")
