"""
Page 10: The Funnel (Execution Diagnostics)

Purpose: Visualize the signal-to-P&L pipeline, identify where alpha leaks,
and quantify dollar impact. Bridges the intelligence layer (Scorecard) with
the financial layer (Trade Analytics).

Data sources:
  - council_history.csv: Intelligence + gates (decisions, compliance, conviction, strategy, P&L)
  - execution_funnel.csv: Execution (order placed, price walk, fill, cancel)
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
    path = _resolve_data_path_for('execution_funnel.csv', ticker)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            # format='mixed' required: backfill rows have tz-aware timestamps
            # while realtime execution events (ORDER_PLACED, PRICE_WALK_STEP) are naive
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True, errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_order_events(ticker: str) -> pd.DataFrame:
    """Load order events for slippage analysis."""
    path = _resolve_data_path_for('order_events.csv', ticker)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True, errors='coerce')
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


# --- Canonical stage ordering for dynamic sections ---
STAGE_SORT_ORDER = {
    'COUNCIL_DECISION': 0,
    'CONVICTION_GATE': 1,
    'COMPLIANCE_AUDIT': 2,
    'DA_REVIEW': 3,
    'CONFIDENCE_THRESHOLD': 4,
    'THESIS_COHERENCE': 5,
    'CAPITAL_CHECK': 6,
    'STRATEGY_SELECTION': 7,
    'ORDER_QUEUED': 8,
    'DRAWDOWN_GATE': 9,
    'LIQUIDITY_GATE': 10,
    'ORDER_PLACED': 11,
    'PRICE_WALK_STEP': 12,
    'ORDER_FILLED': 13,
    'ORDER_PARTIAL_FILL': 14,
    'ORDER_CANCELLED': 15,
    'POSITION_OPENED': 16,
    'RISK_TRIGGER': 17,
    'POSITION_CLOSED': 18,
    'PNL_RECONCILED': 19,
    'TMS_ORPHAN_DETECTED': 20,
}


def build_dynamic_stage_order(df: pd.DataFrame) -> list:
    """Build stage order from stages actually present in data, sorted canonically."""
    if df.empty or 'stage' not in df.columns:
        return []
    present = df['stage'].dropna().unique().tolist()
    return sorted(present, key=lambda s: STAGE_SORT_ORDER.get(s, 999))


# --- Cascading Funnel Builder ---
def build_true_funnel(council_df: pd.DataFrame, funnel_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Build a monotonically-decreasing cascading funnel from two data sources.

    Top half (Intelligence + Gates): derived from council_history columns.
    Bottom half (Execution): derived from execution_funnel.csv aggregate stage counts.
    Final stage (P&L): derived from council_history pnl_realized.

    Returns DataFrame with columns: stage, survivors, drop, drop_pct, source_label
    """
    min_score = config.get('strategy', {}).get('min_weighted_score_magnitude', 0.20)

    stages = []

    # 1. All Council Decisions
    n_total = len(council_df)
    stages.append({'stage': 'Council Decisions', 'survivors': n_total,
                   'source_label': 'council_history'})

    # 2. Actionable (BULLISH/BEARISH)
    if 'master_decision' in council_df.columns:
        actionable = council_df[council_df['master_decision'].isin(['BULLISH', 'BEARISH'])]
    else:
        actionable = council_df
    n_actionable = len(actionable)
    stages.append({'stage': 'Actionable (Bull/Bear)', 'survivors': n_actionable,
                   'source_label': 'council_history'})

    # 3. Compliance Passed (subset of actionable)
    if 'compliance_approved' in actionable.columns:
        compliant = actionable[
            actionable['compliance_approved'].astype(str).str.lower().isin(['true', '1', 'yes'])
        ]
    else:
        compliant = actionable
    n_compliant = len(compliant)
    stages.append({'stage': 'Compliance Passed', 'survivors': n_compliant,
                   'source_label': 'council_history'})

    # 4. Conviction Gate Passed (subset of compliant)
    if 'weighted_score' in compliant.columns:
        ws = pd.to_numeric(compliant['weighted_score'], errors='coerce').fillna(0)
        convicted = compliant[ws.abs() >= min_score]
    else:
        convicted = compliant
    n_convicted = len(convicted)
    stages.append({'stage': f'Conviction Gate (\u2265{min_score})', 'survivors': n_convicted,
                   'source_label': 'council_history'})

    # 5. Strategy Selected (subset of convicted)
    if 'strategy_type' in convicted.columns:
        strategized = convicted[
            convicted['strategy_type'].notna() &
            ~convicted['strategy_type'].isin(['NONE', '', 'N/A'])
        ]
    else:
        strategized = convicted
    n_strategy = len(strategized)
    stages.append({'stage': 'Strategy Selected', 'survivors': n_strategy,
                   'source_label': 'council_history'})

    # 6. Orders Placed (from execution_funnel.csv)
    n_placed = 0
    if not funnel_df.empty and 'stage' in funnel_df.columns:
        n_placed = len(funnel_df[funnel_df['stage'] == 'ORDER_PLACED'])
    stages.append({'stage': 'Orders Placed', 'survivors': n_placed,
                   'source_label': 'execution_funnel'})

    # 7. Orders Filled
    n_filled = 0
    if not funnel_df.empty and 'stage' in funnel_df.columns:
        n_filled = len(funnel_df[funnel_df['stage'] == 'ORDER_FILLED'])
    stages.append({'stage': 'Orders Filled', 'survivors': n_filled,
                   'source_label': 'execution_funnel'})

    # 8. P&L Resolved (from council_history)
    n_pnl = 0
    n_profitable = 0
    if 'pnl_realized' in council_df.columns:
        pnl_series = pd.to_numeric(council_df['pnl_realized'], errors='coerce')
        n_pnl = int(pnl_series.notna().sum())
        n_profitable = int((pnl_series > 0).sum())
    stages.append({'stage': 'P&L Resolved', 'survivors': n_pnl,
                   'source_label': 'council_history',
                   '_n_profitable': n_profitable})

    # Build DataFrame with drop calculations
    result = pd.DataFrame(stages)
    result['drop'] = -result['survivors'].diff().fillna(0).astype(int).clip(upper=0)
    result.loc[0, 'drop'] = 0
    result['drop_pct'] = 0.0
    for i in range(1, len(result)):
        prev = result.iloc[i - 1]['survivors']
        if prev > 0:
            result.loc[result.index[i], 'drop_pct'] = (
                (prev - result.iloc[i]['survivors']) / prev * 100
            )

    return result


# ============================================================
# Compute funnel cascade once — reused by waterfall + leak table
# ============================================================
config = get_config()
funnel_cascade = build_true_funnel(council_df, funnel_df, config)


# ============================================================
# ROW 1: KPI CARDS
# ============================================================
st.subheader("📈 Key Metrics")

def calc_funnel_kpis(df: pd.DataFrame, ch: pd.DataFrame) -> dict:
    """Calculate KPI metrics from funnel and council data."""
    kpis = {}

    if not df.empty and 'stage' in df.columns:
        decisions = df[df['stage'] == 'COUNCIL_DECISION']
        actionable = decisions[decisions['outcome'] == 'PASS']
        filled = df[df['stage'] == 'ORDER_FILLED']
        cancelled = df[df['stage'] == 'ORDER_CANCELLED']
        placed = df[df['stage'] == 'ORDER_PLACED']

        n_actionable = len(actionable)
        n_filled = len(filled)
        n_placed = len(placed)
        n_cancelled = len(cancelled)

        # Signal-to-Trade %
        kpis['signal_to_trade_pct'] = (n_filled / n_actionable * 100) if n_actionable > 0 else 0

        # Fill Rate (of placed orders)
        kpis['fill_rate_pct'] = (n_filled / n_placed * 100) if n_placed > 0 else 0

        # Avg Slippage (relative to credit/debit)
        filled_with_prices = filled.dropna(subset=['fill_price', 'initial_limit'])
        if not filled_with_prices.empty:
            fill_p = filled_with_prices['fill_price'].astype(float)
            init_p = filled_with_prices['initial_limit'].astype(float)
            slippage = (fill_p - init_p).abs()
            slippage_pct = (slippage / init_p.abs().replace(0, np.nan) * 100).dropna()
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

        # Alpha left on table: unfilled orders that passed all gates
        kpis['alpha_left_count'] = n_cancelled
        kpis['alpha_left_pct'] = (n_cancelled / n_placed * 100) if n_placed > 0 else 0

    else:
        kpis['signal_to_trade_pct'] = 0
        kpis['fill_rate_pct'] = 0
        kpis['avg_slippage_pct'] = 0
        kpis['conviction_block_pct'] = 0
        kpis['avg_walk_steps'] = 0
        kpis['alpha_left_count'] = 0
        kpis['alpha_left_pct'] = 0

    # Signal win rate from council_history
    if not ch.empty and 'pnl_realized' in ch.columns:
        resolved = ch[ch['pnl_realized'].notna()]
        kpis['signal_win_rate'] = (resolved['pnl_realized'] > 0).mean() * 100 if not resolved.empty else 0
        kpis['n_resolved'] = len(resolved)
    else:
        kpis['signal_win_rate'] = 0
        kpis['n_resolved'] = 0

    return kpis


kpis = calc_funnel_kpis(funnel_df, council_df)

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("📡 Signal-to-Trade", f"{kpis['signal_to_trade_pct']:.0f}%",
          help="% of actionable council signals that resulted in filled orders")
k2.metric("⛽ Fill Rate", f"{kpis['fill_rate_pct']:.0f}%",
          help="% of placed orders that filled (vs timed out/cancelled)")
k3.metric("📉 Avg Slippage", f"{kpis['avg_slippage_pct']:.1f}%",
          help="Average slippage as % of credit/debit (fill vs initial limit)")
k4.metric("🛡️ Conviction Blocks", f"{kpis['conviction_block_pct']:.0f}%",
          help="% of directional signals blocked by conviction gate")
k5.metric("👣 Avg Walk Steps", f"{kpis['avg_walk_steps']:.0f}",
          help="Average adaptive walk steps per placed order (est.)")
k6.metric("🎯 Signal Win Rate", f"{kpis['signal_win_rate']:.0f}%",
          help=f"Directional accuracy of resolved signals (n={kpis['n_resolved']})")
k7.metric("💸 Alpha Left on Table", f"{kpis['alpha_left_count']}",
          delta=f"-{kpis['alpha_left_pct']:.0f}% of placed" if kpis['alpha_left_count'] > 0 else None,
          delta_color="inverse",
          help="Orders that passed all gates but failed to fill — potential alpha lost to adaptive walk timeout")


# ============================================================
# ROW 2: FUNNEL WATERFALL (Cascading)
# ============================================================
st.subheader("🌊 Funnel Waterfall — Where Do Signals Die?")

if not funnel_cascade.empty and funnel_cascade['survivors'].sum() > 0:
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_cascade['stage'],
        x=funnel_cascade['survivors'],
        textinfo="value+percent initial+percent previous",
        marker=dict(
            color=[
                '#2ecc71', '#27ae60', '#1abc9c', '#16a085',  # Green: intelligence gates
                '#2980b9',                                     # Blue: strategy
                '#3498db', '#5dade2',                          # Light blue: execution
                '#f39c12',                                     # Gold: P&L
            ][:len(funnel_cascade)],
        ),
        connector=dict(line=dict(color="gray", dash="dot", width=1)),
    ))
    fig_funnel.update_layout(
        height=450,
        margin=dict(t=20, b=20, l=10, r=10),
        funnelmode="stack",
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    # Survival summary
    first_count = funnel_cascade.iloc[0]['survivors']
    filled_row = funnel_cascade[funnel_cascade['stage'] == 'Orders Filled']
    filled_count = int(filled_row['survivors'].values[0]) if not filled_row.empty else 0
    pnl_row = funnel_cascade[funnel_cascade['stage'] == 'P&L Resolved']
    n_profitable = int(pnl_row['_n_profitable'].values[0]) if not pnl_row.empty and '_n_profitable' in pnl_row.columns else 0
    n_pnl = int(pnl_row['survivors'].values[0]) if not pnl_row.empty else 0

    surv_pct = filled_count / max(first_count, 1) * 100
    win_rate = n_profitable / max(n_pnl, 1) * 100

    cap_col1, cap_col2, cap_col3 = st.columns(3)
    cap_col1.caption(f"End-to-end survival: **{filled_count}/{first_count} ({surv_pct:.0f}%)**")
    cap_col2.caption(f"P&L resolved: **{n_pnl}** trades")
    cap_col3.caption(f"Win rate (among resolved): **{win_rate:.0f}%**")

    # Source boundary indicator
    has_realtime = False
    if not funnel_df.empty and 'source' in funnel_df.columns:
        has_realtime = (funnel_df['source'] == 'REALTIME').any()

    if has_realtime:
        st.caption("Includes REALTIME execution data with full cycle tracing.")
    else:
        st.caption(
            "Execution stages (Orders Placed/Filled) are from backfilled data. "
            "Intelligence gates (top half) are reconstructed from council_history columns. "
            "Per-signal tracing requires REALTIME data."
        )

    # --- Regime Comparison ---
    regime_col = 'regime'
    if not funnel_df.empty and regime_col in funnel_df.columns:
        regime_values = funnel_df[regime_col].dropna().unique()
        regime_values = [r for r in regime_values if r and str(r).upper() != 'UNKNOWN']
        if len(regime_values) >= 2:
            with st.expander("Regime Comparison", expanded=False):
                regime_rows = []
                for regime in sorted(regime_values):
                    rdf = funnel_df[funnel_df[regime_col] == regime]
                    for stage in build_dynamic_stage_order(rdf):
                        sdf = rdf[rdf['stage'] == stage]
                        passed = len(sdf[sdf['outcome'] == 'PASS'])
                        blocked = len(sdf[sdf['outcome'] == 'BLOCK'])
                        info = len(sdf[sdf['outcome'] == 'INFO'])
                        passed += info
                        total = passed + blocked
                        if total > 0:
                            regime_rows.append({
                                'Regime': str(regime).title(),
                                'Stage': stage.replace('_', ' ').title(),
                                'Passed': passed,
                                'Blocked': blocked,
                                'Total': total,
                                'Block Rate': f"{blocked / total * 100:.0f}%" if total > 0 else "0%",
                            })
                if regime_rows:
                    rg_df = pd.DataFrame(regime_rows)
                    fig_regime = px.bar(
                        rg_df, x='Stage', y='Blocked', color='Regime',
                        barmode='group', text='Blocked',
                        labels={'Blocked': 'Blocked Count'},
                        title='Block Counts by Regime',
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_regime.update_layout(
                        xaxis_tickangle=-45,
                        height=350,
                        margin=dict(t=40, b=80),
                    )
                    st.plotly_chart(fig_regime, use_container_width=True)

                    # Regime survival table
                    regime_survival = []
                    for regime in sorted(regime_values):
                        rdf = funnel_df[funnel_df[regime_col] == regime]
                        decisions = rdf[(rdf['stage'] == 'COUNCIL_DECISION')]
                        n_dec = len(decisions[decisions['outcome'].isin(['PASS', 'BLOCK', 'INFO'])])
                        n_filled = len(rdf[rdf['stage'] == 'ORDER_FILLED'])
                        regime_survival.append({
                            'Regime': str(regime).title(),
                            'Decisions': n_dec,
                            'Filled': n_filled,
                            'Survival %': f"{n_filled / max(n_dec, 1) * 100:.1f}%",
                        })
                    st.dataframe(
                        pd.DataFrame(regime_survival),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Regime": st.column_config.TextColumn("🎭 Regime", help="The market regime at the time of the decision."),
                            "Decisions": st.column_config.NumberColumn("⚖️ Decisions", help="Total number of council decisions in this regime."),
                            "Filled": st.column_config.NumberColumn("⛽ Filled", help="Total number of filled orders in this regime."),
                            "Survival %": st.column_config.TextColumn("🏁 Survival %", help="End-to-end survival rate from signal to fill."),
                        }
                    )

                    # Source-awareness badge
                    if 'source' in funnel_df.columns:
                        n_realtime = (funnel_df['source'] == 'REALTIME').sum()
                        n_total_events = len(funnel_df)
                        if n_realtime < n_total_events * 0.5:
                            st.caption(
                                f"Regime survival based on aggregate counts. "
                                f"{n_realtime}/{n_total_events} events are REALTIME (remainder is backfill). "
                                f"Per-signal regime tracing improves as REALTIME data accumulates."
                            )
else:
    st.info("Insufficient data to build funnel. Need council history or funnel events.")


# ============================================================
# ROW 3: SIGNAL vs OUTCOME MATRIX
# ============================================================
st.subheader("🎯 Signal vs Outcome — Skill or Luck?")

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
        labels = ['Skill', 'Lucky', 'Market Risk', 'Bad Call']
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
                'Market Risk': '#e74c3c', 'Bad Call': '#95a5a6',
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
        qc1.metric("✅ Skill", quad_counts.get('Skill', 0), help="Good process + positive P&L")
        qc2.metric("🍀 Lucky", quad_counts.get('Lucky', 0), help="Weak process + positive P&L")
        qc3.metric("📉 Market Risk", quad_counts.get('Market Risk', 0),
                    help="Good process, adverse market move — irreducible market risk")
        qc4.metric("❌ Bad Call", quad_counts.get('Bad Call', 0), help="Weak process + negative P&L")
else:
    st.info("Insufficient council history data for matrix analysis.")


# ============================================================
# ROW 4: EXECUTION EFFICIENCY + POSITION LIFECYCLE
# ============================================================
tab_exec, tab_lifecycle = st.tabs(["Execution Efficiency", "Position Lifecycle"])

with tab_exec:
    if not funnel_df.empty and 'stage' in funnel_df.columns:
        # Slippage analysis
        filled = funnel_df[funnel_df['stage'] == 'ORDER_FILLED'].copy()
        if not filled.empty and 'fill_price' in filled.columns and 'initial_limit' in filled.columns:
            filled_clean = filled.dropna(subset=['fill_price', 'initial_limit'])
            if not filled_clean.empty:
                fill_p = filled_clean['fill_price'].astype(float)
                init_p = filled_clean['initial_limit'].astype(float)
                filled_clean['slippage_abs'] = (fill_p - init_p)
                filled_clean['slippage_pct'] = (filled_clean['slippage_abs'].abs() / init_p.abs().replace(0, np.nan) * 100)

                # Summary stats
                slip_col1, slip_col2 = st.columns(2)
                with slip_col1:
                    st.markdown("**Slippage Summary (% of credit/debit)**")
                    slip_stats = filled_clean['slippage_pct'].dropna()
                    if not slip_stats.empty:
                        ss1, ss2, ss3, ss4 = st.columns(4)
                        ss1.metric("📊 Mean", f"{slip_stats.mean():.1f}%", help="Mean slippage as % of credit/debit")
                        ss2.metric("🎯 Median", f"{slip_stats.median():.1f}%", help="Median slippage as % of credit/debit")
                        ss3.metric("🔝 P75", f"{slip_stats.quantile(0.75):.1f}%", help="75th percentile of slippage as % of credit/debit")
                        ss4.metric("🚀 Max", f"{slip_stats.max():.1f}%", help="Maximum slippage as % of credit/debit")

                with slip_col2:
                    st.markdown("**Absolute Slippage (cents/ticks)**")
                    abs_stats = filled_clean['slippage_abs'].dropna()
                    if not abs_stats.empty:
                        sa1, sa2, sa3, sa4 = st.columns(4)
                        sa1.metric("📊 Mean", f"{abs_stats.mean():.2f}", help="Mean absolute slippage in cents/ticks")
                        sa2.metric("🎯 Median", f"{abs_stats.median():.2f}", help="Median absolute slippage in cents/ticks")
                        sa3.metric("✅ Favorable", f"{(abs_stats < 0).sum()}", help="Fills better than initial limit")
                        sa4.metric("⚠️ Adverse", f"{(abs_stats > 0).sum()}", help="Fills worse than initial limit")

                # Histogram
                st.markdown("**Slippage Distribution (% of credit/debit)**")
                fig_slip = px.histogram(
                    filled_clean, x='slippage_pct',
                    nbins=20, color_discrete_sequence=['#3498db'],
                    labels={'slippage_pct': 'Slippage % (relative to credit/debit)'},
                )
                fig_slip.update_layout(height=300, margin=dict(t=20))
                st.plotly_chart(fig_slip, use_container_width=True)

        # Top unfilled orders
        cancelled = funnel_df[funnel_df['stage'] == 'ORDER_CANCELLED'].copy()
        if not cancelled.empty:
            st.markdown("**Recent Unfilled Orders**")
            display_cols = ['timestamp', 'cycle_id', 'contract', 'detail', 'walk_away_price', 'initial_limit']
            avail_cols = [c for c in display_cols if c in cancelled.columns]
            _cancelled_display = cancelled[avail_cols].tail(10).sort_values('timestamp', ascending=False).copy()
            if 'timestamp' in _cancelled_display.columns:
                _cancelled_display['timestamp'] = _cancelled_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(
                _cancelled_display,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.TextColumn("🕒 Time", help="Time when the order was cancelled."),
                    "cycle_id": st.column_config.TextColumn("🆔 Cycle ID", help="Unique identifier for the trading cycle."),
                    "contract": st.column_config.TextColumn("📜 Contract", help="The futures contract for the order."),
                    "detail": st.column_config.TextColumn("📝 Detail", help="Additional details about the cancellation."),
                    "walk_away_price": st.column_config.NumberColumn("🚪 Walk Away", format="$%.2f", help="The price at which the adaptive walk would have stopped."),
                    "initial_limit": st.column_config.NumberColumn("🎯 Initial Limit", format="$%.2f", help="The initial limit price of the order."),
                }
            )
    else:
        st.info("No execution data yet. Funnel events will appear after the next trading cycle.")

with tab_lifecycle:
    if not funnel_df.empty and 'stage' in funnel_df.columns:
        # Exit reason distribution — only from REALTIME data (backfill can't reconstruct real reasons)
        closed = funnel_df[funnel_df['stage'] == 'POSITION_CLOSED'].copy()
        if 'source' in closed.columns:
            realtime_closed = closed[closed['source'] == 'REALTIME']
        else:
            realtime_closed = pd.DataFrame()

        n_realtime_exits = len(realtime_closed)
        if n_realtime_exits >= 10 and 'detail' in realtime_closed.columns:
            realtime_closed = realtime_closed.copy()
            realtime_closed['exit_reason'] = realtime_closed['detail'].str.extract(
                r'exit_reason=([^,]+)', expand=False
            ).fillna('Unknown')
            reason_counts = realtime_closed['exit_reason'].value_counts()

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
            total_closed = len(closed)
            st.info(
                f"Exit reason data requires live trading cycles with instrumentation. "
                f"Currently **{n_realtime_exits}/{total_closed}** exits have proper classification. "
                f"Showing after 10+ REALTIME exits accumulate."
            )

        # Risk triggers
        risk_events = funnel_df[funnel_df['stage'] == 'RISK_TRIGGER']
        if not risk_events.empty:
            st.markdown("**Recent Risk Triggers**")
            display_cols = ['timestamp', 'contract', 'detail']
            avail_cols = [c for c in display_cols if c in risk_events.columns]
            _risk_display = risk_events[avail_cols].tail(10).sort_values('timestamp', ascending=False).copy()
            if 'timestamp' in _risk_display.columns:
                _risk_display['timestamp'] = _risk_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(
                _risk_display,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.TextColumn("🕒 Time", help="Time when the risk trigger occurred."),
                    "contract": st.column_config.TextColumn("📜 Contract", help="The contract affected by the risk event."),
                    "detail": st.column_config.TextColumn("📝 Detail", help="Descriptive details of the risk trigger."),
                }
            )
    else:
        st.info("No lifecycle data yet.")


# ============================================================
# ROW 5: TOP LEAKS TABLE (with corrected denominators)
# ============================================================
st.subheader("🚰 Top Alpha Leaks — Ranked by Impact")

# Extract survivor counts from the cascade for proper denominators
_count_for = {row['stage']: row['survivors'] for _, row in funnel_cascade.iterrows()}
n_actionable_decisions = _count_for.get('Actionable (Bull/Bear)', 1)
n_compliance_passed = _count_for.get('Compliance Passed', 1)
# Find conviction gate count (label includes threshold value)
_conviction_labels = [k for k in _count_for if 'Conviction' in k]
n_conviction_passed = _count_for.get(_conviction_labels[0], 1) if _conviction_labels else 1

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

    # Leak 2: Compliance blocks (denominator = actionable decisions)
    n_compl_block = len(funnel_df[(funnel_df['stage'] == 'COMPLIANCE_AUDIT') & (funnel_df['outcome'] == 'BLOCK')])
    if n_compl_block > 0:
        leak_data.append({
            'Leak Category': 'Compliance Veto',
            'Count': n_compl_block,
            '% of Pipeline': f"{n_compl_block/max(n_actionable_decisions,1)*100:.0f}%",
            'Diagnosis': 'Hallucination check or risk limit breach',
            'Config Key': 'compliance.*',
        })

    # Leak 3: Conviction gate blocks (denominator = compliance passed)
    n_conv_block = len(funnel_df[(funnel_df['stage'] == 'CONVICTION_GATE') & (funnel_df['outcome'] == 'BLOCK')])
    if n_conv_block > 0:
        leak_data.append({
            'Leak Category': 'Conviction Gate',
            'Count': n_conv_block,
            '% of Pipeline': f"{n_conv_block/max(n_compliance_passed,1)*100:.0f}%",
            'Diagnosis': 'Weighted score below threshold',
            'Config Key': 'strategy.min_weighted_score_magnitude',
        })

    # Leak 4: Confidence threshold blocks (denominator = conviction passed)
    n_conf_block = len(funnel_df[(funnel_df['stage'] == 'CONFIDENCE_THRESHOLD') & (funnel_df['outcome'] == 'BLOCK')])
    if n_conf_block > 0:
        leak_data.append({
            'Leak Category': 'Confidence Threshold',
            'Count': n_conf_block,
            '% of Pipeline': f"{n_conf_block/max(n_conviction_passed,1)*100:.0f}%",
            'Diagnosis': 'Signal confidence below min_confidence_threshold',
            'Config Key': 'risk_management.min_confidence_threshold',
        })

    # Leak 5: Liquidity gate blocks
    n_liq_block = len(funnel_df[(funnel_df['stage'] == 'LIQUIDITY_GATE') & (funnel_df['outcome'] == 'BLOCK')])
    if n_liq_block > 0:
        n_strategy_selected = _count_for.get('Strategy Selected', 1)
        leak_data.append({
            'Leak Category': 'Liquidity Gate',
            'Count': n_liq_block,
            '% of Pipeline': f"{n_liq_block/max(n_strategy_selected,1)*100:.0f}%",
            'Diagnosis': 'Bid-ask spread too wide or volume too low',
            'Config Key': 'risk_management.max_spread_pct',
        })

    # Leak 6: Drawdown gate blocks
    n_dd_block = len(funnel_df[(funnel_df['stage'] == 'DRAWDOWN_GATE') & (funnel_df['outcome'] == 'BLOCK')])
    if n_dd_block > 0:
        leak_data.append({
            'Leak Category': 'Drawdown Circuit Breaker',
            'Count': n_dd_block,
            '% of Pipeline': f"{n_dd_block/max(n_placed,1)*100:.0f}%",
            'Diagnosis': 'Cumulative drawdown exceeded threshold',
            'Config Key': 'drawdown_circuit_breaker.*',
        })

    if leak_data:
        leak_df = pd.DataFrame(leak_data).sort_values('Count', ascending=False)
        st.dataframe(
            leak_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Leak Category": st.column_config.TextColumn("🚰 Leak Category", help="The stage or component where alpha was lost."),
                "Count": st.column_config.NumberColumn("🔢 Count", help="Number of signals blocked or lost at this stage."),
                "% of Pipeline": st.column_config.TextColumn("📊 % of Pipeline", help="The relative impact of this leak on the total signal flow."),
                "Diagnosis": st.column_config.TextColumn("🩺 Diagnosis", help="The underlying reason for the leak."),
                "Config Key": st.column_config.TextColumn("⚙️ Config Key", help="The configuration parameter governing this gate."),
            }
        )
    else:
        st.success("No significant leaks detected in the current period.")
else:
    st.info("Leak analysis requires funnel event data. Will populate after trading cycles run with instrumentation.")


# ============================================================
# RAW DATA EXPLORER + CYCLE DRILL-DOWN
# ============================================================
with st.expander("Raw Funnel Data", expanded=False):
    if not funnel_df.empty:
        # Filters row
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            _dynamic_stages = build_dynamic_stage_order(funnel_df)
            stages = ['ALL'] + _dynamic_stages
            sel_stage = st.selectbox("Filter by Stage", stages)
        with filter_col2:
            cycle_ids = funnel_df['cycle_id'].dropna().unique().tolist() if 'cycle_id' in funnel_df.columns else []
            cycle_options = ['ALL'] + sorted(set(cycle_ids), reverse=True)[:50]
            sel_cycle = st.selectbox("Drill into Cycle ID", cycle_options,
                                     help="Select a cycle to see all its events end-to-end")

        display_df = funnel_df.copy()
        if sel_stage != 'ALL':
            display_df = display_df[display_df['stage'] == sel_stage]
        if sel_cycle != 'ALL':
            display_df = display_df[display_df['cycle_id'] == sel_cycle]

        # When drilling into a cycle, show events in chronological order
        sort_asc = (sel_cycle != 'ALL')
        _raw_display = display_df.sort_values('timestamp', ascending=sort_asc).head(200).copy()
        if 'timestamp' in _raw_display.columns:
            _raw_display['timestamp'] = _raw_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(
            _raw_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.TextColumn("🕒 Time", help="Timestamp of the funnel event."),
                "cycle_id": st.column_config.TextColumn("🆔 Cycle ID", help="The unique cycle ID."),
                "stage": st.column_config.TextColumn("🎭 Stage", help="The execution stage name."),
                "outcome": st.column_config.TextColumn("🏁 Outcome", help="PASS/BLOCK outcome at this stage."),
                "detail": st.column_config.TextColumn("📝 Detail", help="Technical metadata for the event."),
                "source": st.column_config.TextColumn("🔌 Source", help="The data source (e.g., orchestrator)."),
                "regime": st.column_config.TextColumn("🎭 Regime", help="Market regime during the event."),
                "fill_price": st.column_config.NumberColumn("💰 Fill Price", format="$%.2f"),
                "initial_limit": st.column_config.NumberColumn("🎯 Limit", format="$%.2f"),
            }
        )

        # Cycle journey summary when a specific cycle is selected
        if sel_cycle != 'ALL' and not display_df.empty:
            journey = display_df.sort_values('timestamp')[['timestamp', 'stage', 'outcome', 'detail']].copy()
            journey['timestamp'] = journey['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.markdown(f"**Cycle Journey: `{sel_cycle}`** ({len(journey)} events)")
            st.dataframe(
                journey,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.TextColumn("🕒 Time", help="Time of day for the event."),
                    "stage": st.column_config.TextColumn("🎭 Stage", help="Execution stage."),
                    "outcome": st.column_config.TextColumn("🏁 Outcome", help="Result at this stage."),
                    "detail": st.column_config.TextColumn("📝 Detail", help="Detailed event metadata."),
                }
            )

        st.download_button(
            "Download Funnel CSV",
            display_df.to_csv(index=False).encode('utf-8'),
            f"execution_funnel_{ticker}.csv",
            "text/csv",
            help="Download the complete execution funnel data as a CSV file for detailed analysis."
        )
    else:
        st.info("No funnel data available.")
