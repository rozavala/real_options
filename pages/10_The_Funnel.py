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
    selected_source = st.sidebar.selectbox("Data Source", sources, index=0, help="Filter the execution funnel by data source (e.g., REALTIME or backfill).")
    if selected_source != 'ALL':
        funnel_df = funnel_df[funnel_df['source'] == selected_source]

# Regime filter
if not funnel_df.empty and 'regime' in funnel_df.columns:
    regimes = ['ALL'] + sorted(funnel_df['regime'].dropna().unique().tolist())
    selected_regime = st.sidebar.selectbox("Regime Filter", regimes, index=0, help="Filter the execution funnel by specific market regimes.")
    if selected_regime != 'ALL':
        funnel_df = funnel_df[funnel_df['regime'] == selected_regime]

# Trading Mode filter (v6 — requires trading_mode_active column in council_history)
_has_trading_mode = (
    not council_df.empty and
    'trading_mode_active' in council_df.columns and
    council_df['trading_mode_active'].astype(str).str.lower().isin(['true', 'false']).any()
)
if _has_trading_mode:
    _tm_options = ['ALL', 'LIVE ONLY', 'OBSERVATION ONLY']
    selected_trading_mode = st.sidebar.selectbox(
        "Trading Mode",
        _tm_options,
        index=1,  # Default: LIVE ONLY — observation data in default view is misleading
        help="LIVE ONLY excludes decisions made when trading_mode=OFF",
    )
    if selected_trading_mode == 'LIVE ONLY':
        council_df = council_df[
            council_df['trading_mode_active'].astype(str).str.lower() == 'true'
        ]
    elif selected_trading_mode == 'OBSERVATION ONLY':
        council_df = council_df[
            council_df['trading_mode_active'].astype(str).str.lower() == 'false'
        ]
else:
    selected_trading_mode = 'ALL'
    if not council_df.empty:
        st.sidebar.caption("⚠️ Trading mode filter unavailable (pre-v6 data)")


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
def build_true_funnel(
    council_df: pd.DataFrame,
    funnel_df: pd.DataFrame,
    config: dict,
    observation_only: bool = False,
) -> pd.DataFrame:
    """
    Build a monotonically-decreasing cascading funnel from two data sources.

    Top half (Intelligence + Gates): derived from council_history columns.
    Bottom half (Execution): derived from execution_funnel.csv aggregate stage counts.
    When observation_only=True, execution and P&L stages are omitted — trading was OFF
    so Orders Placed/Filled are zero by design; showing them looks broken.

    Returns DataFrame with columns: stage, survivors, drop, drop_pct, source_label
    """
    min_score = config.get('strategy', {}).get('min_weighted_score_magnitude', 0.20)

    stages = []

    # 1. All Council Decisions
    n_total = len(council_df)
    stages.append({'stage': 'Council Decisions', 'survivors': n_total,
                   'source_label': 'council_history'})

    # 2. Actionable — matches order_manager.py gate logic:
    # Directional (BULL/BEAR) OR volatility plays (NEUTRAL + VOLATILITY prediction_type) pass through.
    # NEUTRAL + DIRECTIONAL = no trade (the "Cash is a Position" path).
    if 'master_decision' in council_df.columns:
        directional = council_df['master_decision'].isin(['BULLISH', 'BEARISH'])

        if 'prediction_type' in council_df.columns:
            vol_play = council_df['prediction_type'].fillna('DIRECTIONAL') == 'VOLATILITY'
        else:
            vol_play = pd.Series(False, index=council_df.index)

        actionable = council_df[directional | vol_play]
    else:
        actionable = council_df
    n_actionable = len(actionable)
    stages.append({'stage': 'Actionable', 'survivors': n_actionable,
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

    # Execution stages are irrelevant in observation-only mode (trading was OFF).
    # Showing zeros mid-funnel looks broken — omit them entirely.
    if not observation_only:
        # 6. Orders Placed — raw count from execution_funnel.csv ORDER_PLACED events.
        # NOTE: Backfill coverage is partial (order_events.csv only goes back to ~mid-Feb).
        # The PNL_RECONCILED fallback was removed (it inflated placed/filled to 100%
        # efficiency using non-trade NEUTRAL rows). Raw counts are honest about coverage.
        n_placed = 0
        if not funnel_df.empty and 'stage' in funnel_df.columns:
            n_placed = len(funnel_df[funnel_df['stage'] == 'ORDER_PLACED'])
        # Cap at n_strategy so the waterfall can't widen across the council→execution boundary.
        n_placed = min(n_placed, n_strategy)
        stages.append({'stage': 'Orders Placed', 'survivors': n_placed,
                       'source_label': 'execution_funnel'})

        # 7. Orders Filled — raw count from ORDER_FILLED events.
        n_filled = 0
        if not funnel_df.empty and 'stage' in funnel_df.columns:
            n_filled = len(funnel_df[funnel_df['stage'] == 'ORDER_FILLED'])
        n_filled = min(n_filled, n_placed)
        stages.append({'stage': 'Orders Filled', 'survivors': n_filled,
                       'source_label': 'execution_funnel'})

        # P&L outcome stages are intentionally omitted from the waterfall.
        # They come from council_history which covers a different record set than
        # execution_funnel ORDER_FILLED — forcing them into the cascade produces
        # misleading 100% win rates when capped. P&L outcomes are shown below
        # in the "Post-Funnel Outcome" section, sourced directly from council_history.

    # Build DataFrame with drop calculations
    result = pd.DataFrame(stages)
    # Backfill survivors_raw for stages that don't need capping (raw == capped)
    if 'survivors_raw' not in result.columns:
        result['survivors_raw'] = result['survivors']
    else:
        result['survivors_raw'] = result['survivors_raw'].fillna(result['survivors'])
    result['survivors_raw'] = result['survivors_raw'].astype(int)
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
_observation_only = (selected_trading_mode == 'OBSERVATION ONLY')
funnel_cascade = build_true_funnel(council_df, funnel_df, config, observation_only=_observation_only)


# ============================================================
# ROW 1: SIGNAL QUALITY KPIs (top — independent of execution)
# ============================================================
st.subheader("🧭 Signal Quality")
st.caption("Independent of execution, strike selection, and option pricing.")


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

    # P&L win rate from council_history
    if not ch.empty and 'pnl_realized' in ch.columns:
        resolved = ch[ch['pnl_realized'].notna()]
        kpis['signal_win_rate'] = (resolved['pnl_realized'] > 0).mean() * 100 if not resolved.empty else 0
        kpis['n_resolved'] = len(resolved)
    else:
        kpis['signal_win_rate'] = 0
        kpis['n_resolved'] = 0

    # Directional accuracy — only directional signals (BULLISH/BEARISH), not vol plays.
    # Vol plays (NEUTRAL + VOLATILITY) don't predict direction so excluding them keeps
    # this metric honest. They get their own metric below.
    kpis['dir_accuracy_pct'] = 0.0
    kpis['dir_accuracy_n'] = 0
    kpis['dir_denominator'] = 0
    kpis['dir_correct_and_profitable_pct'] = 0.0
    kpis['dir_correct_and_profitable_n'] = 0
    if not ch.empty and 'actual_trend_direction' in ch.columns and 'master_decision' in ch.columns:
        # actual_trend_direction may use UP/DOWN or BULLISH/BEARISH vocabulary — normalise both.
        _DIR_NORM = {'UP': 'UP', 'DOWN': 'DOWN', 'BULLISH': 'UP', 'BEARISH': 'DOWN'}
        _directional = ch[ch['master_decision'].isin(['BULLISH', 'BEARISH'])].copy()
        _dir_resolved = _directional[
            _directional['actual_trend_direction'].str.upper().isin(_DIR_NORM)
        ].copy()
        _n_dir_resolved = len(_dir_resolved)
        if _n_dir_resolved > 0:
            _actual = _dir_resolved['actual_trend_direction'].str.upper().map(_DIR_NORM)
            _predicted = _dir_resolved['master_decision'].str.upper().map(
                {'BULLISH': 'UP', 'BEARISH': 'DOWN'}
            )
            _correct_mask = (_actual == _predicted)
            _n_correct = int(_correct_mask.sum())
            kpis['dir_accuracy_pct'] = _n_correct / _n_dir_resolved * 100
            kpis['dir_accuracy_n'] = _n_correct
            kpis['dir_denominator'] = _n_dir_resolved
            # Bridge: correct direction AND profitable (requires pnl_realized)
            if 'pnl_realized' in _dir_resolved.columns:
                _pnl = pd.to_numeric(_dir_resolved['pnl_realized'], errors='coerce')
                _both = int((_correct_mask & (_pnl > 0)).sum())
                kpis['dir_correct_and_profitable_n'] = _both
                kpis['dir_correct_and_profitable_pct'] = _both / _n_dir_resolved * 100

    # Vol play hit rate — straddles/condors: correct if pnl_realized > 0
    kpis['vol_hit_rate_pct'] = 0.0
    kpis['vol_hit_n'] = 0
    kpis['vol_denominator'] = 0
    if not ch.empty and 'prediction_type' in ch.columns and 'pnl_realized' in ch.columns:
        _vol = ch[ch['prediction_type'].fillna('DIRECTIONAL') == 'VOLATILITY'].copy()
        _vol_resolved = _vol[pd.to_numeric(_vol['pnl_realized'], errors='coerce').notna()]
        _n_vol = len(_vol_resolved)
        if _n_vol > 0:
            _vol_pnl = pd.to_numeric(_vol_resolved['pnl_realized'], errors='coerce')
            kpis['vol_hit_rate_pct'] = (_vol_pnl > 0).sum() / _n_vol * 100
            kpis['vol_hit_n'] = int((_vol_pnl > 0).sum())
            kpis['vol_denominator'] = _n_vol

    return kpis


kpis = calc_funnel_kpis(funnel_df, council_df)

sq1, sq2, sq3 = st.columns(3)
_dir_label = (f"{kpis['dir_accuracy_n']}/{kpis['dir_denominator']} resolved"
              if kpis['dir_denominator'] > 0 else "no resolved signals")
sq1.metric(
    "🧭 Directional Accuracy",
    f"{kpis['dir_accuracy_pct']:.0f}%" if kpis['dir_denominator'] > 0 else "—",
    help=f"Of directional signals (BULLISH/BEARISH) with a known outcome, % that matched actual "
         f"price direction. Excludes vol plays. ({_dir_label})",
)
_vol_label = (f"{kpis['vol_hit_n']}/{kpis['vol_denominator']} resolved"
              if kpis['vol_denominator'] > 0 else "no vol plays")
sq2.metric(
    "⚡ Vol Play Hit Rate",
    f"{kpis['vol_hit_rate_pct']:.0f}%" if kpis['vol_denominator'] > 0 else "—",
    help=f"% of resolved volatility plays (straddles/condors) with positive P&L. "
         f"({_vol_label})",
)
sq3.metric(
    "🎯 P&L Win Rate",
    f"{kpis['signal_win_rate']:.0f}%" if kpis['n_resolved'] > 0 else "—",
    help=f"% of all resolved trades (directional + vol plays) with positive P&L. "
         f"n={kpis['n_resolved']} resolved. A gap between Directional Accuracy and this "
         f"metric suggests option structure (theta, strikes, entry timing) is consuming edge.",
)

# Auto-diagnosis: answer the three questions in plain English.
# Leakage signal: directional accuracy is high but P&L win rate is low
# (correct-direction trades aren't converting to profit — option structure issue).
if kpis['dir_denominator'] >= 5:
    _dir_ok = kpis['dir_accuracy_pct'] >= 55
    _pnl_ok = kpis['signal_win_rate'] >= 50

    if _dir_ok and _pnl_ok:
        st.success(
            f"✅ **System is working**: signals are accurate ({kpis['dir_accuracy_pct']:.0f}%) "
            f"and the P&L win rate is {kpis['signal_win_rate']:.0f}%."
        )
    elif _dir_ok and not _pnl_ok:
        st.warning(
            f"⚠️ **Option structure may be leaking edge**: directional accuracy is "
            f"{kpis['dir_accuracy_pct']:.0f}% but P&L win rate is only "
            f"{kpis['signal_win_rate']:.0f}%. Correct-direction calls aren't converting to "
            f"profit — review strike selection, theta decay, and entry timing."
        )
    elif not _dir_ok:
        st.warning(
            f"⚠️ **Signal quality is the bottleneck**: directional accuracy is {kpis['dir_accuracy_pct']:.0f}% "
            f"(below 55% threshold). Focus on improving council decision quality before optimising execution."
        )

with st.expander("📈 Execution KPIs", expanded=False):
    k1, k2, k3, k4, k5, k6 = st.columns(6)
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
    k6.metric("💸 Alpha Left on Table", f"{kpis['alpha_left_count']}",
              delta=f"-{kpis['alpha_left_pct']:.0f}% of placed" if kpis['alpha_left_count'] > 0 else None,
              delta_color="inverse",
              help="Orders that passed all gates but failed to fill — potential alpha lost to adaptive walk timeout")


# ============================================================
# ROW 2: FUNNEL WATERFALL (Cascading)
# ============================================================
st.subheader("🌊 Funnel Waterfall — Where Do Signals Die?")

# Conviction-gate disclaimer — the gate threshold is applied retroactively to all
# historical data, but many pre-gate decisions were actually executed. Show a
# banner when there are below-conviction rows with non-zero P&L so users understand
# the waterfall is applying current standards to historical trades.
if not council_df.empty and 'weighted_score' in council_df.columns and 'pnl_realized' in council_df.columns:
    _min_ws = config.get('strategy', {}).get('min_weighted_score_magnitude', 0.20)
    _ws_all = pd.to_numeric(council_df['weighted_score'], errors='coerce').fillna(0)
    _pnl_all = pd.to_numeric(council_df['pnl_realized'], errors='coerce')
    _below_with_pnl = int(
        ((_ws_all.abs() < _min_ws) & _pnl_all.notna() & (_pnl_all != 0)).sum()
    )
    if _below_with_pnl > 0:
        st.info(
            f"ℹ️ **Historical threshold note**: {_below_with_pnl} decisions below the current "
            f"conviction gate (|weighted_score| < {_min_ws}) have realized P&L — they were "
            f"executed before this gate existed or with a different threshold. The waterfall "
            f"shows them as blocked retroactively."
        )

if not funnel_cascade.empty and funnel_cascade['survivors'].sum() > 0:
    # Build custom text labels: "N (XX% of initial)"
    first_count = int(funnel_cascade.iloc[0]['survivors'])
    # Build per-stage colors dynamically so adding stages never breaks color mapping.
    # Color scheme: green gradient = intelligence/gates, blue = execution, gold/orange = P&L outcomes.
    _COLOR_BY_SOURCE = {
        'council_history_gate': ['#2ecc71', '#27ae60', '#1abc9c', '#16a085', '#2980b9'],
        'execution':            ['#3498db', '#5dade2'],
        'pnl':                  ['#f39c12', '#e67e22'],
    }
    _gate_color_idx = 0
    _exec_color_idx = 0
    _pnl_color_idx = 0
    _stage_colors = []
    for _, _sr in funnel_cascade.iterrows():
        _src = _sr.get('source_label', 'council_history')
        if _src == 'execution_funnel':
            _stage_colors.append(_COLOR_BY_SOURCE['execution'][_exec_color_idx % 2])
            _exec_color_idx += 1
        elif _sr['stage'] in ('P&L Resolved', 'Profitable'):
            _stage_colors.append(_COLOR_BY_SOURCE['pnl'][_pnl_color_idx % 2])
            _pnl_color_idx += 1
        else:
            _stage_colors.append(_COLOR_BY_SOURCE['council_history_gate'][
                _gate_color_idx % len(_COLOR_BY_SOURCE['council_history_gate'])
            ])
            _gate_color_idx += 1

    custom_text = []
    for idx in range(len(funnel_cascade)):
        row = funnel_cascade.iloc[idx]
        val = int(row['survivors'])
        pct_initial = val / max(first_count, 1) * 100
        if idx == 0:
            custom_text.append(f"<b>{val}</b>")
        else:
            prev_val = int(funnel_cascade.iloc[idx - 1]['survivors'])
            pct_prev = val / max(prev_val, 1) * 100
            custom_text.append(f"<b>{val}</b>  ({pct_initial:.0f}% of initial, {pct_prev:.0f}% of prev)")

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_cascade['stage'],
        x=funnel_cascade['survivors'],
        text=custom_text,
        textinfo="text",
        textfont=dict(size=14),
        marker=dict(color=_stage_colors),
        connector=dict(line=dict(color="gray", dash="dot", width=1)),
    ))
    fig_funnel.update_layout(
        height=max(500, 60 * len(funnel_cascade)),
        margin=dict(t=20, b=20, l=10, r=10),
        funnelmode="stack",
        font=dict(size=13),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    # Observation mode banner — execution stages hidden since trading was OFF
    if _observation_only:
        st.info(
            "📊 **Observation Mode** — trading was OFF during this period. "
            "Execution stages are hidden (Orders Placed/Filled would be zero by design). "
            "Use Signal Quality metrics above to assess signal quality before going live."
        )

    # --- Capping Warning (improved: explain root cause) ---
    capped_rows = funnel_cascade[
        funnel_cascade.get('survivors_raw', funnel_cascade['survivors']) > funnel_cascade['survivors']
    ] if 'survivors_raw' in funnel_cascade.columns else pd.DataFrame()
    if not capped_rows.empty:
        cap_msgs = []
        for row in capped_rows.to_dict('records'):
            cap_msgs.append(
                f"**{row['stage']}**: showing {int(row['survivors'])} "
                f"(raw council_history count: {int(row['survivors_raw'])})"
            )
        st.warning(
            "⚠️ **Data source mismatch** — some outcome stages were capped to maintain funnel order.\n\n"
            + "\n\n".join(cap_msgs) + "\n\n"
            "**Root cause**: `council_history.csv` and `execution_funnel.csv` cover different "
            "record sets for this date range. Common causes: (1) trading mode was OFF during "
            "part of the period — use the Trading Mode filter to isolate live periods; "
            "(2) backfill was run on a narrower date range than the current display window; "
            "(3) some cycles completed before funnel instrumentation was deployed.\n\n"
            "**Fix**: re-run backfill with `--all` flag, or narrow the date range to the live period.",
            icon=None,
        )

    # --- Survival Summary ---
    filled_row = funnel_cascade[funnel_cascade['stage'] == 'Orders Filled']
    filled_count = int(filled_row['survivors'].values[0]) if not filled_row.empty else 0
    surv_pct = filled_count / max(first_count, 1) * 100

    if not _observation_only:
        st.caption(
            f"Signal-to-fill survival: **{filled_count}/{first_count} ({surv_pct:.0f}%)**  "
            f"·  P&L outcomes shown below (independent data source)"
        )

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

    # ── POST-FUNNEL OUTCOME: Did the surviving signals make money? ──
    st.markdown("---")
    st.subheader("💰 Post-Funnel Outcome — Are We Making Money?")
    st.caption("Outcome of all resolved trades (from council_history, independent of funnel tracing).")

    if not council_df.empty and 'pnl_realized' in council_df.columns:
        pnl_series = pd.to_numeric(council_df['pnl_realized'], errors='coerce')
        resolved = council_df[pnl_series.notna()].copy()
        resolved['pnl'] = pnl_series[pnl_series.notna()]

        n_resolved = len(resolved)
        n_wins = int((resolved['pnl'] > 0).sum())
        n_losses = int((resolved['pnl'] <= 0).sum())
        total_pnl = resolved['pnl'].sum()
        avg_pnl = resolved['pnl'].mean() if n_resolved > 0 else 0
        avg_win = resolved.loc[resolved['pnl'] > 0, 'pnl'].mean() if n_wins > 0 else 0
        avg_loss = resolved.loc[resolved['pnl'] <= 0, 'pnl'].mean() if n_losses > 0 else 0
        win_rate_pnl = n_wins / max(n_resolved, 1) * 100

        # Row 1: Big-picture P&L metrics
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("📊 Resolved Trades", f"{n_resolved}",
                   help="Total trades with realized P&L outcome.")
        p2.metric("🏆 Win Rate", f"{win_rate_pnl:.0f}%",
                   delta=f"{n_wins}W / {n_losses}L",
                   help="Percentage of resolved trades with positive P&L.")
        p3.metric("💵 Total P&L", f"{total_pnl:+.1f} cents",
                   delta="profitable" if total_pnl > 0 else "unprofitable",
                   delta_color="normal" if total_pnl > 0 else "inverse",
                   help="Sum of all realized P&L in cents (KC) or ticks (NG).")
        p4.metric("📐 Avg P&L/Trade", f"{avg_pnl:+.2f} cents",
                   help="Average realized P&L per resolved trade.")

        # Row 2: Win/Loss asymmetry
        a1, a2, a3 = st.columns(3)
        a1.metric("✅ Avg Win", f"+{avg_win:.2f} cents" if avg_win > 0 else "\u2014",
                   help="Average P&L of winning trades.")
        a2.metric("❌ Avg Loss", f"{avg_loss:.2f} cents" if n_losses > 0 else "\u2014",
                   help="Average P&L of losing trades.")

        # Profit factor: gross wins / abs(gross losses)
        gross_wins = resolved.loc[resolved['pnl'] > 0, 'pnl'].sum()
        gross_losses = abs(resolved.loc[resolved['pnl'] <= 0, 'pnl'].sum())
        profit_factor = gross_wins / max(gross_losses, 0.01)
        a3.metric("⚖️ Profit Factor", f"{profit_factor:.2f}x",
                   help="Gross wins / gross losses. >1.0 = profitable system. "
                        ">1.5 = good. >2.0 = excellent.")

        # Mini P&L distribution chart
        if n_resolved >= 5:
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Histogram(
                x=resolved['pnl'],
                nbinsx=25,
                marker_color='#3498db',
                name='P&L Distribution',
            ))
            fig_pnl.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
            fig_pnl.add_vline(x=avg_pnl, line_dash="dot", line_color="#f39c12",
                              annotation_text=f"Mean: {avg_pnl:+.2f}",
                              annotation_position="top right")
            fig_pnl.update_layout(
                height=250,
                margin=dict(t=30, b=30, l=40, r=20),
                xaxis_title="P&L (cents)",
                yaxis_title="Count",
                showlegend=False,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        # --- Direction → P&L 2×2 Bridge (merged into outcome section) ---
        if (
            'actual_trend_direction' in council_df.columns and
            'master_decision' in council_df.columns
        ):
            _DIR_NORM_B = {'UP': 'UP', 'DOWN': 'DOWN', 'BULLISH': 'UP', 'BEARISH': 'DOWN'}
            _bridge_df = council_df[
                council_df['master_decision'].isin(['BULLISH', 'BEARISH']) &
                council_df['actual_trend_direction'].str.upper().isin(_DIR_NORM_B)
            ].copy()
            _bridge_df['pnl'] = pd.to_numeric(_bridge_df['pnl_realized'], errors='coerce')
            _bridge_df = _bridge_df[_bridge_df['pnl'].notna()]

            if len(_bridge_df) >= 4:
                st.markdown("**Direction → P&L** — did correct direction translate to profit?")
                st.caption(
                    "ℹ️ Under current instrumentation `actual_trend_direction` is derived from "
                    "the same exit-price comparison used to compute P&L, so Correct+Loss and "
                    "Wrong+Profitable are structurally zero for directional plays. "
                    "This bridge will become fully diagnostic once direction is measured against "
                    "an independent price window (future enhancement)."
                )
                _actual_b = _bridge_df['actual_trend_direction'].str.upper().map(_DIR_NORM_B)
                _predicted_b = _bridge_df['master_decision'].str.upper().map({'BULLISH': 'UP', 'BEARISH': 'DOWN'})
                _bridge_df['dir_correct'] = (_actual_b == _predicted_b)
                _bridge_df['profitable'] = (_bridge_df['pnl'] > 0)

                _cells = {
                    ('Correct', 'Profitable'):   int((_bridge_df['dir_correct'] & _bridge_df['profitable']).sum()),
                    ('Correct', 'Loss'):         int((_bridge_df['dir_correct'] & ~_bridge_df['profitable']).sum()),
                    ('Wrong', 'Profitable'):     int((~_bridge_df['dir_correct'] & _bridge_df['profitable']).sum()),
                    ('Wrong', 'Loss'):           int((~_bridge_df['dir_correct'] & ~_bridge_df['profitable']).sum()),
                }
                _n_total_b = sum(_cells.values())

                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric(
                    "🎯 Correct + Profitable",
                    f"{_cells[('Correct', 'Profitable')]}",
                    delta=f"{_cells[('Correct', 'Profitable')] / max(_n_total_b, 1) * 100:.0f}% of resolved",
                    delta_color="normal",
                    help="Got direction right AND made money. Pure skill.",
                )
                bc2.metric(
                    "⚠️ Correct + Loss",
                    f"{_cells[('Correct', 'Loss')]}",
                    delta=f"{_cells[('Correct', 'Loss')] / max(_n_total_b, 1) * 100:.0f}% of resolved",
                    delta_color="inverse",
                    help="Got direction right but still lost money. "
                         "Indicates option structure problems: theta decay eating gains, "
                         "strikes too far OTM, or move magnitude too small to overcome premium.",
                )
                bc3.metric(
                    "🍀 Wrong + Profitable",
                    f"{_cells[('Wrong', 'Profitable')]}",
                    delta=f"{_cells[('Wrong', 'Profitable')] / max(_n_total_b, 1) * 100:.0f}% of resolved",
                    delta_color="off",
                    help="Got direction wrong but still made money. Lucky — unsustainable.",
                )
                bc4.metric(
                    "❌ Wrong + Loss",
                    f"{_cells[('Wrong', 'Loss')]}",
                    delta=f"{_cells[('Wrong', 'Loss')] / max(_n_total_b, 1) * 100:.0f}% of resolved",
                    delta_color="inverse",
                    help="Got direction wrong and lost money. Bad call.",
                )

                _correct_loss_pct = _cells[('Correct', 'Loss')] / max(_cells[('Correct', 'Profitable')] + _cells[('Correct', 'Loss')], 1) * 100
                if _correct_loss_pct > 30 and _n_total_b >= 10:
                    st.warning(
                        f"⚠️ **Option structure leakage detected**: {_correct_loss_pct:.0f}% of directionally "
                        f"correct calls still lost money. This suggests theta decay, OTM strikes, or premium "
                        f"costs are eroding edge even when direction is right."
                    )

                # Breakdown of Correct+Loss quadrant — which conditions cluster here?
                _correct_loss_df = _bridge_df[_bridge_df['dir_correct'] & ~_bridge_df['profitable']]
                if len(_correct_loss_df) >= 2:
                    with st.expander(
                        f"🔍 Correct Direction but Lost ({len(_correct_loss_df)} trades) — where does the edge leak?",
                        expanded=(_correct_loss_pct > 30 and _n_total_b >= 10),
                    ):
                        _group_cols = [c for c in ['strategy_type', 'entry_regime', 'master_decision', 'volatility_level']
                                       if c in _correct_loss_df.columns and _correct_loss_df[c].notna().any()]

                        if _group_cols:
                            for _gc in _group_cols:
                                _grp = (
                                    _correct_loss_df.groupby(_gc)['pnl']
                                    .agg(count='count', avg_pnl='mean', total_pnl='sum')
                                    .reset_index()
                                    .sort_values('count', ascending=False)
                                )
                                _grp.columns = [_gc.replace('_', ' ').title(), 'Count', 'Avg P&L', 'Total P&L']
                                st.caption(f"By **{_gc.replace('_', ' ')}**")
                                st.dataframe(
                                    _grp.style.format({'Avg P&L': '{:+.2f}', 'Total P&L': '{:+.2f}'}),
                                    hide_index=True,
                                    use_container_width=True,
                                )

                        # Raw rows for drill-down
                        _raw_cols = [c for c in ['timestamp', 'contract', 'master_decision',
                                                  'strategy_type', 'entry_regime', 'pnl',
                                                  'master_confidence', 'weighted_score']
                                     if c in _correct_loss_df.columns]
                        st.caption("Individual trades")
                        _cl_display = _correct_loss_df[_raw_cols].sort_values('pnl').copy()
                        if 'timestamp' in _cl_display.columns:
                            _cl_display['timestamp'] = _cl_display['timestamp'].astype(str).str[:16]
                        st.dataframe(
                            _cl_display.style.format({'pnl': '{:+.2f}', 'master_confidence': '{:.2f}',
                                                      'weighted_score': '{:.2f}'}),
                            hide_index=True,
                            use_container_width=True,
                        )
    else:
        st.info("No P&L data available yet. Will populate after trades are resolved.")

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
# ROW 3: SIGNAL vs OUTCOME MATRIX (in expander — detail view)
# ============================================================
with st.expander("🎯 Signal vs Outcome — Skill or Luck?", expanded=False):
    st.caption("Process quality (confidence × |weighted score|) vs P&L outcome.")
    if not council_df.empty and 'pnl_realized' in council_df.columns and 'weighted_score' in council_df.columns:
        resolved = council_df[council_df['pnl_realized'].notna()].copy()
        if not resolved.empty:
            resolved['process_score'] = resolved['master_confidence'].fillna(0.5) * resolved['weighted_score'].abs().fillna(0)
            resolved['pnl'] = resolved['pnl_realized'].astype(float)

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
# ⚡ Bolt: vectorized dict generation via dict(zip()) is ~40x faster than .iterrows()
_count_for = dict(zip(funnel_cascade['stage'], funnel_cascade['survivors']))
n_actionable_decisions = _count_for.get('Actionable', _count_for.get('Actionable (Bull/Bear)', 1))
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
            sel_stage = st.selectbox("Filter by Stage", stages, help="Filter the raw funnel data to show only events from a specific execution stage.")
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
