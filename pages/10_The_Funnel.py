"""
Page 10: The Funnel — Signal-to-P&L Diagnostic

Answers three questions:
  1. Are we providing good signals? (directional accuracy)
  2. Are we making money from them? (P&L win rate, total P&L)
  3. What's blocking us? (signal quality / execution leakage / tail risk)

Data sources:
  - council_history.csv: Intelligence + gates (decisions, compliance, conviction, strategy, P&L)
  - execution_funnel.csv: Execution (order placed, price walk, fill, cancel)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
st.title("🔬 The Funnel")

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
    st.warning(f"No funnel or council data for {ticker}.")
    st.stop()

# Date filter
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

# Trading Mode filter
_has_trading_mode = (
    not council_df.empty and
    'trading_mode_active' in council_df.columns and
    council_df['trading_mode_active'].astype(str).str.lower().isin(['true', 'false']).any()
)
if _has_trading_mode:
    _tm_options = ['ALL', 'LIVE ONLY', 'OBSERVATION ONLY']
    selected_trading_mode = st.sidebar.selectbox(
        "Trading Mode", _tm_options, index=1,
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


# ============================================================
# CONSTANTS
# ============================================================

# Normalise actual_trend_direction which may use UP/DOWN or BULLISH/BEARISH
DIR_NORM = {'UP': 'UP', 'DOWN': 'DOWN', 'BULLISH': 'UP', 'BEARISH': 'DOWN'}

STAGE_SORT_ORDER = {
    'COUNCIL_DECISION': 0, 'CONVICTION_GATE': 1, 'COMPLIANCE_AUDIT': 2,
    'DA_REVIEW': 3, 'CONFIDENCE_THRESHOLD': 4, 'THESIS_COHERENCE': 5,
    'CAPITAL_CHECK': 6, 'STRATEGY_SELECTION': 7, 'ORDER_QUEUED': 8,
    'DRAWDOWN_GATE': 9, 'LIQUIDITY_GATE': 10, 'ORDER_PLACED': 11,
    'PRICE_WALK_STEP': 12, 'ORDER_FILLED': 13, 'ORDER_PARTIAL_FILL': 14,
    'ORDER_CANCELLED': 15, 'POSITION_OPENED': 16, 'RISK_TRIGGER': 17,
    'POSITION_CLOSED': 18, 'PNL_RECONCILED': 19, 'TMS_ORPHAN_DETECTED': 20,
}


def build_dynamic_stage_order(df: pd.DataFrame) -> list:
    """Build stage order from stages actually present in data, sorted canonically."""
    if df.empty or 'stage' not in df.columns:
        return []
    present = df['stage'].dropna().unique().tolist()
    return sorted(present, key=lambda s: STAGE_SORT_ORDER.get(s, 999))


# ============================================================
# FUNNEL BUILDER (unchanged logic)
# ============================================================
def build_true_funnel(
    council_df: pd.DataFrame,
    funnel_df: pd.DataFrame,
    config: dict,
    observation_only: bool = False,
) -> pd.DataFrame:
    """Build monotonically-decreasing cascading funnel from two data sources."""
    min_score = config.get('strategy', {}).get('min_weighted_score_magnitude', 0.20)
    stages = []

    n_total = len(council_df)
    stages.append({'stage': 'Council Decisions', 'survivors': n_total,
                   'source_label': 'council_history'})

    # Actionable
    if 'master_decision' in council_df.columns:
        directional = council_df['master_decision'].isin(['BULLISH', 'BEARISH'])
        vol_play = (council_df.get('prediction_type', pd.Series('DIRECTIONAL', index=council_df.index))
                    .fillna('DIRECTIONAL') == 'VOLATILITY')
        actionable = council_df[directional | vol_play]
    else:
        actionable = council_df
    stages.append({'stage': 'Actionable', 'survivors': len(actionable),
                   'source_label': 'council_history'})

    # Compliance
    if 'compliance_approved' in actionable.columns:
        compliant = actionable[
            actionable['compliance_approved'].astype(str).str.lower().isin(['true', '1', 'yes'])
        ]
    else:
        compliant = actionable
    stages.append({'stage': 'Compliance Passed', 'survivors': len(compliant),
                   'source_label': 'council_history'})

    # Conviction Gate
    if 'weighted_score' in compliant.columns:
        ws = pd.to_numeric(compliant['weighted_score'], errors='coerce').fillna(0)
        convicted = compliant[ws.abs() >= min_score]
    else:
        convicted = compliant
    stages.append({'stage': f'Conviction Gate (\u2265{min_score})', 'survivors': len(convicted),
                   'source_label': 'council_history'})

    # Strategy Selected
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

    if not observation_only:
        n_placed = 0
        if not funnel_df.empty and 'stage' in funnel_df.columns:
            n_placed = len(funnel_df[funnel_df['stage'] == 'ORDER_PLACED'])
        n_placed = min(n_placed, n_strategy)
        stages.append({'stage': 'Orders Placed', 'survivors': n_placed,
                       'source_label': 'execution_funnel'})

        n_filled = 0
        if not funnel_df.empty and 'stage' in funnel_df.columns:
            n_filled = len(funnel_df[funnel_df['stage'] == 'ORDER_FILLED'])
        n_filled = min(n_filled, n_placed)
        stages.append({'stage': 'Orders Filled', 'survivors': n_filled,
                       'source_label': 'execution_funnel'})

    result = pd.DataFrame(stages)
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
# DIAGNOSIS ENGINE — compute all metrics once
# ============================================================
def compute_diagnosis(council_df: pd.DataFrame, funnel_df: pd.DataFrame) -> dict:
    """Single function computing all metrics for the three questions."""
    d = {
        # Q1: Signal quality
        'dir_accuracy_pct': 0.0, 'dir_correct': 0, 'dir_resolved': 0,
        'vol_hit_pct': 0.0, 'vol_wins': 0, 'vol_resolved': 0,
        # Q2: Making money
        'n_resolved': 0, 'n_wins': 0, 'n_losses': 0,
        'win_rate_pct': 0.0, 'total_pnl': 0.0, 'avg_pnl': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0,
        # Q3: Bottleneck components
        'correct_profitable': 0, 'correct_loss': 0,
        'wrong_profitable': 0, 'wrong_loss': 0,
        'unresolved_count': 0, 'unresolved_wins': 0, 'unresolved_losses': 0,
        'alpha_capture_pct': 0.0,
        'worst_trade_pnl': 0.0, 'worst_3_pct_of_loss': 0.0,
        'tail_gate_met': False,
        # Execution (for expander)
        'signal_to_trade_pct': 0.0, 'fill_rate_pct': 0.0,
        'avg_slippage_pct': 0.0, 'conviction_block_pct': 0.0,
        'avg_walk_steps': 0.0, 'alpha_left_count': 0, 'alpha_left_pct': 0.0,
        # Verdict
        'bottleneck': 'insufficient_data', 'bottleneck_msg': '',
        'fix_target': '',
    }

    if council_df.empty:
        return d

    # council_df is pre-filtered to filled trades only (traded_df) by the caller.
    traded_df = council_df

    # --- Q1: Directional accuracy (filled trades only) ---
    if 'actual_trend_direction' in traded_df.columns and 'master_decision' in traded_df.columns:
        directional = traded_df[traded_df['master_decision'].isin(['BULLISH', 'BEARISH'])].copy()
        dir_resolved = directional[
            directional['actual_trend_direction'].str.upper().isin(DIR_NORM)
        ].copy()
        dir_unresolved = directional[
            ~directional['actual_trend_direction'].str.upper().isin(DIR_NORM)
        ].copy()
        n_dir = len(dir_resolved)
        if n_dir > 0:
            actual = dir_resolved['actual_trend_direction'].str.upper().map(DIR_NORM)
            predicted = dir_resolved['master_decision'].str.upper().map({'BULLISH': 'UP', 'BEARISH': 'DOWN'})
            correct_mask = (actual == predicted)
            d['dir_correct'] = int(correct_mask.sum())
            d['dir_resolved'] = n_dir
            d['dir_accuracy_pct'] = d['dir_correct'] / n_dir * 100

            # 2x2 matrix (directional only — resolved trades)
            if 'pnl_realized' in dir_resolved.columns:
                pnl = pd.to_numeric(dir_resolved['pnl_realized'], errors='coerce')
                has_pnl = pnl.notna()
                profitable = pnl > 0
                d['correct_profitable'] = int((correct_mask & profitable & has_pnl).sum())
                d['correct_loss'] = int((correct_mask & ~profitable & has_pnl).sum())
                d['wrong_profitable'] = int((~correct_mask & profitable & has_pnl).sum())
                d['wrong_loss'] = int((~correct_mask & ~profitable & has_pnl).sum())
                n_correct_with_pnl = d['correct_profitable'] + d['correct_loss']
                if n_correct_with_pnl > 0:
                    d['alpha_capture_pct'] = d['correct_profitable'] / n_correct_with_pnl * 100

        # Unresolved directional trades (market flat — actual_trend = NEUTRAL/NaN)
        if 'pnl_realized' in dir_unresolved.columns:
            unr_pnl = pd.to_numeric(dir_unresolved['pnl_realized'], errors='coerce').dropna()
            d['unresolved_count'] = len(unr_pnl)
            d['unresolved_wins'] = int((unr_pnl > 0).sum())
            d['unresolved_losses'] = int((unr_pnl <= 0).sum())

    # --- Q1: Vol play hit rate (filled trades only) ---
    if 'prediction_type' in traded_df.columns and 'pnl_realized' in traded_df.columns:
        vol = traded_df[traded_df['prediction_type'].fillna('DIRECTIONAL') == 'VOLATILITY']
        vol_pnl = pd.to_numeric(vol['pnl_realized'], errors='coerce').dropna()
        if len(vol_pnl) > 0:
            d['vol_wins'] = int((vol_pnl > 0).sum())
            d['vol_resolved'] = len(vol_pnl)
            d['vol_hit_pct'] = d['vol_wins'] / d['vol_resolved'] * 100

    # --- Q2: P&L metrics (filled trades only, including vol plays) ---
    if 'pnl_realized' in traded_df.columns:
        pnl_all = pd.to_numeric(traded_df['pnl_realized'], errors='coerce').dropna()
        if len(pnl_all) > 0:
            wins = pnl_all[pnl_all > 0]
            losses = pnl_all[pnl_all <= 0]
            d['n_resolved'] = len(pnl_all)
            d['n_wins'] = len(wins)
            d['n_losses'] = len(losses)
            d['win_rate_pct'] = d['n_wins'] / d['n_resolved'] * 100
            d['total_pnl'] = float(pnl_all.sum())
            d['avg_pnl'] = float(pnl_all.mean())
            d['avg_win'] = float(wins.mean()) if len(wins) > 0 else 0.0
            d['avg_loss'] = float(losses.mean()) if len(losses) > 0 else 0.0
            gross_wins = float(wins.sum()) if len(wins) > 0 else 0.0
            gross_losses = abs(float(losses.sum())) if len(losses) > 0 else 0.0
            d['profit_factor'] = (gross_wins / gross_losses) if gross_losses > 0 else float('inf')

            # Tail risk
            d['worst_trade_pnl'] = float(pnl_all.min())
            neg = pnl_all[pnl_all < 0]
            if len(neg) >= 8:
                d['tail_gate_met'] = True
                worst_3 = neg.nsmallest(3)
                d['worst_3_pct_of_loss'] = abs(float(worst_3.sum()) / float(neg.sum())) * 100

    # --- Execution metrics (from funnel_df) ---
    if not funnel_df.empty and 'stage' in funnel_df.columns:
        decisions = funnel_df[funnel_df['stage'] == 'COUNCIL_DECISION']
        actionable = decisions[decisions['outcome'] == 'PASS']
        placed = funnel_df[funnel_df['stage'] == 'ORDER_PLACED']
        filled = funnel_df[funnel_df['stage'] == 'ORDER_FILLED']
        cancelled = funnel_df[funnel_df['stage'] == 'ORDER_CANCELLED']

        n_act = len(actionable)
        n_pl = len(placed)
        n_fi = len(filled)
        n_ca = len(cancelled)

        d['signal_to_trade_pct'] = (n_fi / n_act * 100) if n_act > 0 else 0
        d['fill_rate_pct'] = (n_fi / n_pl * 100) if n_pl > 0 else 0
        d['alpha_left_count'] = n_ca
        d['alpha_left_pct'] = (n_ca / n_pl * 100) if n_pl > 0 else 0

        conviction_total = funnel_df[funnel_df['stage'] == 'CONVICTION_GATE']
        conviction_blocks = conviction_total[conviction_total['outcome'] == 'BLOCK']
        d['conviction_block_pct'] = (len(conviction_blocks) / len(conviction_total) * 100) if len(conviction_total) > 0 else 0

        walk_steps = funnel_df[funnel_df['stage'] == 'PRICE_WALK_STEP']
        d['avg_walk_steps'] = (len(walk_steps) * 3 / n_pl) if n_pl > 0 else 0

        filled_with_prices = filled.dropna(subset=['fill_price', 'initial_limit'])
        if not filled_with_prices.empty:
            fill_p = filled_with_prices['fill_price'].astype(float)
            init_p = filled_with_prices['initial_limit'].astype(float)
            slippage_pct = ((fill_p - init_p).abs() / init_p.abs().replace(0, np.nan) * 100).dropna()
            d['avg_slippage_pct'] = float(slippage_pct.mean()) if not slippage_pct.empty else 0

    # --- Determine bottleneck ---
    d = _determine_bottleneck(d)
    return d


def _determine_bottleneck(d: dict) -> dict:
    """Determine primary bottleneck. Priority: tail_risk > execution > signal_quality > sizing > none."""
    if d['n_resolved'] < 5:
        d['bottleneck'] = 'insufficient_data'
        d['bottleneck_msg'] = f"Need 5+ resolved trades for diagnosis (currently {d['n_resolved']})"
        d['fix_target'] = ''
        return d

    # Tail risk: worst 3 trades > 50% of all losses (only when >= 8 losses)
    if d['tail_gate_met'] and d['worst_3_pct_of_loss'] > 50:
        d['bottleneck'] = 'tail_risk'
        d['bottleneck_msg'] = (
            f"Worst 3 trades account for {d['worst_3_pct_of_loss']:.0f}% of total losses"
        )
        d['fix_target'] = "Add max loss stops or reduce position size on low-conviction trades"
        return d

    # Execution leakage: correct direction but losing money
    n_correct_with_pnl = d['correct_profitable'] + d['correct_loss']
    if n_correct_with_pnl >= 5 and d['alpha_capture_pct'] < 70:
        d['bottleneck'] = 'execution'
        d['bottleneck_msg'] = (
            f"Only {d['alpha_capture_pct']:.0f}% of correct signals made money "
            f"({d['correct_loss']} correct signals lost money)"
        )
        d['fix_target'] = "Review strike selection, theta decay, entry timing, and hold duration"
        return d

    # Signal quality
    if d['dir_resolved'] >= 5 and d['dir_accuracy_pct'] < 55:
        d['bottleneck'] = 'signal_quality'
        d['bottleneck_msg'] = (
            f"Directional accuracy is {d['dir_accuracy_pct']:.0f}% (below 55% threshold)"
        )
        if d['avg_loss'] != 0 and d['avg_win'] != 0:
            breakeven = abs(d['avg_loss']) / (d['avg_win'] + abs(d['avg_loss'])) * 100
            d['fix_target'] = f"Need {breakeven:.0f}% accuracy to break even at current win/loss ratio"
        else:
            d['fix_target'] = "Improve council decision quality"
        return d

    # Win/loss sizing
    if d['profit_factor'] < 1.0:
        d['bottleneck'] = 'sizing'
        d['bottleneck_msg'] = (
            f"Wins average {d['avg_win']:.2f} but losses average {d['avg_loss']:.2f}"
        )
        d['fix_target'] = "Let winners run longer or cut losers faster"
        return d

    d['bottleneck'] = 'none'
    d['bottleneck_msg'] = "System is performing well"
    d['fix_target'] = "Consider scaling up position size"
    return d


# ============================================================
# COMPUTE ONCE
# ============================================================
config = get_config()
_observation_only = (selected_trading_mode == 'OBSERVATION ONLY')
funnel_cascade = build_true_funnel(council_df, funnel_df, config, observation_only=_observation_only)

# Build traded_df: council decisions that resulted in actual fills.
# Used by Q1/Q2 metrics, Outcome Breakdown, P&L histogram, and Skill/Luck scatter.
if (not funnel_df.empty and 'stage' in funnel_df.columns
        and 'cycle_id' in funnel_df.columns and 'cycle_id' in council_df.columns):
    _filled_cycles = set(funnel_df[funnel_df['stage'] == 'ORDER_FILLED']['cycle_id'].dropna())
    traded_df = council_df[council_df['cycle_id'].isin(_filled_cycles)] if _filled_cycles else council_df.iloc[0:0]
else:
    traded_df = council_df  # fallback when no funnel data

diag = compute_diagnosis(traded_df, funnel_df)


# ============================================================
# RENDER: VERDICT BANNER
# ============================================================
_bn = diag['bottleneck']
if _bn == 'insufficient_data':
    st.info(f"📊 {diag['bottleneck_msg']}")
elif _bn == 'tail_risk':
    st.error(f"💥 **Tail risk**: {diag['bottleneck_msg']}")
elif _bn == 'execution':
    st.warning(f"⚠️ **Execution is leaking alpha**: {diag['bottleneck_msg']}")
elif _bn == 'signal_quality':
    st.warning(f"⚠️ **Signal quality is the bottleneck**: {diag['bottleneck_msg']}")
elif _bn == 'sizing':
    st.warning(f"📐 **Win/loss sizing issue**: {diag['bottleneck_msg']}")
else:
    st.success(f"✅ {diag['bottleneck_msg']}")

if diag['fix_target']:
    st.caption(f"**Recommended fix**: {diag['fix_target']}")


# ============================================================
# RENDER: Q1 & Q2 SIDE-BY-SIDE
# ============================================================
q1_col, q2_col = st.columns(2)

with q1_col:
    st.subheader("Q1: Are signals good?")
    _q1a, _q1b = st.columns(2)
    _q1a.metric(
        "🧭 Directional Accuracy",
        f"{diag['dir_accuracy_pct']:.0f}%" if diag['dir_resolved'] > 0 else "—",
        delta=f"{diag['dir_correct']}/{diag['dir_resolved']} correct" if diag['dir_resolved'] > 0 else None,
        delta_color="off",
    )
    _q1b.metric(
        "⚡ Vol Play Hit Rate",
        f"{diag['vol_hit_pct']:.0f}%" if diag['vol_resolved'] > 0 else "—",
        delta=f"{diag['vol_wins']}/{diag['vol_resolved']} profitable" if diag['vol_resolved'] > 0 else None,
        delta_color="off",
    )
    # Show trade breakdown so the denominator is transparent
    _n_traded = len(traded_df)
    _n_dir = len(traded_df[traded_df['master_decision'].isin(['BULLISH', 'BEARISH'])]) if 'master_decision' in traded_df.columns else 0
    _n_vol = len(traded_df[traded_df['prediction_type'].fillna('DIRECTIONAL') == 'VOLATILITY']) if 'prediction_type' in traded_df.columns else 0
    _n_unresolved = _n_dir - diag['dir_resolved']
    _parts = []
    if diag['dir_resolved'] > 0:
        _parts.append(f"{diag['dir_resolved']} directional resolved")
    if _n_unresolved > 0:
        _parts.append(f"{_n_unresolved} unresolved (market flat)")
    if _n_vol > 0:
        _parts.append(f"{_n_vol} vol plays")
    if _parts:
        st.caption(f"Based on **{_n_traded} filled trades**: {', '.join(_parts)}")

with q2_col:
    st.subheader("Q2: Are we making money?")
    _q2a, _q2b = st.columns(2)
    _q2a.metric(
        "🏆 Win Rate",
        f"{diag['win_rate_pct']:.0f}%" if diag['n_resolved'] > 0 else "—",
        delta=f"{diag['n_wins']}W / {diag['n_losses']}L" if diag['n_resolved'] > 0 else None,
        delta_color="off",
    )
    # P&L is in underlying price units (cents/lb for KC, $/mmBtu for NG).
    # This measures signal quality (how much did price move in our direction),
    # NOT actual dollar P&L — spreads capture only a fraction of the move.
    try:
        from config.commodity_profiles import get_commodity_profile
        _profile = get_commodity_profile(ticker)
        _pnl_unit = _profile.contract.unit
    except Exception:
        _pnl_unit = "pts"

    _pnl_color = "normal" if diag['total_pnl'] >= 0 else "inverse"
    _q2b.metric(
        "💰 Signal P&L",
        f"{diag['total_pnl']:+.2f} {_pnl_unit}" if diag['n_resolved'] > 0 else "—",
        delta=f"{diag['avg_pnl']:+.3f}/trade ({diag['n_resolved']} trades)" if diag['n_resolved'] > 0 else None,
        delta_color=_pnl_color,
        help="Underlying price movement in our predicted direction. Measures signal quality, not actual dollar P&L (spreads capture only a fraction of the move).",
    )

st.markdown("---")


# ============================================================
# RENDER: Q3 — WHAT'S BLOCKING US?
# ============================================================
st.subheader("Q3: What's blocking us?")

_b1, _b2, _b3 = st.columns(3)

with _b1:
    _sig_ok = diag['dir_accuracy_pct'] >= 55 or diag['dir_resolved'] < 5
    _sig_icon = "✅" if _sig_ok else "⚠️"
    st.metric(
        f"{_sig_icon} Signal Quality",
        f"{diag['dir_accuracy_pct']:.0f}%" if diag['dir_resolved'] > 0 else "—",
        help="Directional accuracy — need 55%+ for viable edge",
    )
    if _bn == 'signal_quality':
        st.caption("← **Primary issue**")

with _b2:
    _n_correct_pnl = diag['correct_profitable'] + diag['correct_loss']
    _alpha_ok = diag['alpha_capture_pct'] >= 70 or _n_correct_pnl < 5
    _alpha_icon = "✅" if _alpha_ok else "⚠️"
    st.metric(
        f"{_alpha_icon} Alpha Capture",
        f"{diag['alpha_capture_pct']:.0f}%" if _n_correct_pnl > 0 else "—",
        help="% of correct-direction signals that made money",
    )
    if diag['correct_loss'] > 0:
        st.caption(f"{diag['correct_loss']} correct signals lost money")
    if _bn == 'execution':
        st.caption("← **Primary issue**")

with _b3:
    _tail_ok = not diag['tail_gate_met'] or diag['worst_3_pct_of_loss'] <= 50
    _tail_icon = "✅" if _tail_ok else "⚠️"
    if diag['tail_gate_met']:
        st.metric(
            f"{_tail_icon} Tail Risk",
            f"{diag['worst_3_pct_of_loss']:.0f}%",
            help="% of total loss from worst 3 trades (< 50% = well-distributed)",
        )
    else:
        st.metric("✅ Tail Risk", "—", help="Need 8+ losing trades to assess")
        st.caption("Insufficient data")
    if _bn == 'tail_risk':
        st.caption("← **Primary issue**")


# ============================================================
# RENDER: OUTCOME BREAKDOWN (2x2 matrix, directional only)
# ============================================================
st.markdown("---")
st.subheader("📋 Outcome Breakdown")

_matrix_resolved = diag['correct_profitable'] + diag['correct_loss'] + diag['wrong_profitable'] + diag['wrong_loss']
_matrix_total = _matrix_resolved + diag['unresolved_count'] + diag['vol_resolved']

if _matrix_total >= 1:
    st.caption(
        "Filled trades breakdown. **Resolved**: market moved clearly up/down. "
        "**Flat market**: direction inconclusive. **Vol plays**: non-directional strategies."
    )
    _m1, _m2, _m3, _m4, _m5 = st.columns(5)
    _m1.metric("🎯 Correct + Profit", diag['correct_profitable'],
               delta=f"{diag['correct_profitable'] / max(_matrix_total, 1) * 100:.0f}%",
               delta_color="normal", help="Correct direction, profitable trade.")
    _m2.metric("⚠️ Correct + Loss", diag['correct_loss'],
               delta=f"{diag['correct_loss'] / max(_matrix_total, 1) * 100:.0f}%",
               delta_color="inverse", help="Right direction but lost money (theta, strikes, timing).")
    _m3.metric("❌ Wrong + Loss", diag['wrong_loss'],
               delta=f"{diag['wrong_loss'] / max(_matrix_total, 1) * 100:.0f}%",
               delta_color="inverse", help="Wrong direction, expected loss.")
    _m4.metric("🔲 Flat Market", diag['unresolved_count'],
               delta=f"{diag['unresolved_wins']}W / {diag['unresolved_losses']}L" if diag['unresolved_count'] > 0 else None,
               delta_color="off",
               help="Market didn't move clearly — direction accuracy can't be assessed.")
    _m5_parts = []
    if diag['vol_resolved'] > 0:
        _m5_parts.append(f"{diag['vol_wins']}/{diag['vol_resolved']} profitable")
    if diag['wrong_profitable'] > 0:
        _m5_parts.append(f"{diag['wrong_profitable']} lucky (wrong+profit)")
    _m5.metric("⚡ Vol / Other", diag['vol_resolved'] + diag['wrong_profitable'],
               delta=", ".join(_m5_parts) if _m5_parts else None,
               delta_color="off",
               help="Vol plays + wrong-direction-but-profitable trades.")
elif diag['n_resolved'] > 0:
    st.info(f"No resolved directional trades yet (only {diag['n_resolved']} total trades)")
else:
    st.info("No resolved trades yet.")


# ============================================================
# RENDER: FUNNEL WATERFALL (visible, not collapsed)
# ============================================================
st.markdown("---")
st.subheader("🌊 Funnel — Where Do Signals Die?")

if not funnel_cascade.empty and funnel_cascade['survivors'].sum() > 0:
    first_count = int(funnel_cascade.iloc[0]['survivors'])

    # Dynamic colors
    _GATE_COLORS = ['#2ecc71', '#27ae60', '#1abc9c', '#16a085', '#2980b9']
    _EXEC_COLORS = ['#3498db', '#5dade2']
    _gi, _ei = 0, 0
    _stage_colors = []
    for _, _sr in funnel_cascade.iterrows():
        if _sr.get('source_label') == 'execution_funnel':
            _stage_colors.append(_EXEC_COLORS[_ei % 2]); _ei += 1
        else:
            _stage_colors.append(_GATE_COLORS[_gi % len(_GATE_COLORS)]); _gi += 1

    custom_text = []
    for idx in range(len(funnel_cascade)):
        row = funnel_cascade.iloc[idx]
        val = int(row['survivors'])
        pct = val / max(first_count, 1) * 100
        if idx == 0:
            custom_text.append(f"<b>{val}</b>")
        else:
            prev_val = int(funnel_cascade.iloc[idx - 1]['survivors'])
            pct_prev = val / max(prev_val, 1) * 100
            custom_text.append(f"<b>{val}</b>  ({pct:.0f}% of initial, {pct_prev:.0f}% of prev)")

    fig = go.Figure(go.Funnel(
        y=funnel_cascade['stage'], x=funnel_cascade['survivors'],
        text=custom_text, textinfo="text", textfont=dict(size=14),
        marker=dict(color=_stage_colors),
        connector=dict(line=dict(color="gray", dash="dot", width=1)),
    ))
    fig.update_layout(
        height=max(400, 55 * len(funnel_cascade)),
        margin=dict(t=20, b=20, l=10, r=10),
        funnelmode="stack", font=dict(size=13),
    )
    st.plotly_chart(fig, use_container_width=True)

    if _observation_only:
        st.info(
            "📊 **Observation Mode** — trading was OFF. "
            "Execution stages hidden. Use Signal Quality metrics above."
        )

    if not _observation_only:
        filled_row = funnel_cascade[funnel_cascade['stage'] == 'Orders Filled']
        filled_count = int(filled_row['survivors'].values[0]) if not filled_row.empty else 0
        st.caption(f"Signal-to-fill survival: **{filled_count}/{first_count} ({filled_count / max(first_count, 1) * 100:.0f}%)**")
else:
    st.info("Insufficient data to build funnel.")


# ============================================================
# COLLAPSED DETAILS
# ============================================================
st.markdown("---")

# --- Execution KPIs ---
with st.expander("📈 Execution KPIs", expanded=False):
    _e1, _e2, _e3, _e4, _e5, _e6 = st.columns(6)
    _e1.metric("📡 Signal-to-Trade", f"{diag['signal_to_trade_pct']:.0f}%",
               help="% of actionable signals → filled orders")
    _e2.metric("⛽ Fill Rate", f"{diag['fill_rate_pct']:.0f}%",
               help="% of placed orders that filled")
    _e3.metric("📉 Avg Walk Cost", f"{diag['avg_slippage_pct']:.1f}%",
               help="Avg fill price vs initial limit (adaptive walk distance). Higher = more price concession to get filled.")
    _e4.metric("🛡️ Conviction Blocks", f"{diag['conviction_block_pct']:.0f}%",
               help="% of signals blocked by conviction gate")
    _e5.metric("👣 Avg Walk Steps", f"{diag['avg_walk_steps']:.0f}",
               help="Avg adaptive walk steps per order")
    _e6.metric("💸 Alpha Left", f"{diag['alpha_left_count']}",
               delta=f"-{diag['alpha_left_pct']:.0f}% of placed" if diag['alpha_left_count'] > 0 else None,
               delta_color="inverse",
               help="Orders that passed gates but didn't fill")

# --- P&L Details ---
with st.expander("💵 P&L Details", expanded=False):
    if diag['n_resolved'] > 0:
        _p1, _p2, _p3, _p4 = st.columns(4)
        _p1.metric("Avg Win", f"+{diag['avg_win']:.2f} {_pnl_unit}" if diag['n_wins'] > 0 else "—")
        _p2.metric("Avg Loss", f"{diag['avg_loss']:.2f} {_pnl_unit}" if diag['n_losses'] > 0 else "—")
        _pf = diag['profit_factor']
        _p3.metric("Profit Factor",
                    f"{_pf:.2f}x" if _pf != float('inf') else "No losses",
                    help=">1.0 = profitable. >1.5 = good. >2.0 = excellent.")
        _p4.metric("Worst Trade", f"{diag['worst_trade_pnl']:+.2f} {_pnl_unit}")

        # P&L histogram (filled trades only)
        if diag['n_resolved'] >= 5 and 'pnl_realized' in traded_df.columns:
            pnl_data = pd.to_numeric(traded_df['pnl_realized'], errors='coerce').dropna()
            if not pnl_data.empty:
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Histogram(x=pnl_data, nbinsx=25, marker_color='#3498db'))
                fig_pnl.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
                fig_pnl.add_vline(x=diag['avg_pnl'], line_dash="dot", line_color="#f39c12",
                                  annotation_text=f"Mean: {diag['avg_pnl']:+.2f}")
                fig_pnl.update_layout(height=250, margin=dict(t=30, b=30, l=40, r=20),
                                      xaxis_title=f"P&L ({_pnl_unit})", yaxis_title="Count", showlegend=False)
                st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.info("No resolved trades yet.")

# --- Correct+Loss Drill-Down ---
if diag['correct_loss'] >= 2 and 'actual_trend_direction' in traded_df.columns:
    with st.expander(
        f"🔍 Correct Direction but Lost ({diag['correct_loss']} trades)",
        expanded=(diag['correct_loss'] >= 3 and _bn == 'execution'),
    ):
        directional = traded_df[traded_df['master_decision'].isin(['BULLISH', 'BEARISH'])].copy()
        dir_resolved = directional[directional['actual_trend_direction'].str.upper().isin(DIR_NORM)].copy()
        if not dir_resolved.empty and 'pnl_realized' in dir_resolved.columns:
            actual = dir_resolved['actual_trend_direction'].str.upper().map(DIR_NORM)
            predicted = dir_resolved['master_decision'].str.upper().map({'BULLISH': 'UP', 'BEARISH': 'DOWN'})
            dir_resolved['pnl'] = pd.to_numeric(dir_resolved['pnl_realized'], errors='coerce')
            correct_loss_df = dir_resolved[(actual == predicted) & (dir_resolved['pnl'] <= 0) & dir_resolved['pnl'].notna()]

            if not correct_loss_df.empty:
                group_cols = [c for c in ['strategy_type', 'entry_regime', 'master_decision']
                              if c in correct_loss_df.columns and correct_loss_df[c].notna().any()]
                for gc in group_cols:
                    grp = (correct_loss_df.groupby(gc)['pnl']
                           .agg(count='count', avg_pnl='mean', total_pnl='sum')
                           .reset_index().sort_values('count', ascending=False))
                    grp.columns = [gc.replace('_', ' ').title(), 'Count', 'Avg P&L', 'Total P&L']
                    st.caption(f"By **{gc.replace('_', ' ')}**")
                    st.dataframe(
                        grp.style.format({'Avg P&L': '{:+.2f}', 'Total P&L': '{:+.2f}'}),
                        hide_index=True, use_container_width=True,
                    )

                raw_cols = [c for c in ['timestamp', 'contract', 'master_decision',
                                        'strategy_type', 'entry_regime', 'pnl',
                                        'master_confidence', 'weighted_score']
                            if c in correct_loss_df.columns]
                st.caption("Individual trades")
                cl_display = correct_loss_df[raw_cols].sort_values('pnl').copy()
                if 'timestamp' in cl_display.columns:
                    cl_display['timestamp'] = cl_display['timestamp'].astype(str).str[:16]
                fmt = {k: v for k, v in {'pnl': '{:+.2f}', 'master_confidence': '{:.2f}',
                                          'weighted_score': '{:.2f}'}.items() if k in cl_display.columns}
                st.dataframe(cl_display.style.format(fmt), hide_index=True, use_container_width=True)

# --- Signal vs Outcome Scatter ---
with st.expander("🎯 Signal vs Outcome — Skill or Luck?", expanded=False):
    if not traded_df.empty and 'pnl_realized' in traded_df.columns and 'weighted_score' in traded_df.columns:
        resolved = traded_df[traded_df['pnl_realized'].notna()].copy()
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
                resolved, x='process_score', y='pnl', color='quadrant',
                size=resolved['pnl'].abs().clip(lower=0.1),
                hover_data=['cycle_id', 'contract', 'strategy_type', 'master_decision'],
                color_discrete_map={'Skill': '#2ecc71', 'Lucky': '#f39c12',
                                    'Market Risk': '#e74c3c', 'Bad Call': '#95a5a6'},
                labels={'process_score': 'Process Score', 'pnl': 'P&L'},
            )
            fig_matrix.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_matrix.add_vline(x=process_median, line_dash="dash", line_color="gray", opacity=0.5)
            fig_matrix.update_layout(height=400, margin=dict(t=20))
            st.plotly_chart(fig_matrix, use_container_width=True)

            quad_counts = resolved['quadrant'].value_counts()
            _qc1, _qc2, _qc3, _qc4 = st.columns(4)
            _qc1.metric("✅ Skill", quad_counts.get('Skill', 0), help="Good process + positive P&L")
            _qc2.metric("🍀 Lucky", quad_counts.get('Lucky', 0), help="Weak process + positive P&L")
            _qc3.metric("📉 Market Risk", quad_counts.get('Market Risk', 0), help="Good process, adverse move")
            _qc4.metric("❌ Bad Call", quad_counts.get('Bad Call', 0), help="Weak process + negative P&L")
    else:
        st.info("Insufficient data for matrix analysis.")

# --- Regime Comparison ---
if not funnel_df.empty and 'regime' in funnel_df.columns:
    regime_values = [r for r in funnel_df['regime'].dropna().unique() if r and str(r).upper() != 'UNKNOWN']
    if len(regime_values) >= 2:
        with st.expander("🎭 Regime Comparison", expanded=False):
            # Build cycle_id → regime mapping from COUNCIL_DECISION rows
            # (ORDER_FILLED rows have regime=UNKNOWN, so we resolve via cycle_id)
            _cycle_regime = {}
            if 'cycle_id' in funnel_df.columns:
                # ⚡ Bolt: vectorized dict generation via .dropna and dict(zip()) is ~40x faster than .iterrows()
                _c_df = funnel_df[(funnel_df['stage'] == 'COUNCIL_DECISION') & (funnel_df['regime'].str.upper() != 'UNKNOWN')].dropna(subset=['cycle_id', 'regime'])
                _cycle_regime = dict(zip(_c_df['cycle_id'], _c_df['regime']))

            regime_survival = []
            for regime in sorted(regime_values):
                rdf = funnel_df[funnel_df['regime'] == regime]
                n_dec = len(rdf[(rdf['stage'] == 'COUNCIL_DECISION') & rdf['outcome'].isin(['PASS', 'BLOCK', 'INFO'])])
                # Count fills whose cycle_id maps to this regime
                fills = funnel_df[funnel_df['stage'] == 'ORDER_FILLED']
                if 'cycle_id' in fills.columns and _cycle_regime:
                    n_filled = int(fills['cycle_id'].map(_cycle_regime).eq(regime).sum())
                else:
                    n_filled = len(rdf[rdf['stage'] == 'ORDER_FILLED'])
                regime_survival.append({
                    'Regime': str(regime).title(),
                    'Decisions': n_dec, 'Filled': n_filled,
                    'Survival %': f"{n_filled / max(n_dec, 1) * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(regime_survival), hide_index=True, use_container_width=True)

# --- Raw Data Explorer ---
with st.expander("📄 Raw Funnel Data", expanded=False):
    if not funnel_df.empty:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            stages = ['ALL'] + build_dynamic_stage_order(funnel_df)
            sel_stage = st.selectbox("Filter by Stage", stages)
        with filter_col2:
            cycle_ids = funnel_df['cycle_id'].dropna().unique().tolist() if 'cycle_id' in funnel_df.columns else []
            cycle_options = ['ALL'] + sorted(set(cycle_ids), reverse=True)[:50]
            sel_cycle = st.selectbox("Drill into Cycle ID", cycle_options)

        display_df = funnel_df.copy()
        if sel_stage != 'ALL':
            display_df = display_df[display_df['stage'] == sel_stage]
        if sel_cycle != 'ALL':
            display_df = display_df[display_df['cycle_id'] == sel_cycle]

        sort_asc = (sel_cycle != 'ALL')
        _raw = display_df.sort_values('timestamp', ascending=sort_asc).head(200).copy()
        if 'timestamp' in _raw.columns:
            _raw['timestamp'] = _raw['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(_raw, hide_index=True, use_container_width=True)

        if sel_cycle != 'ALL' and not display_df.empty:
            journey = display_df.sort_values('timestamp')[['timestamp', 'stage', 'outcome', 'detail']].copy()
            journey['timestamp'] = journey['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.markdown(f"**Cycle Journey: `{sel_cycle}`** ({len(journey)} events)")
            st.dataframe(journey, hide_index=True, use_container_width=True)

        st.download_button(
            "Download CSV", display_df.to_csv(index=False).encode('utf-8'),
            f"execution_funnel_{ticker}.csv", "text/csv",
            help="Download execution funnel data as CSV",
        )
    else:
        st.info("No funnel data available.")
