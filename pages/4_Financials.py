"""
Page 4: Trade Analytics (Per-Commodity)

Purpose: Strategy performance, trade breakdown, and execution ledger for the
selected commodity. Account-level financials (equity curve, benchmarks, risk
metrics) live on the Portfolio home page.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import (
    load_trade_data,
    load_council_history,
    get_config,
    grade_decision_quality
)

st.set_page_config(layout="wide", page_title="Trade Analytics | Real Options")

from _commodity_selector import selected_commodity
ticker = selected_commodity()

st.title("\U0001f4c8 Trade Analytics")
st.caption("Per-commodity strategy performance, trade breakdown, and execution ledger")

# --- Load Data ---
trade_df = load_trade_data(ticker=ticker)
council_df = load_council_history(ticker=ticker)
config = get_config()

st.markdown("---")

# === SECTION 1: Key Metrics (per-commodity) ===
st.subheader("\U0001f4ca Performance Summary")

metric_cols = st.columns(3)

with metric_cols[0]:
    # Primary: Use trade_ledger if available
    # Fallback: Count reconciled trades from council_history
    if not trade_df.empty:
        trade_count = len(trade_df)
    elif not council_df.empty and 'pnl_realized' in council_df.columns:
        # Count trades that have been reconciled (have P&L data)
        trade_count = council_df['pnl_realized'].notna().sum()
    else:
        trade_count = 0
    st.metric("Total Trades", trade_count, help="Total number of reconciled trades with P&L data.")

with metric_cols[1]:
    # Calculate win rate from council history
    if not council_df.empty and 'pnl_realized' in council_df.columns:
        reconciled = council_df[pd.notna(council_df['pnl_realized'])]
        if not reconciled.empty:
            win_rate = (reconciled['pnl_realized'] > 0).mean() * 100
            st.metric("Win Rate", f"{win_rate:.1f}%", help="Percentage of reconciled trades that resulted in a positive P&L.")
        else:
            st.metric("Win Rate", "N/A", help="Needs reconciled trade data to calculate.")
    else:
        st.metric("Win Rate", "N/A", help="Needs reconciled trade data to calculate.")

with metric_cols[2]:
    # Sum realized P&L from reconciled trades
    if not council_df.empty and 'pnl_realized' in council_df.columns:
        realized_pnl = council_df['pnl_realized'].fillna(0).sum()
        st.metric(
            "Realized P&L",
            f"${realized_pnl:+,.2f}",
            help="Sum of P&L from reconciled trades in council_history"
        )
    else:
        st.metric("Realized P&L", "$0.00")

# Pre-compute graded trades for sections that need it
graded_fin = grade_decision_quality(council_df) if not council_df.empty else pd.DataFrame()

st.markdown("---")


# === SECTION 2: Strategy Efficiency (ROI by Strategy Type) ===
st.subheader("\U0001f3af Strategy Efficiency by Type")

if not council_df.empty and 'strategy_type' in council_df.columns and 'pnl_realized' in council_df.columns:
    # Group by strategy type
    strategy_perf = council_df.groupby('strategy_type').agg({
        'pnl_realized': ['sum', 'mean', 'count']
    })
    strategy_perf.columns = ['Total P&L', 'Avg P&L', 'Trade Count']
    strategy_perf = strategy_perf.reset_index()

    # Create bar chart
    fig = px.bar(
        strategy_perf,
        x='strategy_type',
        y='Total P&L',
        color='Total P&L',
        color_continuous_scale='RdYlGn',
        text='Trade Count',
        title='P&L by Strategy Type'
    )

    fig.update_traces(texttemplate='%{text} trades', textposition='outside')
    st.plotly_chart(fig, width="stretch")

    # Detailed table
    st.dataframe(strategy_perf)

    # Rolling win rate by strategy (time dimension)
    if not graded_fin.empty and 'strategy_type' in graded_fin.columns:
        strat_resolved = graded_fin[graded_fin['outcome'].isin(['WIN', 'LOSS'])].copy()
        strat_resolved = strat_resolved.sort_values('timestamp').reset_index(drop=True)

        if len(strat_resolved) > 20:
            strat_resolved['is_win'] = (strat_resolved['outcome'] == 'WIN').astype(int)
            strat_resolved['trade_num'] = range(1, len(strat_resolved) + 1)

            strategies_present = strat_resolved['strategy_type'].unique()
            strat_colors = {
                'BULL_CALL_SPREAD': '#00CC96',
                'BEAR_PUT_SPREAD': '#EF553B',
                'LONG_STRADDLE': '#AB63FA',
                'IRON_CONDOR': '#636EFA',
            }
            strat_pretty = {
                'BULL_CALL_SPREAD': 'Bull Call Spread',
                'BEAR_PUT_SPREAD': 'Bear Put Spread',
                'LONG_STRADDLE': 'Long Straddle',
                'IRON_CONDOR': 'Iron Condor',
            }

            fig_roll = go.Figure()
            _declining = []

            for strat in strategies_present:
                s_df = strat_resolved[strat_resolved['strategy_type'] == strat].copy()
                if len(s_df) < 5:
                    continue
                s_df['rolling_wr'] = s_df['is_win'].rolling(window=min(10, len(s_df)), min_periods=3).mean() * 100
                s_df = s_df[s_df['rolling_wr'].notna()]

                if s_df.empty:
                    continue

                pretty = strat_pretty.get(strat, strat)
                fig_roll.add_trace(go.Scatter(
                    x=s_df['trade_num'], y=s_df['rolling_wr'],
                    name=pretty,
                    line=dict(color=strat_colors.get(strat, '#FFFFFF'), width=2)
                ))

                # Check for declining trend
                if len(s_df) >= 10:
                    first_q = s_df['rolling_wr'].iloc[:len(s_df)//3].mean()
                    last_q = s_df['rolling_wr'].iloc[-len(s_df)//3:].mean()
                    if first_q - last_q > 15:
                        _declining.append(f"{pretty} win rate declining (was {first_q:.0f}%, now {last_q:.0f}%)")

            if fig_roll.data:
                fig_roll.add_hline(y=50, line_dash="dot", line_color="gray")
                fig_roll.update_layout(
                    title='Rolling Win Rate by Strategy (window=10)',
                    height=350,
                    xaxis=dict(title='Trade #'),
                    yaxis=dict(title='Win Rate %', range=[0, 100]),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=0, r=0, t=60, b=0)
                )
                st.plotly_chart(fig_roll, width='stretch')

                for warning_msg in _declining:
                    st.warning(warning_msg)

# === Trade Type Breakdown ===
st.subheader("\U0001f4ca Directional vs Volatility Performance")

if 'prediction_type' in council_df.columns:
    type_perf = council_df.groupby('prediction_type').agg({
        'pnl_realized': ['sum', 'mean', 'count']
    })
    type_perf.columns = ['Total P&L', 'Avg P&L', 'Trade Count']

    col1, col2 = st.columns(2)

    with col1:
        dir_data = type_perf.loc['DIRECTIONAL'] if 'DIRECTIONAL' in type_perf.index else None
        if dir_data is not None:
            st.metric("Directional P&L", f"${dir_data['Total P&L']:,.2f}", help="Total P&L from directional trades (Bull Call / Bear Put Spreads).")
            st.caption(f"{int(dir_data['Trade Count'])} trades | Avg: ${dir_data['Avg P&L']:,.2f}")
        else:
            st.metric("Directional P&L", "$0.00", help="Total P&L from directional trades (Bull Call / Bear Put Spreads).")
            st.caption("No directional trades yet")

    with col2:
        vol_data = type_perf.loc['VOLATILITY'] if 'VOLATILITY' in type_perf.index else None
        if vol_data is not None:
            st.metric("Volatility P&L", f"${vol_data['Total P&L']:,.2f}", help="Total P&L from volatility trades (Long Straddle / Iron Condor).")
            st.caption(f"{int(vol_data['Trade Count'])} trades | Avg: ${vol_data['Avg P&L']:,.2f}")
        else:
            st.metric("Volatility P&L", "$0.00", help="Total P&L from volatility trades (Long Straddle / Iron Condor).")
            st.caption("No volatility trades yet")
else:
    st.info("No trade type data available.")

st.markdown("---")

# === Win/Loss Ratio (per-commodity, from graded trades) ===
st.subheader("Win/Loss Ratio")
st.caption("Are winners bigger than losers?")

if not graded_fin.empty:
    pnl_c = 'pnl' if 'pnl' in graded_fin.columns else 'pnl_realized'
    if pnl_c in graded_fin.columns:
        fin_resolved = graded_fin[graded_fin['outcome'].isin(['WIN', 'LOSS'])].copy()
        fin_resolved['_pnl'] = pd.to_numeric(fin_resolved[pnl_c], errors='coerce')
        fin_resolved = fin_resolved[fin_resolved['_pnl'].notna()]

        if len(fin_resolved) >= 10:
            avg_win = fin_resolved.loc[fin_resolved['_pnl'] > 0, '_pnl'].mean()
            avg_loss = fin_resolved.loc[fin_resolved['_pnl'] < 0, '_pnl'].abs().mean()
            if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss > 0:
                _wl_ratio = avg_win / avg_loss
                st.metric(
                    "Win/Loss Ratio", f"{_wl_ratio:.2f}",
                    help=">1.5 means winners are bigger than losers"
                )
            else:
                st.metric("Win/Loss Ratio", "N/A", help="Need both winning and losing trades")
        else:
            st.info("Need 10+ graded trades with P&L for Win/Loss ratio.")
    else:
        st.info("No P&L data available.")
else:
    st.info("No graded trades available.")

st.markdown("---")

# === Trade Ledger ===
st.subheader("\U0001f4cb Trade Ledger")

if not trade_df.empty:
    # Primary: Show from trade_ledger.csv
    display_cols = ['timestamp', 'local_symbol', 'action', 'quantity', 'price', 'total_value_usd']
    display_cols = [c for c in display_cols if c in trade_df.columns]
    st.dataframe(
        trade_df[display_cols].sort_values('timestamp', ascending=False).head(50),
        width="stretch"
    )

elif not council_df.empty and 'pnl_realized' in council_df.columns:
    # Fallback: Show reconciled trades from council_history
    reconciled_trades = council_df[council_df['pnl_realized'].notna()].copy()

    if not reconciled_trades.empty:
        display_cols = ['timestamp', 'contract', 'strategy_type', 'master_decision',
                        'entry_price', 'exit_price', 'pnl_realized']
        display_cols = [c for c in display_cols if c in reconciled_trades.columns]

        st.dataframe(
            reconciled_trades[display_cols].sort_values('timestamp', ascending=False).head(50),
            width="stretch"
        )
        st.caption("\u26a0\ufe0f Showing reconciled trades from council_history (trade_ledger.csv is empty)")
    else:
        st.info("No trades recorded.")
else:
    st.info("No trades recorded.")
