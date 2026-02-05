"""
Page 4: Financial Performance (ROI & Audit)

Purpose: The "Rearview Mirror" - Institutional-grade reporting on actual dollars.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import calendar
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import (
    load_equity_data,
    load_trade_data,
    load_council_history,
    fetch_benchmark_data,
    fetch_live_dashboard_data,
    get_config,
    DEFAULT_STARTING_CAPITAL
)

st.set_page_config(layout="wide", page_title="Financials | Mission Control")

st.title("📈 Financial Performance")
st.caption("ROI & Audit - Institutional-grade reporting on actual dollars gained or lost")

# --- Load Data ---
equity_df = load_equity_data()
trade_df = load_trade_data()
council_df = load_council_history()
config = get_config()

# E4 FIX: Get starting capital from config/profile
from dashboard_utils import get_starting_capital
starting_capital = get_starting_capital(config)

if not equity_df.empty:
    # If we have equity history, use the first point as truth
    starting_capital = equity_df.iloc[0]['total_value_usd']

st.markdown("---")

# === SECTION 1: Key Metrics ===
st.subheader("📊 Performance Summary")

metric_cols = st.columns(6)

with metric_cols[0]:
    if config:
        live_data = fetch_live_dashboard_data(config)
        net_liq = live_data['NetLiquidation']
        st.metric("Net Liquidation", f"${net_liq:,.0f}")
    else:
        net_liq = equity_df['total_value_usd'].iloc[-1] if not equity_df.empty else 0
        st.metric("Net Liquidation", f"${net_liq:,.0f}")

with metric_cols[1]:
    total_return = ((net_liq - starting_capital) / starting_capital) * 100 if starting_capital > 0 else 0
    st.metric("Total Return", f"{total_return:+.2f}%")

with metric_cols[2]:
    total_pnl = net_liq - starting_capital
    st.metric("Total P&L", f"${total_pnl:+,.0f}")

with metric_cols[3]:
    # Primary: Use trade_ledger if available
    # Fallback: Count reconciled trades from council_history
    if not trade_df.empty:
        trade_count = len(trade_df)
    elif not council_df.empty and 'pnl_realized' in council_df.columns:
        # Count trades that have been reconciled (have P&L data)
        trade_count = council_df['pnl_realized'].notna().sum()
    else:
        trade_count = 0
    st.metric("Total Trades", trade_count)

with metric_cols[4]:
    # Calculate win rate from council history
    if not council_df.empty and 'pnl_realized' in council_df.columns:
        reconciled = council_df[pd.notna(council_df['pnl_realized'])]
        if not reconciled.empty:
            win_rate = (reconciled['pnl_realized'] > 0).mean() * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        else:
            st.metric("Win Rate", "N/A")
    else:
        st.metric("Win Rate", "N/A")

with metric_cols[5]:
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

st.markdown("---")

# === SECTION 2: Interactive Equity Curve ===
st.subheader("📈 Equity Curve")

if not equity_df.empty:
    # Prepare data
    equity_df = equity_df.sort_values('timestamp')

    # Get trade markers
    trade_markers = None
    if not council_df.empty:
        trade_markers = council_df[['timestamp', 'master_decision', 'entry_price']].copy()
        trade_markers['timestamp'] = pd.to_datetime(trade_markers['timestamp'])

    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Equity Curve', 'Drawdown')
    )

    # Equity line
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['total_value_usd'],
            mode='lines',
            name='Equity',
            line=dict(color='#636EFA', width=2)
        ),
        row=1, col=1
    )

    # Add trade markers if available
    if trade_markers is not None and not trade_markers.empty:
        buys = trade_markers[trade_markers['master_decision'] == 'BULLISH']
        sells = trade_markers[trade_markers['master_decision'] == 'BEARISH']

        # Merge with equity to get Y values
        if not buys.empty:
            buy_equity = pd.merge_asof(
                buys.sort_values('timestamp'),
                equity_df[['timestamp', 'total_value_usd']].sort_values('timestamp'),
                on='timestamp'
            )
            fig.add_trace(
                go.Scatter(
                    x=buy_equity['timestamp'],
                    y=buy_equity['total_value_usd'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=12, color='#00CC96')
                ),
                row=1, col=1
            )

        if not sells.empty:
            sell_equity = pd.merge_asof(
                sells.sort_values('timestamp'),
                equity_df[['timestamp', 'total_value_usd']].sort_values('timestamp'),
                on='timestamp'
            )
            fig.add_trace(
                go.Scatter(
                    x=sell_equity['timestamp'],
                    y=sell_equity['total_value_usd'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=12, color='#EF553B')
                ),
                row=1, col=1
            )

    # Drawdown calculation
    equity_df['peak'] = equity_df['total_value_usd'].cummax()
    equity_df['drawdown'] = (equity_df['total_value_usd'] - equity_df['peak']) / equity_df['peak'] * 100

    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['drawdown'],
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#EF553B', width=1)
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    st.plotly_chart(fig, width="stretch")

    # Max Drawdown stat
    max_dd = equity_df['drawdown'].min()
    st.caption(f"Maximum Drawdown: {max_dd:.2f}%")

else:
    st.warning("No equity data available. Ensure equity_logger.py is running.")

st.markdown("---")


# === SECTION 3: Strategy Efficiency (ROI by Strategy Type) ===
st.subheader("🎯 Strategy Efficiency by Type")

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

# === NEW: Trade Type Breakdown ===
st.subheader("📊 Directional vs Volatility Performance")

if 'prediction_type' in council_df.columns:
    type_perf = council_df.groupby('prediction_type').agg({
        'pnl_realized': ['sum', 'mean', 'count']
    })
    type_perf.columns = ['Total P&L', 'Avg P&L', 'Trade Count']

    col1, col2 = st.columns(2)

    with col1:
        dir_data = type_perf.loc['DIRECTIONAL'] if 'DIRECTIONAL' in type_perf.index else None
        if dir_data is not None:
            st.metric("Directional P&L", f"${dir_data['Total P&L']:,.2f}")
            st.caption(f"{int(dir_data['Trade Count'])} trades | Avg: ${dir_data['Avg P&L']:,.2f}")
        else:
            st.metric("Directional P&L", "$0.00")
            st.caption("No directional trades yet")

    with col2:
        vol_data = type_perf.loc['VOLATILITY'] if 'VOLATILITY' in type_perf.index else None
        if vol_data is not None:
            st.metric("Volatility P&L", f"${vol_data['Total P&L']:,.2f}")
            st.caption(f"{int(vol_data['Trade Count'])} trades | Avg: ${vol_data['Avg P&L']:,.2f}")
        else:
            st.metric("Volatility P&L", "$0.00")
            st.caption("No volatility trades yet")
else:
    st.info("No trade type data available.")

st.markdown("---")

# === SECTION 4: Monthly Returns Heatmap ===
st.subheader("📅 Monthly Returns Heatmap")
st.caption("Calendar view of monthly performance - standard hedge fund reporting format")

if not equity_df.empty:
    # Calculate monthly returns
    equity_df['month'] = equity_df['timestamp'].dt.to_period('M')

    monthly = equity_df.groupby('month').agg({
        'total_value_usd': ['first', 'last']
    })
    monthly.columns = ['start', 'end']
    monthly['return'] = ((monthly['end'] - monthly['start']) / monthly['start']) * 100

    # Pivot for heatmap
    monthly = monthly.reset_index()
    monthly['year'] = monthly['month'].dt.year
    monthly['month_num'] = monthly['month'].dt.month
    monthly['month_name'] = monthly['month_num'].apply(lambda x: calendar.month_abbr[x])

    if len(monthly) > 1:
        pivot = monthly.pivot(index='year', columns='month_name', values='return')

        # Reorder columns to calendar order
        month_order = [calendar.month_abbr[i] for i in range(1, 13)]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

        fig = px.imshow(
            pivot,
            labels=dict(x="Month", y="Year", color="Return %"),
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            aspect='auto'
        )

        fig.update_layout(height=300)

        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough monthly data for heatmap.")
else:
    st.info("Equity data not available for monthly analysis.")

st.markdown("---")

# === SECTION 5: Benchmark Comparison ===
st.subheader("📊 Benchmark Comparison")

if not equity_df.empty:
    start_date = equity_df['timestamp'].min()
    end_date = equity_df['timestamp'].max()

    benchmark_df = fetch_benchmark_data(start_date, end_date)

    if not benchmark_df.empty:
        # Calculate bot returns normalized
        bot_returns = (equity_df.set_index('timestamp')['total_value_usd'] / starting_capital - 1) * 100
        bot_returns = bot_returns.resample('D').last().dropna()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=bot_returns.index,
            y=bot_returns.values,
            name='Mission Control',
            line=dict(color='#636EFA', width=2)
        ))

        if 'SPY' in benchmark_df.columns:
            fig.add_trace(go.Scatter(
                x=benchmark_df.index,
                y=benchmark_df['SPY'],
                name='S&P 500',
                line=dict(color='#FFA15A', width=1, dash='dot')
            ))

        # E5 FIX: Commodity-agnostic benchmark
        from config import get_active_profile
        profile = get_active_profile(config)
        benchmark_col = f"{profile.ticker}=F"

        if benchmark_col in benchmark_df.columns:
            fig.add_trace(go.Scatter(
                x=benchmark_df.index,
                y=benchmark_df[benchmark_col],
                name=f'{profile.name} Futures',
                line=dict(color='#00CC96', width=1, dash='dot')
            ))

        fig.update_layout(
            title='Returns vs Benchmarks',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Could not fetch benchmark data.")
else:
    st.info("Equity data required for benchmark comparison.")

st.markdown("---")

# === SECTION 6: Trade Ledger ===
st.subheader("📋 Trade Ledger")

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
        st.caption("⚠️ Showing reconciled trades from council_history (trade_ledger.csv is empty)")
    else:
        st.info("No trades recorded.")
else:
    st.info("No trades recorded.")
