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

st.set_page_config(layout="wide", page_title="Financials | Coffee Bot")

st.title("📈 Financial Performance")
st.caption("ROI & Audit - Institutional-grade reporting on actual dollars gained or lost")

# --- Load Data ---
equity_df = load_equity_data()
trade_df = load_trade_data()
council_df = load_council_history()
config = get_config()

# Get starting capital
starting_capital = DEFAULT_STARTING_CAPITAL
if not equity_df.empty:
    starting_capital = equity_df.iloc[0]['total_value_usd']

st.markdown("---")

# === SECTION 1: Key Metrics ===
st.subheader("📊 Performance Summary")

metric_cols = st.columns(5)

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
    trade_count = len(trade_df) if not trade_df.empty else 0
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

    st.plotly_chart(fig, use_container_width=True)

    # Max Drawdown stat
    max_dd = equity_df['drawdown'].min()
    st.caption(f"Maximum Drawdown: {max_dd:.2f}%")

else:
    st.warning("No equity data available. Ensure equity_logger.py is running.")

st.markdown("---")

# === SECTION 3: Strategy Efficiency (ROI by Strategy Type) ===
st.subheader("🎯 Strategy Efficiency")
st.caption("ROI breakdown by position/combo ID - Are certain strategies more profitable?")

if not trade_df.empty and 'position_id' in trade_df.columns:
    # Group by position_id (combo)
    strategy_perf = trade_df.groupby('position_id').agg({
        'total_value_usd': 'sum',
        'quantity': 'sum',
        'timestamp': 'count'
    }).rename(columns={'timestamp': 'trade_count'})

    if not strategy_perf.empty:
        strategy_perf = strategy_perf.sort_values('total_value_usd', ascending=True)

        fig = px.bar(
            strategy_perf.reset_index(),
            x='position_id',
            y='total_value_usd',
            color='total_value_usd',
            color_continuous_scale='RdYlGn',
            title='P&L by Position/Combo'
        )

        fig.update_layout(
            xaxis_title='Position ID',
            yaxis_title='P&L ($)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Trade ledger empty or missing position_id column.")

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

        st.plotly_chart(fig, use_container_width=True)
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
            name='Coffee Bot',
            line=dict(color='#636EFA', width=2)
        ))

        if 'SPY' in benchmark_df.columns:
            fig.add_trace(go.Scatter(
                x=benchmark_df.index,
                y=benchmark_df['SPY'],
                name='S&P 500',
                line=dict(color='#FFA15A', width=1, dash='dot')
            ))

        if 'KC=F' in benchmark_df.columns:
            fig.add_trace(go.Scatter(
                x=benchmark_df.index,
                y=benchmark_df['KC=F'],
                name='Coffee Futures',
                line=dict(color='#00CC96', width=1, dash='dot')
            ))

        fig.update_layout(
            title='Returns vs Benchmarks',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Could not fetch benchmark data.")
else:
    st.info("Equity data required for benchmark comparison.")

st.markdown("---")

# === SECTION 6: Trade Ledger ===
st.subheader("📋 Trade Ledger")

if not trade_df.empty:
    st.dataframe(
        trade_df.sort_values('timestamp', ascending=False),
        use_container_width=True
    )
else:
    st.info("No trades recorded.")
