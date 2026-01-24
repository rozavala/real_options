"""
Page 6: Signal Overlay (Decision vs. Price)

Purpose: Deep forensic analysis of decisions.
Visualizes WHERE signals were generated and HOW price evolved.
Optimized for performance using vectorized operations (merge_asof).
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import load_council_history

st.set_page_config(layout="wide", page_title="Signal Analysis | Coffee Bot")

st.title("🎯 Signal Overlay Analysis")
st.caption("Forensic analysis of Council decisions against Coffee Futures price action")

# --- 1. CONFIGURATION & SIDEBAR ---

AGENT_MAPPING = {
    "👑 Master Decision": "master_decision",
    "🤖 ML Signal": "ml_signal",
    "🌱 Agronomist (Weather)": "meteorologist_sentiment",
    "💹 Macro Economist": "macro_sentiment",
    "📈 Fundamentalist": "fundamentalist_sentiment",
    "🐦 Sentiment (News/X)": "sentiment_sentiment",
    "📐 Technical Analyst": "technical_sentiment",
    "📊 Volatility Analyst": "volatility_sentiment",
    "🌍 Geopolitical": "geopolitical_sentiment"
}

with st.sidebar:
    st.header("🔬 Analysis Settings")

    timeframe = st.selectbox(
        "Timeframe",
        options=['1m', '5m', '15m', '30m', '1h', '1d', '1wk'],
        index=1,
        format_func=lambda x: f"{x} Candles"
    )

    # Limits for Yahoo Finance reliability
    max_days = 59 if timeframe in ['1m', '5m', '15m', '30m', '1h'] else 730
    default_lookback = 7 if timeframe in ['1m', '5m'] else 90

    lookback_days = st.slider(
        "Lookback Period (Days)",
        min_value=1,
        max_value=max_days,
        value=default_lookback
    )

    st.markdown("---")
    st.header("🕵️ Signal Source")

    selected_agent_label = st.selectbox(
        "Decision Maker",
        options=list(AGENT_MAPPING.keys()),
        index=0
    )
    selected_agent_col = AGENT_MAPPING[selected_agent_label]

    st.markdown("---")
    st.header("Visuals")
    show_labels = st.toggle("Always Show Signal Labels", value=True)
    hide_gaps = st.toggle("Remove Non-Trading Gaps", value=True)
    show_confidence = st.toggle("Show Confidence Scores", value=True)

# --- 2. DATA FUNCTIONS ---

@st.cache_data(ttl=300)
def fetch_price_history_extended(ticker="KC=F", period="5d", interval="5m"):
    """Fetches historical OHLC data with robustness."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None

        # Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize Timezone to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        return df
    except Exception as e:
        st.error(f"Failed to load price data: {e}")
        return None

def calculate_rangebreaks(df, interval_str):
    """Calculates gaps > 3.5x interval to hide non-trading periods."""
    if df is None or df.empty: return []

    # Milliseconds per interval
    dt_mapping = {
        '1m': 60*1000, '5m': 300*1000, '15m': 900*1000,
        '30m': 1800*1000, '1h': 3600*1000, '1d': 24*3600*1000
    }
    expected_ms = dt_mapping.get(interval_str, 3600*1000)

    # Detect gaps
    diffs = df.index.to_series().diff().dt.total_seconds() * 1000
    gaps = diffs[diffs > expected_ms * 3.5]

    bounds = []
    for idx, gap_ms in gaps.items():
        gap_end = idx
        gap_start = idx - pd.Timedelta(milliseconds=gap_ms)
        bounds.append(dict(values=[gap_start, gap_end]))
    return bounds

def process_signals_for_agent(history_df, agent_col, start_date):
    """Clean, filter, and format signals for the selected agent."""
    if history_df.empty: return pd.DataFrame()

    df = history_df.copy()

    # 1. Strict Timestamp Cleaning
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # 2. Timezone Standardization (UTC)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    # 3. Date Filtering
    # Ensure start_date is UTC compatible
    ts_start = pd.Timestamp(start_date)
    if ts_start.tz is None: ts_start = ts_start.tz_localize('UTC')
    else: ts_start = ts_start.tz_convert('UTC')

    df = df[df['timestamp'] >= ts_start].copy()
    if df.empty: return pd.DataFrame()

    # 4. Handle Column Mapping (Legacy Support)
    if agent_col not in df.columns:
        if agent_col == 'master_decision': agent_col = 'direction' # Legacy
        else: return pd.DataFrame()

    # 5. Extract & Fill Data
    df['plot_direction'] = df[agent_col].fillna('NEUTRAL').astype(str).str.upper()

    # Confidence: Fallback to master_confidence or 0.5, ensure no NaNs
    if 'master_confidence' in df.columns:
        df['plot_confidence'] = pd.to_numeric(df['master_confidence'], errors='coerce').fillna(0.5)
    else:
        df['plot_confidence'] = 0.5

    # 6. Label Logic (Resolve Volatility Strategies)
    if agent_col == 'master_decision':
        def resolve_label(row):
            d = row['plot_direction']
            s = str(row.get('strategy_type', '')).upper()
            if d == 'NEUTRAL':
                if 'CONDOR' in s: return 'CONDOR'
                if 'STRADDLE' in s: return 'STRADDLE'
            return d
        df['plot_label'] = df.apply(resolve_label, axis=1)
    else:
        df['plot_label'] = df['plot_direction']

    # 7. Colors & Symbols
    color_map = {
        'BULLISH': '#00CC96', 'UP': '#00CC96', 'LONG': '#00CC96',
        'BEARISH': '#EF553B', 'DOWN': '#EF553B', 'SHORT': '#EF553B',
        'CONDOR': '#AB63FA', 'STRADDLE': '#AB63FA' # Volatility
    }
    symbol_map = {
        'BULLISH': 'triangle-up', 'UP': 'triangle-up',
        'BEARISH': 'triangle-down', 'DOWN': 'triangle-down',
        'CONDOR': 'diamond', 'STRADDLE': 'diamond'
    }

    df['marker_color'] = df['plot_label'].map(color_map).fillna('#636EFA') # Default Blue
    df['marker_symbol'] = df['plot_label'].map(symbol_map).fillna('circle')

    return df

# --- 3. MAIN EXECUTION ---

end_date = datetime.now()
start_date = end_date - timedelta(days=lookback_days)

with st.spinner("Crunching data..."):
    # Determine YFinance period
    if lookback_days <= 1: yf_period = "1d"
    elif lookback_days <= 5: yf_period = "5d"
    elif lookback_days <= 29: yf_period = "1mo"
    elif lookback_days <= 59: yf_period = "1mo"
    else: yf_period = "2y"

    price_df = fetch_price_history_extended(ticker="KC=F", period=yf_period, interval=timeframe)
    council_df = load_council_history()

# --- 4. PLOTTING ---

if price_df is not None and not price_df.empty:

    # Filter price data to slider range
    cutoff_dt = pd.Timestamp(datetime.now(price_df.index.tz) - timedelta(days=lookback_days))
    price_df = price_df[price_df.index >= cutoff_dt]

    # Process Signals
    signals = process_signals_for_agent(council_df, selected_agent_col, start_date)

    # === ALIGNMENT ENGINE (Vectorized merge_asof) ===
    # This replaces the slow loop. It snaps every signal to the nearest candle timestamp.

    plot_df = pd.DataFrame()
    if not signals.empty:
        # Prepare for merge
        signals = signals.sort_values('timestamp')
        price_idx = pd.DataFrame(index=price_df.index).reset_index()
        # Explicitly rename to avoid ambiguity and ensure we retrieve the CANDLE timestamp
        price_idx.columns = ['candle_timestamp']

        # Merge_asof requires sorted timestamps
        # direction='nearest' snaps signal to closest candle
        # tolerance=pd.Timedelta('4h') ignores signals > 4h away from any candle
        merged = pd.merge_asof(
            signals,
            price_idx,
            left_on='timestamp',
            right_on='candle_timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(hours=4)
        )

        # Only keep signals that successfully matched a candle
        matched_signals = merged.dropna(subset=['candle_timestamp'])

        if not matched_signals.empty:
            # Now we look up price values for Y-axis positioning efficiently
            # We set index to timestamp to map quickly
            aligned_prices = price_df.loc[matched_signals['candle_timestamp']]

            # Construct display DF
            plot_df = matched_signals.copy()
            # Use aligned candle timestamp for plotting so markers sit on candles
            plot_df['plot_x'] = matched_signals['candle_timestamp']

            plot_df['candle_high'] = aligned_prices['High'].values
            plot_df['candle_low'] = aligned_prices['Low'].values

            # Vectorized Y-position logic
            plot_df['y_pos'] = np.where(
                plot_df['marker_symbol'] == 'triangle-down',
                plot_df['candle_high'] * 1.0005,
                plot_df['candle_low'] * 0.9995
            )

            plot_df['text_pos'] = np.where(
                plot_df['marker_symbol'] == 'triangle-down',
                "top center",
                "bottom center"
            )

            # Construct Hover Text
            plot_df['hover'] = (
                "<b>" + plot_df['plot_label'] + "</b><br>" +
                "Time: " + plot_df['timestamp'].dt.strftime('%d %b %H:%M') + "<br>" +
                "Conf: " + plot_df['plot_confidence'].apply(lambda x: f"{x:.2f}") + "<br>" +
                "<i>" + plot_df['rationale'].astype(str).str.slice(0, 150) + "...</i>"
            )

    # === DRAW CHARTS ===

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])

    # 1. Candle Trace
    fig.add_trace(go.Candlestick(
        x=price_df.index, open=price_df['Open'], high=price_df['High'],
        low=price_df['Low'], close=price_df['Close'], name="Coffee"
    ), row=1, col=1)

    # 2. Signal Trace
    if not plot_df.empty:
        fig.add_trace(go.Scatter(
            x=plot_df['plot_x'],
            y=plot_df['y_pos'],
            mode='markers+text' if show_labels else 'markers',
            text=plot_df['plot_label'] if show_labels else None,
            textposition=plot_df['text_pos'],
            textfont=dict(size=10, color='white'),
            marker=dict(symbol=plot_df['marker_symbol'], color=plot_df['marker_color'], size=14, line=dict(width=1, color='black')),
            hovertext=plot_df['hover'],
            hoverinfo="text",
            name="Signals"
        ), row=1, col=1)

    # 3. Volume Trace
    if 'Volume' in price_df.columns:
        fig.add_trace(go.Bar(
            x=price_df.index, y=price_df['Volume'],
            marker_color='rgba(255,255,255,0.1)', name='Volume'
        ), row=2, col=1)

    # 4. Confidence Trace
    if not plot_df.empty and show_confidence:
        fig.add_trace(go.Scatter(
            x=plot_df['plot_x'],
            y=plot_df['plot_confidence'],
            mode='markers',
            marker=dict(color='#00CC96', size=6),
            name='Confidence'
        ), row=2, col=1)

    # Layout Updates
    rangebreaks = calculate_rangebreaks(price_df, timeframe) if hide_gaps else []

    fig.update_xaxes(rangebreaks=rangebreaks)
    fig.update_layout(
        height=800, xaxis_rangeslider_visible=False, template="plotly_dark",
        title=f"Coffee Analysis | {selected_agent_label} | {timeframe}",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📝 Raw Signal Log", expanded=True):
        if not plot_df.empty:
            display_cols = ['timestamp', 'plot_label', 'plot_confidence', 'rationale', 'strategy_type']
            st.dataframe(
                plot_df[[c for c in display_cols if c in plot_df.columns]].sort_values('timestamp', ascending=False),
                column_config={"timestamp": st.column_config.DatetimeColumn("Time (UTC)", format="D MMM HH:mm")},
                use_container_width=True
            )
        else:
            st.info("No signals found in this window.")

else:
    st.warning("No market data available. Check lookback period or internet connection.")
