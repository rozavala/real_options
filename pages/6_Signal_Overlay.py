"""
Page 6: Signal Overlay (Decision vs. Price)

Purpose: Deep forensic analysis of decisions.
Visualizes WHERE signals were generated and HOW price evolved,
with support for specific Agent inspection and Gap removal.
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

# Mapping of Friendly Names -> DataFrame Columns
# NOTE: Logistics removed as column does not exist in current schema
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

    # Defaults set to your request: 5m interval, last 7 days (updated from 3 for better context)
    timeframe = st.selectbox(
        "Timeframe",
        options=['1m', '5m', '15m', '30m', '1h', '1d', '1wk'],
        index=1, # Default to 5m
        format_func=lambda x: f"{x} Candles"
    )

    # Dynamic lookback max based on timeframe (Yahoo limits)
    max_days = 59 if timeframe in ['1m', '5m', '15m', '30m', '1h'] else 730

    # Default lookback logic: 7 days for intraday to prevent timeouts, 90 for daily
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
    hide_gaps = st.toggle("Remove Non-Trading Gaps", value=True, help="Hides weekends and nights where no data exists")
    show_confidence = st.toggle("Show Confidence Scores", value=True)

# --- 2. DATA FUNCTIONS ---

@st.cache_data(ttl=300)
def fetch_price_history_extended(ticker="KC=F", period="5d", interval="5m"):
    """Fetches historical OHLC data with robustness for short intervals."""
    try:
        # yfinance requires specific period strings for intraday
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

        if df.empty:
            return None

        # Standardize Columns (Flatten MultiIndex if present)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure UTC timezone for consistency
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        return df
    except Exception as e:
        st.error(f"Failed to load price data: {e}")
        return None

def calculate_rangebreaks(df, interval_str):
    """
    Dynamically calculates gaps in the data to hide them in Plotly.
    This works better than hardcoding hours for Futures.
    """
    if df is None or df.empty:
        return []

    # Calculate dt based on interval string
    dt_mapping = {
        '1m': 60*1000, '2m': 120*1000, '5m': 300*1000,
        '15m': 900*1000, '30m': 1800*1000, '1h': 3600*1000,
        '1d': 24*3600*1000
    }

    # Default to 1 hour if unknown
    expected_diff_ms = dt_mapping.get(interval_str, 3600*1000)

    # Find gaps significantly larger than the interval (e.g., 3x)
    # This detects weekends and overnight breaks automatically
    # We use index.to_series() to handle potential index type issues
    diff_series = df.index.to_series().diff().dt.total_seconds() * 1000
    gaps = diff_series[diff_series > expected_diff_ms * 3.5]

    bounds = []
    for idx, diff_val in gaps.items():
        # Gap is from previous_index to current_index
        # We need to find the previous timestamp.
        # Since we can't easily index 'prev', we construct the rangebreak
        # from (current - diff) to current
        gap_end = idx
        gap_start = idx - pd.Timedelta(milliseconds=diff_val)

        # Add a small buffer so we don't clip the candles themselves
        bounds.append(dict(values=[gap_start, gap_end]))

    return bounds

def process_signals_for_agent(history_df, agent_col, start_date):
    """
    Extracts signals specifically for the selected agent/master.
    Handles 'Neutral' vs 'Strategy' logic for the Master.
    """
    if history_df.empty:
        return pd.DataFrame()

    # 1. Normalize Timestamp
    df = history_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ensure UTC
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    # Filter Date
    ts_start = pd.Timestamp(start_date)
    if ts_start.tz is None:
        ts_start = ts_start.tz_localize('UTC')
    else:
        ts_start = ts_start.tz_convert('UTC')

    df = df[df['timestamp'] >= ts_start].copy()

    # 2. Extract Signal Columns
    # We need: direction, confidence, rationale, strategy

    # Check if column exists
    if agent_col not in df.columns:
        # Fallback for older logs
        if agent_col == 'master_decision':
            agent_col = 'direction' # Legacy
        else:
            return pd.DataFrame() # Agent not found in logs

    # Create standardized columns
    df['plot_direction'] = df[agent_col].fillna('NEUTRAL').astype(str).str.upper()

    # Confidence mapping varies by agent
    # Master has 'master_confidence', agents might just be text,
    # but 'brier_scoring.py' suggests we might have scores.
    # For now, default to master_confidence if specific agent confidence is missing
    if 'master_confidence' in df.columns:
        df['plot_confidence'] = df['master_confidence']
    else:
        df['plot_confidence'] = 0.5

    # 3. SPECIAL LOGIC: Handle "Neutral" Master Decisions that are actually Volatility Trades
    if agent_col == 'master_decision':
        # If Master says NEUTRAL but Strategy is Iron Condor/Straddle, label it!
        def resolve_label(row):
            d = row['plot_direction']
            s = str(row.get('strategy_type', '')).upper()

            if d == 'NEUTRAL':
                if 'CONDOR' in s: return 'CONDOR'
                if 'STRADDLE' in s: return 'STRADDLE'
            return d

        df['plot_label'] = df.apply(resolve_label, axis=1)
    else:
        # For sub-agents, just use their sentiment
        df['plot_label'] = df['plot_direction']

    # 4. Map Colors
    def get_color(label):
        if label in ['BULLISH', 'UP', 'LONG']: return '#00CC96' # Green
        if label in ['BEARISH', 'DOWN', 'SHORT']: return '#EF553B' # Red
        if label in ['CONDOR', 'STRADDLE']: return '#AB63FA' # Purple (Volatility)
        return '#636EFA' # Blue (Neutral/Unknown)

    def get_symbol(label):
        if label in ['BULLISH', 'UP', 'LONG']: return 'triangle-up'
        if label in ['BEARISH', 'DOWN', 'SHORT']: return 'triangle-down'
        if label in ['CONDOR', 'STRADDLE']: return 'diamond'
        return 'circle'

    df['marker_color'] = df['plot_label'].apply(get_color)
    df['marker_symbol'] = df['plot_label'].apply(get_symbol)

    return df

# --- 3. MAIN EXECUTION ---

# Calc Dates
end_date = datetime.now()
start_date = end_date - timedelta(days=lookback_days)

# Data Fetching
with st.spinner(f"Fetching {timeframe} data for {lookback_days} days..."):
    # Approx yfinance period string
    if lookback_days <= 1: yf_period = "1d"
    elif lookback_days <= 5: yf_period = "5d"
    elif lookback_days <= 29: yf_period = "1mo"
    elif lookback_days <= 59: yf_period = "1mo" # Max for 5m is 60d
    else: yf_period = "2y" # Daily

    price_df = fetch_price_history_extended(ticker="KC=F", period=yf_period, interval=timeframe)
    council_df = load_council_history()

# --- 4. DATA PROCESSING & PLOTTING ---

if price_df is not None and not price_df.empty:

    # Trim price data to slider EXACTLY
    # (yfinance period='5d' might return 5 trading days, which is > 5 calendar days)
    # We filter to ensure chart matches user expectation
    cutoff_dt = pd.Timestamp(datetime.now(price_df.index.tz) - timedelta(days=lookback_days))
    price_df = price_df[price_df.index >= cutoff_dt]

    # Process Signals
    if not council_df.empty:
        signals = process_signals_for_agent(council_df, selected_agent_col, start_date)
    else:
        signals = pd.DataFrame()

    # Create Subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25]
    )

    # --- TRACE 1: CANDLESTICK ---
    fig.add_trace(go.Candlestick(
        x=price_df.index,
        open=price_df['Open'],
        high=price_df['High'],
        low=price_df['Low'],
        close=price_df['Close'],
        name="Coffee (KC=F)"
    ), row=1, col=1)

    # --- TRACE 2: SIGNALS ---
    if not signals.empty:
        plot_data = []

        # Align signals to nearest available candle
        # This fixes "Signals floating in empty space" errors
        valid_indices = price_df.index

        for idx, row in signals.iterrows():
            try:
                # Find nearest timestamp in price data
                # Because index is sorted, we can use searchsorted or get_indexer
                # We use get_indexer with method='nearest'
                target_ts = row['timestamp']

                # Ensure TZ match
                if target_ts.tz != valid_indices.tz:
                     target_ts = target_ts.tz_convert(valid_indices.tz)

                loc_idx = valid_indices.get_indexer([target_ts], method='nearest')[0]

                # Check proximity (don't map a signal from Sunday to Friday's close if it's too far)
                matched_ts = valid_indices[loc_idx]
                time_diff = abs(matched_ts - target_ts)

                # Tolerance: 2x the interval or 4 hours, whichever is larger
                # Prevents massive jumps
                if time_diff > timedelta(hours=4):
                    continue

                price_at_signal = price_df.iloc[loc_idx]

                # Dynamic Y-positioning
                # Use .item() to extract scalar from Series if necessary
                high_val = price_at_signal['High'] if np.isscalar(price_at_signal['High']) else price_at_signal['High'].item()
                low_val = price_at_signal['Low'] if np.isscalar(price_at_signal['Low']) else price_at_signal['Low'].item()

                if row['marker_symbol'] == 'triangle-down':
                    y_pos = high_val + (high_val * 0.0005)
                    text_pos = "top center"
                else:
                    y_pos = low_val - (low_val * 0.0005)
                    text_pos = "bottom center"

                rationale = str(row.get('rationale', 'No rationale provided.'))

                # Build Hover Text
                hover = (
                    f"<b>{row['plot_label']}</b><br>"
                    f"Time: {row['timestamp'].strftime('%d %b %H:%M')}<br>"
                    f"Conf: {row['plot_confidence']:.2f}<br>"
                    f"<i>{rationale[:150]}...</i>"
                )

                plot_data.append({
                    'x': matched_ts,
                    'y': y_pos,
                    'color': row['marker_color'],
                    'symbol': row['marker_symbol'],
                    'label': row['plot_label'] if show_labels else "",
                    'text_pos': text_pos,
                    'hover': hover
                })

            except Exception as e:
                # print(f"Error plotting signal: {e}")
                continue

        if plot_data:
            pdf = pd.DataFrame(plot_data)

            fig.add_trace(go.Scatter(
                x=pdf['x'],
                y=pdf['y'],
                mode='markers+text' if show_labels else 'markers',
                text=pdf['label'],
                textposition=pdf['text_pos'],
                textfont=dict(size=10, color='white'),
                marker=dict(
                    symbol=pdf['symbol'],
                    color=pdf['color'],
                    size=14,
                    line=dict(width=1, color='black')
                ),
                hovertext=pdf['hover'],
                hoverinfo="text",
                name=f"{selected_agent_label} Signals"
            ), row=1, col=1)

    # --- TRACE 3: CONFIDENCE / VOLUME ---
    if 'Volume' in price_df.columns:
         fig.add_trace(go.Bar(
            x=price_df.index,
            y=price_df['Volume'],
            marker_color='rgba(255, 255, 255, 0.1)',
            name='Volume'
        ), row=2, col=1)

    # Overlay Confidence: Simplified logic - use signal locations
    if not signals.empty and show_confidence and 'plot_confidence' in signals.columns:
        # Filter signals to those that were actually plotted (matched)
        # We can reuse 'pdf' from above if it exists
        if 'pdf' in locals() and not pdf.empty:
            # We need to get the confidence values corresponding to the plotted points.
            # Since pdf was built from signals iteration, we can reconstruct or map.
            # Simpler: just loop through plot_data and extract confidence from hover string or re-lookup
            # But re-lookup is painful.
            # Better: add 'confidence' to plot_data

            # Let's rebuild plot_data with confidence quickly
            # (In production I'd refactor the loop above, but trying to minimize diff risk)
            # Actually, let's just use the loop logic again for confidence trace
            pass # See below

            # Alternative: iterate pdf and find matching signal.
            # Let's just do a clean pass for confidence trace
            conf_data = []
            for idx, row in signals.iterrows():
                try:
                    target_ts = row['timestamp']
                    if target_ts.tz != valid_indices.tz:
                        target_ts = target_ts.tz_convert(valid_indices.tz)
                    loc_idx = valid_indices.get_indexer([target_ts], method='nearest')[0]
                    matched_ts = valid_indices[loc_idx]
                    if abs(matched_ts - target_ts) > timedelta(hours=4): continue

                    conf_data.append({
                        'x': matched_ts,
                        'y': row['plot_confidence']
                    })
                except: continue

            if conf_data:
                cdf = pd.DataFrame(conf_data)
                fig.add_trace(go.Scatter(
                    x=cdf['x'],
                    y=cdf['y'],
                    mode='markers',
                    marker=dict(color='#00CC96', size=6),
                    name='Confidence'
                ), row=2, col=1)

    # --- LAYOUT & GAP REMOVAL ---

    # Gap Detection
    rangebreaks = []
    if hide_gaps:
        rangebreaks = calculate_rangebreaks(price_df, timeframe)

    fig.update_xaxes(
        rangebreaks=rangebreaks,
        row=1, col=1
    )
    fig.update_xaxes(
        rangebreaks=rangebreaks,
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        title=f"Coffee Analysis | {selected_agent_label} | {timeframe}",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- DATAFRAME VIEW ---
    with st.expander("📝 Raw Signal Log", expanded=True):
        if not signals.empty:
            st.dataframe(
                signals[['timestamp', 'plot_label', 'plot_confidence', 'rationale', 'strategy_type']].sort_values('timestamp', ascending=False),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time (UTC)", format="D MMM HH:mm"),
                    "plot_label": "Signal",
                    "plot_confidence": st.column_config.ProgressColumn("Conf", min_value=0, max_value=1),
                    "rationale": st.column_config.TextColumn("Reasoning", width="large")
                },
                use_container_width=True
            )
        else:
            st.info("No signals found for this agent in the selected window.")

else:
    st.warning("Unable to fetch market data. Market may be closed or Lookback is too short for this interval.")
