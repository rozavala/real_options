"""
Page 6: Signal Overlay (Decision vs. Price)

Purpose: Visual backtesting tool to see WHERE signals were generated on the price chart
and HOW the price evolved afterwards.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import load_council_history

st.set_page_config(layout="wide", page_title="Signal Analysis | Coffee Bot")

st.title("🎯 Signal Overlay Analysis")
st.caption("Visualizing Council decisions against Coffee Futures price action")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Chart Settings")
    timeframe = st.selectbox(
        "Timeframe",
        options=['1h', '1d', '1wk'],
        index=1,
        format_func=lambda x: "Hourly" if x == '1h' else "Daily" if x == '1d' else "Weekly"
    )

    lookback_days = st.slider("Lookback Period (Days)", min_value=7, max_value=365, value=90)

    st.markdown("---")
    show_confidence = st.toggle("Show Confidence Scores", value=True)
    show_neutral = st.toggle("Show Neutral Signals", value=False)

# --- Data Loading Functions ---

@st.cache_data(ttl=3600)
def fetch_price_history(ticker="KC=F", period="1y", interval="1d"):
    """Fetches historical OHLC data for the background chart."""
    try:
        # Buffer the start date slightly to ensure we cover the lookback
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # Flatten multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"Failed to load price data: {e}")
        return None

def process_signals(history_df, start_date):
    """Filters and formats council history for plotting."""
    if history_df.empty:
        return history_df

    df = history_df.copy()

    # Column Mapping (Legacy Support)
    # council_history.csv uses master_decision/confidence/reasoning
    column_mapping = {
        'master_decision': 'direction',
        'master_confidence': 'confidence',
        'master_reasoning': 'rationale'
    }
    df = df.rename(columns=column_mapping)

    # Ensure columns exist
    required = ['direction', 'confidence', 'rationale', 'strategy_type']
    for col in required:
        if col not in df.columns:
            df[col] = None

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter by date
    # Make start_date timezone-aware to match history_df (usually UTC)
    if df['timestamp'].dt.tz is not None:
        if pd.Timestamp(start_date).tz is None:
            # If start_date is naive, localize it to the dataframe's timezone
            # Assuming dataframe is in UTC or similar consistent timezone
            start_date = pd.Timestamp(start_date).tz_localize(df['timestamp'].dt.tz)
        else:
             # If start_date has tz, convert to df tz
             start_date = pd.Timestamp(start_date).tz_convert(df['timestamp'].dt.tz)

    df = df[df['timestamp'] >= start_date].copy()

    # Map colors and markers
    color_map = {
        'BULLISH': '#00CC96',  # Green
        'BEARISH': '#EF553B',  # Red
        'NEUTRAL': '#636EFA'   # Blue
    }

    symbol_map = {
        'BULLISH': 'triangle-up',
        'BEARISH': 'triangle-down',
        'NEUTRAL': 'circle'
    }

    df['marker_color'] = df['direction'].map(color_map).fillna('grey')
    df['marker_symbol'] = df['direction'].map(symbol_map).fillna('circle')

    return df

# --- Main Logic ---

# 1. Calculate Date Range
end_date = datetime.now()
start_date = end_date - timedelta(days=lookback_days)

# 2. Fetch Data
with st.spinner("Loading market data and signal history..."):
    # Convert lookback to yfinance period format approx
    yf_period = "2y" if lookback_days > 365 else "1y" if lookback_days > 30 else "1mo"
    price_df = fetch_price_history(ticker="KC=F", period=yf_period, interval=timeframe)

    council_df = load_council_history()

# 3. Process Data
if price_df is not None and not price_df.empty:
    # Slice price df to slider range exactly
    # Handle timezone differences between price_df and start_date
    price_start_date = pd.Timestamp(start_date)
    if price_df.index.tz is not None:
        if price_start_date.tz is None:
            price_start_date = price_start_date.tz_localize(price_df.index.tz)
        else:
            price_start_date = price_start_date.tz_convert(price_df.index.tz)

    price_df = price_df.loc[price_df.index >= price_start_date]

    if not council_df.empty:
        signals = process_signals(council_df, start_date)

        # Filter neutrals if requested
        if not show_neutral:
            signals = signals[signals['direction'] != 'NEUTRAL']
    else:
        signals = pd.DataFrame()

    # --- Plotting ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # A. Candlestick Chart (Prices)
    fig.add_trace(go.Candlestick(
        x=price_df.index,
        open=price_df['Open'],
        high=price_df['High'],
        low=price_df['Low'],
        close=price_df['Close'],
        name="Coffee Futures"
    ), row=1, col=1)

    # B. Signal Overlay
    if not signals.empty:
        # We need to align signals to the price Y-axis.
        # Since signals don't inherently have a "price" column in standard logs,
        # we interpolate or use the closest available price from price_df.

        # Merge closest price to signal timestamp
        signals_plottable = []
        for idx, row in signals.iterrows():
            # Find closest timestamp in price_df
            # Note: price_df.index is likely DatetimeIndex.
            # We need to ensure we can find nearest.
            # Using get_indexer with method='nearest' requires sorted index.

            try:
                # Ensure row timestamp is compatible with price_df index (timezone wise)
                ts = row['timestamp']
                if price_df.index.tz is not None and ts.tz is None:
                    ts = ts.tz_localize(price_df.index.tz)
                elif price_df.index.tz is None and ts.tz is not None:
                    ts = ts.tz_localize(None) # Make naive
                elif price_df.index.tz != ts.tz:
                    ts = ts.tz_convert(price_df.index.tz)

                closest_idx = price_df.index.get_indexer([ts], method='nearest')[0]

                # Check if closest index is reasonably close (e.g. within 2 days)
                # otherwise we might be plotting a signal on a price from weeks ago if gaps exist
                closest_time = price_df.index[closest_idx]
                time_diff = abs(closest_time - ts)

                if time_diff > timedelta(days=2):
                    continue

                closest_price = price_df.iloc[closest_idx]

                # Create hover text
                rationale_text = str(row.get('rationale',''))
                rationale_short = (rationale_text[:100] + '...') if len(rationale_text) > 100 else rationale_text

                confidence_val = row['confidence'] if pd.notna(row['confidence']) else 0.0

                hover_text = (
                    f"<b>{row['direction']}</b> ({confidence_val:.0%})<br>"
                    f"Date: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"Reason: {rationale_short}<br>"
                    f"Strategy: {row.get('strategy_type', 'N/A')}"
                )

                # Determine Y position (slightly above/below candle)
                y_pos = closest_price['High'] * 1.005 if row['direction'] == 'BEARISH' else closest_price['Low'] * 0.995

                signals_plottable.append({
                    'x': closest_time, # Snap to candle time for visual clarity
                    'y': y_pos,
                    'color': row['marker_color'],
                    'symbol': row['marker_symbol'],
                    'text': hover_text,
                    'direction': row['direction']
                })
            except Exception as e:
                # print(f"Error plotting signal: {e}")
                continue

        plot_df = pd.DataFrame(signals_plottable)

        if not plot_df.empty:
            # Add Scatter trace for signals
            fig.add_trace(go.Scatter(
                x=plot_df['x'],
                y=plot_df['y'],
                mode='markers',
                marker=dict(
                    symbol=plot_df['symbol'],
                    color=plot_df['color'],
                    size=12,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=plot_df['text'],
                hoverinfo='text',
                name='Trade Signals'
            ), row=1, col=1)

    # C. Confidence/Volume Lower Chart
    # If we have volume in price_df
    if 'Volume' in price_df.columns:
        fig.add_trace(go.Bar(
            x=price_df.index,
            y=price_df['Volume'],
            marker_color='rgba(200,200,200,0.5)',
            name='Volume'
        ), row=2, col=1)

    # Overlay Confidence scores on bottom chart if enabled
    if not signals.empty and show_confidence:
        # Align confidence to time like before
        conf_x = []
        conf_y = []
        conf_color = []

        for idx, row in signals.iterrows():
             try:
                 ts = row['timestamp']
                 if price_df.index.tz is not None and ts.tz is None:
                    ts = ts.tz_localize(price_df.index.tz)
                 elif price_df.index.tz is None and ts.tz is not None:
                    ts = ts.tz_localize(None)
                 elif price_df.index.tz != ts.tz:
                    ts = ts.tz_convert(price_df.index.tz)

                 closest_idx = price_df.index.get_indexer([ts], method='nearest')[0]

                 # Similarity check
                 closest_time = price_df.index[closest_idx]
                 if abs(closest_time - ts) > timedelta(days=2): continue

                 conf_x.append(closest_time)
                 conf_y.append(row['confidence'])
                 conf_color.append(row['marker_color'])
             except:
                 continue

        if conf_x:
            fig.add_trace(go.Scatter(
                x=conf_x,
                y=conf_y,
                mode='markers+lines',
                marker=dict(color=conf_color, size=8),
                line=dict(color='grey', width=1, dash='dot'),
                name='Signal Confidence'
            ), row=2, col=1)

    # --- Layout Update ---
    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        title_text=f"Coffee Futures (KC=F) - {timeframe} Candles",
        hovermode='x unified',
        template="plotly_dark"  # Matches your dashboard theme
    )

    # Y-Axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume / Conf", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # --- Drill Down Section ---
    st.markdown("### 📝 Signal Detail Log")

    # Show dataframe of visible signals
    if not signals.empty:
        # Create a cleaner view for the table
        display_cols = ['timestamp', 'direction', 'confidence', 'strategy_type', 'rationale']
        # Filter cols that actually exist
        display_cols = [c for c in display_cols if c in signals.columns]

        table_df = signals[display_cols].sort_values('timestamp', ascending=False)

        st.dataframe(
            table_df,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time (UTC)", format="D MMM, HH:mm"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                "rationale": st.column_config.TextColumn("Council Rationale", width="large"),
            },
            use_container_width=True,
            height=300
        )
    else:
        st.info("No signals found in the selected period.")

else:
    st.warning("Could not load price data. Please check your internet connection or yfinance availability.")
