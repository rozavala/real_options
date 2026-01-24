"""
Page 6: Signal Overlay (Decision vs. Price)

Purpose: Deep forensic analysis of decisions.
Visualizes WHERE signals were generated and HOW price evolved.

v3.1 - Final Feature Set:
- OVERLAP FIX: Explicitly DROP rows outside core trading hours.
- VISUALS: Added Session Markers (🔔) and Day/Week Separators.
- COMPLIANCE: Fixed st.dataframe deprecation warnings.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time
import sys
import os
import numpy as np
import holidays
import pytz

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import load_council_history, grade_decision_quality

st.set_page_config(layout="wide", page_title="Signal Analysis | Coffee Bot")

st.title("🎯 Signal Overlay Analysis")
st.caption("Forensic analysis of Council decisions against Coffee Futures price action")

# === CONFIGURATION CONSTANTS ===

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

COLOR_MAP = {
    'BULLISH': '#00CC96', 'UP': '#00CC96', 'LONG': '#00CC96',
    'BEARISH': '#EF553B', 'DOWN': '#EF553B', 'SHORT': '#EF553B',
    'IRON_CONDOR': '#636EFA', 'CONDOR': '#636EFA',
    'LONG_STRADDLE': '#AB63FA', 'STRADDLE': '#AB63FA',
    'NEUTRAL': '#888888', 'HOLD': '#888888',
}

SYMBOL_MAP = {
    'BULLISH': 'triangle-up', 'UP': 'triangle-up', 'LONG': 'triangle-up',
    'BEARISH': 'triangle-down', 'DOWN': 'triangle-down', 'SHORT': 'triangle-down',
    'IRON_CONDOR': 'hourglass', 'CONDOR': 'hourglass',
    'LONG_STRADDLE': 'diamond-wide', 'STRADDLE': 'diamond-wide',
    'NEUTRAL': 'circle', 'HOLD': 'circle',
}

# === SIDEBAR CONFIGURATION ===

with st.sidebar:
    st.header("🔬 Analysis Settings")

    timeframe = st.selectbox(
        "Timeframe",
        options=['5m', '15m', '30m', '1h', '1d'],
        index=0,  # Default: 5m
        format_func=lambda x: f"{x} Candles"
    )

    max_days = 59 if timeframe in ['5m', '15m', '30m', '1h'] else 730
    default_lookback = 3  # 72 hours

    lookback_days = st.slider(
        "Lookback Period (Days)",
        min_value=1,
        max_value=max_days,
        value=default_lookback,
        help="Default: 3 days (72 hours)"
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
    hide_gaps = st.toggle("Remove Non-Trading Gaps", value=True, help="Hides nights/weekends")
    show_confidence = st.toggle("Show Confidence Scores", value=True)
    show_outcomes = st.toggle("Highlight Win/Loss", value=True)


# === DATA FUNCTIONS ===

@st.cache_data(ttl=300)
def fetch_price_history_extended(ticker="KC=F", period="5d", interval="5m"):
    """Fetches historical OHLC data, sorted and converted to NY Time."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 1. Standardize to UTC first
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
            
        # 2. Convert to NY Time
        df.index = df.index.tz_convert('America/New_York')

        # 3. Strict Sorting & Dedup
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        return df
    except Exception as e:
        st.error(f"Failed to load price data: {e}")
        return None

def filter_market_hours(df):
    """
    NUCLEAR FIX for Overlaps:
    Explicitly drop rows outside of core trading hours (03:30 - 13:30 ET).
    """
    if df is None or df.empty:
        return df
    
    start_time = time(3, 30)
    end_time = time(14, 0) # Buffer to include settlement
    
    mask = (df.index.time >= start_time) & (df.index.time <= end_time)
    return df.loc[mask]

def calculate_market_hours_rangebreaks(df: pd.DataFrame, interval_str: str) -> list:
    """Calculates rangebreaks using NY Time bounds."""
    if df is None or df.empty:
        return []
    
    rangebreaks = []
    
    # 1. Weekend exclusion
    rangebreaks.append(dict(bounds=["sat", "mon"]))
    
    # 2. Holiday exclusion (NY Time)
    start_year = df.index.min().year
    end_year = df.index.max().year + 1
    us_holidays = holidays.US(years=range(start_year, end_year), observed=True)
    
    for holiday_date in us_holidays.keys():
        h_ts = pd.Timestamp(holiday_date).tz_localize('America/New_York')
        rangebreaks.append(dict(values=[h_ts.isoformat()]))
    
    # 3. Non-trading hours (intraday only)
    if interval_str in ['5m', '15m', '30m', '1h']:
        rangebreaks.append(dict(bounds=[14, 3.5], pattern="hour"))
    
    return rangebreaks

def get_marker_size(confidence: float, base_size: int = 14) -> int:
    """Scale marker size by confidence."""
    scale = 0.7 + (confidence * 0.6)
    return int(base_size * scale)

def process_signals_for_agent(history_df, agent_col, start_date):
    """Clean, filter, and format signals."""
    if history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()

    # 1. Timestamp cleaning
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # 2. Timezone: Convert to NY
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

    # 3. Date filtering
    cutoff_ts = pd.Timestamp(start_date)
    if cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize('America/New_York')
    else:
        cutoff_ts = cutoff_ts.tz_convert('America/New_York')

    df = df[df['timestamp'] >= cutoff_ts].copy()
    if df.empty:
        return pd.DataFrame()

    # 4. Column mapping
    if agent_col not in df.columns:
        if agent_col == 'master_decision':
            agent_col = 'direction'
        else:
            return pd.DataFrame()

    # 5. Extract direction
    df['plot_direction'] = df[agent_col].fillna('NEUTRAL').astype(str).str.upper()

    # 6. Confidence
    if 'master_confidence' in df.columns:
        df['plot_confidence'] = pd.to_numeric(df['master_confidence'], errors='coerce').fillna(0.5)
    else:
        df['plot_confidence'] = 0.5

    # 7. Label resolution
    if agent_col == 'master_decision':
        def resolve_label(row):
            d = str(row['plot_direction']).upper()
            s = str(row.get('strategy_type', '')).upper().strip()
            
            if 'IRON_CONDOR' in s: return 'IRON_CONDOR'
            if 'STRADDLE' in s: return 'LONG_STRADDLE'
            if 'BULL_CALL' in s: return 'BULLISH'
            if 'BEAR_PUT' in s: return 'BEARISH'
            return d if d in ['BULLISH', 'BEARISH'] else 'NEUTRAL'
        
        df['plot_label'] = df.apply(resolve_label, axis=1)
    else:
        df['plot_label'] = df['plot_direction']

    # 8. Colors & Symbols
    df['marker_color'] = df['plot_label'].map(COLOR_MAP).fillna('#888888')
    df['marker_symbol'] = df['plot_label'].map(SYMBOL_MAP).fillna('circle')
    df['marker_size'] = df['plot_confidence'].apply(lambda c: get_marker_size(c, base_size=14))

    return df

def build_hover_text(row):
    """Build rich hover text."""
    parts = [
        f"<b>{row.get('plot_label', 'SIGNAL')}</b>",
        f"Time: {row['timestamp'].strftime('%d %b %H:%M')} ET",
        f"Confidence: {row.get('plot_confidence', 0.5):.0%}",
    ]
    
    if 'strategy_type' in row and pd.notna(row.get('strategy_type')):
        parts.append(f"Strategy: {row['strategy_type']}")
    
    if 'outcome' in row and row.get('outcome') in ['WIN', 'LOSS']:
        outcome_emoji = '✅' if row['outcome'] == 'WIN' else '❌'
        parts.append(f"Outcome: {outcome_emoji} {row['outcome']}")
    
    if 'pnl_realized' in row and pd.notna(row.get('pnl_realized')) and row.get('pnl_realized') != 0:
        pnl = float(row['pnl_realized'])
        parts.append(f"P&L: {pnl:+.4f}")
    
    rationale = str(row.get('master_reasoning', row.get('rationale', '')))[:150]
    if rationale and rationale != 'nan':
        parts.append(f"<i>{rationale}...</i>")
    
    return "<br>".join(parts)


# === MAIN EXECUTION ===

end_date = datetime.now()
start_date = end_date - timedelta(days=lookback_days)

with st.spinner("Loading market data and signals..."):
    # Determine YFinance period
    if lookback_days <= 1: yf_period = "1d"
    elif lookback_days <= 5: yf_period = "5d"
    elif lookback_days <= 29: yf_period = "1mo"
    elif lookback_days <= 59: yf_period = "2mo"
    else: yf_period = "2y"

    price_df = fetch_price_history_extended(ticker="KC=F", period=yf_period, interval=timeframe)
    council_df = load_council_history()


# === PLOTTING ===

if price_df is not None and not price_df.empty:

    # 1. Filter Date Range
    current_time_et = datetime.now().astimezone(pytz.timezone('America/New_York'))
    cutoff_dt = current_time_et - timedelta(days=lookback_days)
    price_df = price_df[price_df.index >= cutoff_dt]

    # 2. NUCLEAR FIX: Filter Market Hours explicitly if hiding gaps
    if hide_gaps and timeframe in ['5m', '15m', '30m', '1h']:
        price_df = filter_market_hours(price_df)

    # Process signals
    signals = process_signals_for_agent(council_df, selected_agent_col, start_date)

    # === ALIGNMENT ENGINE ===
    plot_df = pd.DataFrame()
    if not signals.empty:
        signals = signals.sort_values('timestamp')
        price_idx = pd.DataFrame(index=price_df.index).reset_index()
        price_idx.columns = ['candle_timestamp']

        merged = pd.merge_asof(
            signals,
            price_idx,
            left_on='timestamp',
            right_on='candle_timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(hours=4)
        )

        matched_signals = merged.dropna(subset=['candle_timestamp'])

        if not matched_signals.empty:
            aligned_prices = price_df.loc[matched_signals['candle_timestamp']]

            plot_df = matched_signals.copy()
            plot_df['plot_x'] = matched_signals['candle_timestamp']
            plot_df['candle_high'] = aligned_prices['High'].values
            plot_df['candle_low'] = aligned_prices['Low'].values

            # Y-position logic
            plot_df['y_pos'] = np.where(
                plot_df['marker_symbol'] == 'triangle-down',
                plot_df['candle_high'] * 1.002,
                plot_df['candle_low'] * 0.998
            )

            plot_df['text_pos'] = np.where(
                plot_df['marker_symbol'] == 'triangle-down',
                "top center",
                "bottom center"
            )

            # Merge outcome data if available
            if show_outcomes:
                graded_df = grade_decision_quality(council_df)
                if not graded_df.empty:
                    graded_subset = graded_df[['timestamp', 'outcome', 'pnl_realized']].copy()
                    graded_subset['timestamp'] = pd.to_datetime(graded_subset['timestamp'], errors='coerce')
                    
                    # Normalize graded data to NY time
                    if graded_subset['timestamp'].dt.tz is None:
                        graded_subset['timestamp'] = graded_subset['timestamp'].dt.tz_localize('UTC')
                    graded_subset['timestamp'] = graded_subset['timestamp'].dt.tz_convert('America/New_York')
                    
                    plot_df = plot_df.merge(
                        graded_subset,
                        on='timestamp',
                        how='left'
                    )

            # Build hover text
            plot_df['hover'] = plot_df.apply(build_hover_text, axis=1)

            # Outcome-based line styling
            if 'outcome' in plot_df.columns:
                plot_df['marker_line_width'] = plot_df['outcome'].apply(
                    lambda x: 3 if x in ['WIN', 'LOSS'] else 1.5
                )
                plot_df['marker_line_color'] = plot_df['outcome'].apply(
                    lambda x: '#00FF00' if x == 'WIN' else '#FF0000' if x == 'LOSS' else 'white'
                )
            else:
                plot_df['marker_line_width'] = 1.5
                plot_df['marker_line_color'] = 'white'

    # === DRAW CHARTS ===

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=price_df.index, open=price_df['Open'], high=price_df['High'],
        low=price_df['Low'], close=price_df['Close'], name="KC Coffee (ET)"
    ), row=1, col=1)

    # 2. Signal markers
    if not plot_df.empty:
        fig.add_trace(go.Scatter(
            x=plot_df['plot_x'],
            y=plot_df['y_pos'],
            mode='markers+text' if show_labels else 'markers',
            text=plot_df['plot_label'] if show_labels else None,
            textposition=plot_df['text_pos'].tolist(),
            textfont=dict(size=9, color='white'),
            marker=dict(
                symbol=plot_df['marker_symbol'].tolist(),
                color=plot_df['marker_color'].tolist(),
                size=plot_df['marker_size'].tolist(),
                line=dict(
                    width=plot_df['marker_line_width'].tolist(),
                    color=plot_df['marker_line_color'].tolist()
                )
            ),
            hovertext=plot_df['hover'],
            hoverinfo="text",
            name="Signals",
            showlegend=False
        ), row=1, col=1)

    # 3. Legend entries
    legend_entries = [
        ('Bullish', '#00CC96', 'triangle-up'),
        ('Bearish', '#EF553B', 'triangle-down'),
        ('Iron Condor', '#636EFA', 'hourglass'),
        ('Long Straddle', '#AB63FA', 'diamond-wide'),
        ('Neutral', '#888888', 'circle'),
    ]
    for name, color, symbol in legend_entries:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol=symbol, color=color, size=12, line=dict(width=1, color='white')),
            name=name, showlegend=True
        ), row=1, col=1)

    # 4. Volume
    if 'Volume' in price_df.columns:
        fig.add_trace(go.Bar(
            x=price_df.index, y=price_df['Volume'],
            marker_color='rgba(255,255,255,0.1)', name='Volume', showlegend=False
        ), row=2, col=1)

    # 5. Confidence trace
    if not plot_df.empty and show_confidence:
        fig.add_trace(go.Scatter(
            x=plot_df['plot_x'],
            y=plot_df['plot_confidence'],
            mode='markers',
            marker=dict(color='#00CC96', size=6),
            name='Confidence', showlegend=False
        ), row=2, col=1)

    # === FEATURE: SESSION MARKERS (Task 4A) ===
    # Draw a Bell Icon 🔔 at the start of each trading session
    if timeframe in ['5m', '15m', '30m']:
        # Group by day
        dates = price_df.index.date
        unique_dates = np.unique(dates)
        
        for date in unique_dates:
            day_data = price_df[price_df.index.date == date]
            if not day_data.empty:
                first_candle = day_data.index[0]
                # Only mark if it's strictly a morning open (e.g. before 4:30 AM ET)
                # This prevents marking late data starts as "Open"
                if first_candle.time() <= time(4, 30):
                    fig.add_annotation(
                        x=first_candle,
                        y=day_data.iloc[0]['High'],
                        text="🔔",
                        showarrow=True,
                        arrowhead=0,
                        ax=0,
                        ay=-25,
                        row=1, col=1
                    )

    # === FEATURE: VISUAL SEPARATORS (Day & Week) ===
    if len(price_df) > 1:
        dates = price_df.index.date
        weekdays = price_df.index.dayofweek
        
        day_changes = np.where(dates[1:] != dates[:-1])[0] + 1
        
        for idx in day_changes:
            ts = price_df.index[idx]
            current_weekday = weekdays[idx]
            prev_weekday = weekdays[idx - 1] if idx > 0 else current_weekday
            
            # Week boundary: Monday (0) or a gap larger than 2 days
            is_week_start = (current_weekday == 0) or (current_weekday < prev_weekday)
            
            if is_week_start:
                # WEEK: Thick Dashed Orange
                fig.add_vline(x=ts, line_width=2, line_dash="dash", line_color="rgba(255, 165, 0, 0.5)", row="all")
            else:
                # DAY: Thin Dotted White
                fig.add_vline(x=ts, line_width=1, line_dash="dot", line_color="rgba(255, 255, 255, 0.2)", row="all")

    # Layout
    rangebreaks = calculate_market_hours_rangebreaks(price_df, timeframe) if hide_gaps else []

    fig.update_xaxes(rangebreaks=rangebreaks)
    fig.update_yaxes(title_text="Confidence", row=2, col=1, range=[0, 1])
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        title=f"Coffee Analysis (ET) | {selected_agent_label} | {timeframe} | Last {lookback_days} Days",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # === RAW SIGNAL LOG ===
    with st.expander("📝 Raw Signal Log", expanded=False):
        if not plot_df.empty:
            display_cols = ['timestamp', 'plot_label', 'plot_confidence', 'strategy_type']
            if 'outcome' in plot_df.columns:
                display_cols.append('outcome')
            if 'pnl_realized' in plot_df.columns:
                display_cols.append('pnl_realized')
            
            display_cols = [c for c in display_cols if c in plot_df.columns]
            
            st.dataframe(
                plot_df[display_cols].sort_values('timestamp', ascending=False),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time (ET)", format="D MMM HH:mm"),
                    "plot_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                },
                # FIX: Use width='stretch' to fix the warning
                width='stretch' 
            )
        else:
            st.info("No signals found in this window.")

else:
    st.warning("No market data available. Check lookback period or internet connection.")
