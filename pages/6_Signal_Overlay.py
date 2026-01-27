"""
Page 6: Signal Overlay (Decision vs. Price)

Purpose: Deep forensic analysis of decisions.
Visualizes WHERE signals were generated and HOW price evolved.

v3.0 - Critical Fixes:
- OVERLAP FIX: Use type='category' for X-axis (eliminates datetime gap bugs)
- Removed unreliable rangebreaks approach
- Added week boundary separators
- Added signal statistics summary
- Added session markers
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
import re

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

# New constant for Month-specific colors (perpetual)
MONTH_COLORS = {
    'H': '#00CC96',  # March - Green
    'K': '#636EFA',  # May - Blue
    'N': '#EF553B',  # July - Red
    'U': '#AB63FA',  # Sep - Purple
    'Z': '#FFA15A',  # Dec - Orange
}

SYMBOL_MAP = {
    'BULLISH': 'triangle-up', 'UP': 'triangle-up', 'LONG': 'triangle-up',
    'BEARISH': 'triangle-down', 'DOWN': 'triangle-down', 'SHORT': 'triangle-down',
    'IRON_CONDOR': 'hourglass', 'CONDOR': 'hourglass',
    'LONG_STRADDLE': 'diamond-wide', 'STRADDLE': 'diamond-wide',
    'NEUTRAL': 'circle', 'HOLD': 'circle',
}

# === DATA HELPER FUNCTIONS ===

def clean_contract_symbol(contract: str) -> str | None:
    """
    Extracts clean yfinance-compatible ticker from various input formats.

    Handles formats found in council_history.csv:
    - 'KCH6 (202603)' -> 'KCH26' (IB localSymbol + date)
    - 'KCH26 (202603)' -> 'KCH26' (already correct)
    - 'KCH26' -> 'KCH26' (clean)
    - 'KCH6' -> 'KCH26' (single-digit year, needs date context)
    - '(202603)' -> None (date only, no ticker)
    - None/empty -> None

    Returns:
        5-character ticker like 'KCH26' or None if unparseable
    """
    if not contract:
        return None

    contract = str(contract).strip()

    # Skip if it's just a date in parentheses like "(202603)"
    if contract.startswith('(') and ')' in contract:
        # Check if there's anything before the parentheses
        if contract.index('(') == 0:
            return None

    contract_upper = contract.upper()

    # Strategy 1: Try to match KC + Month + 2-digit year directly (e.g., KCH26)
    match_full = re.search(r'KC([FGHJKMNQUVXZ])(\d{2})(?!\d)', contract_upper)
    if match_full:
        month_code = match_full.group(1)
        year_2digit = match_full.group(2)
        return f"KC{month_code}{year_2digit}"

    # Strategy 2: Match KC + Month + 1-digit year AND extract year from date portion
    # Pattern: "KCH6 (202603)" -> extract H from KCH6, extract 26 from 202603
    match_ib_format = re.search(r'KC([FGHJKMNQUVXZ])(\d)(?:\s*\((\d{4})(\d{2})\))?', contract_upper)
    if match_ib_format:
        month_code = match_ib_format.group(1)
        single_year = match_ib_format.group(2)

        # If we have the date portion, extract the proper 2-digit year
        if match_ib_format.group(3) and match_ib_format.group(4):
            full_year = match_ib_format.group(3)  # e.g., "2026"
            year_2digit = full_year[2:4]  # e.g., "26"
            return f"KC{month_code}{year_2digit}"
        else:
            # No date portion - assume 202X decade (prepend "2" to make "26" from "6")
            year_2digit = "2" + single_year
            return f"KC{month_code}{year_2digit}"

    # Strategy 3: Try to extract from date portion if month code exists somewhere
    # This handles edge cases where format is unusual
    match_date = re.search(r'\((\d{4})(\d{2})\)', contract)
    match_month = re.search(r'KC([FGHJKMNQUVXZ])', contract_upper)
    if match_date and match_month:
        year_2digit = match_date.group(1)[2:4]  # "2026" -> "26"
        month_code = match_month.group(1)
        return f"KC{month_code}{year_2digit}"

    return None


def get_available_contracts(council_df: pd.DataFrame) -> list[str]:
    """
    Extract unique, clean contract symbols from council history.
    Returns sorted list with most recent contracts first.

    Example output: ['KCK26', 'KCH26']
    """
    if council_df.empty or 'contract' not in council_df.columns:
        return []

    # Get unique raw contracts
    raw_contracts = council_df['contract'].dropna().unique().tolist()

    # Clean each one
    cleaned = []
    for raw in raw_contracts:
        clean = clean_contract_symbol(raw)
        if clean and len(clean) == 5:  # Valid: KC + month + 2 digits
            cleaned.append(clean)

    # Deduplicate
    contracts = sorted(set(cleaned))

    # Sort by year (desc) then month (asc)
    month_order = 'FGHJKMNQUVXZ'

    def sort_key(symbol: str) -> tuple:
        """Sort by year descending, then month ascending."""
        try:
            if len(symbol) == 5 and symbol.startswith('KC'):
                month = symbol[2]
                year = int(symbol[3:5])
                month_idx = month_order.find(month)
                return (-year, month_idx if month_idx >= 0 else 99)
        except:
            pass
        return (0, 99)

    return sorted(contracts, key=sort_key)


def contract_to_yfinance_ticker(contract: str) -> str:
    """
    Convert contract symbol to yfinance ticker.

    yfinance Coffee futures format: KC{MONTH}{YY}.NYB
    Examples:
        - KCH26.NYB (March 2026)
        - KCK26.NYB (May 2026)

    Args:
        contract: Raw contract string (e.g., 'KCH6 (202603)' or 'FRONT_MONTH')

    Returns:
        Valid yfinance ticker (e.g., 'KCH26.NYB') or 'KC=F' for front month/fallback
    """
    # Front month option
    if contract == 'FRONT_MONTH' or not contract:
        return 'KC=F'

    # Clean the symbol
    clean_symbol = clean_contract_symbol(contract)

    # Validate: must be exactly 5 chars (KC + month + 2-digit year)
    if not clean_symbol or len(clean_symbol) != 5:
        return 'KC=F'

    # Validate format: KC + valid month + 2 digits
    if not re.match(r'^KC[FGHJKMNQUVXZ]\d{2}$', clean_symbol):
        return 'KC=F'

    # Add yfinance suffix for ICE/NYBOT
    return f"{clean_symbol}.NYB"


def get_contract_display_name(contract: str) -> str:
    """
    Get human-readable display name for contract.

    Examples:
        'FRONT_MONTH' -> '📊 Front Month (Continuous)'
        'KCH26' -> 'KCH26 (Mar 2026)'
        'KCH6 (202603)' -> 'KCH26 (Mar 2026)'
    """
    if contract == 'FRONT_MONTH':
        return '📊 Front Month (Continuous)'

    month_names = {
        'F': 'Jan', 'G': 'Feb', 'H': 'Mar', 'J': 'Apr',
        'K': 'May', 'M': 'Jun', 'N': 'Jul', 'Q': 'Aug',
        'U': 'Sep', 'V': 'Oct', 'X': 'Nov', 'Z': 'Dec'
    }

    clean_symbol = clean_contract_symbol(contract)
    if clean_symbol and len(clean_symbol) == 5:
        month_code = clean_symbol[2]
        year = clean_symbol[3:5]
        month_name = month_names.get(month_code, '???')
        return f"{clean_symbol} ({month_name} 20{year})"

    # Fallback: return original if can't parse
    return str(contract) if contract else 'Unknown'

def get_contract_color(contract: str, default_color: str) -> str:
    """Get color for contract based on month code."""
    if not contract or contract == 'FRONT_MONTH':
        return default_color

    clean = clean_contract_symbol(contract)
    if clean and len(clean) >= 3:
        month_code = clean[2]
        return MONTH_COLORS.get(month_code, default_color)

    return default_color


# === SIDEBAR CONFIGURATION ===

with st.sidebar:
    st.header("🔬 Analysis Settings")

    timeframe = st.selectbox(
        "Timeframe",
        options=['5m', '15m', '30m', '1h', '1d'],
        index=0,
        format_func=lambda x: f"{x} Candles"
    )

    max_days = 59 if timeframe in ['5m', '15m', '30m', '1h'] else 730
    default_lookback = 3

    lookback_days = st.slider(
        "Lookback Period (Days)",
        min_value=1,
        max_value=max_days,
        value=default_lookback,
        help="Default: 3 days (72 hours)"
    )

    st.markdown("---")

    # === NEW: CONTRACT SELECTOR ===
    st.header("📜 Contract")

    # Load council data early to get available contracts
    council_df_for_contracts = load_council_history()
    available_contracts = get_available_contracts(council_df_for_contracts)

    # Build options list: Front Month + specific contracts
    contract_options = ['FRONT_MONTH'] + available_contracts

    # Format function for display
    contract_format = lambda c: get_contract_display_name(c)

    selected_contract = st.selectbox(
        "Price Data Source",
        options=contract_options,
        index=0,  # Default to Front Month
        format_func=contract_format,
        help="Select which contract's price data to display"
    )

    # Signal filter option
    if selected_contract != 'FRONT_MONTH':
        filter_signals_to_contract = st.checkbox(
            f"Only show {selected_contract} signals",
            value=True,
            help="When checked, only shows signals targeting this specific contract"
        )
    else:
        filter_signals_to_contract = False

    # Multi-contract mode (optional - more advanced)
    show_all_signals = st.checkbox(
        "Show signals from all contracts",
        value=False,
        help="Shows signals from all contracts overlaid on the selected price chart"
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
    filter_to_market_hours = st.toggle("Filter to Market Hours Only", value=True,
                                        help="Show only 03:30-13:30 ET candles")
    show_labels = st.toggle("Show Signal Labels", value=True)
    show_day_separators = st.toggle("Show Day/Week Separators", value=True)
    show_confidence = st.toggle("Show Confidence Scores", value=True)
    show_outcomes = st.toggle("Highlight Win/Loss", value=True)


# === DATA FUNCTIONS ===

@st.cache_data(ttl=300)
def fetch_price_history_extended(ticker="KC=F", period="5d", interval="5m"):
    """
    Fetches historical OHLC data from yfinance, converted to NY Time.

    Args:
        ticker: yfinance ticker (e.g., 'KC=F' or 'KCH26.NYB')
        period: lookback period
        interval: candle interval

    Returns:
        DataFrame with OHLC data, or None if fetch failed
    """
    import logging

    try:
        # Suppress yfinance error logging temporarily
        yf_logger = logging.getLogger('yfinance')
        original_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)

        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        finally:
            # Restore original logging level
            yf_logger.setLevel(original_level)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize to UTC then convert to NY
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        df.index = df.index.tz_convert('America/New_York')

        # Clean: dedupe and sort
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        return df

    except Exception as e:
        # Silent fail - we'll fall back to front month
        return None


def filter_market_hours(df):
    """
    Filter to ICE Coffee core trading hours (03:30 - 13:30 ET).
    With categorical X-axis, this effectively hides the gaps.
    """
    if df is None or df.empty:
        return df

    start_time = time(3, 30)
    end_time = time(13, 30)
    
    mask = (df.index.time >= start_time) & (df.index.time <= end_time)
    return df.loc[mask]


def filter_non_trading_days(df):
    """
    Remove weekend days, US market holidays, and rows with missing OHLC data.

    Coffee futures don't trade on weekends or holidays. YFinance may return:
    - Rows for weekends with NaN OHLC but Volume=0
    - Rows for holidays with NaN OHLC but Volume=0

    This function filters all three cases.

    Args:
        df: DataFrame with DatetimeIndex in America/New_York timezone

    Returns:
        DataFrame filtered to trading days with valid price data only
    """
    if df is None or df.empty:
        return df

    # 1. Filter out weekends (Monday=0, Sunday=6)
    df = df[df.index.weekday < 5]

    # 2. Filter out US holidays (ICE follows NYSE calendar)
    unique_years = df.index.year.unique()
    us_holidays = holidays.US(years=list(unique_years), observed=True)

    # FIX: Convert holidays object to a set of dates for proper filtering
    holiday_dates = set(us_holidays.keys())
    # Use df.index.date to compare with holiday dates (using normalize() keeps it as Timestamp which mismatches)
    df = df[~pd.Index(df.index.date).isin(holiday_dates)]

    # 3. CRITICAL: Filter out rows with NaN in OHLC columns
    # YFinance returns these for non-trading periods
    if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    return df


def get_marker_size(confidence: float, base_size: int = 14) -> int:
    """Scale marker size by confidence."""
    scale = 0.7 + (confidence * 0.6)
    return int(base_size * scale)


def process_signals_for_agent(history_df, agent_col, start_date, contract_filter=None, show_all=False):
    """
    Clean, filter, and format signals.

    Args:
        history_df: Council history DataFrame
        agent_col: Column name for agent to display
        start_date: Cutoff date
        contract_filter: If set, only show signals for this contract (e.g., 'KCH26')
        show_all: If True, show all contracts (overrides contract_filter)
    """
    if history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()

    # Timestamp cleaning
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Timezone: Convert to NY to match price data
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

    # Date filtering
    cutoff_ts = pd.Timestamp(start_date)
    if cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize('America/New_York')
    else:
        cutoff_ts = cutoff_ts.tz_convert('America/New_York')

    df = df[df['timestamp'] >= cutoff_ts].copy()
    if df.empty:
        return pd.DataFrame()

    # === NEW: CONTRACT FILTERING (With Regex Cleaning) ===
    # Clean the 'contract' column in the DataFrame for reliable filtering
    if 'contract' in df.columns:
        df['contract_clean'] = df['contract'].apply(clean_contract_symbol)
    else:
        df['contract_clean'] = None

    if not show_all and contract_filter and contract_filter != 'FRONT_MONTH':
        # Clean the filter target as well
        target_contract = clean_contract_symbol(contract_filter)

        # Filter using the clean column
        if target_contract:
            df = df[df['contract_clean'] == target_contract]
            if df.empty:
                return pd.DataFrame()

    # Add contract info for display (use original or clean?)
    # Using clean makes it cleaner in hover text, fallback to "Unknown"
    df['signal_contract'] = df['contract_clean'].fillna('Unknown')

    # Column mapping
    if agent_col not in df.columns:
        if agent_col == 'master_decision':
            agent_col = 'direction'
        else:
            return pd.DataFrame()

    # Extract direction
    df['plot_direction'] = df[agent_col].fillna('NEUTRAL').astype(str).str.upper()

    # Confidence
    if 'master_confidence' in df.columns:
        df['plot_confidence'] = pd.to_numeric(df['master_confidence'], errors='coerce').fillna(0.5)
    else:
        df['plot_confidence'] = 0.5

    # Label resolution (strategy-aware)
    if agent_col == 'master_decision':
        def resolve_label(row):
            d = str(row['plot_direction']).upper()
            s = str(row.get('strategy_type', '')).upper().strip()
            
            if 'IRON_CONDOR' in s:
                return 'IRON_CONDOR'
            if 'STRADDLE' in s:
                return 'LONG_STRADDLE'
            if 'BULL_CALL' in s:
                return 'BULLISH'
            if 'BEAR_PUT' in s:
                return 'BEARISH'

            return d if d in ['BULLISH', 'BEARISH'] else 'NEUTRAL'
        
        df['plot_label'] = df.apply(resolve_label, axis=1)
    else:
        df['plot_label'] = df['plot_direction']

    # Colors & Symbols
    df['marker_color'] = df['plot_label'].map(COLOR_MAP).fillna('#888888')
    df['marker_symbol'] = df['plot_label'].map(SYMBOL_MAP).fillna('circle')

    # Override colors if show_all is True to distinguish contracts
    if show_all:
        df['marker_color'] = df['signal_contract'].apply(lambda c: get_contract_color(c, '#888888'))

    df['marker_size'] = df['plot_confidence'].apply(lambda c: get_marker_size(c, base_size=14))

    return df


def build_hover_text(row):
    """Build rich hover text."""
    parts = [
        f"<b>{row.get('plot_label', 'SIGNAL')}</b>",
        f"Time: {row['timestamp'].strftime('%b %d %H:%M')} ET",
    ]
    
    # Add contract info
    if 'signal_contract' in row and row.get('signal_contract') not in [None, 'Unknown', 'nan']:
        parts.append(f"Contract: {row['signal_contract']}")

    parts.append(f"Confidence: {row.get('plot_confidence', 0.5):.0%}")

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

# Determine yfinance ticker based on contract selection
yf_ticker = contract_to_yfinance_ticker(selected_contract)

# Debug info (helps troubleshooting)
if selected_contract != 'FRONT_MONTH':
    clean = clean_contract_symbol(selected_contract)
    st.caption(f"🔍 Debug: `{selected_contract}` → cleaned: `{clean}` → yfinance: `{yf_ticker}`")

with st.spinner(f"Loading {get_contract_display_name(selected_contract)} data..."):
    # Determine period
    if lookback_days <= 1: yf_period = "5d"
    elif lookback_days <= 5: yf_period = "10d"
    elif lookback_days <= 29: yf_period = "1mo"
    elif lookback_days <= 59: yf_period = "2mo"
    else: yf_period = "2y"

    # Fetch price data for selected contract
    price_df = fetch_price_history_extended(ticker=yf_ticker, period=yf_period, interval=timeframe)

    # Fallback: If specific contract has no data, try front month
    if price_df is None or price_df.empty:
        if selected_contract != 'FRONT_MONTH':
            st.warning(f"⚠️ No price data available for {selected_contract}. Falling back to front month.")
            price_df = fetch_price_history_extended(ticker='KC=F', period=yf_period, interval=timeframe)
            # Update display to show fallback
            actual_ticker_display = "Front Month (Fallback)"
        else:
            actual_ticker_display = None
    else:
        actual_ticker_display = get_contract_display_name(selected_contract)

    council_df = load_council_history()


# === PLOTTING ===

if price_df is not None and not price_df.empty:

    # 1. Filter Date Range (UPDATED & FIXED)
    current_time_et = datetime.now().astimezone(pytz.timezone('America/New_York'))
    
    # Initialize US Holidays
    us_holidays = holidays.US(years=[current_time_et.year, current_time_et.year - 1], observed=True)
    
    # Calculate cutoff by counting back N trading days
    cutoff_dt = current_time_et
    days_counted = 0
    
    while days_counted < lookback_days:
        cutoff_dt -= timedelta(days=1)
        # Check if weekday (0-4) and not a holiday
        if cutoff_dt.weekday() < 5 and cutoff_dt.date() not in us_holidays:
            days_counted += 1
            
    # CRITICAL FIX: Force the start time to Midnight (00:00:00)
    # This prevents the filter from clipping the Morning Session of the first day.
    cutoff_dt = cutoff_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Apply the filter (>= Midnight captures the whole day)
    price_df = price_df[price_df.index >= cutoff_dt]

    # 2. Remove non-trading days (weekends/holidays)
    pre_filter_count = len(price_df)
    price_df = filter_non_trading_days(price_df)

    # Check for filtered rows
    if len(price_df) < pre_filter_count:
        removed_count = pre_filter_count - len(price_df)
        removed_pct = (removed_count / pre_filter_count) * 100
        st.caption(f"🗓️ Filtered {removed_count} candles ({removed_pct:.1f}%) - weekends/holidays/invalid data")

    # Warn if we ended up with no data after filtering
    if price_df.empty:
        st.warning(f"⚠️ No trading data available for the selected {lookback_days}-day lookback period. Try increasing the lookback.")

    if len(price_df) < pre_filter_count:
        removed_count = pre_filter_count - len(price_df)
        removed_pct = (removed_count / pre_filter_count) * 100
        st.caption(f"🗓️ Filtered {removed_count} candles ({removed_pct:.1f}%) - weekends/holidays/invalid data")
    elif len(price_df) == 0:
        st.warning("⚠️ No valid trading data available for the selected period.")

    # 3. Optionally filter to market hours
    original_count = len(price_df)
    if filter_to_market_hours and timeframe in ['5m', '15m', '30m', '1h']:
        price_df = filter_market_hours(price_df)

    if len(price_df) < original_count:
        st.caption(f"📊 Showing {len(price_df):,} candles (filtered from {original_count:,} to market hours only)")

    # 3. Process signals with contract filter
    contract_filter = selected_contract if filter_signals_to_contract else None
    signals = process_signals_for_agent(
        council_df,
        selected_agent_col,
        start_date,
        contract_filter=contract_filter,
        show_all=show_all_signals
    )

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

            # Merge outcome data
            if show_outcomes:
                graded_df = grade_decision_quality(council_df)
                if not graded_df.empty:
                    graded_subset = graded_df[['timestamp', 'outcome', 'pnl_realized']].copy()
                    graded_subset['timestamp'] = pd.to_datetime(graded_subset['timestamp'], errors='coerce')
                    
                    if graded_subset['timestamp'].dt.tz is None:
                        graded_subset['timestamp'] = graded_subset['timestamp'].dt.tz_localize('UTC')
                    graded_subset['timestamp'] = graded_subset['timestamp'].dt.tz_convert('America/New_York')
                    
                    plot_df = plot_df.merge(graded_subset, on='timestamp', how='left')

            # Build hover text
            plot_df['hover'] = plot_df.apply(build_hover_text, axis=1)

            # Outcome-based styling
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

    # === PRICE SUMMARY ===
    if len(price_df) >= 2:
        first_close = price_df['Close'].iloc[0]
        last_close = price_df['Close'].iloc[-1]
        pct_change = ((last_close - first_close) / first_close) * 100
        high = price_df['High'].max()
        low = price_df['Low'].min()

        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Period Change", f"{pct_change:+.2f}%")
        with summary_cols[1]:
            st.metric("High", f"${high:.2f}")
        with summary_cols[2]:
            st.metric("Low", f"${low:.2f}")
        with summary_cols[3]:
            st.metric("Range", f"${high - low:.2f}")

    # === DRAW CHARTS ===
    # CREATE SUBPLOTS WITH SECONDARY Y-AXIS
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        specs=[
            [{"secondary_y": False}],  # Row 1: Price chart
            [{"secondary_y": True}]    # Row 2: Confidence (left) + Volume (right)
        ]
    )

    # 1. Candlestick (Row 1)
    fig.add_trace(go.Candlestick(
        x=price_df.index,
        open=price_df['Open'],
        high=price_df['High'],
        low=price_df['Low'],
        close=price_df['Close'],
        name="KC Coffee (ET)"
    ), row=1, col=1)

    # 2. Signal markers (Row 1)
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

    # 3. Legend entries (Row 1)
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

    # 4. Volume (Row 2 - SECONDARY Y-AXIS on RIGHT)
    if 'Volume' in price_df.columns:
        fig.add_trace(go.Bar(
            x=price_df.index,
            y=price_df['Volume'],
            marker_color='rgba(100, 150, 255, 0.4)',  # Slightly more visible
            name='Volume',
            showlegend=False,
        ), row=2, col=1, secondary_y=True)

    # 5. Confidence (Row 2 - PRIMARY Y-AXIS on LEFT)
    if not plot_df.empty and show_confidence:
        fig.add_trace(go.Scatter(
            x=plot_df['plot_x'],
            y=plot_df['plot_confidence'],
            mode='markers+lines',
            line=dict(color='#00CC96', width=1.5, dash='dot'),
            marker=dict(color='#00CC96', size=8, symbol='circle'),
            name='Confidence',
            showlegend=False,
        ), row=2, col=1, secondary_y=False)

    # === DAY & WEEK SEPARATORS (FIXED VISIBILITY) ===
    if show_day_separators and len(price_df) > 1:
        dates = price_df.index.date
        weekdays = price_df.index.dayofweek

        day_changes = np.where(dates[1:] != dates[:-1])[0] + 1
        
        for idx in day_changes:
            ts = price_df.index[idx]
            current_weekday = weekdays[idx]
            prev_weekday = weekdays[idx - 1] if idx > 0 else current_weekday

            is_week_start = (current_weekday == 0) or (current_weekday < prev_weekday)

            if is_week_start:
                # WEEK SEPARATOR - Prominent orange
                fig.add_vline(
                    x=ts,
                    line_width=2,
                    line_dash="dash",
                    line_color="rgba(255, 180, 80, 0.8)",
                    row="all"
                )
                fig.add_annotation(
                    x=ts,
                    y=1.02,
                    yref="paper",
                    text="📅 Week",
                    showarrow=False,
                    font=dict(size=9, color="rgba(255, 180, 80, 0.9)"),
                    xanchor="left"
                )
            else:
                # DAY SEPARATOR - Visible gray
                fig.add_vline(
                    x=ts,
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(180, 180, 180, 0.6)",
                    row="all"
                )

    # === SESSION MARKERS (MARKET OPEN/CLOSE) ===
    if timeframe in ['5m', '15m', '30m']:
        # Find market open times (03:30 ET)
        for date in price_df.index.date:
            date_data = price_df[price_df.index.date == date]
            if not date_data.empty:
                first_candle = date_data.index[0]
                # Market Open marker
                if first_candle.time() <= time(4, 0):  # Within first 30 min
                    fig.add_annotation(
                        x=first_candle,
                        y=date_data.loc[first_candle, 'High'] * 1.003,
                        text="🔔",
                        showarrow=False,
                        font=dict(size=10),
                        row=1, col=1
                    )

    # Determine title based on contract
    if actual_ticker_display:
        chart_title = f"{actual_ticker_display} | {selected_agent_label} | {timeframe} | Last {lookback_days} Days"
    else:
        chart_title = f"Coffee Analysis (ET) | {selected_agent_label} | {timeframe} | Last {lookback_days} Days"

    # Add signal contract info if showing multiple
    if show_all_signals and not signals.empty:
        unique_contracts = signals['signal_contract'].unique()
        if len(unique_contracts) > 1:
            chart_title += f" | Signals: {', '.join(unique_contracts)}"

    # === LAYOUT (CRITICAL: type='category' fixes overlaps) ===
    fig.update_xaxes(
        type='category',
        tickformat='%b %d\n%H:%M',
        nticks=15,
        tickangle=0,
        row=1, col=1
    )

    fig.update_xaxes(
        type='category',
        tickformat='%H:%M',
        nticks=15,
        row=2, col=1
    )

    # Row 1 Y-axis
    fig.update_yaxes(title_text="Price (¢/lb)", row=1, col=1)

    # Row 2 Primary Y-axis (Confidence) - LEFT
    fig.update_yaxes(
        title_text="Confidence",
        range=[0, 1],
        tickformat='.0%',
        side='left',
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=2, col=1,
        secondary_y=False
    )

    # Row 2 Secondary Y-axis (Volume) - RIGHT
    fig.update_yaxes(
        title_text="Volume",
        side='right',
        showgrid=True,
        gridcolor='rgba(100, 150, 255, 0.1)',
        tickformat=',d',
        row=2, col=1,
        secondary_y=True
    )

    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        title=chart_title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width='stretch')

    # === SIGNAL STATISTICS ===
    if not plot_df.empty:
        st.markdown("---")
        st.subheader("📊 Signal Statistics")

        stat_cols = st.columns(5)

        total_signals = len(plot_df)
        bullish = plot_df['plot_label'].isin(['BULLISH', 'LONG']).sum()
        bearish = plot_df['plot_label'].isin(['BEARISH', 'SHORT']).sum()
        vol_trades = plot_df['plot_label'].isin(['IRON_CONDOR', 'LONG_STRADDLE']).sum()
        neutral = total_signals - bullish - bearish - vol_trades

        with stat_cols[0]:
            st.metric("Total Signals", total_signals)
        with stat_cols[1]:
            st.metric("🟢 Bullish", int(bullish))
        with stat_cols[2]:
            st.metric("🔴 Bearish", int(bearish))
        with stat_cols[3]:
            st.metric("🟣 Volatility", int(vol_trades))
        with stat_cols[4]:
            st.metric("⚪ Neutral", int(neutral))

        if 'outcome' in plot_df.columns:
            wins = (plot_df['outcome'] == 'WIN').sum()
            losses = (plot_df['outcome'] == 'LOSS').sum()
            graded = wins + losses
            if graded > 0:
                win_rate = wins / graded * 100
                st.caption(f"📈 Win Rate: **{win_rate:.1f}%** ({wins}W / {losses}L from {graded} graded signals)")

    # === DOWNLOAD SECTION ===
    st.markdown("---")
    with st.expander("💾 Download Chart Data"):
        # Prepare data for download
        download_df = price_df.copy()
        download_df.index.name = "Date_ET"
        
        # Merge signal info if it exists and matches timestamps
        if 'plot_df' in locals() and not plot_df.empty:
            # Create a simplified signal frame
            sig_export = plot_df.set_index('plot_x')[['plot_label', 'plot_confidence', 'strategy_type', 'outcome']]
            # Merge left to keep all price rows
            download_df = download_df.join(sig_export, how='left')
        
        # Convert to CSV
        csv = download_df.to_csv().encode('utf-8')
        
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"coffee_data_{lookback_days}d_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv',
        )

    # === RAW SIGNAL LOG ===
    with st.expander("📝 Raw Signal Log", expanded=False):
        if not plot_df.empty:
            display_cols = ['timestamp', 'plot_label', 'plot_confidence', 'strategy_type']
            if 'outcome' in plot_df.columns:
                display_cols.append('outcome')
            if 'pnl_realized' in plot_df.columns:
                display_cols.append('pnl_realized')
            if 'signal_contract' in plot_df.columns:
                display_cols.insert(1, 'signal_contract')
            
            display_cols = [c for c in display_cols if c in plot_df.columns]
            
            st.dataframe(
                plot_df[display_cols].sort_values('timestamp', ascending=False),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time (ET)", format="D MMM HH:mm"),
                    "plot_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                },
                width='stretch'
            )
        else:
            st.info("No signals found in this window.")

else:
    st.warning("No market data available. Check lookback period or internet connection.")
