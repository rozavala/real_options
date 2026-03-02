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
from datetime import datetime, timedelta, time
import sys
import os
import numpy as np
import holidays
import pytz
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import load_council_history, grade_decision_quality
from trading_bot.timestamps import parse_ts_column
from trading_bot.data_providers import get_data_source_label
from config import get_active_profile
from dashboard_utils import get_config

st.set_page_config(layout="wide", page_title="Signal Analysis | Real Options")

from _commodity_selector import selected_commodity
ticker = selected_commodity()

# E1: Dynamic profile loading
config = get_config()
profile = get_active_profile(config)

# Derive price display decimals from tick size (KC=0.05 → 2dp, CC=1.0 → 0dp)
import math
_tick = profile.contract.tick_size
_price_decimals = max(0, -math.floor(math.log10(_tick))) if _tick < 1 else 0
_price_fmt = f",.{_price_decimals}f"

st.title("🎯 Signal Overlay Analysis")
st.caption("Forensic analysis of Council decisions against futures price action")

# === CONFIGURATION CONSTANTS ===

# E6: Removed archived ML Signal agent
AGENT_MAPPING = {
    "👑 Master Decision": "master_decision",
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

# Month-specific colors for all 12 contract months (F-Z)
MONTH_COLORS = {
    'F': '#19D3F3',  # Jan - Cyan
    'G': '#B6E880',  # Feb - Lime
    'H': '#00CC96',  # Mar - Green
    'J': '#FF6692',  # Apr - Pink
    'K': '#636EFA',  # May - Blue
    'M': '#FECB52',  # Jun - Yellow
    'N': '#EF553B',  # Jul - Red
    'Q': '#FF97FF',  # Aug - Magenta
    'U': '#AB63FA',  # Sep - Purple
    'V': '#00B5F7',  # Oct - Sky blue
    'X': '#72B7B2',  # Nov - Teal
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
        Ticker like 'KCH26' or None if unparseable
    """
    if not contract:
        return None

    contract = str(contract).strip()
    _tk = profile.contract.symbol  # e.g., 'KC', 'CC'

    # Skip if it's just a date in parentheses like "(202603)"
    if contract.startswith('(') and ')' in contract:
        # Check if there's anything before the parentheses
        if contract.index('(') == 0:
            return None

    contract_upper = contract.upper()

    # Strategy 1: Try to match TICKER + Month + 2-digit year directly (e.g., KCH26)
    match_full = re.search(rf'{_tk}([FGHJKMNQUVXZ])(\d{{2}})(?!\d)', contract_upper)
    if match_full:
        month_code = match_full.group(1)
        year_2digit = match_full.group(2)
        return f"{_tk}{month_code}{year_2digit}"

    # Strategy 2: Match TICKER + Month + 1-digit year AND extract year from date portion
    match_ib_format = re.search(rf'{_tk}([FGHJKMNQUVXZ])(\d)(?:\s*\((\d{{4}})(\d{{2}})\))?', contract_upper)
    if match_ib_format:
        month_code = match_ib_format.group(1)
        single_year = match_ib_format.group(2)

        # If we have the date portion, extract the proper 2-digit year
        if match_ib_format.group(3) and match_ib_format.group(4):
            full_year = match_ib_format.group(3)
            year_2digit = full_year[2:4]
            return f"{_tk}{month_code}{year_2digit}"
        else:
            year_2digit = "2" + single_year
            return f"{_tk}{month_code}{year_2digit}"

    # Strategy 3: Try to extract from date portion if month code exists somewhere
    match_date = re.search(r'\((\d{4})(\d{2})\)', contract)
    match_month = re.search(rf'{_tk}([FGHJKMNQUVXZ])', contract_upper)
    if match_date and match_month:
        year_2digit = match_date.group(1)[2:4]
        month_code = match_month.group(1)
        return f"{_tk}{month_code}{year_2digit}"

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
        _tk_len = len(profile.contract.symbol) + 3  # ticker + month + 2-digit year
        if clean and len(clean) == _tk_len:
            cleaned.append(clean)

    # Deduplicate
    contracts = sorted(set(cleaned))

    # Sort by expiration date ASCENDING (soonest first)
    # This matches operator mental model: "what's trading NOW is at the top"
    month_order = 'FGHJKMNQUVXZ'

    _tk = profile.contract.symbol
    _tk_len = len(_tk)

    def sort_key(symbol: str) -> tuple:
        """Sort by year ascending, then month ascending (soonest expiry first)."""
        try:
            if symbol.startswith(_tk) and len(symbol) == _tk_len + 3:
                month = symbol[_tk_len]
                year = int(symbol[_tk_len + 1:_tk_len + 3])
                month_idx = month_order.find(month)
                return (year, month_idx if month_idx >= 0 else 99)
        except (ValueError, IndexError):
            pass
        return (99, 99)  # Unknown contracts sort last

    return sorted(contracts, key=sort_key)


def resolve_front_month_ticker(config_path: str = "config.json") -> tuple[str, str]:
    """
    Resolve the trading system's actual front month contract using the
    same DTE rules as the execution layer.

    This ensures the Signal Overlay chart matches what the trading system
    actually trades, not just the nearest calendar month.

    Commodity-agnostic: Uses CommodityProfile.min_dte and contract_months.

    Returns:
        Tuple of (yfinance_ticker, display_symbol)
        e.g., ('KCK26.NYB', 'KCK26') or ('KC=F', 'FRONT_MONTH') as fallback
    """
    try:
        from config import get_active_profile
        from config_loader import load_config

        cfg = load_config()
        profile = get_active_profile(cfg)
        min_dte = profile.min_dte  # e.g., 45

        # Get the valid contract month codes for this commodity
        valid_months = profile.contract.contract_months  # e.g., ['H', 'K', 'N', 'U', 'Z']
        ticker = profile.contract.symbol  # e.g., 'KC'

        # Generate candidate contracts for the next ~2 years
        from datetime import datetime, timedelta
        today = datetime.now()

        # Month code to calendar month mapping
        month_code_to_num = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }

        candidates = []
        for year_offset in range(0, 3):  # Current year + next 2
            year = today.year + year_offset
            year_2digit = year % 100

            for month_code in valid_months:
                month_num = month_code_to_num.get(month_code)
                if not month_num:
                    continue

                # Approximate expiration: ~20th of the contract month
                # (exact date varies by exchange, but 20th is a safe approximation
                # for determining DTE eligibility — the real filter happens in IB)
                try:
                    from calendar import monthrange
                    # Use 3rd Friday as rough expiry estimate for ICE coffee
                    approx_expiry = datetime(year, month_num, 19)
                except ValueError:
                    continue

                dte = (approx_expiry - today).days

                if dte >= min_dte:
                    symbol = f"{ticker}{month_code}{year_2digit}"
                    candidates.append((dte, symbol))

        if candidates:
            # Sort by DTE ascending — first one is the trading front month
            candidates.sort(key=lambda x: x[0])
            front_symbol = candidates[0][1]
            # Use exchange-specific suffix (NYB for ICE/NYBOT, NYM for NYMEX, etc.)
            suffix_map = {'ICE': 'NYB', 'NYBOT': 'NYB', 'NYMEX': 'NYM', 'COMEX': 'CMX', 'CME': 'CME'}
            suffix = suffix_map.get(profile.contract.exchange, 'NYB')
            yf_ticker = f"{front_symbol}.{suffix}"
            return (yf_ticker, front_symbol)

    except Exception as e:
        import logging
        logging.warning(f"Front month resolution failed, falling back to {profile.contract.symbol}=F: {e}")

    # Fallback: use yfinance continuous contract
    return (f'{profile.contract.symbol}=F', 'FRONT_MONTH')


def contract_to_yfinance_ticker(contract: str) -> str:
    """
    Convert contract symbol to yfinance ticker.

    Format: {TICKER}{MONTH}{YY}.{EXCHANGE_SUFFIX}
    Examples:
        - KCH26.NYB (Coffee March 2026)
        - KCK26.NYB (Coffee May 2026)

    Args:
        contract: Raw contract string (e.g., 'KCH6 (202603)' or 'FRONT_MONTH')

    Returns:
        Valid yfinance ticker or '{ticker}=F' for front month/fallback
    """
    # E1: Commodity-agnostic ticker
    fallback_ticker = f"{profile.ticker}=F"

    # Front month option — resolve using trading system's DTE rules
    if contract == 'FRONT_MONTH' or not contract:
        yf_ticker, _ = resolve_front_month_ticker()
        return yf_ticker

    # Clean the symbol
    clean_symbol = clean_contract_symbol(contract)

    # Validate: must be exactly 5 chars (TICKER + month + 2-digit year)
    if not clean_symbol or len(clean_symbol) != (len(profile.ticker) + 3):
        return fallback_ticker

    # Validate format: TICKER + valid month + 2 digits
    pattern = rf'^{profile.ticker}[FGHJKMNQUVXZ]\d{{2}}$'
    if not re.match(pattern, clean_symbol):
        return fallback_ticker

    # Add yfinance suffix based on exchange
    # ICE -> NYB (for KC/CC), NYMEX -> NYM (for CL/NG), COMEX -> CMX (for GC/SI)
    suffix_map = {'ICE': 'NYB', 'NYBOT': 'NYB', 'NYMEX': 'NYM', 'COMEX': 'CMX', 'CME': 'CME'}
    suffix = suffix_map.get(profile.contract.exchange, 'NYB')

    return f"{clean_symbol}.{suffix}"


def get_contract_display_name(contract: str) -> str:
    """
    Get human-readable display name for contract.

    Examples:
        'FRONT_MONTH' -> '📊 Front Month (Continuous)'
        'KCH26' -> 'KCH26 (Mar 2026)'
        'CCK26' -> 'CCK26 (May 2026)'
    """
    _tk_len = len(profile.contract.symbol)  # 2 for KC/CC
    _expected_len = _tk_len + 3  # ticker + month_code + 2-digit year

    if contract == 'FRONT_MONTH':
        _, resolved_symbol = resolve_front_month_ticker()
        if resolved_symbol and resolved_symbol != 'FRONT_MONTH':
            month_names = {
                'F': 'Jan', 'G': 'Feb', 'H': 'Mar', 'J': 'Apr',
                'K': 'May', 'M': 'Jun', 'N': 'Jul', 'Q': 'Aug',
                'U': 'Sep', 'V': 'Oct', 'X': 'Nov', 'Z': 'Dec'
            }
            if len(resolved_symbol) == _expected_len:
                mc = resolved_symbol[_tk_len]
                yr = resolved_symbol[_tk_len + 1:_tk_len + 3]
                mn = month_names.get(mc, '???')
                return f'📊 Front Month ({resolved_symbol} · {mn} 20{yr})'
        return '📊 Front Month (Continuous)'

    month_names = {
        'F': 'Jan', 'G': 'Feb', 'H': 'Mar', 'J': 'Apr',
        'K': 'May', 'M': 'Jun', 'N': 'Jul', 'Q': 'Aug',
        'U': 'Sep', 'V': 'Oct', 'X': 'Nov', 'Z': 'Dec'
    }

    clean_symbol = clean_contract_symbol(contract)
    if clean_symbol and len(clean_symbol) == _expected_len:
        month_code = clean_symbol[_tk_len]
        year = clean_symbol[_tk_len + 1:_tk_len + 3]
        month_name = month_names.get(month_code, '???')
        return f"{clean_symbol} ({month_name} 20{year})"

    # Fallback: return original if can't parse
    return str(contract) if contract else 'Unknown'

def get_contract_color(contract: str, default_color: str) -> str:
    """Get color for contract based on month code."""
    if not contract or contract == 'FRONT_MONTH':
        return default_color

    clean = clean_contract_symbol(contract)
    sym_len = len(profile.contract.symbol)  # 2 for KC/CC/NG, dynamic for future tickers
    if clean and len(clean) > sym_len:
        month_code = clean[sym_len]
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
        min_value=0,
        max_value=max_days,
        value=default_lookback,
        help="0 = Today only. Default: 3 days."
    )

    st.markdown("---")

    # === NEW: CONTRACT SELECTOR ===
    st.header("📜 Contract")

    # Load council data early to get available contracts
    council_df_for_contracts = load_council_history(ticker=ticker)
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
                                        help=f"Show only {profile.contract.trading_hours_et} ET candles")
    show_labels = st.toggle("Show Signal Labels", value=True)
    show_day_separators = st.toggle("Show Day/Week Separators", value=True)
    show_confidence = st.toggle("Show Confidence Scores", value=True)
    show_outcomes = st.toggle("Highlight Win/Loss", value=True)
    show_regime_overlay = st.toggle("Show Regime Overlay", value=True,
                                     help="Shade chart background by market regime")


# === DATA FUNCTIONS ===

@st.cache_data(ttl=300)
def fetch_price_history_extended(ticker="KC=F", period="5d", interval="5m",
                                  commodity_ticker=None, exchange=None,
                                  contract=None, lookback_days=3):
    """
    Fetches historical OHLC data, converted to NY Time.

    Primary path: Databento (when commodity_ticker/exchange provided and API key set)
    Fallback: yfinance

    Args:
        ticker: yfinance ticker (e.g., 'KC=F' or 'KCH26.NYB') — used for yfinance fallback
        period: lookback period (yfinance format, e.g., '1mo')
        interval: candle interval (e.g., '5m', '1h', '1d')
        commodity_ticker: e.g., 'KC', 'NG' — enables Databento path
        exchange: e.g., 'ICE', 'NYMEX' — enables Databento path
        contract: e.g., 'FRONT_MONTH', 'KCH26' — passed to Databento
        lookback_days: number of days to fetch — used by Databento

    Returns:
        DataFrame with OHLC data in NY timezone, or None if fetch failed
    """
    import logging

    # Path 1: Databento (when commodity info provided)
    if commodity_ticker and exchange:
        try:
            from trading_bot.data_providers import get_price_data
            df = get_price_data(
                commodity_ticker, exchange,
                contract or 'FRONT_MONTH',
                interval, lookback_days,
            )
            if df is not None and not df.empty:
                return df
            logging.getLogger(__name__).warning(
                f"Databento returned empty for {commodity_ticker}/{contract}, trying yfinance"
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Databento path failed: {e}, trying yfinance"
            )

    # Path 2: yfinance (legacy fallback)
    try:
        import yfinance as yf

        yf_logger = logging.getLogger('yfinance')
        original_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)

        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        finally:
            yf_logger.setLevel(original_level)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize timezone to NY
        # Note: Yahoo daily candles arrive as tz-naive midnight timestamps.
        # These represent calendar dates, NOT midnight UTC — localizing as UTC
        # would shift them to 7 PM ET previous day, breaking signal alignment.
        # Intraday candles from Yahoo/Databento DO carry UTC semantics.
        if df.index.tz is None:
            if timeframe == '1d':
                df.index = df.index.tz_localize('America/New_York')
            else:
                df.index = df.index.tz_localize('UTC')
                df.index = df.index.tz_convert('America/New_York')
        else:
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
    Filter to core trading hours (from commodity profile) in ET.
    With categorical X-axis, this effectively hides the gaps.
    """
    if df is None or df.empty:
        return df

    # Use profile trading hours, fallback to wide window
    try:
        from config.commodity_profiles import parse_trading_hours
        start_time, end_time = parse_trading_hours(profile.contract.trading_hours_et)
    except Exception:
        start_time = time(3, 30)
        end_time = time(13, 30)

    mask = (df.index.time >= start_time) & (df.index.time <= end_time)
    return df.loc[mask]


def filter_non_trading_days(df):
    """
    Remove weekend days, US market holidays, and rows with missing OHLC data.

    Commodity futures don't trade on weekends or holidays. YFinance may return:
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

    # Timestamp cleaning (handles mixed formats)
    df['timestamp'] = parse_ts_column(df['timestamp'])
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
# start_date for signals is computed later alongside the price cutoff (after anchor normalization)

# Determine yfinance ticker based on contract selection
yf_ticker = contract_to_yfinance_ticker(selected_contract)

# Debug info (helps troubleshooting)
if selected_contract != 'FRONT_MONTH':
    clean = clean_contract_symbol(selected_contract)
    st.caption(f"🔍 Debug: `{selected_contract}` → cleaned: `{clean}` → yfinance: `{yf_ticker}`")

with st.spinner(f"Loading {get_contract_display_name(selected_contract)} data..."):
    # Determine period
    # FIX: Use 1mo minimum to ensure we catch enough trading days (skipping weekends/holidays)
    if lookback_days <= 29: yf_period = "1mo"
    elif lookback_days <= 59: yf_period = "2mo"
    else: yf_period = "2y"

    # Fetch price data for selected contract (Databento primary, yfinance fallback)
    price_df = fetch_price_history_extended(
        ticker=yf_ticker, period=yf_period, interval=timeframe,
        commodity_ticker=profile.contract.symbol, exchange=profile.contract.exchange,
        contract=selected_contract, lookback_days=lookback_days,
    )

    # Fallback: If specific contract has no data, try continuous front month
    if price_df is None or price_df.empty:
        continuous_ticker = f'{profile.contract.symbol}=F'
        if yf_ticker != continuous_ticker:
            # Resolved FRONT_MONTH or specific contract had no data — try continuous contract
            st.warning(f"⚠️ No price data for `{yf_ticker}`. Falling back to continuous contract.")
            price_df = fetch_price_history_extended(
                ticker=continuous_ticker, period=yf_period, interval=timeframe,
                commodity_ticker=profile.contract.symbol, exchange=profile.contract.exchange,
                contract='FRONT_MONTH', lookback_days=lookback_days,
            )
            actual_ticker_display = "Front Month (Fallback)"
        else:
            actual_ticker_display = None
    else:
        actual_ticker_display = get_contract_display_name(selected_contract)

    council_df = load_council_history(ticker=ticker)


# === PLOTTING ===

if price_df is not None and not price_df.empty:

    # 1. Filter Date Range (UPDATED & FIXED)
    current_time_et = datetime.now().astimezone(pytz.timezone('America/New_York'))
    
    # Initialize US Holidays
    us_holidays = holidays.US(years=[current_time_et.year, current_time_et.year - 1], observed=True)
    
    # Calculate cutoff by counting back N trading days.
    #
    # Step 1: Find the "anchor" trading day. On weekdays this is today; on weekends
    # or holidays it's the most recent trading day (e.g. Friday when viewed on Sunday).
    # This ensures lookback=N always yields N+1 trading days regardless of the day
    # of the week (the anchor day + N prior trading days).
    #
    # Step 2: Count back N additional trading days from that anchor.
    anchor_dt = current_time_et
    while anchor_dt.weekday() >= 5 or anchor_dt.date() in us_holidays:
        anchor_dt -= timedelta(days=1)

    cutoff_dt = anchor_dt
    days_counted = 0
    while days_counted < lookback_days:
        cutoff_dt -= timedelta(days=1)
        if cutoff_dt.weekday() < 5 and cutoff_dt.date() not in us_holidays:
            days_counted += 1

    # Force start time to midnight so we capture the full first trading day
    cutoff_dt = cutoff_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # Align signal start_date with price cutoff so signals match the visible price window
    start_date = cutoff_dt

    # Apply the filter (>= Midnight captures the whole day)
    price_df = price_df[price_df.index >= cutoff_dt]

    # 2. Remove non-trading days (weekends/holidays)
    pre_filter_count = len(price_df)
    price_df = filter_non_trading_days(price_df)

    # Check for filtered rows
    if pre_filter_count > 0 and len(price_df) < pre_filter_count:
        removed_count = pre_filter_count - len(price_df)
        removed_pct = (removed_count / pre_filter_count) * 100
        st.caption(f"🗓️ Filtered {removed_count} candles ({removed_pct:.1f}%) - weekends/holidays/invalid data")

    # Early exit if no data after filtering (e.g., lookback=0 on a weekend)
    if price_df.empty:
        st.warning(f"⚠️ No trading data available for the selected {lookback_days}-day lookback period. Try increasing the lookback.")
        st.stop()

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

    # === X-AXIS PREPARATION (Numerical Index for Reliable Candlestick Rendering) ===
    # Use numerical indices (0, 1, 2, ...) for x-axis - this is the most reliable approach for candlesticks
    # Store string labels for tick display
    price_df['str_index'] = price_df.index.strftime('%Y-%m-%d %H:%M')
    price_df['num_index'] = range(len(price_df))

    # Create mapping from timestamp to numerical index for signal alignment
    timestamp_to_num = dict(zip(price_df.index, price_df['num_index']))

    # === ALIGNMENT ENGINE ===
    plot_df = pd.DataFrame()
    if not signals.empty:
        signals = signals.sort_values('timestamp')
        price_idx = pd.DataFrame(index=price_df.index).reset_index()
        price_idx.columns = ['candle_timestamp']

        # Dynamic tolerance: tighter for intraday, wider for daily
        # Yahoo 5m data often starts 30-75 min after ICE market open (4:15 ET),
        # so early signals need ~90 min tolerance. Daily candles sit at midnight ET.
        _tolerance_map = {'5m': 'minutes=90', '15m': 'minutes=90', '30m': 'hours=2', '1h': 'hours=2', '1d': 'hours=20'}
        _tol_str = _tolerance_map.get(timeframe, 'hours=4')
        _tol = pd.Timedelta(**{_tol_str.split('=')[0]: int(_tol_str.split('=')[1])})

        merged = pd.merge_asof(
            signals,
            price_idx,
            left_on='timestamp',
            right_on='candle_timestamp',
            direction='nearest',
            tolerance=_tol
        )

        matched_signals = merged.dropna(subset=['candle_timestamp'])
        unmatched_count = len(signals) - len(matched_signals)

        if unmatched_count > 0:
            # Show which dates have signals but no price data
            unmatched = merged[merged['candle_timestamp'].isna()]
            unmatched_dates = unmatched['timestamp'].dt.date.unique()
            price_dates = set(price_df.index.date)
            missing_dates = sorted(d for d in unmatched_dates if d not in price_dates)
            if missing_dates:
                date_str = ', '.join(str(d) for d in missing_dates[:5])
                st.caption(
                    f"ℹ️ {unmatched_count} signal(s) not shown — Yahoo Finance has no {timeframe} data for {date_str}. "
                    f"Try the 1d timeframe to see them."
                )
            else:
                st.caption(f"ℹ️ {unmatched_count} signal(s) outside alignment tolerance ({_tol}) — not shown on chart.")

        if not matched_signals.empty:
            aligned_prices = price_df.loc[matched_signals['candle_timestamp']]

            plot_df = matched_signals.copy()
            plot_df['plot_x'] = matched_signals['candle_timestamp']
            # Create string version for display and numerical version for plotting
            plot_df['plot_x_str'] = plot_df['plot_x'].dt.strftime('%Y-%m-%d %H:%M')
            # Map timestamps to numerical indices for reliable plotting
            plot_df['plot_x_num'] = plot_df['plot_x'].map(timestamp_to_num)

            plot_df['candle_high'] = aligned_prices['High'].values
            plot_df['candle_low'] = aligned_prices['Low'].values

            plot_df['y_pos'] = np.where(
                plot_df['marker_symbol'] == 'triangle-up',
                plot_df['candle_high'] * 1.003,
                plot_df['candle_low'] * 0.997
            )

            # Clip signal positions to visible price range so outliers can't stretch the Y-axis
            price_floor = price_df['Low'].min() * 0.995
            price_ceil = price_df['High'].max() * 1.005
            plot_df['y_pos'] = plot_df['y_pos'].clip(lower=price_floor, upper=price_ceil)

            plot_df['text_pos'] = np.where(
                plot_df['marker_symbol'] == 'triangle-up',
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
            st.metric("Period Change", f"{pct_change:+.2f}%", help="Percentage change in price over the selected period.")
        with summary_cols[1]:
            st.metric("High", f"${high:{_price_fmt}}", help="Highest price observed in the selected period.")
        with summary_cols[2]:
            st.metric("Low", f"${low:{_price_fmt}}", help="Lowest price observed in the selected period.")
        with summary_cols[3]:
            st.metric("Range", f"${high - low:{_price_fmt}}", help="Difference between High and Low prices in the selected period.")

    st.caption(f"Data source: {get_data_source_label()}")

    # === DRAW CHARTS ===
    # CRITICAL: Plotly go.Candlestick fails to render in make_subplots when
    # shared_xaxes=True or secondary_y specs are present (rendering bug).
    # Must use shared_xaxes=False with NO specs to get candlestick working.
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick (Row 1)
    # CRITICAL: Use numerical x-axis for reliable candlestick rendering
    chart_commodity_label = f"{actual_ticker_display or 'Futures'} (ET)"
    fig.add_trace(go.Candlestick(
        x=price_df['num_index'],
        open=price_df['Open'],
        high=price_df['High'],
        low=price_df['Low'],
        close=price_df['Close'],
        name=chart_commodity_label,
        increasing=dict(line=dict(color='#00CC96', width=1), fillcolor='#00CC96'),
        decreasing=dict(line=dict(color='#EF553B', width=1), fillcolor='#EF553B'),
    ), row=1, col=1)

    # 2. Signal markers (Row 1)
    if not plot_df.empty:
        fig.add_trace(go.Scatter(
            x=plot_df['plot_x_num'],
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

    # 4. Volume (Row 2)
    if 'Volume' in price_df.columns:
        fig.add_trace(go.Bar(
            x=price_df['num_index'],
            y=price_df['Volume'],
            marker_color='rgba(100, 150, 255, 0.4)',
            name='Volume',
            showlegend=False,
        ), row=2, col=1)

    # 5. Confidence (Row 2 — overlaid on volume axis, scaled to visible range)
    if not plot_df.empty and show_confidence:
        # Scale confidence (0-1) to volume range so both are visible on same axis
        max_vol = price_df['Volume'].max() if 'Volume' in price_df.columns and not price_df['Volume'].empty else 1
        max_vol = max_vol if pd.notna(max_vol) and max_vol > 0 else 1.0
        scaled_confidence = plot_df['plot_confidence'] * max_vol
        fig.add_trace(go.Scatter(
            x=plot_df['plot_x_num'],
            y=scaled_confidence,
            mode='markers+lines',
            line=dict(color='#00CC96', width=1.5, dash='dot'),
            marker=dict(color='#00CC96', size=8, symbol='circle'),
            name='Confidence',
            showlegend=False,
        ), row=2, col=1)

    # === DAY & WEEK SEPARATORS (FIXED VISIBILITY) ===
    if show_day_separators and len(price_df) > 1:
        dates = price_df.index.date
        weekdays = price_df.index.dayofweek

        day_changes = np.where(dates[1:] != dates[:-1])[0] + 1

        for idx in day_changes:
            ts_num = price_df['num_index'].iloc[idx]  # Use numerical index
            current_weekday = weekdays[idx]
            prev_weekday = weekdays[idx - 1] if idx > 0 else current_weekday

            is_week_start = (current_weekday == 0) or (current_weekday < prev_weekday)

            if is_week_start:
                # WEEK SEPARATOR - Prominent orange
                fig.add_vline(
                    x=ts_num,
                    line_width=2,
                    line_dash="dash",
                    line_color="rgba(255, 180, 80, 0.8)",
                    row="all"
                )
                fig.add_annotation(
                    x=ts_num,
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
                    x=ts_num,
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
                first_candle_num = date_data['num_index'].iloc[0]  # Use numerical index
                # Market Open marker
                if first_candle.time() <= time(4, 0):  # Within first 30 min
                    fig.add_annotation(
                        x=first_candle_num,
                        y=date_data.loc[first_candle, 'High'] * 1.003,
                        text="🔔",
                        showarrow=False,
                        font=dict(size=10),
                        row=1, col=1
                    )

    # === REGIME OVERLAY (Background Shading) ===
    if show_regime_overlay:
        try:
            from dashboard_utils import _resolve_data_path as _rdp
            _ds_path = _rdp('decision_signals.csv')
            if os.path.exists(_ds_path):
                ds_df = pd.read_csv(_ds_path)
                if not ds_df.empty and 'regime' in ds_df.columns and 'timestamp' in ds_df.columns:
                    ds_df['timestamp'] = parse_ts_column(ds_df['timestamp'])
                    ds_df = ds_df.dropna(subset=['timestamp', 'regime'])
                    if ds_df['timestamp'].dt.tz is None:
                        ds_df['timestamp'] = ds_df['timestamp'].dt.tz_localize('UTC')
                    ds_df['timestamp'] = ds_df['timestamp'].dt.tz_convert('America/New_York')
                    ds_df = ds_df.sort_values('timestamp')

                    regime_colors = {
                        'bullish': 'rgba(0, 204, 150, 0.06)',
                        'bearish': 'rgba(239, 85, 59, 0.06)',
                        'neutral': 'rgba(136, 136, 136, 0.04)',
                        'range_bound': 'rgba(255, 161, 90, 0.06)',
                        'high_volatility': 'rgba(171, 99, 250, 0.06)',
                    }

                    # Snap each signal to its nearest candle via merge_asof
                    _price_ts = pd.DataFrame({
                        'candle_ts': price_df.index,
                        'num_idx': price_df['num_index'].values
                    }).sort_values('candle_ts')
                    ds_df = ds_df.sort_values('timestamp')
                    ds_df = pd.merge_asof(
                        ds_df, _price_ts,
                        left_on='timestamp', right_on='candle_ts',
                        direction='nearest'
                    )
                    ds_df = ds_df.dropna(subset=['num_idx'])

                    if not ds_df.empty:
                        # Build regime spans: consecutive signals with same regime
                        ds_df = ds_df.sort_values('num_idx')
                        prev_regime = None
                        span_start = None
                        for _, sig_row in ds_df.iterrows():
                            regime = str(sig_row['regime']).lower().strip()
                            idx = sig_row['num_idx']
                            if regime != prev_regime:
                                # Close previous span
                                if prev_regime is not None and span_start is not None:
                                    color = regime_colors.get(prev_regime, 'rgba(136, 136, 136, 0.04)')
                                    fig.add_vrect(
                                        x0=span_start, x1=idx,
                                        fillcolor=color, layer="below", line_width=0,
                                        row=1, col=1
                                    )
                                span_start = idx
                                prev_regime = regime
                        # Close final span
                        if prev_regime is not None and span_start is not None:
                            final_x = price_df['num_index'].max()
                            color = regime_colors.get(prev_regime, 'rgba(136, 136, 136, 0.04)')
                            fig.add_vrect(
                                x0=span_start, x1=final_x,
                                fillcolor=color, layer="below", line_width=0,
                                row=1, col=1
                            )
        except Exception:
            pass  # Silently skip if regime data unavailable

        if show_regime_overlay:
            st.caption(
                "Regime overlay: "
                "\U0001f7e2 Green = bullish | "
                "\U0001f534 Red = bearish | "
                "\u26aa Gray = neutral | "
                "\U0001f7e0 Orange = range-bound | "
                "\U0001f7e3 Purple = high volatility"
            )

    # Determine title based on contract
    if actual_ticker_display:
        chart_title = f"{actual_ticker_display} | {selected_agent_label} | {timeframe} | Last {lookback_days} Days"
    else:
        chart_title = f"Market Analysis (ET) | {selected_agent_label} | {timeframe} | Last {lookback_days} Days"

    # Add signal contract info if showing multiple
    if show_all_signals and not signals.empty:
        unique_contracts = signals['signal_contract'].unique()
        if len(unique_contracts) > 1:
            chart_title += f" | Signals: {', '.join(unique_contracts)}"

    # === LAYOUT (Numerical X-Axis with Custom Tick Labels) ===
    # Use numerical indices for reliable candlestick rendering, with custom tick labels for display
    # Select ~15 evenly spaced tick positions
    n_ticks = min(15, len(price_df))
    if n_ticks > 0:
        tick_indices = np.linspace(0, len(price_df) - 1, n_ticks, dtype=int)
        tick_vals = price_df['num_index'].iloc[tick_indices].tolist()
        tick_text = price_df['str_index'].iloc[tick_indices].tolist()
    else:
        tick_vals = []
        tick_text = []

    # Set explicit x-axis range to prevent autorange issues with small datasets
    x_range = [-0.5, max(len(price_df) - 0.5, 0.5)]

    # Row 1: hide x-axis labels (row 2 shows them) since shared_xaxes is off
    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=tick_text,
        range=x_range,
        showticklabels=False,
        row=1, col=1
    )

    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=45,
        range=x_range,
        row=2, col=1
    )

    # Row 1 Y-axis - Set explicit range from price data only.
    # autorange=False is critical: without it, Plotly can stretch the axis to
    # accommodate outlier signal markers that fall outside the price range.
    if not price_df.empty:
        y_min = price_df['Low'].min()
        y_max = price_df['High'].max()
        y_range = y_max - y_min
        # Add 5% buffer on each side, with tick-based minimum to prevent zero-range issues
        # (fixed 0.5 was fine for KC/CC but consumed 40% of NG's $3 price range)
        y_buffer = max(y_range * 0.05, _tick * 10)
        fig.update_yaxes(
            title_text=f"Price ({profile.name})",
            range=[y_min - y_buffer, y_max + y_buffer],
            autorange=False,
            row=1, col=1
        )
    else:
        fig.update_yaxes(title_text=f"Price ({profile.name})", row=1, col=1)

    # Row 2 Y-axis (Volume)
    fig.update_yaxes(
        title_text="Volume",
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        tickformat=',d',
        row=2, col=1,
    )

    # CRITICAL: Fully disable candlestick rangeslider to prevent it from eating row 1's space.
    # go.Candlestick auto-creates a rangeslider that reserves vertical space even when hidden.
    # With fewer data points the effect is more pronounced, squishing the price chart to near-zero.
    fig.update_layout(
        height=800,
        xaxis=dict(
            rangeslider=dict(visible=False, thickness=0),
        ),
        xaxis2=dict(
            rangeslider=dict(visible=False, thickness=0),
        ),
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
            st.metric("Total Signals", total_signals, help="Total number of agent signals generated.")
        with stat_cols[1]:
            st.metric("🟢 Bullish", int(bullish), help="Number of bullish signals.")
        with stat_cols[2]:
            st.metric("🔴 Bearish", int(bearish), help="Number of bearish signals.")
        with stat_cols[3]:
            st.metric("🟣 Volatility", int(vol_trades), help="Number of volatility expansion signals.")
        with stat_cols[4]:
            st.metric("⚪ Neutral", int(neutral), help="Number of neutral or conflicting signals.")

        if 'outcome' in plot_df.columns:
            wins = (plot_df['outcome'] == 'WIN').sum()
            losses = (plot_df['outcome'] == 'LOSS').sum()
            graded = wins + losses
            if graded > 0:
                win_rate = wins / graded * 100
                st.caption(f"📈 Win Rate: **{win_rate:.1f}%** ({wins}W / {losses}L from {graded} graded signals)")

    # === STRATEGY PERFORMANCE TABLE ===
    if not plot_df.empty and 'outcome' in plot_df.columns and 'strategy_type' in plot_df.columns:
        graded_signals = plot_df[plot_df['outcome'].isin(['WIN', 'LOSS'])].copy()
        if not graded_signals.empty:
            st.markdown("---")
            st.subheader("🎯 Strategy Performance")

            strat_stats = graded_signals.groupby('strategy_type').agg(
                Signals=('outcome', 'count'),
                Wins=('outcome', lambda x: (x == 'WIN').sum()),
                Losses=('outcome', lambda x: (x == 'LOSS').sum()),
            ).reset_index()
            strat_stats.rename(columns={'strategy_type': 'Strategy'}, inplace=True)
            strat_stats['Win Rate%'] = (strat_stats['Wins'] / strat_stats['Signals'] * 100).round(1)
            strat_stats = strat_stats.sort_values('Win Rate%', ascending=False).reset_index(drop=True)

            st.dataframe(
                strat_stats,
                column_config={
                    "Win Rate%": st.column_config.ProgressColumn("Win Rate%", min_value=0, max_value=100, format="%.1f%%"),
                },
                hide_index=True,
                width="stretch"
            )

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
            file_name=f"{profile.contract.symbol.lower()}_data_{lookback_days}d_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
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
