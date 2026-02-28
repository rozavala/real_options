"""
Price data providers for the trading system.

Architecture: yfinance-first, Databento gap-fill, disk cache.

1. yfinance (free) fetches the full range
2. Gap detection compares against exchange-specific trading calendar
3. Missing dates are filled from persistent disk cache (parquet)
4. Only truly uncached missing dates hit Databento (licensed, ~$500/GB ICE)
5. Databento results are cached to disk — each missing day is paid for once, ever

Module-level TTL cache prevents redundant API calls within a session.
"""

import os
import time as _time
import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Tuple

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

NY_TZ = pytz.timezone('America/New_York')

# =============================================================================
# MODULE-LEVEL TTL CACHE
# =============================================================================

_cache: Dict[Tuple, Tuple[float, pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _cache_key(
    ticker: str, exchange: str, contract: str, timeframe: str, lookback_days: int,
) -> Tuple:
    return (ticker.upper(), exchange.upper(), contract.upper(), timeframe, lookback_days)


def _cache_get(key: Tuple) -> Optional[pd.DataFrame]:
    if key in _cache:
        ts, df = _cache[key]
        if _time.time() - ts < _CACHE_TTL_SECONDS:
            return df.copy()
        del _cache[key]
    return None


def _cache_set(key: Tuple, df: pd.DataFrame):
    _cache[key] = (_time.time(), df.copy())


# =============================================================================
# ABSTRACT BASE
# =============================================================================

class PriceDataProvider(ABC):
    """Abstract base for OHLCV data providers."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        exchange: str,
        contract: str,
        timeframe: str,
        lookback_days: int,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data.

        Returns DataFrame with:
        - DatetimeIndex in America/New_York timezone
        - Columns: Open, High, Low, Close, Volume (capitalized)
        - Sorted by timestamp ascending

        Returns None on failure.
        """
        ...


# =============================================================================
# DATABENTO PROVIDER
# =============================================================================

class DatabentoPriceProvider(PriceDataProvider):
    """
    Fetches OHLCV data from Databento (licensed exchange data).

    Handles ICE vs CME differences:
    - ICE: parent symbology (KC.FUT) + filter outrights + pick highest-volume contract
    - CME/NYMEX: continuous symbology (NG.c.0) -- straightforward
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy init -- only imports/connects when first needed."""
        if self._client is None:
            import databento as db
            key = os.environ.get('DATABENTO_API_KEY', '')
            if not key:
                raise ValueError("DATABENTO_API_KEY not set")
            self._client = db.Historical(key=key)
        return self._client

    def fetch_ohlcv(
        self,
        ticker: str,
        exchange: str,
        contract: str,
        timeframe: str,
        lookback_days: int,
    ) -> Optional[pd.DataFrame]:
        from config.databento_mappings import (
            get_dataset,
            get_schema_and_resample,
            parse_our_contract_to_databento,
            is_outright_symbol,
            estimate_cost,
        )

        dataset = get_dataset(exchange)
        if not dataset:
            logger.warning(f"Databento: no dataset mapping for exchange '{exchange}'")
            return None

        schema, resample_freq = get_schema_and_resample(timeframe)
        symbol, stype_in = parse_our_contract_to_databento(ticker, exchange, contract)

        end = datetime.now(pytz.UTC)
        # +3 buffer: weekends (2) + dataset availability lag (1).
        # Caller trims to exact lookback; extra days add ~$0.04 cost.
        start = end - timedelta(days=lookback_days + 3)

        is_parent = stype_in == 'parent'
        est_cost = estimate_cost(schema, lookback_days, is_parent=is_parent)
        logger.info(
            f"Databento: fetching {ticker} ({symbol}) from {dataset}, "
            f"schema={schema}, est. ${est_cost:.4f}"
        )

        try:
            client = self._get_client()

            # Cap end to dataset's available range to avoid 422 errors
            # (historical data has a delay — today's data may not be available yet)
            try:
                ds_range = client.metadata.get_dataset_range(dataset=dataset)
                # Response is a dict with 'end' key as ISO string
                end_str = None
                if isinstance(ds_range, dict):
                    end_str = ds_range.get('end')
                elif hasattr(ds_range, 'end'):
                    end_str = ds_range.end
                if end_str and isinstance(end_str, str):
                    # Parse "2026-02-24T23:05:00.000000000Z" — strip nanoseconds
                    clean = end_str.replace('Z', '+00:00')
                    # Remove nanosecond precision (Python can't parse >6 fractional digits)
                    import re as _re
                    clean = _re.sub(r'(\.\d{6})\d+', r'\1', clean)
                    available_end = datetime.fromisoformat(clean)
                    if end > available_end:
                        logger.debug(
                            f"Databento: capping end from {end.isoformat()} "
                            f"to {available_end.isoformat()}"
                        )
                        end = available_end
            except Exception as range_err:
                logger.debug(f"Databento: could not query dataset range: {range_err}")

            data = client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                stype_in=stype_in,
                schema=schema,
                start=start.strftime('%Y-%m-%dT%H:%M:%S'),
                end=end.strftime('%Y-%m-%dT%H:%M:%S'),
            )
            df = data.to_df()
        except Exception as e:
            err_type = type(e).__name__
            logger.warning(f"Databento API error ({err_type}): {e}")
            return None

        if df is None or df.empty:
            logger.warning(f"Databento: empty response for {ticker} ({symbol})")
            return None

        # --- ICE parent symbology: filter outrights, pick front month ---
        if is_parent and 'symbol' in df.columns:
            # Filter to outrights only (exclude calendar spreads)
            # Vectorized alternative to apply(is_outright_symbol) for performance
            df = df[df['symbol'].str.strip().str.endswith('!')]
            if df.empty:
                logger.warning(f"Databento: no outright contracts found for {ticker}")
                return None

            # Pick the contract with highest total volume (front month proxy)
            if contract == 'FRONT_MONTH' or not contract:
                vol_by_sym = df.groupby('symbol')['volume'].sum()
                front_symbol = vol_by_sym.idxmax()
                df = df[df['symbol'] == front_symbol]
                logger.info(
                    f"Databento: selected front month '{front_symbol}' (highest volume)"
                )

        # --- Normalize columns ---
        df = self._normalize(df, resample_freq)

        logger.info(f"Databento: {len(df)} bars fetched for {ticker} (est. ${est_cost:.4f})")
        return df

    def _normalize(
        self, df: pd.DataFrame, resample_freq: Optional[str],
    ) -> pd.DataFrame:
        """Normalize Databento output to standard OHLCV format in NY timezone."""
        # Databento uses 'ts_event' as timestamp; may be a column or the index.
        if 'ts_event' in df.columns:
            df = df.set_index('ts_event')

        # Ensure timezone-aware UTC, then convert to NY
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(NY_TZ)

        # Rename columns to standard OHLCV (Databento uses lowercase)
        col_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
        }
        df = df.rename(columns=col_map)

        # Keep only OHLCV columns
        ohlcv_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        df = df[ohlcv_cols]

        # Filter zero-price bars (ICE emits non-trading period bars with OHLCV = 0)
        if 'Open' in df.columns and 'Close' in df.columns:
            before = len(df)
            df = df[(df['Open'] > 0) & (df['Close'] > 0)]
            dropped = before - len(df)
            if dropped > 0:
                logger.debug(f"Databento: filtered {dropped} zero-price bars")

        # Resample if needed (5m/15m/30m from 1m bars)
        if resample_freq:
            df = df.resample(resample_freq).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
            }).dropna(subset=['Open'])

        # Sort and deduplicate
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        return df


# =============================================================================
# YFINANCE FALLBACK PROVIDER
# =============================================================================

class YFinancePriceProvider(PriceDataProvider):
    """
    Fallback provider using yfinance (unofficial Yahoo Finance scraper).

    Known limitations:
    - ICE commodity futures often have gaps (missing trading days)
    - Intraday data limited to ~60 days
    - Unofficial API, subject to breaking changes
    """

    # Exchange suffix map for yfinance tickers
    _EXCHANGE_SUFFIX = {
        'ICE': 'NYB',
        'NYBOT': 'NYB',
        'NYMEX': 'NYM',
        'CME': 'CME',
        'COMEX': 'CMX',
    }

    def fetch_ohlcv(
        self,
        ticker: str,
        exchange: str,
        contract: str,
        timeframe: str,
        lookback_days: int,
    ) -> Optional[pd.DataFrame]:
        import yfinance as yf

        yf_ticker = self._resolve_yf_ticker(ticker, exchange, contract)
        period = self._lookback_to_period(lookback_days, timeframe)

        logger.info(f"yfinance: fetching {yf_ticker} period={period} interval={timeframe}")

        try:
            yf_logger = logging.getLogger('yfinance')
            original_level = yf_logger.level
            yf_logger.setLevel(logging.CRITICAL)
            try:
                df = yf.download(
                    yf_ticker, period=period, interval=timeframe,
                    progress=False, auto_adjust=True,
                )
            finally:
                yf_logger.setLevel(original_level)

            if df is None or df.empty:
                logger.warning(f"yfinance: no data for {yf_ticker}")
                return None

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Normalize timezone to NY
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            df.index = df.index.tz_convert(NY_TZ)

            # Deduplicate and sort
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()

            logger.info(f"yfinance: {len(df)} bars fetched for {yf_ticker}")
            return df

        except Exception as e:
            logger.warning(f"yfinance error: {e}")
            return None

    def _resolve_yf_ticker(self, ticker: str, exchange: str, contract: str) -> str:
        """
        Build yfinance ticker from commodity info.

        FRONT_MONTH -> continuous contract (e.g., KC=F)
        Specific    -> contract + exchange suffix (e.g., KCH26.NYB)
        """
        if contract == 'FRONT_MONTH' or not contract:
            return f'{ticker}=F'

        suffix = self._EXCHANGE_SUFFIX.get(exchange.upper(), 'NYB')
        return f'{contract}.{suffix}'

    @staticmethod
    def _lookback_to_period(lookback_days: int, timeframe: str) -> str:
        """Convert lookback days to yfinance period string."""
        if timeframe in ('5m', '15m', '30m', '1h'):
            if lookback_days <= 29:
                return '1mo'
            elif lookback_days <= 59:
                return '2mo'
            return '2y'
        # Daily
        if lookback_days <= 29:
            return '1mo'
        elif lookback_days <= 89:
            return '3mo'
        elif lookback_days <= 364:
            return '1y'
        return '2y'


# =============================================================================
# DISK CACHE (per-day parquet files — pay Databento once per missing day)
# =============================================================================

_DISK_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'databento_cache',
)


def _disk_cache_path(ticker: str, timeframe: str, dt: date) -> str:
    return os.path.join(
        _DISK_CACHE_DIR, f"{ticker.upper()}_{timeframe}_{dt.isoformat()}.parquet",
    )


def _load_from_disk_cache(ticker: str, timeframe: str, dt: date) -> Optional[pd.DataFrame]:
    path = _disk_cache_path(ticker, timeframe, dt)
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                return df
        except Exception as e:
            logger.debug(f"Disk cache read error for {path}: {e}")
    return None


def _save_to_disk_cache(ticker: str, timeframe: str, dt: date, df: pd.DataFrame):
    if df.empty:
        return
    os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
    path = _disk_cache_path(ticker, timeframe, dt)
    try:
        df.to_parquet(path)
    except Exception as e:
        logger.debug(f"Disk cache write error for {path}: {e}")


# =============================================================================
# GAP DETECTION
# =============================================================================

def _find_missing_trading_days(
    df: Optional[pd.DataFrame], exchange: str, lookback_days: int,
) -> List[date]:
    """Compare actual dates in df against expected trading days."""
    from trading_bot.calendars import is_trading_day

    today = datetime.now(NY_TZ).date()
    start_date = today - timedelta(days=lookback_days + 3)

    expected: List[date] = []
    d = start_date
    while d < today:  # Only completed days (exclude today)
        if is_trading_day(d, exchange):
            expected.append(d)
        d += timedelta(days=1)

    if df is not None and not df.empty:
        actual_dates = set(df.index.date)
    else:
        actual_dates = set()

    missing = [d for d in expected if d not in actual_dates]
    return sorted(missing)


# =============================================================================
# PUBLIC API
# =============================================================================

_databento_provider: Optional[DatabentoPriceProvider] = None
_yfinance_provider: Optional[YFinancePriceProvider] = None
_last_source: str = "none"
_last_gap_fill_count: int = 0


def get_price_data(
    ticker: str,
    exchange: str,
    contract: str = 'FRONT_MONTH',
    timeframe: str = '5m',
    lookback_days: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV price data: yfinance primary, Databento gap-fill.

    1. In-memory TTL cache (5 min)
    2. yfinance (free) for full range
    3. Gap detection against exchange trading calendar
    4. Disk cache for previously-fetched Databento days
    5. Databento API only for truly uncached missing dates
    6. Merge, deduplicate, sort

    Returns DataFrame with DatetimeIndex (NY tz) and OHLCV columns, or None.
    """
    global _databento_provider, _yfinance_provider, _last_source, _last_gap_fill_count

    # 1. In-memory TTL cache
    key = _cache_key(ticker, exchange, contract, timeframe, lookback_days)
    cached = _cache_get(key)
    if cached is not None:
        logger.debug(f"Cache hit for {ticker}/{contract}/{timeframe}")
        return cached

    # 2. yfinance first (free)
    if _yfinance_provider is None:
        _yfinance_provider = YFinancePriceProvider()

    yf_df = None
    try:
        yf_df = _yfinance_provider.fetch_ohlcv(
            ticker, exchange, contract, timeframe, lookback_days,
        )
    except Exception as e:
        logger.warning(f"yfinance failed for {ticker}/{contract}: {e}")

    # 3. Gap detection
    missing_days = _find_missing_trading_days(yf_df, exchange, lookback_days)

    if not missing_days:
        # No gaps — return yfinance data directly (zero cost)
        if yf_df is not None and not yf_df.empty:
            _cache_set(key, yf_df)
            _last_source = "yfinance"
            _last_gap_fill_count = 0
            return yf_df

    # 4 + 5. Fill gaps from disk cache, then Databento
    gap_frames: List[pd.DataFrame] = []
    uncached_missing: List[date] = []

    for d in missing_days:
        disk_df = _load_from_disk_cache(ticker, timeframe, d)
        if disk_df is not None:
            gap_frames.append(disk_df)
        else:
            uncached_missing.append(d)

    # 5. Databento API — only for truly uncached missing dates
    if uncached_missing and os.environ.get('DATABENTO_API_KEY'):
        if _databento_provider is None:
            _databento_provider = DatabentoPriceProvider()

        api_start = min(uncached_missing)
        api_end = max(uncached_missing) + timedelta(days=1)
        api_lookback = (api_end - api_start).days

        logger.info(
            f"Databento gap-fill: {len(uncached_missing)} uncached days "
            f"({api_start} to {max(uncached_missing)}) for {ticker}"
        )

        try:
            db_df = _databento_provider.fetch_ohlcv(
                ticker, exchange, contract, timeframe, api_lookback,
            )
            if db_df is not None and not db_df.empty:
                # Cache each day separately to disk
                for d in uncached_missing:
                    day_mask = db_df.index.date == d
                    day_df = db_df[day_mask]
                    if not day_df.empty:
                        _save_to_disk_cache(ticker, timeframe, d, day_df)
                gap_frames.append(db_df)
        except Exception as e:
            logger.warning(f"Databento gap-fill failed for {ticker}: {e}")

    # 6. Merge all frames
    all_frames = [f for f in [yf_df] + gap_frames if f is not None and not f.empty]

    if not all_frames:
        # yfinance failed and no gap-fill available — try Databento as sole source
        if os.environ.get('DATABENTO_API_KEY'):
            if _databento_provider is None:
                _databento_provider = DatabentoPriceProvider()
            try:
                df = _databento_provider.fetch_ohlcv(
                    ticker, exchange, contract, timeframe, lookback_days,
                )
                if df is not None and not df.empty:
                    _cache_set(key, df)
                    _last_source = "databento"
                    _last_gap_fill_count = 0
                    return df
            except Exception as e:
                logger.warning(f"Databento sole-source also failed: {e}")

        _last_source = "none"
        _last_gap_fill_count = 0
        return None

    if len(all_frames) == 1:
        merged = all_frames[0]
    else:
        merged = pd.concat(all_frames)
        merged = merged[~merged.index.duplicated(keep='last')]
        merged = merged.sort_index()

    _cache_set(key, merged)
    filled = len(missing_days) - len(uncached_missing) + (
        len(uncached_missing) if gap_frames else 0
    )
    _last_gap_fill_count = filled if gap_frames else 0

    if gap_frames:
        _last_source = "yfinance+databento"
    elif yf_df is not None and not yf_df.empty:
        _last_source = "yfinance"
    else:
        _last_source = "databento"

    return merged


def get_data_source_label() -> str:
    """
    Return a label indicating which data source was last used.

    For dashboard display.
    """
    if _last_source == "yfinance":
        return "yfinance (free)"
    elif _last_source == "yfinance+databento":
        n = _last_gap_fill_count
        return f"yfinance + Databento gap-fill ({n} day{'s' if n != 1 else ''})"
    elif _last_source == "databento":
        return "Databento (licensed)"
    return "No data source available"
