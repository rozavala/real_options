"""
Price data providers for the trading system.

Primary: Databento (licensed ICE/CME exchange data)
Fallback: yfinance (unofficial Yahoo Finance scraper)

Module-level TTL cache prevents redundant API calls within a session.
"""

import os
import time as _time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

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
            df = df[df['symbol'].apply(is_outright_symbol)]
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
# PUBLIC API
# =============================================================================

_databento_provider: Optional[DatabentoPriceProvider] = None
_yfinance_provider: Optional[YFinancePriceProvider] = None
_last_source: str = "none"


def get_price_data(
    ticker: str,
    exchange: str,
    contract: str = 'FRONT_MONTH',
    timeframe: str = '5m',
    lookback_days: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV price data with caching and automatic fallback.

    Primary: Databento (if DATABENTO_API_KEY is set)
    Fallback: yfinance

    Returns DataFrame with DatetimeIndex (NY tz) and OHLCV columns, or None.
    """
    global _databento_provider, _yfinance_provider, _last_source

    # Check cache first
    key = _cache_key(ticker, exchange, contract, timeframe, lookback_days)
    cached = _cache_get(key)
    if cached is not None:
        logger.debug(f"Cache hit for {ticker}/{contract}/{timeframe}")
        return cached

    # Try Databento first
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
                return df
            logger.warning(
                f"Databento returned empty for {ticker}/{contract}, "
                f"falling back to yfinance"
            )
        except Exception as e:
            logger.warning(
                f"Databento failed for {ticker}/{contract}: {e}, "
                f"falling back to yfinance"
            )

    # Fallback to yfinance
    if _yfinance_provider is None:
        _yfinance_provider = YFinancePriceProvider()
    try:
        df = _yfinance_provider.fetch_ohlcv(
            ticker, exchange, contract, timeframe, lookback_days,
        )
        if df is not None and not df.empty:
            _cache_set(key, df)
            _last_source = "yfinance"
            return df
    except Exception as e:
        logger.warning(f"yfinance also failed for {ticker}/{contract}: {e}")

    _last_source = "none"
    return None


def get_data_source_label() -> str:
    """
    Return a label indicating which data source was last used.

    For dashboard display (e.g., "Databento" or "yfinance (fallback)").
    """
    labels = {
        "databento": "Databento (licensed exchange data)",
        "yfinance": "yfinance (fallback)",
        "none": "No data source available",
    }
    return labels.get(_last_source, _last_source)
