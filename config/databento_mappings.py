"""
Databento API symbol and dataset mappings for commodity futures.

Maps the trading system's commodity profiles to Databento's API parameters.
Handles the differences between ICE and CME/NYMEX symbology.

NOTE: ICE continuous symbology (e.g., KC.c.0) is NOT available for historical
API queries as of Feb 2026. Use parent symbology + outright filtering instead.
"""

import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exchange -> Databento dataset
# ---------------------------------------------------------------------------
EXCHANGE_TO_DATASET = {
    'ICE': 'IFUS.IMPACT',
    'NYBOT': 'IFUS.IMPACT',
    'NYMEX': 'GLBX.MDP3',
    'CME': 'GLBX.MDP3',
    'COMEX': 'GLBX.MDP3',
}

# ---------------------------------------------------------------------------
# Timeframe -> (Databento schema, optional resample frequency)
# Sub-hourly timeframes fetch 1-minute bars and resample client-side.
# ---------------------------------------------------------------------------
TIMEFRAME_TO_SCHEMA = {
    '5m': ('ohlcv-1m', '5min'),
    '15m': ('ohlcv-1m', '15min'),
    '30m': ('ohlcv-1m', '30min'),
    '1h': ('ohlcv-1h', None),
    '1d': ('ohlcv-1d', None),
}

# ---------------------------------------------------------------------------
# Month helpers
# ---------------------------------------------------------------------------
MONTH_CODE_TO_LETTER = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z',
}
LETTER_TO_MONTH_NUM = {v: k for k, v in MONTH_CODE_TO_LETTER.items()}

# Regex for our contract symbols (e.g., KCH26, NGK26, CCZ25)
_CONTRACT_RE = re.compile(r'^([A-Z]{2,4})([FGHJKMNQUVXZ])(\d{2})$')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset(exchange: str) -> Optional[str]:
    """Map exchange name to Databento dataset."""
    return EXCHANGE_TO_DATASET.get(exchange.upper())


def get_schema_and_resample(timeframe: str) -> Tuple[str, Optional[str]]:
    """
    Get Databento schema and optional resample frequency for a timeframe.

    Returns:
        (schema, resample_freq) -- resample_freq is None if no resampling needed.
    """
    if timeframe not in TIMEFRAME_TO_SCHEMA:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Valid: {list(TIMEFRAME_TO_SCHEMA.keys())}"
        )
    return TIMEFRAME_TO_SCHEMA[timeframe]


def _is_ice_exchange(exchange: str) -> bool:
    """Check if exchange uses ICE symbology."""
    return exchange.upper() in ('ICE', 'NYBOT')


def build_front_month_symbol(ticker: str, exchange: str) -> Tuple[str, str]:
    """
    Build Databento front-month symbol and stype_in for a commodity.

    ICE commodities use parent symbology (e.g., KC.FUT) because ICE continuous
    symbology (KC.c.0) is NOT available for historical API as of Feb 2026.

    CME/NYMEX commodities use continuous symbology (e.g., NG.c.0).

    Returns:
        (symbol, stype_in) -- e.g., ('KC.FUT', 'parent') or ('NG.c.0', 'continuous')
    """
    if _is_ice_exchange(exchange):
        # ICE: parent symbology -- returns all contracts including spreads.
        # Caller must filter to outrights (symbols ending with '!').
        return (f'{ticker}.FUT', 'parent')
    else:
        # CME/NYMEX: continuous front-month symbology works directly.
        return (f'{ticker}.c.0', 'continuous')


def build_specific_contract_symbol(
    ticker: str, exchange: str, month_letter: str, year_2digit: int,
) -> Tuple[str, str]:
    """
    Build Databento symbol for a specific contract month.

    ICE format: "{ticker}  FM{month}00{yy}!" (TWO spaces between ticker and FM)
    CME format: "{ticker}{month}{last_digit_of_year}" (e.g., NGK5 for May 2025)

    Returns:
        (symbol, stype_in)
    """
    if _is_ice_exchange(exchange):
        # ICE raw symbols use TWO spaces, e.g., "KC  FMK0026!"
        symbol = f"{ticker}  FM{month_letter}00{year_2digit:02d}!"
        return (symbol, 'raw_symbol')
    else:
        # CME uses single-digit year in raw symbols
        last_digit = year_2digit % 10
        symbol = f"{ticker}{month_letter}{last_digit}"
        return (symbol, 'raw_symbol')


def parse_our_contract_to_databento(
    ticker: str, exchange: str, contract: str,
) -> Tuple[str, str]:
    """
    Convert our contract identifier to Databento (symbol, stype_in).

    Args:
        ticker: Commodity ticker (e.g., 'KC', 'NG')
        exchange: Exchange name (e.g., 'ICE', 'NYMEX')
        contract: Either 'FRONT_MONTH' or a specific contract like 'KCH26'

    Returns:
        (symbol, stype_in) tuple for Databento API
    """
    if contract == 'FRONT_MONTH' or not contract:
        return build_front_month_symbol(ticker, exchange)

    match = _CONTRACT_RE.match(contract.upper())
    if not match:
        logger.warning(
            f"Cannot parse contract '{contract}', falling back to front month"
        )
        return build_front_month_symbol(ticker, exchange)

    month_letter = match.group(2)
    year_2digit = int(match.group(3))

    return build_specific_contract_symbol(ticker, exchange, month_letter, year_2digit)


def is_outright_symbol(raw_symbol: str) -> bool:
    """
    Check if a Databento raw symbol is an outright (not a spread).

    ICE parent symbology returns both outrights and spreads (e.g., calendar
    spreads like ``KC  FMK0026-KC  FMN0026``).  Outrights end with ``!``.
    """
    return raw_symbol.strip().endswith('!')


def estimate_cost(schema: str, num_days: int, is_parent: bool = False) -> float:
    """
    Rough cost estimate for a Databento query.

    Based on observed costs from API testing (Feb 2026):
    - ohlcv-1m, ICE parent, 1 day ~ $0.12
    - ohlcv-1h, CME continuous, 1 day ~ $0.01
    - ohlcv-1d, any, 1 day ~ $0.001
    """
    per_day = {
        'ohlcv-1m': 0.12 if is_parent else 0.03,
        'ohlcv-1h': 0.02 if is_parent else 0.01,
        'ohlcv-1d': 0.005,
    }
    return per_day.get(schema, 0.05) * num_days
