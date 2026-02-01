"""
Centralized timestamp parsing and formatting for Mission Control.

WHY THIS EXISTS:
Our CSV files contain timestamps written by different subsystems over time:
  - Old: "2026-01-20 12:30:45" (naive, no TZ, no microseconds)
  - New: "2026-01-30 12:30:45.374747+00:00" (ISO8601 with microseconds + TZ)

Calling pd.to_datetime() without format='mixed' crashes on mixed-format columns.
This module provides a single, tested entry point for ALL timestamp operations.

USAGE:
    from trading_bot.timestamps import parse_ts_column, format_ts

    # Reading timestamps from any CSV column:
    df['timestamp'] = parse_ts_column(df['timestamp'])

    # Writing timestamps to CSV:
    row['timestamp'] = format_ts()              # current UTC time
    row['timestamp'] = format_ts(some_datetime)  # specific time
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Standard format for all NEW writes (ISO8601, always UTC, always TZ-aware)
_WRITE_FORMAT = '%Y-%m-%d %H:%M:%S+00:00'


def parse_ts_column(series: pd.Series) -> pd.Series:
    """
    Parse a pandas Series of timestamp strings into timezone-aware UTC datetimes.

    Handles ALL formats found in our CSV files:
      - "2026-01-20 12:30:45"                    (naive)
      - "2026-01-20 12:30:45+00:00"              (TZ-aware)
      - "2026-01-30 12:30:45.374747+00:00"       (microseconds + TZ)
      - "2026-01-20T12:30:45Z"                   (ISO8601 with T and Z)
      - NaN / None / empty string                (returned as NaT)

    Returns:
        pd.Series of datetime64[ns, UTC]

    Example:
        df['timestamp'] = parse_ts_column(df['timestamp'])
    """
    return pd.to_datetime(series, utc=True, format='mixed')


def parse_ts_single(value: str) -> Optional[datetime]:
    """
    Parse a single timestamp string into a timezone-aware UTC datetime.

    Useful for non-pandas contexts (e.g., individual row processing).

    Returns:
        datetime with tzinfo=UTC, or None if unparseable
    """
    if not value or str(value).strip() in ('', 'nan', 'None', 'NaT'):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, format='mixed')
        return ts.to_pydatetime()
    except Exception:
        logger.warning(f"Could not parse timestamp: {value!r}")
        return None


def format_ts(dt: Optional[datetime] = None) -> str:
    """
    Format a datetime for CSV writing. Standardized to UTC.

    If no datetime provided, returns current UTC time.
    Strips microseconds for cleaner CSV output while remaining
    fully parseable by parse_ts_column().

    Returns:
        String like "2026-01-30 12:30:45+00:00"
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Ensure TZ-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC if different TZ
        dt = dt.astimezone(timezone.utc)

    return dt.strftime(_WRITE_FORMAT)
