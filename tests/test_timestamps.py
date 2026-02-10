"""Tests for centralized timestamp parsing and formatting."""

import sys
import os
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
import pytz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot.timestamps import parse_ts_column, parse_ts_single, format_ts, format_ib_datetime


class TestParseTimestampColumn:
    """Tests for parse_ts_column — the core read-side utility."""

    def test_naive_format(self):
        """Old-style timestamps without TZ or microseconds."""
        s = pd.Series(["2026-01-20 12:30:45"])
        result = parse_ts_column(s)
        assert result.iloc[0].year == 2026
        assert result.iloc[0].month == 1
        assert result.iloc[0].tzinfo is not None  # Should be UTC

    def test_iso8601_with_microseconds_and_tz(self):
        """New-style timestamps with microseconds and TZ offset."""
        s = pd.Series(["2026-01-30 12:30:45.374747+00:00"])
        result = parse_ts_column(s)
        assert result.iloc[0].microsecond == 374747
        assert result.iloc[0].tzinfo is not None

    def test_mixed_formats_in_same_column(self):
        """THE KEY TEST: mixed formats must not crash."""
        s = pd.Series([
            "2026-01-20 12:30:45",                     # naive
            "2026-01-30 12:30:45.374747+00:00",        # ISO8601 full
            "2026-01-25T08:00:00Z",                     # ISO8601 with T and Z
            "2026-01-28 15:00:00+00:00",                # TZ-aware, no microseconds
        ])
        result = parse_ts_column(s)
        assert len(result) == 4
        assert result.notna().all()
        assert all(ts.tzinfo is not None for ts in result)

    def test_nan_handling(self):
        """NaN values should become NaT, not crash."""
        s = pd.Series(["2026-01-20 12:30:45", None, ""])
        result = parse_ts_column(s)
        assert result.iloc[0].year == 2026
        # NaN/None/empty should be NaT
        assert pd.isna(result.iloc[1])

    def test_large_mixed_column(self):
        """Simulate realistic data: 63+ rows where format changes mid-column."""
        naive_rows = [f"2026-01-{d:02d} 10:00:00" for d in range(1, 32)]
        iso_rows = [f"2026-01-{d:02d} 10:00:00.{d*11111:06d}+00:00" for d in range(1, 32)]
        s = pd.Series(naive_rows + iso_rows)  # 62 rows, mixed
        result = parse_ts_column(s)
        assert len(result) == 62
        assert result.notna().all()


class TestParseSingleTimestamp:
    """Tests for parse_ts_single — for non-pandas contexts."""

    def test_naive_string(self):
        result = parse_ts_single("2026-01-20 12:30:45")
        assert result is not None
        assert result.tzinfo is not None

    def test_iso8601_string(self):
        result = parse_ts_single("2026-01-30 12:30:45.374747+00:00")
        assert result is not None
        assert result.microsecond == 374747

    def test_none_and_empty(self):
        assert parse_ts_single(None) is None
        assert parse_ts_single("") is None
        assert parse_ts_single("nan") is None


class TestFormatTimestamp:
    """Tests for format_ts — the write-side standardizer."""

    def test_default_current_time(self):
        result = format_ts()
        assert "+00:00" in result  # Must include TZ
        assert "T" not in result   # We use space separator

    def test_specific_datetime(self):
        dt = datetime(2026, 1, 30, 12, 30, 45, tzinfo=timezone.utc)
        result = format_ts(dt)
        assert result == "2026-01-30 12:30:45+00:00"

    def test_naive_datetime_assumed_utc(self):
        dt = datetime(2026, 1, 30, 12, 30, 45)
        result = format_ts(dt)
        assert "+00:00" in result

    def test_round_trip(self):
        """Write → Read cycle must produce identical datetime."""
        original = datetime(2026, 6, 15, 9, 30, 0, tzinfo=timezone.utc)
        written = format_ts(original)
        parsed = parse_ts_single(written)
        assert parsed.year == original.year
        assert parsed.month == original.month
        assert parsed.day == original.day
        assert parsed.hour == original.hour
        assert parsed.minute == original.minute


class TestFormatIbDatetime:
    """Tests for format_ib_datetime() — IB API endDateTime parameter formatting."""

    def test_none_returns_empty_string(self):
        """None input → '' (IB interprets as 'now')."""
        assert format_ib_datetime(None) == ''
        # We cannot test format_ib_datetime() without arguments directly
        # because the type hint says dt: Optional[datetime] = None,
        # but if we call it without arguments it uses the default value.
        assert format_ib_datetime() == ''

    def test_utc_datetime(self):
        """UTC datetime → dash format without suffix."""
        dt = datetime(2026, 2, 1, 16, 0, 0, tzinfo=timezone.utc)
        result = format_ib_datetime(dt)
        assert result == '20260201-16:00:00'
        # CRITICAL: Must NOT contain ' UTC' suffix
        assert ' UTC' not in result
        assert result.count('-') == 1  # Only the date-time dash

    def test_naive_datetime_assumed_utc(self):
        """Naive datetime → treated as UTC."""
        dt = datetime(2026, 1, 30, 16, 0, 0)
        result = format_ib_datetime(dt)
        assert result == '20260130-16:00:00'

    def test_non_utc_converted(self):
        """Non-UTC timezone → converted to UTC before formatting."""
        ny = pytz.timezone('America/New_York')
        dt = ny.localize(datetime(2026, 1, 30, 11, 0, 0))  # 11:00 ET = 16:00 UTC
        result = format_ib_datetime(dt)
        assert result == '20260130-16:00:00'

    def test_with_microseconds(self):
        """Microseconds are truncated (IB doesn't use them)."""
        dt = datetime(2026, 2, 1, 16, 30, 45, 123456, tzinfo=timezone.utc)
        result = format_ib_datetime(dt)
        assert result == '20260201-16:30:45'
        assert '.' not in result  # No microseconds

    def test_reconciliation_scenario(self):
        """Reproduce the exact scenario that was failing: exit_time + 2 days."""
        # Friday Jan 30 at 16:00 UTC (weekly close)
        exit_time = datetime(2026, 1, 30, 16, 0, 0, tzinfo=timezone.utc)
        # +2 days for the endDateTime buffer
        end_dt = exit_time + timedelta(days=2)
        result = format_ib_datetime(end_dt)
        assert result == '20260201-16:00:00'
        # This is what was being sent as '20260201-16:00:00 UTC' (broken)

    def test_midnight_utc(self):
        """Midnight UTC edge case."""
        dt = datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = format_ib_datetime(dt)
        assert result == '20260315-00:00:00'

    def test_end_of_day(self):
        """23:59:59 edge case."""
        dt = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        result = format_ib_datetime(dt)
        assert result == '20261231-23:59:59'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
