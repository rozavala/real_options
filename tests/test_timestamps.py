"""Tests for centralized timestamp parsing and formatting."""

import sys
import os
import pytest
import pandas as pd
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot.timestamps import parse_ts_column, parse_ts_single, format_ts


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
