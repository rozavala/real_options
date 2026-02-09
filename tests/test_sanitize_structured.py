"""
Tests for timestamp parsing utilities.
"""
import unittest
import os
import pandas as pd

# Adjust import path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')


class TestParseTimestampCoerce(unittest.TestCase):
    """Test that parse_ts_column with errors='coerce' handles bad data."""

    def test_coerce_returns_nat_for_bad_values(self):
        """Bad timestamp values should become NaT instead of crashing."""
        from trading_bot.timestamps import parse_ts_column

        series = pd.Series([
            '2026-01-30 09:00:00+00:00',
            'KC-c0f3b12c',  # Bad value
            '2026-01-30 10:00:00+00:00',
        ])

        result = parse_ts_column(series, errors='coerce')

        # First and third should parse fine
        self.assertFalse(pd.isna(result.iloc[0]))
        self.assertFalse(pd.isna(result.iloc[2]))

        # Second should be NaT
        self.assertTrue(pd.isna(result.iloc[1]))

    def test_raise_mode_still_crashes(self):
        """Default mode should still raise on bad values (backward compat)."""
        from trading_bot.timestamps import parse_ts_column

        series = pd.Series(['KC-c0f3b12c', '2026-01-30 09:00:00+00:00'])

        with self.assertRaises(Exception):
            parse_ts_column(series)  # default errors='raise'


if __name__ == '__main__':
    unittest.main()
