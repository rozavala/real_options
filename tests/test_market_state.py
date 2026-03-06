"""Unit tests for the three-tier market state resolver.

Tests cover: _in_time_window(), get_market_state(), is_passive_mode(),
_minutes_until_next_break() — with mocked time for reproducibility.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pytz

from trading_bot.utils import (
    _in_time_window,
    get_market_state,
    is_passive_mode,
    _minutes_until_next_break,
)
from config.commodity_profiles import MarketStatesConfig


# ---------------------------------------------------------------------------
# _in_time_window() — pure function tests (no mocking needed)
# ---------------------------------------------------------------------------

class TestInTimeWindow:
    """Test same-day and overnight time window matching."""

    def test_same_day_inside(self):
        # 09:00-14:30 — 10:00 (600 min) is inside
        assert _in_time_window(600, "09:00-14:30") is True

    def test_same_day_at_start(self):
        # 09:00 = 540 min — start is inclusive
        assert _in_time_window(540, "09:00-14:30") is True

    def test_same_day_at_end(self):
        # 14:30 = 870 min — end is exclusive
        assert _in_time_window(870, "09:00-14:30") is False

    def test_same_day_before(self):
        assert _in_time_window(500, "09:00-14:30") is False

    def test_same_day_after(self):
        assert _in_time_window(900, "09:00-14:30") is False

    def test_overnight_inside_evening(self):
        # 18:00-09:00 — 20:00 (1200 min) is inside
        assert _in_time_window(1200, "18:00-09:00") is True

    def test_overnight_inside_morning(self):
        # 18:00-09:00 — 06:00 (360 min) is inside
        assert _in_time_window(360, "18:00-09:00") is True

    def test_overnight_at_start(self):
        # 18:00 = 1080 min — inclusive
        assert _in_time_window(1080, "18:00-09:00") is True

    def test_overnight_at_end(self):
        # 09:00 = 540 min — exclusive
        assert _in_time_window(540, "18:00-09:00") is False

    def test_overnight_outside_daytime(self):
        # 12:00 (720 min) — outside 18:00-09:00
        assert _in_time_window(720, "18:00-09:00") is False

    def test_midnight_boundary(self):
        # Midnight = 0 min — should be inside 18:00-09:00
        assert _in_time_window(0, "18:00-09:00") is True

    def test_maintenance_break(self):
        # 17:00-18:00 — 17:30 (1050 min) is inside
        assert _in_time_window(1050, "17:00-18:00") is True
        # 16:59 (1019 min) is outside
        assert _in_time_window(1019, "17:00-18:00") is False


# ---------------------------------------------------------------------------
# get_market_state() — mocked time tests
# ---------------------------------------------------------------------------

def _make_ng_config():
    """Return a minimal config dict that loads the NG profile."""
    return {"symbol": "NG"}


def _make_kc_config():
    """Return a minimal config dict that loads the KC profile."""
    return {"symbol": "KC"}


def _mock_now(year, month, day, hour, minute):
    """Create a timezone-aware NY datetime for mocking."""
    ny_tz = pytz.timezone('America/New_York')
    return ny_tz.localize(datetime(year, month, day, hour, minute, 0))


class TestGetMarketStateNG:
    """Test state resolution for NG (has passive windows)."""

    def _patch_now(self, ny_dt):
        """Patch datetime.now to return the given NY time."""
        return patch('trading_bot.utils.datetime', wraps=datetime,
                     **{'now.return_value': ny_dt})

    def test_saturday_sleeping(self):
        # Saturday 10:00 ET → SLEEPING
        ny_dt = _mock_now(2026, 3, 7, 10, 0)  # March 7, 2026 = Saturday
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'SLEEPING'

    def test_sunday_before_open_sleeping(self):
        # Sunday 15:00 ET → SLEEPING (before 18:00 Globex open)
        ny_dt = _mock_now(2026, 3, 8, 15, 0)  # March 8, 2026 = Sunday
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'SLEEPING'

    def test_sunday_at_1800_passive(self):
        # Sunday 18:00 ET → PASSIVE (NG opens, but 18:00 is passive window)
        ny_dt = _mock_now(2026, 3, 8, 18, 0)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'PASSIVE'

    def test_monday_active_window(self):
        # Monday 10:00 ET → ACTIVE (within 09:00-14:30)
        ny_dt = _mock_now(2026, 3, 9, 10, 0)  # Monday
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'ACTIVE'

    def test_monday_passive_morning(self):
        # Monday 07:00 ET → PASSIVE (within 18:00-09:00 overnight window)
        ny_dt = _mock_now(2026, 3, 9, 7, 0)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'PASSIVE'

    def test_monday_passive_afternoon(self):
        # Monday 15:00 ET → PASSIVE (within 14:30-17:00 window)
        ny_dt = _mock_now(2026, 3, 9, 15, 0)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'PASSIVE'

    def test_maintenance_break_sleeping(self):
        # Monday 17:30 ET → SLEEPING (within 17:00-18:00 maintenance break)
        ny_dt = _mock_now(2026, 3, 9, 17, 30)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'SLEEPING'

    def test_after_maintenance_passive(self):
        # Monday 18:00 ET → PASSIVE (maintenance ends, back to passive)
        ny_dt = _mock_now(2026, 3, 9, 18, 0)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'PASSIVE'

    def test_active_boundary_start(self):
        # Monday 09:00 ET → ACTIVE (start of active window, inclusive)
        ny_dt = _mock_now(2026, 3, 9, 9, 0)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'ACTIVE'

    def test_active_boundary_end(self):
        # Monday 14:30 ET → PASSIVE (end of active window is exclusive,
        # falls into 14:30-17:00 passive window)
        ny_dt = _mock_now(2026, 3, 9, 14, 30)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_ng_config()) == 'PASSIVE'


class TestGetMarketStateKC:
    """Test state resolution for KC (no passive windows)."""

    def _patch_now(self, ny_dt):
        return patch('trading_bot.utils.datetime', wraps=datetime,
                     **{'now.return_value': ny_dt})

    def test_active_during_trading(self):
        # Monday 10:00 ET → ACTIVE (within 04:15-13:30)
        ny_dt = _mock_now(2026, 3, 9, 10, 0)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_kc_config()) == 'ACTIVE'

    def test_sleeping_after_hours(self):
        # Monday 14:00 ET → SLEEPING (outside 04:15-13:30, no passive windows)
        ny_dt = _mock_now(2026, 3, 9, 14, 0)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_kc_config()) == 'SLEEPING'

    def test_sleeping_weekend(self):
        # Saturday → SLEEPING
        ny_dt = _mock_now(2026, 3, 7, 10, 0)
        with self._patch_now(ny_dt):
            assert get_market_state(_make_kc_config()) == 'SLEEPING'


class TestIsPassiveMode:
    """Test is_passive_mode() convenience function."""

    def test_passive_true(self):
        ny_dt = _mock_now(2026, 3, 9, 7, 0)  # Monday 07:00 → NG PASSIVE
        with patch('trading_bot.utils.datetime', wraps=datetime,
                   **{'now.return_value': ny_dt}):
            assert is_passive_mode(_make_ng_config()) is True

    def test_passive_false_active(self):
        ny_dt = _mock_now(2026, 3, 9, 10, 0)  # Monday 10:00 → NG ACTIVE
        with patch('trading_bot.utils.datetime', wraps=datetime,
                   **{'now.return_value': ny_dt}):
            assert is_passive_mode(_make_ng_config()) is False


# ---------------------------------------------------------------------------
# _minutes_until_next_break() tests
# ---------------------------------------------------------------------------

class TestMinutesUntilNextBreak:
    """Test maintenance break countdown."""

    def test_ng_before_break(self):
        # Monday 16:00 → 60 min until 17:00 break
        ny_dt = _mock_now(2026, 3, 9, 16, 0)
        with patch('trading_bot.utils.datetime', wraps=datetime,
                   **{'now.return_value': ny_dt}):
            result = _minutes_until_next_break(_make_ng_config())
            assert result == 60

    def test_ng_right_before_break(self):
        # Monday 16:50 → 10 min until break
        ny_dt = _mock_now(2026, 3, 9, 16, 50)
        with patch('trading_bot.utils.datetime', wraps=datetime,
                   **{'now.return_value': ny_dt}):
            result = _minutes_until_next_break(_make_ng_config())
            assert result == 10

    def test_kc_no_breaks(self):
        # KC has no maintenance breaks → 9999
        ny_dt = _mock_now(2026, 3, 9, 10, 0)
        with patch('trading_bot.utils.datetime', wraps=datetime,
                   **{'now.return_value': ny_dt}):
            result = _minutes_until_next_break(_make_kc_config())
            assert result == 9999

    def test_no_config(self):
        assert _minutes_until_next_break(None) == 9999
