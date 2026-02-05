import pytest
import os
from datetime import datetime, timezone, timedelta
import pytz
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
from ib_insync import Contract

# Import the module under test
from trading_bot.reconciliation import _calculate_actual_exit_time, reconcile_council_history

# Global test config with schedule offset
TEST_CONFIG = {
    'symbol': 'KC',
    'exchange': 'NYBOT',
    'schedule': {
        'position_close_hour': 11,
        'position_close_minute': 0,
        'offset_minutes': -10
    }
}

def test_exit_time_monday_entry():
    """Monday entry → Next trading day (Tuesday) at CLOSE_TIME (10:50 ET)."""
    ny = pytz.timezone('America/New_York')
    # Monday Jan 26, 2026 at 9:00 AM ET
    entry = ny.localize(datetime(2026, 1, 26, 9, 0)).astimezone(pytz.UTC)

    exit_time = _calculate_actual_exit_time(entry, TEST_CONFIG)
    exit_ny = exit_time.astimezone(ny)

    # Should be Tuesday Jan 27 at 10:50 ET
    assert exit_ny.weekday() == 1  # Tuesday
    assert exit_ny.date() == datetime(2026, 1, 27).date()
    assert exit_ny.hour == 10
    assert exit_ny.minute == 50


def test_exit_time_friday_entry():
    """Friday entry → Same day at CLOSE_TIME (10:50 ET)."""
    ny = pytz.timezone('America/New_York')
    # Friday Jan 30, 2026 at 4:53 AM ET (09:53 UTC)
    entry = ny.localize(datetime(2026, 1, 30, 4, 53)).astimezone(pytz.UTC)

    exit_time = _calculate_actual_exit_time(entry, TEST_CONFIG)
    exit_ny = exit_time.astimezone(ny)

    # Should be Friday Jan 30 at 10:50 ET
    assert exit_ny.weekday() == 4  # Friday
    assert exit_ny.date() == datetime(2026, 1, 30).date()
    assert exit_ny.hour == 10
    assert exit_ny.minute == 50


def test_exit_time_thursday_before_holiday():
    """Thursday entry before Friday holiday → Same day at CLOSE_TIME (10:50 ET)."""
    ny = pytz.timezone('America/New_York')
    # Jan 1, 2027 is a Friday (New Year's Day)
    # So Thursday Dec 31, 2026 entry should close Thursday
    entry = ny.localize(datetime(2026, 12, 31, 9, 0)).astimezone(pytz.UTC)

    exit_time = _calculate_actual_exit_time(entry, TEST_CONFIG)
    exit_ny = exit_time.astimezone(ny)

    # Should be Thursday Dec 31 at 10:50 ET
    assert exit_ny.date() == datetime(2026, 12, 31).date()
    assert exit_ny.hour == 10
    assert exit_ny.minute == 50


def test_exit_time_thursday_normal():
    """Normal Thursday entry → Next trading day (Friday) at CLOSE_TIME (10:50 ET)."""
    ny = pytz.timezone('America/New_York')
    # Thursday Jan 29, 2026 at 9:00 AM ET
    entry = ny.localize(datetime(2026, 1, 29, 9, 0)).astimezone(pytz.UTC)

    exit_time = _calculate_actual_exit_time(entry, TEST_CONFIG)
    exit_ny = exit_time.astimezone(ny)

    # Should be Friday Jan 30 at 10:50 ET
    assert exit_ny.weekday() == 4  # Friday
    assert exit_ny.date() == datetime(2026, 1, 30).date()
    assert exit_ny.hour == 10
    assert exit_ny.minute == 50


def test_exit_time_wednesday_spanning_weekend():
    """Late Wednesday entry → Next trading day (Thursday) at CLOSE_TIME."""
    ny = pytz.timezone('America/New_York')
    # Wednesday at 9:00 ET
    entry = ny.localize(datetime(2026, 1, 28, 9, 0)).astimezone(pytz.UTC)

    exit_time = _calculate_actual_exit_time(entry, TEST_CONFIG)
    exit_ny = exit_time.astimezone(ny)

    # Should be Thursday at 10:50 ET
    assert exit_ny.weekday() == 3  # Thursday
    assert exit_ny.hour == 10
    assert exit_ny.minute == 50


@pytest.mark.asyncio
async def test_reconcile_friday_entry_uses_friday_close():
    """Friday entry should use Friday's bar (weekly close)."""
    ny = pytz.timezone('America/New_York')
    entry_time = datetime(2026, 1, 30, 14, 0, 0, tzinfo=timezone.utc)  # Fri 9AM ET

    # Mock data
    mock_csv_data = pd.DataFrame({
        'timestamp': [entry_time],
        'contract': ['KCH6 (202603)'],
        'entry_price': [340.0],
        'master_decision': ['BULLISH'],
        'prediction_type': ['DIRECTIONAL'],
        'strategy_type': [''],
        'exit_price': [None],
        'exit_timestamp': [None],
        'pnl_realized': [None],
        'actual_trend_direction': [None],
        'volatility_outcome': [None]
    })

    mock_ib = MagicMock()
    mock_ib.reqContractDetailsAsync = AsyncMock()
    mock_details = MagicMock()
    mock_details.contract = Contract(symbol='KC', lastTradeDateOrContractMonth='202603')
    mock_ib.reqContractDetailsAsync.return_value = [mock_details]

    # IB returns both Friday and Monday bars
    mock_bar_fri = MagicMock()
    mock_bar_fri.date = datetime(2026, 1, 30).date()  # Friday
    mock_bar_fri.close = 335.0

    mock_bar_mon = MagicMock()
    mock_bar_mon.date = datetime(2026, 2, 2).date()   # Monday
    mock_bar_mon.close = 345.0

    mock_ib.reqHistoricalDataAsync = AsyncMock(
        return_value=[mock_bar_fri, mock_bar_mon]
    )

    # Use TEST_CONFIG which includes the schedule offset
    config = TEST_CONFIG

    original_exists = os.path.exists
    def exists_side_effect(path):
        if 'council_history.csv' in str(path):
            return True
        return original_exists(path)

    with patch('pandas.read_csv', return_value=mock_csv_data), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('os.path.exists', side_effect=exists_side_effect):

        await reconcile_council_history(config, ib=mock_ib)

        # Check that the exit price was set to Friday's close (335.0)
        exit_price_val = mock_csv_data.iloc[0]['exit_price']
        assert exit_price_val == 335.0, f"Expected 335.0, got {exit_price_val}"

        # Trend should be BEARISH (340 -> 335)
        trend_val = mock_csv_data.iloc[0]['actual_trend_direction']
        assert trend_val == 'BEARISH', f"Expected BEARISH, got {trend_val}"
