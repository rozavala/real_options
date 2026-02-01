import pytest
import os
from datetime import datetime, timezone, timedelta
import pytz
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
from ib_insync import Contract

# Import the module under test
from trading_bot.reconciliation import _calculate_actual_exit_time, reconcile_council_history

def test_exit_time_monday_entry():
    """Monday entry → standard entry+27h exit (Tuesday)."""
    ny = pytz.timezone('America/New_York')
    # Monday Jan 26, 2026 at 9:00 AM ET
    entry = ny.localize(datetime(2026, 1, 26, 9, 0)).astimezone(pytz.UTC)
    exit_time = _calculate_actual_exit_time(entry, {})
    exit_ny = exit_time.astimezone(ny)
    # Should be Tuesday ~12:00 ET (entry+27h)
    assert exit_ny.weekday() == 1  # Tuesday
    assert exit_ny.date() == datetime(2026, 1, 27).date()


def test_exit_time_friday_entry():
    """Friday entry → same Friday 12:45 ET (weekly close)."""
    ny = pytz.timezone('America/New_York')
    # Friday Jan 30, 2026 at 4:53 AM ET (09:53 UTC)
    entry = ny.localize(datetime(2026, 1, 30, 4, 53)).astimezone(pytz.UTC)
    exit_time = _calculate_actual_exit_time(entry, {})
    exit_ny = exit_time.astimezone(ny)
    # Should be Friday Jan 30 at 12:45 ET
    assert exit_ny.weekday() == 4  # Friday
    assert exit_ny.hour == 12
    assert exit_ny.minute == 45
    assert exit_ny.date() == datetime(2026, 1, 30).date()


def test_exit_time_thursday_before_holiday():
    """Thursday entry before Friday holiday → Thursday 12:45 ET."""
    ny = pytz.timezone('America/New_York')
    # Jan 1, 2027 is a Friday (New Year's Day)
    # So Thursday Dec 31, 2026 entry should close Thursday
    entry = ny.localize(datetime(2026, 12, 31, 9, 0)).astimezone(pytz.UTC)
    exit_time = _calculate_actual_exit_time(entry, {})
    exit_ny = exit_time.astimezone(ny)
    # Should be Thursday Dec 31 at 12:45 ET (Friday is holiday)
    assert exit_ny.date() == datetime(2026, 12, 31).date()
    assert exit_ny.hour == 12
    assert exit_ny.minute == 45


def test_exit_time_thursday_normal():
    """Normal Thursday entry (Friday is not a holiday) → Friday exit."""
    ny = pytz.timezone('America/New_York')
    # Thursday Jan 29, 2026 at 9:00 AM ET → entry+27h = Friday 12:00 ET
    entry = ny.localize(datetime(2026, 1, 29, 9, 0)).astimezone(pytz.UTC)
    exit_time = _calculate_actual_exit_time(entry, {})
    exit_ny = exit_time.astimezone(ny)
    # Default +27h lands on Friday, which is a trading day
    assert exit_ny.weekday() == 4  # Friday
    assert exit_ny.date() == datetime(2026, 1, 30).date()


def test_exit_time_wednesday_spanning_weekend():
    """Late Wednesday entry where +27h would land on Friday."""
    ny = pytz.timezone('America/New_York')
    # Wednesday at 9:00 ET → +27h = Thursday 12:00 ET (normal)
    entry = ny.localize(datetime(2026, 1, 28, 9, 0)).astimezone(pytz.UTC)
    exit_time = _calculate_actual_exit_time(entry, {})
    exit_ny = exit_time.astimezone(ny)
    # Should be Thursday 12:00 ET (standard)
    assert exit_ny.weekday() == 3  # Thursday


@pytest.mark.asyncio
async def test_reconcile_friday_entry_uses_friday_close():
    """Friday entry should use Friday's bar (weekly close), not Monday's."""
    ny = pytz.timezone('America/New_York')
    entry_time = datetime(2026, 1, 30, 14, 0, 0, tzinfo=timezone.utc)  # Fri 9AM ET

    # Mock data needs to have required columns to avoid errors
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
    mock_bar_mon.close = 345.0  # Big gap up Monday

    # The order of return values depends on how reqHistoricalDataAsync works, usually sorted
    mock_ib.reqHistoricalDataAsync = AsyncMock(
        return_value=[mock_bar_fri, mock_bar_mon]
    )

    config = {'symbol': 'KC', 'exchange': 'NYBOT'}

    # Patch pandas read_csv AND existence checks
    # Use side_effect to only mock council_history.csv existence, letting other checks (like holidays locales) pass through
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
        # Because the calculation for Friday entry sets target exit to Friday 12:45
        # And bar matching should pick Friday's bar

        exit_price_val = mock_csv_data.iloc[0]['exit_price']
        assert exit_price_val == 335.0, f"Expected 335.0, got {exit_price_val}"

        # Trend should be BEARISH (340 -> 335)
        trend_val = mock_csv_data.iloc[0]['actual_trend_direction']
        assert trend_val == 'BEARISH', f"Expected BEARISH, got {trend_val}"
