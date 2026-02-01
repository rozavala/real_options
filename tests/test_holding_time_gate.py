import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytz

from trading_bot.utils import hours_until_weekly_close


def test_monday_morning():
    """Monday morning → no forced close → inf."""
    ny = pytz.timezone('America/New_York')
    mock_now = datetime(2026, 1, 26, 14, 0, 0, tzinfo=timezone.utc)  # Mon 9 AM ET
    with patch('trading_bot.utils.datetime') as mock_dt:
        mock_dt.now.return_value = mock_now
        result = hours_until_weekly_close()
    assert result == float('inf')


def test_friday_9am():
    """Friday 9 AM ET → ~3.75h until 12:45."""
    ny = pytz.timezone('America/New_York')
    # Friday Jan 30, 2026 at 9:00 AM ET = 14:00 UTC
    mock_now = datetime(2026, 1, 30, 14, 0, 0, tzinfo=timezone.utc)
    with patch('trading_bot.utils.datetime') as mock_dt:
        mock_dt.now.return_value = mock_now
        result = hours_until_weekly_close()
    assert 3.5 <= result <= 4.0  # ~3.75h


def test_friday_1pm():
    """Friday 1 PM ET → past close → 0."""
    mock_now = datetime(2026, 1, 30, 18, 0, 0, tzinfo=timezone.utc)  # Fri 1 PM ET
    with patch('trading_bot.utils.datetime') as mock_dt:
        mock_dt.now.return_value = mock_now
        result = hours_until_weekly_close()
    assert result == 0.0


def test_thursday_before_holiday():
    """Thursday before Friday holiday → should report hours until Thursday close."""
    # Dec 31, 2026 is Thursday; Jan 1, 2027 is Friday (New Year's)
    mock_now = datetime(2026, 12, 31, 14, 0, 0, tzinfo=timezone.utc)  # Thu 9 AM ET
    with patch('trading_bot.utils.datetime') as mock_dt:
        mock_dt.now.return_value = mock_now
        result = hours_until_weekly_close()
    assert 3.5 <= result <= 4.0  # ~3.75h until 12:45 ET


def test_wednesday():
    """Normal Wednesday → no forced close → inf."""
    mock_now = datetime(2026, 1, 28, 14, 0, 0, tzinfo=timezone.utc)  # Wed 9 AM ET
    with patch('trading_bot.utils.datetime') as mock_dt:
        mock_dt.now.return_value = mock_now
        result = hours_until_weekly_close()
    assert result == float('inf')


def test_gate_blocks_friday():
    """Integration: generate_and_execute_orders should skip on Friday."""
    from unittest.mock import AsyncMock
    from trading_bot.order_manager import generate_and_execute_orders

    config = {'risk_management': {'min_holding_hours': 6.0}}

    with patch('trading_bot.order_manager.is_market_open', return_value=True), \
         patch('trading_bot.order_manager.hours_until_weekly_close', return_value=3.75), \
         patch('trading_bot.order_manager.generate_and_queue_orders') as mock_gen, \
         patch('trading_bot.order_manager.send_pushover_notification'):

        import asyncio
        asyncio.run(generate_and_execute_orders(config))

        mock_gen.assert_not_called()  # Should have been skipped


def test_gate_allows_thursday():
    """Integration: generate_and_execute_orders should run on Thursday."""
    from unittest.mock import AsyncMock
    from trading_bot.order_manager import generate_and_execute_orders

    config = {'risk_management': {'min_holding_hours': 6.0}}

    with patch('trading_bot.order_manager.is_market_open', return_value=True), \
         patch('trading_bot.order_manager.hours_until_weekly_close', return_value=float('inf')), \
         patch('trading_bot.order_manager.generate_and_queue_orders') as mock_gen, \
         patch('trading_bot.order_manager.place_queued_orders'), \
         patch('trading_bot.order_manager.send_pushover_notification'):

        import asyncio
        asyncio.run(generate_and_execute_orders(config))

        mock_gen.assert_called_once()  # Should proceed
