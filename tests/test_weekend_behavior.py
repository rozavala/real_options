import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta, time
import pytz

# Import modules to test
# We need to be careful with imports that might trigger top-level code or side effects.
# Using mock for imports if necessary, but we can import functions directly if safe.
from orchestrator import get_next_task, start_monitoring, schedule
from trading_bot.order_manager import generate_and_execute_orders
from trading_bot.utils import is_market_open

# --- Test get_next_task Weekend Skipping ---

def test_get_next_task_saturday():
    """
    If today is Saturday, get_next_task should schedule for Monday.
    """
    # Define a simple schedule: Task at 9:00 AM NY
    mock_task = MagicMock(__name__="mock_task")
    test_schedule = {time(9, 0): mock_task}

    # Mock Current Time: Saturday, Jan 17, 2026 at 10:00 AM NY
    # 2026-01-17 is Saturday.
    ny_tz = pytz.timezone('America/New_York')
    utc = pytz.UTC

    # 10:00 AM NY on Saturday
    now_ny = ny_tz.localize(datetime(2026, 1, 17, 10, 0, 0))
    now_utc = now_ny.astimezone(utc)

    # Expected Next Run: Monday, Jan 19, 2026 at 9:00 AM NY
    expected_ny = ny_tz.localize(datetime(2026, 1, 19, 9, 0, 0))
    expected_utc = expected_ny.astimezone(utc)

    # Call function
    next_run_utc, next_task = get_next_task(now_utc, test_schedule)

    assert next_task == mock_task
    assert next_run_utc == expected_utc

def test_get_next_task_sunday():
    """
    If today is Sunday, get_next_task should schedule for Monday.
    """
    mock_task = MagicMock(__name__="mock_task")
    test_schedule = {time(9, 0): mock_task}

    # Mock Current Time: Sunday, Jan 18, 2026 at 10:00 AM NY
    ny_tz = pytz.timezone('America/New_York')
    utc = pytz.UTC

    now_ny = ny_tz.localize(datetime(2026, 1, 18, 10, 0, 0))
    now_utc = now_ny.astimezone(utc)

    # Expected Next Run: Monday, Jan 19, 2026 at 9:00 AM NY
    expected_ny = ny_tz.localize(datetime(2026, 1, 19, 9, 0, 0))
    expected_utc = expected_ny.astimezone(utc)

    next_run_utc, next_task = get_next_task(now_utc, test_schedule)

    assert next_task == mock_task
    assert next_run_utc == expected_utc

def test_get_next_task_friday_after_task():
    """
    If today is Friday and task passed, next task should be Monday.
    """
    mock_task = MagicMock(__name__="mock_task")
    test_schedule = {time(9, 0): mock_task}

    # Mock Current Time: Friday, Jan 16, 2026 at 10:00 AM NY (Task was 9:00 AM)
    ny_tz = pytz.timezone('America/New_York')
    utc = pytz.UTC

    now_ny = ny_tz.localize(datetime(2026, 1, 16, 10, 0, 0))
    now_utc = now_ny.astimezone(utc)

    # Expected Next Run: Monday, Jan 19, 2026 at 9:00 AM NY
    # Because Friday 9AM is passed. Saturday/Sunday skipped.
    expected_ny = ny_tz.localize(datetime(2026, 1, 19, 9, 0, 0))
    expected_utc = expected_ny.astimezone(utc)

    next_run_utc, next_task = get_next_task(now_utc, test_schedule)

    assert next_task == mock_task
    assert next_run_utc == expected_utc

# --- Test Early Exits ---

@pytest.mark.asyncio
async def test_start_monitoring_early_exit():
    """
    start_monitoring should exit early if market is closed.
    """
    config = {'notifications': {}}

    with patch('orchestrator.is_market_open', return_value=False) as mock_is_open:
        with patch('orchestrator.asyncio.create_subprocess_exec') as mock_proc:
            await start_monitoring(config)

            mock_is_open.assert_called_once()
            mock_proc.assert_not_called()

@pytest.mark.asyncio
async def test_generate_and_execute_orders_early_exit():
    """
    generate_and_execute_orders should exit early if market is closed.
    """
    config = {'notifications': {}}

    with patch('trading_bot.order_manager.is_market_open', return_value=False) as mock_is_open:
        with patch('trading_bot.order_manager.generate_and_queue_orders') as mock_gen:
            await generate_and_execute_orders(config)

            mock_is_open.assert_called_once()
            mock_gen.assert_not_called()
