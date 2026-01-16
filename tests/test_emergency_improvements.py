import pytest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import pytz

# Import the code to test
# We need to import orchestrator but it has top-level execution code if not careful.
# It seems orchestrator.py has if __name__ == "__main__", which is good.
from orchestrator import is_market_open, run_emergency_cycle
from trading_bot.sentinels import SentinelTrigger
from trading_bot.state_manager import StateManager

# --- Unit Tests for is_market_open ---

def test_is_market_open_weekend():
    # Saturday
    with patch('orchestrator.datetime') as mock_date:
        # 2026-01-17 is a Saturday. UTC 5 PM = 12 PM EST.
        mock_date.now.return_value = datetime(2026, 1, 17, 17, 0, 0, tzinfo=timezone.utc)
        mock_date.side_effect = datetime
        assert is_market_open() == False

def test_is_market_open_sunday_closed():
    # Sunday before 6 PM
    with patch('orchestrator.datetime') as mock_date:
        # 2026-01-18 is a Sunday. UTC 22:59 = 5:59 PM EST.
        mock_date.now.return_value = datetime(2026, 1, 18, 22, 59, 0, tzinfo=timezone.utc)
        assert is_market_open() == False

def test_is_market_open_sunday_open():
    # Sunday after 6 PM
    with patch('orchestrator.datetime') as mock_date:
        # 2026-01-18 is a Sunday. UTC 23:01 = 6:01 PM EST.
        mock_date.now.return_value = datetime(2026, 1, 18, 23, 1, 0, tzinfo=timezone.utc)
        # Still false because we check core hours (04:15 - 13:30)
        assert is_market_open() == False

def test_is_market_open_core_hours():
    # Wednesday 10 AM EST = 3 PM UTC
    with patch('orchestrator.datetime') as mock_date:
        # 2026-01-14 is a Wednesday
        mock_date.now.return_value = datetime(2026, 1, 14, 15, 0, 0, tzinfo=timezone.utc)
        assert is_market_open() == True

def test_is_market_open_daily_break():
    # Wednesday 5:30 PM EST = 10:30 PM UTC
    with patch('orchestrator.datetime') as mock_date:
        # 2026-01-14 is a Wednesday
        mock_date.now.return_value = datetime(2026, 1, 14, 22, 30, 0, tzinfo=timezone.utc)
        assert is_market_open() == False

def test_is_market_open_after_hours():
    # Wednesday 3 PM EST = 8 PM UTC (After 1:30 PM close)
    with patch('orchestrator.datetime') as mock_date:
        # 2026-01-14 is a Wednesday
        mock_date.now.return_value = datetime(2026, 1, 14, 20, 0, 0, tzinfo=timezone.utc)
        assert is_market_open() == False

# --- Integration Test for run_emergency_cycle ---

@pytest.mark.asyncio
async def test_emergency_cycle_queues_when_closed():
    # Mock Market Closed
    with patch('orchestrator.is_market_open', return_value=False):
        # Mock StateManager
        with patch('orchestrator.StateManager') as mock_sm:
            # Mock IB
            mock_ib = MagicMock()

            trigger = SentinelTrigger("TestSentinel", "Test Reason", {})
            config = {}

            await run_emergency_cycle(trigger, config, mock_ib)

            # Verify queue_deferred_trigger was called
            mock_sm.queue_deferred_trigger.assert_called_once_with(trigger)

            # Verify NO other actions taken (e.g. no notifications sent via EMERGENCY_LOCK path)
            # We can check if EMERGENCY_LOCK was acquired.
            # Since the check is before the lock, we can't easily check lock acquisition directly without spying on the lock.
            # But we can assume if queue called and return happened, we are good.

            # We can mock logger to ensure "Queuing" message
            # But verifying the mock call is sufficient.

@pytest.mark.asyncio
async def test_emergency_cycle_runs_when_open():
     # Mock Market Open
    with patch('orchestrator.is_market_open', return_value=True):
         with patch('orchestrator.StateManager') as mock_sm:
             with patch('orchestrator.EMERGENCY_LOCK') as mock_lock:
                 # We need to mock the context manager of the lock
                 mock_lock.__aenter__.return_value = None
                 mock_lock.__aexit__.return_value = None

                 mock_ib = MagicMock()
                 trigger = SentinelTrigger("TestSentinel", "Test Reason", {})
                 config = {'notifications': {}}

                 # Mock get_active_futures to return empty so it exits early safely
                 with patch('orchestrator.get_active_futures', return_value=[]):
                     await run_emergency_cycle(trigger, config, mock_ib)

                 # Verify deferred trigger NOT queued
                 mock_sm.queue_deferred_trigger.assert_not_called()
