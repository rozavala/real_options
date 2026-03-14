import pytest
import time
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock
from orchestrator import TriggerDeduplicator
from trading_bot.sentinels import SentinelTrigger

@pytest.fixture
def dedup(tmp_path):
    # Use a temp file for state
    state_file = tmp_path / "dedup_state.json"
    return TriggerDeduplicator(window_seconds=7200, state_file=str(state_file))

def test_post_cycle_debounce(dedup):
    """Test that global debounce blocks different sentinel sources."""
    trigger1 = SentinelTrigger("WeatherSentinel", "Test", {}, severity=5)

    # Should process initially
    assert dedup.should_process(trigger1) == True

    # Set Post-Cycle Debounce
    dedup.set_cooldown("POST_CYCLE", 1800)

    # Verify debounce is active
    assert dedup.cooldowns['POST_CYCLE'] > time.time()

    # Try different source
    trigger2 = SentinelTrigger("LogisticsSentinel", "Different", {}, severity=5)

    # Should be blocked
    assert dedup.should_process(trigger2) == False

    # Metrics check
    assert dedup.metrics['filtered_post_cycle'] > 0

def test_critical_severity_bypass(dedup):
    """Test that CRITICAL severity bypasses post-cycle debounce."""
    dedup.set_cooldown("POST_CYCLE", 1800)

    # Normal severity (blocked)
    trigger_norm = SentinelTrigger("WeatherSentinel", "Normal", {}, severity=5)
    assert dedup.should_process(trigger_norm) == False

    # Critical severity (bypass)
    trigger_crit = SentinelTrigger("WeatherSentinel", "Critical", {}, severity=9)
    assert dedup.should_process(trigger_crit) == True

def test_payload_deduplication(dedup):
    """Test content-based deduplication with unsorted keys."""
    payload1 = {"a": 1, "b": 2}
    payload2 = {"b": 2, "a": 1} # Different order

    trigger1 = SentinelTrigger("Source", "Reason", payload1)
    trigger2 = SentinelTrigger("Source", "Reason", payload2)

    assert dedup.should_process(trigger1) == True

    # Should be detected as duplicate because keys are sorted in hash
    assert dedup.should_process(trigger2) == False
    assert dedup.metrics['filtered_duplicate_content'] > 0
