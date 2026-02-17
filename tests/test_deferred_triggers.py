"""Tests for deferred trigger file-locking and atomicity in StateManager."""

import json
import os
import pytest
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

from trading_bot.state_manager import StateManager


@pytest.fixture(autouse=True)
def temp_triggers_file(tmp_path, monkeypatch):
    """Redirect deferred triggers to a temp file for each test."""
    triggers_file = str(tmp_path / "deferred_triggers.json")
    lock_file = str(tmp_path / ".deferred_triggers.lock")
    monkeypatch.setattr(StateManager, "DEFERRED_TRIGGERS_FILE", triggers_file)
    monkeypatch.setattr(StateManager, "_DEFERRED_LOCK_FILE", lock_file)
    return triggers_file


def _make_trigger(source="test_sentinel", reason="price spike", payload=None):
    return SimpleNamespace(
        source=source,
        reason=reason,
        payload=payload or {"price": 350.0},
    )


def test_queue_and_get_round_trip(temp_triggers_file):
    """Queue a trigger, get it back, verify contents."""
    trigger = _make_trigger(source="PriceSentinel", reason="5% move")
    StateManager.queue_deferred_trigger(trigger)

    result = StateManager.get_deferred_triggers()
    assert len(result) == 1
    assert result[0]["source"] == "PriceSentinel"
    assert result[0]["reason"] == "5% move"


def test_file_cleared_after_get(temp_triggers_file):
    """After get_deferred_triggers, file should be empty."""
    StateManager.queue_deferred_trigger(_make_trigger())
    StateManager.get_deferred_triggers()

    # File should exist but contain empty list
    assert os.path.exists(temp_triggers_file)
    with open(temp_triggers_file) as f:
        assert json.load(f) == []


def test_multiple_queues_accumulate(temp_triggers_file):
    """Multiple queue calls should accumulate triggers."""
    StateManager.queue_deferred_trigger(_make_trigger(source="A"))
    StateManager.queue_deferred_trigger(_make_trigger(source="B"))
    StateManager.queue_deferred_trigger(_make_trigger(source="C"))

    result = StateManager.get_deferred_triggers()
    assert len(result) == 3
    sources = {t["source"] for t in result}
    assert sources == {"A", "B", "C"}


def test_expired_triggers_filtered(temp_triggers_file):
    """Triggers older than max_age_hours should be discarded."""
    # Write a trigger with an old timestamp directly to the file
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    triggers = [
        {"source": "old", "reason": "stale", "payload": {}, "timestamp": old_ts},
        {"source": "fresh", "reason": "current", "payload": {}, "timestamp": fresh_ts},
    ]
    with open(temp_triggers_file, "w") as f:
        json.dump(triggers, f)

    result = StateManager.get_deferred_triggers(max_age_hours=72.0)
    assert len(result) == 1
    assert result[0]["source"] == "fresh"


def test_get_empty_returns_empty_list(temp_triggers_file):
    """get_deferred_triggers on empty/missing file returns []."""
    result = StateManager.get_deferred_triggers()
    assert result == []


def test_queue_after_get_not_lost(temp_triggers_file):
    """A trigger queued after a get should persist (no data loss)."""
    StateManager.queue_deferred_trigger(_make_trigger(source="first"))
    StateManager.get_deferred_triggers()  # clears

    StateManager.queue_deferred_trigger(_make_trigger(source="second"))
    result = StateManager.get_deferred_triggers()
    assert len(result) == 1
    assert result[0]["source"] == "second"
