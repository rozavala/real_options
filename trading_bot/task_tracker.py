"""
Task Completion Tracker — Tracks which scheduled tasks have run today.

Used by the recovery system to distinguish "missed because we weren't running"
from "already completed before a crash." Resets automatically on each new
trading day (NY timezone).

Design principles:
- Fail-safe: Read/write failures never crash the orchestrator
- Consistent: Uses fcntl file locking (same as StateManager)
- Commodity-agnostic: No hardcoded symbols or schedules
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Optional

import pytz

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

logger = logging.getLogger(__name__)

TRACKER_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'data', 'task_completions.json'
)


def _get_trading_date() -> str:
    """Current trading date in NY timezone as YYYY-MM-DD string."""
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(timezone.utc).astimezone(ny_tz).strftime('%Y-%m-%d')


def _load_tracker() -> dict:
    """Load tracker state from disk. Returns empty dict on any failure."""
    if not os.path.exists(TRACKER_FILE):
        return {}
    try:
        with open(TRACKER_FILE, 'r') as f:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                if HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.warning(f"TaskTracker load failed (non-fatal): {e}")
        return {}


def _save_tracker(data: dict):
    """Save tracker state to disk with atomic write."""
    temp_file = TRACKER_FILE + ".tmp"
    try:
        os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
        with open(temp_file, 'w') as f:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)
        os.replace(temp_file, TRACKER_FILE)
    except Exception as e:
        logger.warning(f"TaskTracker save failed (non-fatal): {e}")


def record_task_completion(task_name: str):
    """
    Record that a task completed successfully today.

    Called after each scheduled task finishes in the main orchestrator loop.
    Auto-resets if the trading date has changed since last write.
    """
    today = _get_trading_date()
    data = _load_tracker()

    # Auto-reset on new trading day
    if data.get('trading_date') != today:
        data = {'trading_date': today, 'completions': {}}

    data['completions'][task_name] = datetime.now(timezone.utc).isoformat()
    _save_tracker(data)
    logger.debug(f"TaskTracker: Recorded completion of {task_name}")


def has_task_completed_today(task_name: str) -> bool:
    """
    Check if a task has already completed today.

    Used during recovery to avoid re-executing non-idempotent tasks
    (e.g., guarded_generate_orders) that already ran before a crash.
    """
    today = _get_trading_date()
    data = _load_tracker()

    if data.get('trading_date') != today:
        return False  # Different day — nothing completed yet

    return task_name in data.get('completions', {})


def get_completions_today() -> dict:
    """Return all task completions for today. Used for logging/dashboard."""
    today = _get_trading_date()
    data = _load_tracker()

    if data.get('trading_date') != today:
        return {}

    return data.get('completions', {})
