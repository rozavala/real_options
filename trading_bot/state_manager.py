import json
import os
import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    import threading
    _fallback_lock = threading.Lock()

logger = logging.getLogger(__name__)

# Default paths — overridden by set_data_dir() for multi-commodity isolation
_BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'KC')
STATE_FILE = os.path.join(_BASE_DATA_DIR, 'state.json')

def _validate_confidence(value: Any) -> float:
    """
    Validates and clamps confidence values to [0.0, 1.0].

    Args:
        value: Raw confidence value (could be negative, >1, or non-numeric)

    Returns:
        float: Clamped confidence value between 0.0 and 1.0
    """
    try:
        conf = float(value)
        if conf < 0.0:
            logger.warning(f"Negative confidence detected ({conf}), clamping to 0.0")
            return 0.0
        if conf > 1.0:
            logger.warning(f"Confidence > 1.0 detected ({conf}), clamping to 1.0")
            return 1.0
        return conf
    except (TypeError, ValueError):
        logger.warning(f"Invalid confidence value ({value}), defaulting to 0.5")
        return 0.5

class StateManager:
    """
    Manages the persistence of agent reports and system state.
    Uses file locking (fcntl on Unix, threading fallback on Windows) for safety.
    Supports TTL-based staleness checks.
    """
    _async_lock = asyncio.Lock()
    REPORT_TTL_SECONDS = 3600  # 1 hour staleness threshold
    DEFERRED_TRIGGERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'KC', 'deferred_triggers.json')

    # Class-level path variables — overridden by set_data_dir() for multi-commodity
    _state_file = None  # When None, falls back to module-level STATE_FILE
    _deferred_triggers_file = None
    _state_lock_file = None
    _deferred_lock_file = None

    @classmethod
    def set_data_dir(cls, data_dir: str):
        """Configure all StateManager paths for a commodity-specific data directory.

        Must be called before any state operations when running multi-commodity.
        """
        global STATE_FILE
        os.makedirs(data_dir, exist_ok=True)
        STATE_FILE = os.path.join(data_dir, 'state.json')
        cls._state_file = STATE_FILE
        cls.DEFERRED_TRIGGERS_FILE = os.path.join(data_dir, 'deferred_triggers.json')
        cls._deferred_triggers_file = cls.DEFERRED_TRIGGERS_FILE
        cls._STATE_LOCK_FILE = os.path.join(data_dir, '.state_global.lock')
        cls._state_lock_file = cls._STATE_LOCK_FILE
        cls._DEFERRED_LOCK_FILE = os.path.join(data_dir, '.deferred_triggers.lock')
        cls._deferred_lock_file = cls._DEFERRED_LOCK_FILE
        logger.info(f"StateManager data_dir set to: {data_dir}")

    @classmethod
    def _save_raw_sync(cls, data: dict):
        """Internal sync save with file locking."""
        temp_file = STATE_FILE + ".tmp"

        # Ensure dir exists
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

        with open(temp_file, 'w') as f:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            elif '_fallback_lock' in globals():
                _fallback_lock.acquire()

            try:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)
                elif '_fallback_lock' in globals():
                    _fallback_lock.release()

        os.replace(temp_file, STATE_FILE)

    @classmethod
    def _load_raw_sync(cls) -> dict:
        """Internal sync load."""
        if not os.path.exists(STATE_FILE):
            return {}

        try:
            with open(STATE_FILE, 'r') as f:
                if HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return {}

    # F2 NOTE: fcntl locks are advisory-only on Linux. This means:
    # - All processes accessing this file MUST use the same locking mechanism
    # - External scripts that read/write state directly will NOT be blocked
    # - For stronger guarantees, consider a SQLite backend (future enhancement)
    #
    # Current mitigation: All code paths go through StateManager.
    # DO NOT add direct file access anywhere else.

    # Single global lock file for ALL state writes to prevent cross-namespace races.
    # Previously used per-namespace locks (data/.state_{ns}.lock) which allowed
    # concurrent writes from different namespaces to clobber each other's data.
    _STATE_LOCK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'KC', '.state_global.lock')

    # Separate lock file for deferred triggers (prevents race between sentinel
    # queue_deferred_trigger and orchestrator get_deferred_triggers).
    _DEFERRED_LOCK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'KC', '.deferred_triggers.lock')

    @classmethod
    def _with_state_lock(cls, fn):
        """Execute fn under the global state file lock (read-modify-write safe)."""
        os.makedirs(os.path.dirname(cls._STATE_LOCK_FILE), exist_ok=True)
        with open(cls._STATE_LOCK_FILE, 'w') as lock_fd:
            if HAS_FCNTL:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            elif '_fallback_lock' in globals():
                _fallback_lock.acquire()
            try:
                return fn()
            finally:
                if HAS_FCNTL:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                elif '_fallback_lock' in globals():
                    _fallback_lock.release()

    @classmethod
    def atomic_state_update(cls, namespace: str, key: str, value: dict):
        """Thread-safe and process-safe state update using global lock."""
        def _do_update():
            state = cls._load_raw_sync()
            if namespace not in state:
                state[namespace] = {}
            state[namespace][key] = {
                'data': value,
                'timestamp': time.time()
            }
            cls._save_raw_sync(state)

        cls._with_state_lock(_do_update)

    @classmethod
    def save_state(cls, updates: Dict[str, Any], namespace: str = "reports"):
        """
        Synchronous save (merges updates). Uses global lock to prevent races.
        """
        def _do_save():
            current = cls._load_raw_sync()
            if namespace not in current:
                current[namespace] = {}

            for key, value in updates.items():
                # Validate confidence if present in the value
                if isinstance(value, dict) and 'confidence' in value:
                    value['confidence'] = _validate_confidence(value['confidence'])
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'confidence' in item:
                            item['confidence'] = _validate_confidence(item['confidence'])

                current[namespace][key] = {
                    "data": value,
                    "timestamp": time.time()
                }

            cls._save_raw_sync(current)

        cls._with_state_lock(_do_save)

    @classmethod
    async def save_state_async(cls, updates: Dict[str, Any], namespace: str = "reports"):
        """Async save wrapper."""
        async with cls._async_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, cls.save_state, updates, namespace)

    @classmethod
    def load_state(cls, namespace: str = "reports", max_age: int = None) -> Dict:
        """
        Synchronous load. Returns dict of key -> data.
        Appends warning string if data is stale.
        """
        state = cls._load_raw_sync().get(namespace, {})
        max_age = max_age or cls.REPORT_TTL_SECONDS

        result = {}
        for key, entry in state.items():
            # v3.1: Auto-migrate legacy entries on read
            if not isinstance(entry, dict) or 'data' not in entry:
                # This is a legacy entry - migrate it
                logger.warning(f"Legacy format detected for {namespace}/{key}. Run migrate_legacy_entries().")
                result[key] = {
                    'data': entry,
                    'age_seconds': float('inf'),
                    'age_hours': float('inf'),
                    'timestamp': 0,
                    'is_available': False,
                    'needs_migration': True
                }
                continue

            age = time.time() - entry.get("timestamp", 0)
            if age < max_age:
                result[key] = entry["data"]
            else:
                # Format stale data with warning
                data_str = str(entry['data'])
                preview = data_str[:100] + "..." if len(data_str) > 100 else data_str
                result[key] = f"STALE ({int(age/60)}min old) - {preview}"
        return result

    @classmethod
    def load_state_with_metadata(cls, namespace: str = "reports") -> Dict[str, Dict]:
        """
        Returns agent reports with age metadata for graduated staleness weighting.

        Returns: {
            'agent_name': {
                'data': <actual report dict or string>,
                'age_seconds': <float>,
                'age_hours': <float>,
                'timestamp': <float>,
                'is_available': <bool>
            }
        }
        """
        raw_state = cls._load_raw_sync()
        namespace_data = raw_state.get(namespace, {})
        now = time.time()
        result = {}

        for key, entry in namespace_data.items():
            if key.startswith('_'):  # Skip internal keys
                continue

            if isinstance(entry, dict) and 'data' in entry and 'timestamp' in entry:
                age_seconds = now - entry['timestamp']
                result[key] = {
                    'data': entry['data'],
                    'age_seconds': age_seconds,
                    'age_hours': age_seconds / 3600,
                    'timestamp': entry['timestamp'],
                    'is_available': True
                }
            else:
                # ┌──────────────────────────────────────────────────┐
                # │ LEGACY FALLBACK (R2 — Advisor Review Fix)        │
                # │                                                  │
                # │ CRITICAL: On first deployment, state.json will   │
                # │ contain entries written by the OLD code, which   │
                # │ lack the {data, timestamp} structure. Rather     │
                # │ than crashing or excluding these agents (which   │
                # │ would cause quorum failure — the exact problem   │
                # │ we're fixing), we assign them:                   │
                # │   age_seconds = inf → staleness_weight = 0.1     │
                # │   is_available = False (logged, not fatal)       │
                # │                                                  │
                # │ This means legacy-format agents get MINIMUM      │
                # │ weight (0.1) but still PARTICIPATE in voting,    │
                # │ preventing quorum collapse on the first run.     │
                # │                                                  │
                # │ After the first full scheduled cycle writes new  │
                # │ format data, this path won't fire again.         │
                # └──────────────────────────────────────────────────┘
                logger.info(f"Legacy format for '{key}' — no timestamp metadata. "
                            f"Assigning floor weight (0.1). Will auto-resolve after next scheduled cycle.")
                result[key] = {
                    'data': entry,
                    'age_seconds': float('inf'),
                    'age_hours': float('inf'),
                    'timestamp': 0,
                    'is_available': False
                }

        return result

    @classmethod
    def load_state_raw(cls, namespace: str = "reports") -> Dict[str, Any]:
        """
        Load state WITHOUT staleness string substitution.
        Returns raw data dicts regardless of age, or empty dict if not found.

        Use this for machine consumers (sentinels, agents) that need dict values.
        Use load_state() for human-facing displays that benefit from STALE warnings.
        """
        state = cls._load_raw_sync().get(namespace, {})
        result = {}
        for key, entry in state.items():
            if not isinstance(entry, dict) or 'data' not in entry:
                continue
            # Return the raw data without any string substitution
            result[key] = entry["data"]
        return result

    @classmethod
    async def load_state_async(cls, namespace: str = "reports", max_age: int = None) -> Dict:
        """Async load wrapper."""
        async with cls._async_lock:
             loop = asyncio.get_running_loop()
             return await loop.run_in_executor(None, cls.load_state, namespace, max_age)

    @classmethod
    def _with_deferred_lock(cls, fn):
        """Execute fn under the deferred triggers file lock (read-modify-write safe)."""
        os.makedirs(os.path.dirname(cls._DEFERRED_LOCK_FILE), exist_ok=True)
        with open(cls._DEFERRED_LOCK_FILE, 'w') as lock_fd:
            if HAS_FCNTL:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            elif '_fallback_lock' in globals():
                _fallback_lock.acquire()
            try:
                return fn()
            finally:
                if HAS_FCNTL:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                elif '_fallback_lock' in globals():
                    _fallback_lock.release()

    @classmethod
    def _write_deferred_triggers(cls, triggers: list):
        """Atomically write deferred triggers via temp file + os.replace."""
        os.makedirs(os.path.dirname(cls.DEFERRED_TRIGGERS_FILE), exist_ok=True)
        temp_file = cls.DEFERRED_TRIGGERS_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(triggers, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, cls.DEFERRED_TRIGGERS_FILE)

    @classmethod
    def queue_deferred_trigger(cls, trigger: Any):
        """Queue a trigger for processing when market opens (file-locked)."""
        def _do_queue():
            triggers = cls._load_deferred_triggers()
            triggers.append({
                'source': trigger.source,
                'reason': trigger.reason,
                'payload': trigger.payload,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            cls._write_deferred_triggers(triggers)
            logger.info(f"Queued deferred trigger from {trigger.source}")

        try:
            cls._with_deferred_lock(_do_queue)
        except Exception as e:
            logger.error(f"Failed to queue deferred trigger: {e}")

    @classmethod
    def get_deferred_triggers(cls, max_age_hours: float = 72.0) -> list:
        """
        Get and clear deferred triggers atomically, filtering expired ones.

        The read+clear is performed under a single lock to prevent triggers
        queued between read and clear from being lost.

        Args:
            max_age_hours: Maximum age in hours (default 72 = 3 days)

        Returns:
            List of valid (non-expired) triggers
        """
        def _do_get_and_clear():
            triggers = cls._load_deferred_triggers()

            if not triggers:
                return []

            # Clear immediately under the same lock
            cls._write_deferred_triggers([])

            now = datetime.now(timezone.utc)
            valid_triggers = []
            expired_count = 0

            for t in triggers:
                try:
                    ts_str = t['timestamp'].replace('Z', '+00:00')
                    trigger_time = datetime.fromisoformat(ts_str)
                    age_hours = (now - trigger_time).total_seconds() / 3600

                    if age_hours <= max_age_hours:
                        valid_triggers.append(t)
                    else:
                        expired_count += 1
                        logger.info(
                            f"Discarding expired trigger from {t['source']} "
                            f"(age: {age_hours:.1f}h > {max_age_hours}h)"
                        )
                except Exception as e:
                    logger.warning(f"Could not parse trigger timestamp: {e}")
                    valid_triggers.append(t)

            if expired_count > 0:
                logger.info(f"Filtered {expired_count} expired deferred triggers")

            return valid_triggers

        try:
            return cls._with_deferred_lock(_do_get_and_clear)
        except Exception as e:
            logger.error(f"Failed to get deferred triggers: {e}")
            return []

    @classmethod
    def _load_deferred_triggers(cls) -> list:
        if not os.path.exists(cls.DEFERRED_TRIGGERS_FILE):
            return []
        try:
            with open(cls.DEFERRED_TRIGGERS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load deferred triggers: {e}")
            return []

    @classmethod
    def log_sentinel_event(cls, trigger: Any):
        """
        Logs a sentinel trigger to the persistent history (Last 5 events).
        Used for fallback when grounded research fails.
        """
        try:
            # Load raw to get clean data (bypass staleness formatting of load_state)
            full_state = cls._load_raw_sync()
            history_ns = full_state.get("sentinel_history", {})
            events_entry = history_ns.get("events", {})

            # events_entry is {"data": [...], "timestamp": ...}
            current_events = events_entry.get("data", []) if isinstance(events_entry, dict) else []
            if not isinstance(current_events, list):
                current_events = []

            # Create new event object
            new_event = {
                "source": trigger.source,
                "reason": trigger.reason,
                "payload": trigger.payload,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Append and Trim
            current_events.append(new_event)
            if len(current_events) > 5:
                current_events = current_events[-5:]

            # Save using standard save_state
            cls.save_state({"events": current_events}, namespace="sentinel_history")
            logger.info(f"Logged sentinel event from {trigger.source} to history.")

        except Exception as e:
            logger.error(f"Failed to log sentinel event: {e}")
