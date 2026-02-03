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

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'state.json')

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
    DEFERRED_TRIGGERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'deferred_triggers.json')

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

    @classmethod
    def save_state(cls, updates: Dict[str, Any], namespace: str = "reports"):
        """
        Synchronous save (merges updates).
        """
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
            if not isinstance(entry, dict) or 'timestamp' not in entry:
                 # Attempt to handle legacy format if strictly necessary, but assuming clean slate/overwrite
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
    def queue_deferred_trigger(cls, trigger: Any):
        """Queue a trigger for processing when market opens."""
        triggers = cls._load_deferred_triggers()
        triggers.append({
            'source': trigger.source,
            'reason': trigger.reason,
            'payload': trigger.payload,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        try:
            os.makedirs(os.path.dirname(cls.DEFERRED_TRIGGERS_FILE), exist_ok=True)
            with open(cls.DEFERRED_TRIGGERS_FILE, 'w') as f:
                json.dump(triggers, f, indent=2)
            logger.info(f"Queued deferred trigger from {trigger.source}")
        except Exception as e:
            logger.error(f"Failed to queue deferred trigger: {e}")

    @classmethod
    def get_deferred_triggers(cls) -> list:
        """Get and clear deferred triggers."""
        triggers = cls._load_deferred_triggers()

        if not triggers:
            return []

        # Clear file
        try:
            with open(cls.DEFERRED_TRIGGERS_FILE, 'w') as f:
                json.dump([], f)
        except Exception as e:
            logger.error(f"Failed to clear deferred triggers: {e}")

        return triggers

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
