import json
import os
import logging
import asyncio
import time
from datetime import datetime
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
    async def load_state_async(cls, namespace: str = "reports", max_age: int = None) -> Dict:
        """Async load wrapper."""
        async with cls._async_lock:
             loop = asyncio.get_running_loop()
             return await loop.run_in_executor(None, cls.load_state, namespace, max_age)
