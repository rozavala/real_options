"""
Self-Healing Infrastructure — Automated recovery from common failures.

Runs as a background task in the orchestrator, checking for known
failure patterns every 5 minutes. ALL checks are fail-safe.

NOTE: Log rotation is NOT handled here. Use RotatingFileHandler in
logging_config.py or OS-level logrotate. See Amendment Log.
"""

import logging
import asyncio
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class SelfHealingMonitor:
    """Background monitor for common failure detection and recovery."""

    CHECK_INTERVAL = 300  # 5 minutes

    def __init__(self, config: dict):
        self.config = config
        self._running = False
        self._recovery_count = 0
        self._last_check = None

    async def run(self):
        """Main loop. Run as asyncio.create_task() in orchestrator."""
        self._running = True
        logger.info("Self-Healing Monitor started")
        while self._running:
            try:
                await self._run_checks()
                self._last_check = datetime.now(timezone.utc)
            except Exception as e:
                logger.error(f"Self-healing check failed: {e}")
            await asyncio.sleep(self.CHECK_INTERVAL)

    def stop(self):
        self._running = False

    async def _run_checks(self):
        await self._check_state_freshness()
        self._check_disk_space()
        self._check_memory_usage()
        self._check_stale_deferred_triggers()
        # NOTE: _check_log_rotation intentionally OMITTED (see Amendment Log)

    async def _check_state_freshness(self):
        """Alert if state.json stale during trading hours."""
        try:
            from trading_bot.utils import is_market_open
        except ImportError:
            return

        if not is_market_open():
            return

        state_file = Path("data/state.json")
        if state_file.exists():
            mtime = datetime.fromtimestamp(
                state_file.stat().st_mtime, tz=timezone.utc
            )
            age = datetime.now(timezone.utc) - mtime
            if age > timedelta(hours=2):
                logger.warning(
                    f"SELF-HEAL: state.json is {age.total_seconds()/3600:.1f}h old. "
                    f"Agents may be using stale data."
                )

    def _check_disk_space(self):
        try:
            usage = shutil.disk_usage("/")
            free_gb = usage.free / (1024**3)
            if free_gb < 1.0:
                logger.critical(f"SELF-HEAL: Disk critically low: {free_gb:.1f}GB free")
            elif free_gb < 5.0:
                logger.warning(f"SELF-HEAL: Disk low: {free_gb:.1f}GB free")
        except Exception as e:
            logger.warning(f"Disk check failed: {e}")

    def _check_memory_usage(self):
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                logger.critical(f"SELF-HEAL: Memory critical: {mem.percent}%")
            elif mem.percent > 80:
                logger.warning(f"SELF-HEAL: Memory high: {mem.percent}%")
        except ImportError:
            pass  # psutil not installed — skip gracefully
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")

    def _check_stale_deferred_triggers(self):
        try:
            from trading_bot.state_manager import StateManager
            deferred = StateManager.get_deferred_triggers()
            if len(deferred) > 10:
                logger.warning(
                    f"SELF-HEAL: {len(deferred)} deferred triggers accumulated. "
                    f"Check process_deferred_triggers."
                )
        except Exception:
            pass

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "recovery_count": self._recovery_count,
        }
