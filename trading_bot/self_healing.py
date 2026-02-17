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
import subprocess  # NEW: for TCP zombie and orphan process detection
import os
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
        self._start_time = datetime.now(timezone.utc)

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
        """Run all health checks. Subprocess calls are non-blocking."""
        await self._check_state_freshness()
        self._check_disk_space()
        self._check_memory_usage()
        self._check_stale_deferred_triggers()
        self._check_log_writability()       # NEW
        await self._check_tcp_zombies()     # NEW — async (uses executor)
        await self._check_orphan_processes()  # NEW — async (uses executor)

    async def _check_state_freshness(self):
        """Alert if state.json stale during trading hours."""
        try:
            from trading_bot.utils import is_market_open
        except ImportError:
            return

        if not is_market_open(self.config):
            return

        state_file = Path("data/state.json")
        if state_file.exists():
            mtime = datetime.fromtimestamp(
                state_file.stat().st_mtime, tz=timezone.utc
            )
            age = datetime.now(timezone.utc) - mtime
            if age > timedelta(hours=4):
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

    def _check_log_writability(self):
        """Verify orchestrator.log has an active handler in the logging system.

        Uses a grace period to avoid false positives during initialization.
        """
        # Skip check during first 60 seconds to avoid initialization race conditions
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        if uptime < 60:
            return

        try:
            # Check if the root logger has any file handlers for orchestrator.log
            root_logger = logging.getLogger()
            has_file_handler = False

            for handler in root_logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    # Check if this handler's filename matches our log file
                    if hasattr(handler, 'baseFilename'):
                        if 'orchestrator.log' in handler.baseFilename:
                            has_file_handler = True
                            break

            if not has_file_handler:
                # Only warn if we're past initialization and the file exists
                log_file = Path("logs/orchestrator.log")
                if log_file.exists():
                    logger.warning(
                        "SELF-HEAL: orchestrator.log exists but no file handler attached. "
                        "Logs may only be going to stdout. "
                        "Check logging_config.py for permission issues."
                    )
                else:
                    logger.debug(
                        "SELF-HEAL: orchestrator.log not found and no file handler. "
                        "This may be expected in certain deployment modes."
                    )
        except Exception as e:
            # Fail-safe: Don't crash the monitor if the check itself fails
            logger.debug(f"Log writability check failed: {e}")

    async def _check_tcp_zombies(self):
        """Detect CLOSE_WAIT accumulation — non-blocking via executor."""
        try:
            port = self.config.get('connection', {}).get('port', 7497)
            loop = asyncio.get_running_loop()

            # AMENDMENT A: Run subprocess in thread pool to avoid blocking event loop
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ['ss', '-tn', 'state', 'close-wait'],
                    capture_output=True, text=True, timeout=5
                )
            )

            if result.returncode != 0:
                return

            close_wait_count = sum(
                1 for line in result.stdout.splitlines()
                if f':{port}' in line
            )

            if close_wait_count >= 10:
                logger.critical(
                    f"SELF-HEAL: {close_wait_count} CLOSE_WAIT connections on port {port}! "
                    f"Connection leak detected. Requesting pool cleanup."
                )
                self._recovery_count += 1
                await self._attempt_connection_cleanup()
            elif close_wait_count >= 5:
                logger.warning(
                    f"SELF-HEAL: {close_wait_count} CLOSE_WAIT connections on port {port}. "
                    f"Monitoring — threshold is 10."
                )
        except FileNotFoundError:
            logger.debug("SELF-HEAL: 'ss' command not found. Skipping TCP zombie check.")
        except Exception as e:
            logger.warning(f"SELF-HEAL: TCP zombie check failed: {e}")

    async def _attempt_connection_cleanup(self):
        """Request connection pool to release all connections."""
        try:
            from trading_bot.connection_pool import IBConnectionPool
            await IBConnectionPool.release_all()
            logger.info("SELF-HEAL: Connection pool release scheduled")
        except Exception as e:
            logger.warning(f"SELF-HEAL: Connection cleanup failed: {e}")

    async def _check_orphan_processes(self):
        """Detect orphaned Python trading processes — non-blocking via executor."""
        try:
            loop = asyncio.get_running_loop()

            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ['pgrep', '-af', 'python.*real_options'],
                    capture_output=True, text=True, timeout=5
                )
            )

            if result.returncode != 0:
                return  # No matches or command failed

            processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]

            if len(processes) > 5:
                logger.warning(
                    f"SELF-HEAL: {len(processes)} Python/real_options processes "
                    f"detected (expected ~2-3). Possible orphans."
                )
        except FileNotFoundError:
            logger.debug("SELF-HEAL: 'pgrep' command not found. Skipping orphan check.")
        except Exception as e:
            logger.debug(f"SELF-HEAL: Orphan process check failed: {e}")

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "recovery_count": self._recovery_count,
        }
