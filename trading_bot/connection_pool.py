import asyncio
import random
from typing import Dict, Optional
from datetime import datetime, timezone
from ib_insync import IB
import logging

logger = logging.getLogger(__name__)


# Compressed client ID range for remote/dev Gateway connections (10-79).
# Avoids collision with production IDs (100-279) when sharing a Gateway.
DEV_CLIENT_ID_BASE = {
    "main": 10,
    "sentinel": 15,
    "emergency": 20,
    "monitor": 25,
    "orchestrator_orders": 30,
    "dashboard_orders": 35,
    "dashboard_close": 40,
    "microstructure": 45,
    "test_utilities": 50,
    "audit": 55,
    "equity_logger": 60,
    "reconciliation": 65,
    "cleanup": 70,
    "drawdown_check": 75,
}
DEV_CLIENT_ID_JITTER = 4   # random(0, 4) for dev; prod uses random(0, 9)
DEV_CLIENT_ID_DEFAULT = 80  # Unknown purposes in dev: 80-84


def _is_remote_gateway(config: dict) -> bool:
    """Check if the configured IB Gateway host is remote (not localhost)."""
    host = config.get('connection', {}).get('host', '127.0.0.1')
    return host not in ('127.0.0.1', 'localhost', '::1')


class IBConnectionPool:
    _instances: Dict[str, IB] = {}
    _locks: Dict[str, asyncio.Lock] = {}
    _reconnect_backoff: Dict[str, float] = {}
    _last_reconnect_attempt: Dict[str, float] = {}

    # Track connection health
    _consecutive_failures: Dict[str, int] = {}
    FORCE_RESET_THRESHOLD = 5  # Force reset after 5 consecutive failures

    # Base client IDs - will add random offset on reconnect
    CLIENT_ID_BASE = {
        "main": 100,
        "sentinel": 110,
        "emergency": 120,
        "monitor": 130,
        "orchestrator_orders": 140,    # Orchestrator's order-related tasks
        "dashboard_orders": 160,       # Dashboard's order generation
        "dashboard_close": 180,        # Dashboard's manual position close
        "microstructure": 200,    # Range: 200-209
        "test_utilities": 220,    # Range: 220-229 (Dashboard IB test button)
        "audit": 230,             # Range: 230-239 (Position audit cycle)
        "drawdown_check": 240,    # Range: 240-249
        "equity_logger": 250,     # Range: 250-259
        "reconciliation": 260,    # Range: 260-269
        "cleanup": 270,           # Range: 270-279
        # Note: position_monitor.py uses 300-399
        # Note: reconciliation uses random 5000-9999
        # DEFAULT for unknown purposes: 280-289 (avoid adding new purposes without explicit ID)
    }

    MAX_RECONNECT_BACKOFF = 600  # 10 minutes
    DISCONNECT_SETTLE_TIME = 3.0  # Seconds to wait after disconnect (allows Gateway cleanup)

    @classmethod
    async def _force_reset_connection(cls, purpose: str):
        """Force reset a specific connection - nuclear option."""
        logger.warning(f"FORCE RESETTING connection: {purpose}")

        # Remove from instances
        old_ib = cls._instances.pop(purpose, None)
        if old_ib:
            try:
                old_ib.disconnect()
                # === NEW: Give Gateway time to cleanup ===
                await asyncio.sleep(3.0)
            except Exception:
                pass

        # Reset backoff
        cls._reconnect_backoff[purpose] = 5.0
        cls._last_reconnect_attempt[purpose] = 0
        cls._consecutive_failures[purpose] = 0

        # Note: We don't reset the lock - that could cause issues

    @classmethod
    async def get_connection(cls, purpose: str, config: dict) -> IB:
        if purpose not in cls._locks:
            cls._locks[purpose] = asyncio.Lock()
            cls._reconnect_backoff[purpose] = 5.0
            cls._last_reconnect_attempt[purpose] = 0
            cls._consecutive_failures[purpose] = 0

        async with cls._locks[purpose]:
            ib = cls._instances.get(purpose)

            # Check if existing connection is alive
            if ib and ib.isConnected():
                # === ZOMBIE DETECTION: Verify connection is actually alive ===
                try:
                    await asyncio.wait_for(ib.reqCurrentTimeAsync(), timeout=5.0)  # Increased from 2.0
                    return ib
                except asyncio.TimeoutError:
                    logger.warning(f"IB ({purpose}) zombie connection detected: reqCurrentTime timed out after 5s")
                    # Force proper cleanup before reconnecting
                    try:
                        ib.disconnect()
                        await asyncio.sleep(3.0)  # Give Gateway time to cleanup
                    except Exception:
                        pass
                    cls._instances.pop(purpose, None)
                    # Fall through to reconnect
                except Exception as e:
                    logger.warning(f"IB ({purpose}) connection health check failed: {e}")
                    # Force proper cleanup before reconnecting
                    try:
                        ib.disconnect()
                        await asyncio.sleep(3.0)
                    except Exception:
                        pass
                    cls._instances.pop(purpose, None)
                    # Fall through to reconnect

            # === BACKOFF CHECK ===
            import time
            now = time.time()
            time_since_last = now - cls._last_reconnect_attempt.get(purpose, 0)
            backoff = cls._reconnect_backoff.get(purpose, 5.0)

            if time_since_last < backoff:
                wait_remaining = backoff - time_since_last
                logger.debug(f"IB ({purpose}) in backoff, {wait_remaining:.0f}s remaining")
                raise ConnectionError(f"Connection {purpose} in backoff period")

            cls._last_reconnect_attempt[purpose] = now

            # === DIRECT CONNECTION (No TCP Pre-Flight) ===
            host = config.get('connection', {}).get('host', '127.0.0.1')
            port = config.get('connection', {}).get('port', 7497)

            # === CLEAN DISCONNECT ===
            if ib:
                try:
                    logger.info(f"Disconnecting stale IB ({purpose}) before reconnect...")
                    ib.disconnect()
                    await asyncio.sleep(cls.DISCONNECT_SETTLE_TIME)
                except Exception as e:
                    logger.warning(f"Disconnect error for {purpose}: {e}")

            # === CONNECT WITH RANDOMIZED CLIENT ID ===
            ib = IB()
            if _is_remote_gateway(config):
                base_id = DEV_CLIENT_ID_BASE.get(purpose, DEV_CLIENT_ID_DEFAULT)
                client_id = base_id + random.randint(0, DEV_CLIENT_ID_JITTER)
            else:
                base_id = cls.CLIENT_ID_BASE.get(purpose, 280)
                client_id = base_id + random.randint(0, 9)

            is_paper = config.get('connection', {}).get('paper', False)
            remote_tag = (" [REMOTE/PAPER]" if is_paper else " [REMOTE]") if _is_remote_gateway(config) else ""
            logger.info(f"Connecting IB ({purpose}) ID: {client_id}{remote_tag}...")

            try:
                await ib.connectAsync(
                    host=host,
                    port=port,
                    clientId=client_id,
                    timeout=15  # Increased from 10 to match verify_system_readiness
                )
                cls._instances[purpose] = ib
                cls._reconnect_backoff[purpose] = 5.0
                cls._consecutive_failures[purpose] = 0  # Reset on success
                logger.info(f"IB ({purpose}) connected successfully with ID {client_id}")

                # Log to state
                try:
                    from trading_bot.state_manager import StateManager
                    StateManager.save_state({
                        f"{purpose}_ib_status": "CONNECTED",
                        "last_ib_success": datetime.now(timezone.utc).isoformat()
                    }, namespace="sensors")
                except Exception:
                    pass

                return ib

            except Exception as e:
                # === FIX: Clean up failed connection to prevent CLOSE-WAIT accumulation ===
                # The IB() instance may have opened a TCP socket even if the API handshake
                # failed (TimeoutError). Without explicit disconnect(), the socket enters
                # CLOSE-WAIT state and accumulates over repeated retry attempts.
                try:
                    logger.debug(f"Cleaning up failed connection attempt for {purpose}")
                    ib.disconnect()
                    await asyncio.sleep(0.5)  # Brief settle for socket cleanup
                except Exception as cleanup_err:
                    logger.debug(f"Cleanup disconnect for {purpose} raised: {cleanup_err}")
                    pass  # Ignore cleanup errors - socket may not have been fully established

                cls._reconnect_backoff[purpose] = min(
                    cls._reconnect_backoff[purpose] * 2,
                    cls.MAX_RECONNECT_BACKOFF
                )
                cls._consecutive_failures[purpose] = cls._consecutive_failures.get(purpose, 0) + 1

                # === ALERTING LOGIC ===
                # Alert exactly once when threshold is reached
                if cls._consecutive_failures[purpose] == cls.FORCE_RESET_THRESHOLD:
                    try:
                        from notifications import send_pushover_notification
                        send_pushover_notification(
                            config.get('notifications', {}),
                            "🚨 IB GATEWAY ZOMBIE DETECTED",
                            f"Gateway accepting TCP but failing API handshake "
                            f"{cls.FORCE_RESET_THRESHOLD} times in a row. "
                            f"Manual restart may be required."
                        )
                    except Exception:
                        pass

                if cls._consecutive_failures.get(purpose, 0) >= cls.FORCE_RESET_THRESHOLD:
                    await cls._force_reset_connection(purpose)

                # === NEW: Log DISCONNECTED status ===
                try:
                    from trading_bot.state_manager import StateManager
                    StateManager.save_state({
                        f"{purpose}_ib_status": "DISCONNECTED",
                        f"{purpose}_last_error": str(e)[:100],
                        "last_ib_failure": datetime.now(timezone.utc).isoformat()
                    }, namespace="sensors")
                except Exception:
                    pass

                logger.error(f"IB ({purpose}) connection failed: {e}. "
                           f"Next retry backoff: {cls._reconnect_backoff[purpose]}s")
                raise

    @classmethod
    async def release_connection(cls, purpose: str):
        """Releases a specific connection from the pool."""
        if purpose not in cls._locks:
            cls._locks[purpose] = asyncio.Lock()

        async with cls._locks[purpose]:
            ib = cls._instances.pop(purpose, None)
            if ib:
                if ib.isConnected():
                    logger.info(f"Disconnecting IB ({purpose})...")
                    ib.disconnect()
                    # === NEW: Give Gateway time to cleanup ===
                    await asyncio.sleep(3.0)
                # Reset backoff when explicitly released
                cls._reconnect_backoff[purpose] = 5.0

    @classmethod
    async def release_all(cls):
        """Release all pooled connections with proper cleanup."""
        for name, ib in list(cls._instances.items()):
            try:
                if ib and ib.isConnected():
                    # Disconnect synchronously to avoid async cleanup issues
                    ib.disconnect()
                    await asyncio.sleep(0.1)  # Small delay for transport cleanup
            except Exception as e:
                logger.warning(f"Connection cleanup failed for {name}: {e}")
            finally:
                cls._instances.pop(name, None)

        cls._instances.clear()
        cls._reconnect_backoff.clear()

        # Force garbage collection to clean up any orphaned references
        import gc
        gc.collect()
