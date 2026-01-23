import asyncio
import random
import socket
from typing import Dict, Optional
from datetime import datetime, timezone
from ib_insync import IB
import logging

logger = logging.getLogger(__name__)


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
        "order_manager": 140,
        "microstructure": 200,
        # Note: position_monitor.py uses 300-399
        # Note: reconciliation uses random 5000-9999
    }

    MAX_RECONNECT_BACKOFF = 600  # 10 minutes
    DISCONNECT_SETTLE_TIME = 2.0  # Seconds to wait after disconnect

    @classmethod
    async def _check_gateway_health(cls, host: str, port: int, timeout: float = 5.0) -> bool:
        """Quick TCP check if Gateway is accepting connections."""
        try:
            loop = asyncio.get_running_loop()
            # Run socket check in executor to avoid blocking
            def _tcp_check():
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                try:
                    result = sock.connect_ex((host, port))
                    return result == 0
                finally:
                    sock.close()

            return await loop.run_in_executor(None, _tcp_check)
        except Exception as e:
            logger.warning(f"Gateway health check failed: {e}")
            return False

    @classmethod
    async def _force_reset_connection(cls, purpose: str):
        """Force reset a specific connection - nuclear option."""
        logger.warning(f"FORCE RESETTING connection: {purpose}")

        # Remove from instances
        old_ib = cls._instances.pop(purpose, None)
        if old_ib:
            try:
                old_ib.disconnect()
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
                # === NEW: Verify connection is actually alive (not zombie) ===
                # AMENDMENT from Mission Control: Use reqCurrentTimeAsync with strict timeout
                # to prevent hanging if Gateway is unresponsive (TCP open, app frozen)
                try:
                    await asyncio.wait_for(ib.reqCurrentTimeAsync(), timeout=2.0)
                    return ib
                except asyncio.TimeoutError:
                    logger.warning(f"IB ({purpose}) zombie connection detected: reqCurrentTime timed out")
                    # Fall through to reconnect
                except Exception as e:
                    logger.warning(f"IB ({purpose}) zombie connection detected: {e}")
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

            # === GATEWAY HEALTH CHECK (NEW) ===
            host = config.get('connection', {}).get('host', '127.0.0.1')
            port = config.get('connection', {}).get('port', 7497)

            if await cls._check_gateway_health(host, port):
                # Gateway is responding! Reset backoff if it was high
                if cls._reconnect_backoff.get(purpose, 5.0) > 60:
                    logger.info(f"Gateway healthy - resetting backoff for {purpose}")
                    cls._reconnect_backoff[purpose] = 5.0
            else:
                # Gateway not responding - don't increase backoff for Gateway-level issues
                logger.error(f"IB Gateway at {host}:{port} not accepting connections")
                cls._consecutive_failures[purpose] = cls._consecutive_failures.get(purpose, 0) + 1

                # Check if we should force reset
                if cls._consecutive_failures.get(purpose, 0) >= cls.FORCE_RESET_THRESHOLD:
                    await cls._force_reset_connection(purpose)

                raise ConnectionError(f"IB Gateway unavailable at {host}:{port}")

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
            base_id = cls.CLIENT_ID_BASE.get(purpose, 200)
            client_id = base_id + random.randint(0, 9)

            logger.info(f"Connecting IB ({purpose}) ID: {client_id}...")

            try:
                await ib.connectAsync(
                    host=host,
                    port=port,
                    clientId=client_id,
                    timeout=30
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
                cls._reconnect_backoff[purpose] = min(
                    cls._reconnect_backoff[purpose] * 2,
                    cls.MAX_RECONNECT_BACKOFF
                )
                cls._consecutive_failures[purpose] = cls._consecutive_failures.get(purpose, 0) + 1

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
                # Reset backoff when explicitly released
                cls._reconnect_backoff[purpose] = 5.0

    @classmethod
    async def release_all(cls):
        for name, ib in list(cls._instances.items()):
            if ib and ib.isConnected():
                logger.info(f"Disconnecting IB ({name})...")
                ib.disconnect()
        cls._instances.clear()
        cls._reconnect_backoff.clear()
