import asyncio
import random
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

    # Base client IDs - will add random offset on reconnect
    CLIENT_ID_BASE = {
        "main": 100,
        "sentinel": 110,
        "emergency": 120,
        "monitor": 130,
        "order_manager": 140,
        "microstructure": 200
    }

    MAX_RECONNECT_BACKOFF = 600  # 10 minutes
    DISCONNECT_SETTLE_TIME = 2.0  # Seconds to wait after disconnect

    @classmethod
    async def get_connection(cls, purpose: str, config: dict) -> IB:
        if purpose not in cls._locks:
            cls._locks[purpose] = asyncio.Lock()
            cls._reconnect_backoff[purpose] = 5.0  # Initial backoff
            cls._last_reconnect_attempt[purpose] = 0

        async with cls._locks[purpose]:
            ib = cls._instances.get(purpose)

            # Check if existing connection is alive
            if ib and ib.isConnected():
                return ib

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

            # === CLEAN DISCONNECT ===
            if ib:
                try:
                    logger.info(f"Disconnecting stale IB ({purpose}) before reconnect...")
                    ib.disconnect()
                    # CRITICAL: Wait for IB Gateway to release the client ID
                    await asyncio.sleep(cls.DISCONNECT_SETTLE_TIME)
                except Exception as e:
                    logger.warning(f"Disconnect error for {purpose}: {e}")

            # === CONNECT WITH RANDOMIZED CLIENT ID ===
            ib = IB()
            base_id = cls.CLIENT_ID_BASE.get(purpose, 200)
            # Add random offset to avoid collision with lingering connections
            client_id = base_id + random.randint(0, 9)

            host = config.get('connection', {}).get('host', '127.0.0.1')
            port = config.get('connection', {}).get('port', 7497)

            logger.info(f"Connecting IB ({purpose}) ID: {client_id}...")

            try:
                await ib.connectAsync(
                    host=host,
                    port=port,
                    clientId=client_id,
                    timeout=30
                )
                cls._instances[purpose] = ib
                # Reset backoff on success
                cls._reconnect_backoff[purpose] = 5.0
                logger.info(f"IB ({purpose}) connected successfully with ID {client_id}")

                # === LOG TO STATE FOR DASHBOARD ===
                try:
                    from trading_bot.state_manager import StateManager
                    StateManager.save_state({
                        f"{purpose}_ib_status": "CONNECTED",
                        "last_ib_success": datetime.now(timezone.utc).isoformat()
                    }, namespace="sensors")
                except Exception:
                    pass  # Don't fail connection on state logging error

                return ib

            except Exception as e:
                # Increase backoff on failure (exponential)
                cls._reconnect_backoff[purpose] = min(
                    cls._reconnect_backoff[purpose] * 2,
                    cls.MAX_RECONNECT_BACKOFF
                )
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
