import asyncio
from typing import Dict, Optional
from ib_insync import IB
import logging

logger = logging.getLogger(__name__)

class IBConnectionPool:
    _instances: Dict[str, IB] = {}
    _locks: Dict[str, asyncio.Lock] = {}

    CLIENT_ID_MAP = {
        "main": 100,
        "sentinel": 101,
        "emergency": 102,
        "monitor": 103,
        "order_manager": 104
    }

    @classmethod
    async def get_connection(cls, purpose: str, config: dict) -> IB:
        if purpose not in cls._locks:
            cls._locks[purpose] = asyncio.Lock()

        async with cls._locks[purpose]:
            ib = cls._instances.get(purpose)

            # Check if existing connection is alive
            if ib and ib.isConnected():
                return ib

            # If not connected, clean up if needed and connect
            if ib:
                try:
                    ib.disconnect()
                except:
                    pass

            ib = IB()
            client_id = cls.CLIENT_ID_MAP.get(purpose, 200)

            host = config.get('connection', {}).get('host', '127.0.0.1')
            port = config.get('connection', {}).get('port', 7497)

            logger.info(f"Connecting IB ({purpose}) ID: {client_id}...")

            await ib.connectAsync(
                host=host,
                port=port,
                clientId=client_id,
                timeout=30
            )

            cls._instances[purpose] = ib
            return ib

    @classmethod
    async def release_connection(cls, purpose: str):
        """Releases a specific connection from the pool."""
        async with cls._locks.get(purpose, asyncio.Lock()):
            ib = cls._instances.pop(purpose, None)
            if ib:
                if ib.isConnected():
                    logger.info(f"Disconnecting IB ({purpose})...")
                    ib.disconnect()

    @classmethod
    async def release_all(cls):
        for name, ib in cls._instances.items():
            if ib.isConnected():
                logger.info(f"Disconnecting IB ({name})...")
                ib.disconnect()
        cls._instances.clear()
