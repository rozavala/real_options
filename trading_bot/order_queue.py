import asyncio
import logging

logger = logging.getLogger(__name__)

class OrderQueueManager:
    """
    Thread-safe asynchronous queue manager for trading orders.
    Ensures atomic operations when adding or retrieving orders.
    """
    def __init__(self):
        self._queue = []
        self._lock = asyncio.Lock()

    async def add(self, item):
        """Add an item to the queue securely."""
        async with self._lock:
            self._queue.append(item)

    async def clear(self):
        """Clear the queue securely."""
        async with self._lock:
            self._queue.clear()

    async def pop_all(self):
        """Atomically retrieve all items and clear the queue."""
        async with self._lock:
            items = list(self._queue)
            self._queue.clear()
            return items

    def __len__(self):
        """Return current length (not thread-safe, for info only)."""
        return len(self._queue)

    def is_empty(self):
        """Check if queue is empty (not thread-safe, for info only)."""
        return len(self._queue) == 0
