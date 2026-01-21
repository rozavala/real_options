"""
Global rate limiter for API providers.
Prevents quota exhaustion through centralized throttling.
"""
import asyncio
import time
import logging
from typing import Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


class TokenBucketLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, rate: int, burst: int = None):
        self.rate = rate
        self.burst = burst if burst is not None else max(1, rate // 10)
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
        self._request_times: deque = deque(maxlen=100)
        self._waiters = 0

    async def acquire(self, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        self._waiters += 1

        try:
            while time.monotonic() < deadline:
                async with self._lock:
                    self._refill()

                    if self.tokens >= 1:
                        self.tokens -= 1
                        self._request_times.append(time.monotonic())
                        logger.debug(
                            f"Rate limiter: Token acquired. "
                            f"Remaining: {self.tokens:.1f}, Queued: {self._waiters - 1}"
                        )
                        return True

                wait_time = 60.0 / self.rate
                remaining = deadline - time.monotonic()

                if self._waiters > 5:
                    logger.info(
                        f"Rate limiter: {self._waiters} requests queued, "
                        f"waiting {wait_time:.1f}s for next slot"
                    )

                await asyncio.sleep(min(wait_time, max(0.1, remaining)))

            logger.warning(f"Rate limiter: Timeout after {timeout}s")
            return False

        finally:
            self._waiters -= 1

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_update
        new_tokens = elapsed * (self.rate / 60.0)
        self.tokens = min(self.burst, self.tokens + new_tokens)
        self.last_update = now

    def get_current_rpm(self) -> float:
        now = time.monotonic()
        minute_ago = now - 60.0
        recent = [t for t in self._request_times if t > minute_ago]
        return len(recent)

    def get_status(self) -> dict:
        return {
            'tokens_available': round(self.tokens, 2),
            'current_rpm': self.get_current_rpm(),
            'queued_requests': self._waiters,
            'rate_limit': self.rate
        }


class GlobalRateLimiter:
    """Singleton rate limiter for all API providers."""
    _instance: Optional['GlobalRateLimiter'] = None
    _limiters: Dict[str, TokenBucketLimiter] = {}
    _initialized = False

    PROVIDER_LIMITS = {
        'gemini': 20,
        'openai': 400,
        'anthropic': 80,
        'xai': 25,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            for provider, rpm in cls.PROVIDER_LIMITS.items():
                cls._limiters[provider] = TokenBucketLimiter(rate=rpm, burst=3)
            cls._initialized = True
            logger.info(f"GlobalRateLimiter initialized: {list(cls.PROVIDER_LIMITS.keys())}")

    @classmethod
    async def acquire(cls, provider: str, timeout: float = 30.0) -> bool:
        cls._ensure_initialized()
        provider = provider.lower()

        if provider not in cls._limiters:
            logger.debug(f"Unknown provider '{provider}', allowing request")
            return True

        limiter = cls._limiters[provider]
        acquired = await limiter.acquire(timeout=timeout)

        if not acquired:
            logger.warning(f"Rate limit timeout for {provider} after {timeout}s")

        return acquired

    @classmethod
    def get_status(cls) -> dict:
        cls._ensure_initialized()
        return {
            provider: limiter.get_status()
            for provider, limiter in cls._limiters.items()
        }


async def acquire_api_slot(provider: str, timeout: float = 30.0) -> bool:
    """Acquire an API slot for the given provider."""
    return await GlobalRateLimiter.acquire(provider, timeout)
