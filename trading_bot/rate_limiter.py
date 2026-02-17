"""
Global rate limiter for API providers.
Prevents quota exhaustion through centralized throttling.

Environment-aware: DEV uses conservative limits (lower burst, testing quotas),
PROD uses production quotas aligned with API provider billing tiers.
"""
import asyncio
import time
import logging
import os
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

    # Default limits - conservative for DEV testing
    DEFAULT_LIMITS_DEV = {
        'gemini': 30,      # Increased from 20 to handle multiple sentinels + analysts
        'openai': 400,
        'anthropic': 80,
        'xai': 25,
    }

    # Production limits - aligned with paid API tiers
    DEFAULT_LIMITS_PROD = {
        'gemini': 60,      # Higher for production workload with 5 cycles/day
        'openai': 500,
        'anthropic': 100,
        'xai': 40,
    }

    PROVIDER_LIMITS = DEFAULT_LIMITS_DEV  # Will be overridden during initialization

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _ensure_initialized(cls, config: Optional[dict] = None):
        """Initialize rate limiters with environment-aware settings."""
        if not cls._initialized:
            # Determine environment
            env_name = os.getenv("ENV_NAME", "DEV").upper()

            # Load config overrides if provided
            if config and 'rate_limits' in config:
                rate_config = config['rate_limits']
                env_limits = rate_config.get('env_limits', {}).get(env_name, {})

                # Merge environment-specific overrides with defaults
                base_limits = cls.DEFAULT_LIMITS_PROD if env_name == "PROD" else cls.DEFAULT_LIMITS_DEV
                cls.PROVIDER_LIMITS = {**base_limits, **env_limits}
            else:
                # No config provided, use defaults based on environment
                cls.PROVIDER_LIMITS = cls.DEFAULT_LIMITS_PROD if env_name == "PROD" else cls.DEFAULT_LIMITS_DEV

            # Create limiters with dynamic burst sizing
            for provider, rpm in cls.PROVIDER_LIMITS.items():
                # Burst = 10% of rate, minimum 3, maximum 10
                burst = max(3, min(10, rpm // 10))
                cls._limiters[provider] = TokenBucketLimiter(rate=rpm, burst=burst)

            cls._initialized = True
            logger.info(
                f"GlobalRateLimiter initialized for {env_name}: "
                f"{[(p, f'{rpm} RPM') for p, rpm in cls.PROVIDER_LIMITS.items()]}"
            )

    @classmethod
    async def acquire(cls, provider: str, timeout: float = 30.0, config: Optional[dict] = None) -> bool:
        cls._ensure_initialized(config)
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
    def get_status(cls, config: Optional[dict] = None) -> dict:
        cls._ensure_initialized(config)
        return {
            provider: limiter.get_status()
            for provider, limiter in cls._limiters.items()
        }


async def acquire_api_slot(provider: str, timeout: float = 30.0, config: Optional[dict] = None) -> bool:
    """Acquire an API slot for the given provider.

    Args:
        provider: API provider name (gemini, openai, anthropic, xai)
        timeout: Maximum wait time in seconds
        config: Optional config dict for rate limit overrides

    Returns:
        True if slot acquired, False if timeout
    """
    return await GlobalRateLimiter.acquire(provider, timeout, config)
