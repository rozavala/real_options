"""Heterogeneous LLM Router for Multi-Model Ensemble.

Implements diverse model allocation based on the architecture spec:
- Gemini 1.5 Pro → Fundamental/Agronomy Analyst (large context, research)
- GPT-4o → Portfolio Manager/Strategist (complex reasoning)
- Claude 3.5 Sonnet → Risk Manager/Compliance (Constitutional AI)
- Gemini Flash → Tier 1 Sentinels (cost-efficient monitoring)

This prevents "Algorithmic Monoculture" and the "Abilene Paradox".
"""

import os
import json
import logging
import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from functools import wraps
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass
import numpy as np
from trading_bot.router_metrics import get_router_metrics
from trading_bot.rate_limiter import acquire_api_slot
from trading_bot.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


class CriticalRPCError(RuntimeError):
    """Raised when all LLM providers fail for a critical role."""
    pass


class ModelProvider(Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"


class AgentRole(Enum):
    """Agent roles in the hierarchy."""
    # Tier 1 - Sentinels
    WEATHER_SENTINEL = "weather_sentinel"
    LOGISTICS_SENTINEL = "logistics_sentinel"
    NEWS_SENTINEL = "news_sentinel"
    PRICE_SENTINEL = "price_sentinel"
    MICROSTRUCTURE_SENTINEL = "microstructure_sentinel"

    # Tier 2 - Analysts
    AGRONOMIST = "agronomist"
    MACRO_ANALYST = "macro"
    GEOPOLITICAL_ANALYST = "geopolitical"
    SENTIMENT_ANALYST = "sentiment"
    TECHNICAL_ANALYST = "technical"
    VOLATILITY_ANALYST = "volatility"
    INVENTORY_ANALYST = "inventory"
    SUPPLY_CHAIN_ANALYST = "supply_chain"

    # Tier 3 - Decision Makers
    PERMABEAR = "permabear"
    PERMABULL = "permabull"
    MASTER_STRATEGIST = "master"
    COMPLIANCE_OFFICER = "compliance"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: float = 60.0 # Explicit default 60s to prevent 3s truncations


class ResponseCache:
    """In-memory cache for LLM responses with role-based TTL."""

    # Default TTLs (can be overridden by config)
    DEFAULT_ROLE_TTL = {
        AgentRole.AGRONOMIST.value: 1800,
        AgentRole.MACRO_ANALYST.value: 1800,
        AgentRole.GEOPOLITICAL_ANALYST.value: 1800,
        AgentRole.INVENTORY_ANALYST.value: 1800,
        AgentRole.SUPPLY_CHAIN_ANALYST.value: 1800,
        AgentRole.VOLATILITY_ANALYST.value: 900,
        AgentRole.TECHNICAL_ANALYST.value: 900,
        AgentRole.SENTIMENT_ANALYST.value: 600,
        AgentRole.PERMABEAR.value: 300,
        AgentRole.PERMABULL.value: 300,
        AgentRole.MASTER_STRATEGIST.value: 300,
        AgentRole.COMPLIANCE_OFFICER.value: 300,
    }

    DEFAULT_TTL = 300  # 5 minutes fallback

    def __init__(self, config: dict = None):
        self.cache = {}
        self.config = config or {}

        # Load environment-specific TTL overrides
        env_name = os.getenv("ENV_NAME", "DEV")
        ttl_config = self.config.get('cache_ttl', {})

        # Environment multiplier (DEV = 0.25x, PROD = 1.0x)
        # Note: 'DEV' usually implies testing, so shorter TTLs are better for iteration.
        # But if 'DEV' means "Deployment", maybe cost saving?
        # Requirement said: "DEV environments can use shorter TTLs for testing and PROD uses longer TTLs for cost efficiency."
        # Actually usually PROD caches longer for cost efficiency. DEV might want NO cache for testing?
        # Or DEV wants shorter cache to see changes?
        # The prompt said: "DEV environments can use shorter TTLs for testing and PROD uses longer TTLs for cost efficiency."
        # Wait, if DEV uses shorter TTL, it refreshes MORE often -> Higher Cost.
        # Maybe "PROD uses longer TTLs for cost efficiency" implies DEV uses shorter?
        # Ah, if I am testing, I want to see results of code changes immediately, so I want SHORT cache or NO cache.
        # So DEV = Short TTL makes sense for dev experience (not cost).

        env_multiplier = ttl_config.get('env_multipliers', {}).get(env_name, 1.0)

        self.ROLE_TTL = {}
        for role, default_ttl in self.DEFAULT_ROLE_TTL.items():
            override = ttl_config.get('roles', {}).get(role, default_ttl)
            self.ROLE_TTL[role] = int(override * env_multiplier)

        logger.info(f"ResponseCache initialized with env={env_name}, multiplier={env_multiplier}")

    def _hash_key(self, prompt: str, role: str) -> str:
        return hashlib.md5(f"{role}:{prompt}".encode()).hexdigest()

    def _get_ttl(self, role: str) -> int:
        """Get TTL for a specific role."""
        return self.ROLE_TTL.get(role, self.DEFAULT_TTL)

    def get(self, prompt: str, role: str) -> Optional[str]:
        key = self._hash_key(prompt, role)
        if key in self.cache:
            entry = self.cache[key]
            ttl = self._get_ttl(role)
            if datetime.now() - entry['timestamp'] < timedelta(seconds=ttl):
                logger.debug(f"Cache HIT for {role} (TTL: {ttl}s)")
                return entry['response']
            # Expired - remove from cache
            del self.cache[key]
            logger.debug(f"Cache EXPIRED for {role}")
        return None

    def set(self, prompt: str, role: str, response: str):
        key = self._hash_key(prompt, role)
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now(),
            'role': role
        }
        logger.debug(f"Cache SET for {role} (TTL: {self._get_ttl(role)}s)")

    def get_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        now = datetime.now()
        stats = {
            'total_entries': len(self.cache),
            'entries_by_role': {},
            'expired_count': 0
        }

        for key, entry in self.cache.items():
            role = entry.get('role', 'unknown')
            ttl = self._get_ttl(role)
            is_expired = (now - entry['timestamp']) >= timedelta(seconds=ttl)

            if role not in stats['entries_by_role']:
                stats['entries_by_role'][role] = {'active': 0, 'expired': 0}

            if is_expired:
                stats['entries_by_role'][role]['expired'] += 1
                stats['expired_count'] += 1
            else:
                stats['entries_by_role'][role]['active'] += 1

        return stats


def with_retry(max_retries=3, backoff_factor=2):
    """
    Decorator to retry async methods with exponential backoff and 429 awareness.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import re
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e)

                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        retry_match = re.search(r'retry in (\d+\.?\d*)s', error_str, re.IGNORECASE)
                        if retry_match:
                            wait_time = float(retry_match.group(1)) + np.random.uniform(0.5, 2.0)
                        else:
                            wait_time = (backoff_factor ** attempt) * 10
                    else:
                        wait_time = backoff_factor ** attempt

                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt+1} failed for {func.__name__}, "
                            f"retrying in {wait_time:.1f}s: {e}"
                        )
                        await asyncio.sleep(wait_time)

            raise last_exception
        return wrapper
    return decorator




class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      response_json: bool = False) -> str:
        pass


class GeminiClient(LLMClient):
    """Google Gemini client."""

    def __init__(self, config: ModelConfig):
        from google import genai
        from google.genai import types

        self.config = config
        api_key = config.api_key or os.environ.get('GEMINI_API_KEY')
        self.client = genai.Client(api_key=api_key)
        self.types = types

    @with_retry()
    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      response_json: bool = False) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        gen_config = self.types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )
        if response_json:
            gen_config.response_mime_type = "application/json"

        # Explicit timeout (Google GenAI might not support timeout kwarg directly here,
        # but usually does via request_options or similar.
        # Standard asyncio.wait_for wrapper is safest if SDK is obscure.)
        # But let's try assuming standard kwargs or wrapping.
        try:
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.config.model_name,
                    contents=full_prompt,
                    config=gen_config
                ),
                timeout=self.config.timeout
            )
            return response.text
        except asyncio.TimeoutError:
            raise RuntimeError(f"Gemini timed out after {self.config.timeout}s")


class OpenAIClient(LLMClient):
    """OpenAI GPT client."""

    def __init__(self, config: ModelConfig):
        from openai import AsyncOpenAI
        api_key = config.api_key or os.environ.get('OPENAI_API_KEY')
        self.client = AsyncOpenAI(api_key=api_key)
        self.config = config

    @with_retry()
    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      response_json: bool = False) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "max_completion_tokens": self.config.max_tokens,
            "timeout": self.config.timeout
        }

        if response_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic Claude client."""

    def __init__(self, config: ModelConfig):
        from anthropic import AsyncAnthropic
        api_key = config.api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = AsyncAnthropic(api_key=api_key)
        self.config = config

    @with_retry()
    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      response_json: bool = False) -> str:
        kwargs = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.config.timeout
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)

        # --- FIX: Validate response integrity before access ---
        if not response.content:
            logger.error(f"Anthropic returned empty content. Stop reason: {response.stop_reason}")
            raise ValueError(f"Empty response from Anthropic (stop_reason: {response.stop_reason})")

        text = response.content[0].text
        if not text or not text.strip():
            logger.error(f"Anthropic returned empty text block. Full response: {response}")
            raise ValueError("Empty text in Anthropic response")
        # --- END FIX ---

        return text


class XAIClient(LLMClient):
    """xAI Grok client (OpenAI-compatible API)."""

    def __init__(self, config: ModelConfig):
        from openai import AsyncOpenAI
        api_key = config.api_key or os.environ.get('XAI_API_KEY')
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        self.config = config

    @with_retry()
    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      response_json: bool = False) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout
        )
        return response.choices[0].message.content


class HeterogeneousRouter:
    """Routes agent requests to appropriate LLM based on role."""

    def __init__(self, config: dict):
        self.config = config
        self.clients: dict[str, LLMClient] = {}
        self.registry = config.get('model_registry', {})

        # Extract API keys
        self.api_keys = {
            ModelProvider.GEMINI: config.get('gemini', {}).get('api_key') or os.environ.get('GEMINI_API_KEY'),
            ModelProvider.OPENAI: config.get('openai', {}).get('api_key') or os.environ.get('OPENAI_API_KEY'),
            ModelProvider.ANTHROPIC: config.get('anthropic', {}).get('api_key') or os.environ.get('ANTHROPIC_API_KEY'),
            ModelProvider.XAI: config.get('xai', {}).get('api_key') or os.environ.get('XAI_API_KEY'),
        }

        # Track available providers
        self.available_providers = set()
        for provider, key in self.api_keys.items():
            if key and key != "YOUR_API_KEY_HERE":
                self.available_providers.add(provider)

        # Initialize Cache
        self.cache = ResponseCache(config)  # Now uses role-based TTL internally
        self.semantic_cache = SemanticCache(config)

        # 1. LOAD MODEL KEYS (With Defaults)
        gem_flash = self.registry.get('gemini', {}).get('flash', 'gemini-3-flash-preview')
        gem_pro = self.registry.get('gemini', {}).get('pro', 'gemini-3-pro-preview')

        anth_pro = self.registry.get('anthropic', {}).get('pro', 'claude-opus-4-5-20251101')

        oai_pro = self.registry.get('openai', {}).get('pro', 'gpt-4o')
        oai_reasoning = self.registry.get('openai', {}).get('reasoning', 'o3-2025-04-16')

        xai_pro = self.registry.get('xai', {}).get('pro', 'grok-4-1-fast-reasoning')
        xai_flash = self.registry.get('xai', {}).get('flash', 'grok-4-fast-non-reasoning')

        # 2. ASSIGN ROLES
        self.assignments = {
            # --- TIER 1: SENTINELS (Speed is Priority) ---
            AgentRole.WEATHER_SENTINEL: (ModelProvider.GEMINI, gem_flash),
            AgentRole.LOGISTICS_SENTINEL: (ModelProvider.GEMINI, gem_flash),
            AgentRole.NEWS_SENTINEL: (ModelProvider.GEMINI, gem_flash),
            AgentRole.PRICE_SENTINEL: (ModelProvider.GEMINI, gem_flash),
            AgentRole.MICROSTRUCTURE_SENTINEL: (ModelProvider.GEMINI, gem_flash),

            # --- TIER 2: ANALYSTS (Depth & Data are Priority) ---
            AgentRole.AGRONOMIST: (ModelProvider.GEMINI, gem_pro),
            AgentRole.INVENTORY_ANALYST: (ModelProvider.GEMINI, gem_pro),
            AgentRole.VOLATILITY_ANALYST: (ModelProvider.GEMINI, gem_pro),
            AgentRole.SUPPLY_CHAIN_ANALYST: (ModelProvider.GEMINI, gem_pro), # NEW ROLE

            AgentRole.MACRO_ANALYST: (ModelProvider.OPENAI, oai_pro),
            AgentRole.GEOPOLITICAL_ANALYST: (ModelProvider.OPENAI, oai_pro),

            AgentRole.TECHNICAL_ANALYST: (ModelProvider.OPENAI, oai_reasoning), # Math/Logic
            AgentRole.SENTIMENT_ANALYST: (ModelProvider.XAI, xai_flash),

            # --- TIER 3: DECISION MAKERS (Safety & Debate) ---
            AgentRole.PERMABULL: (ModelProvider.XAI, xai_pro),             # Contrarian
            AgentRole.PERMABEAR: (ModelProvider.ANTHROPIC, anth_pro),      # Safety (Opus)
            AgentRole.COMPLIANCE_OFFICER: (ModelProvider.ANTHROPIC, anth_pro), # Veto (Opus)

            AgentRole.MASTER_STRATEGIST: (ModelProvider.OPENAI, oai_pro),  # Synthesis
        }

        logger.info(f"HeterogeneousRouter initialized. Available: {[p.value for p in self.available_providers]}")

    def _get_client(self, provider: ModelProvider, model_name: str) -> LLMClient:
        """Get or create client for provider."""
        cache_key = f"{provider.value}:{model_name}"

        if cache_key not in self.clients:
            config = ModelConfig(
                provider=provider,
                model_name=model_name,
                api_key=self.api_keys.get(provider)
            )

            if provider == ModelProvider.GEMINI:
                self.clients[cache_key] = GeminiClient(config)
            elif provider == ModelProvider.OPENAI:
                self.clients[cache_key] = OpenAIClient(config)
            elif provider == ModelProvider.ANTHROPIC:
                self.clients[cache_key] = AnthropicClient(config)
            elif provider == ModelProvider.XAI:
                self.clients[cache_key] = XAIClient(config)

        return self.clients[cache_key]

    def _get_fallback_models(self, role: AgentRole, primary_provider: ModelProvider) -> list[tuple[ModelProvider, str]]:
        """
        Returns a list of (Provider, ModelID) tuples for fallback.
        Logic: Tier 3 roles NEVER fall back to 'Flash' models.
        """

        # Helper to safely get model ID from registry or return a default
        def get_model(prov_key, tier_key, default):
            return self.registry.get(prov_key, {}).get(tier_key, default)

        # TIER 3: DECISION MAKERS (High IQ Required)
        if role in [AgentRole.MASTER_STRATEGIST, AgentRole.COMPLIANCE_OFFICER, AgentRole.PERMABEAR, AgentRole.PERMABULL]:
            # If OpenAI Pro fails, try Anthropic Opus, then Gemini Pro.
            # If Anthropic Pro fails (Primary), try OpenAI Pro, then Gemini Pro.
            fallbacks = []

            # Note: We hardcode the Pro models here to ensure quality
            anth_pro = get_model('anthropic', 'pro', 'claude-3-opus-20240229')
            oai_pro = get_model('openai', 'pro', 'gpt-4o')
            gem_pro = get_model('gemini', 'pro', 'gemini-1.5-pro-preview-0409')

            # CRITICAL: NEVER fallback to Flash models for Tier 3
            if primary_provider == ModelProvider.OPENAI:
                fallbacks.append((ModelProvider.ANTHROPIC, anth_pro))
                fallbacks.append((ModelProvider.GEMINI, gem_pro))
            elif primary_provider == ModelProvider.ANTHROPIC:
                fallbacks.append((ModelProvider.OPENAI, oai_pro))
                fallbacks.append((ModelProvider.GEMINI, gem_pro))
            elif primary_provider == ModelProvider.XAI:
                fallbacks.append((ModelProvider.OPENAI, oai_pro))
                fallbacks.append((ModelProvider.ANTHROPIC, anth_pro))
            else:
                 fallbacks.append((ModelProvider.ANTHROPIC, anth_pro))
                 fallbacks.append((ModelProvider.OPENAI, oai_pro))

            return fallbacks

        # TIER 2: DEEP ANALYSTS
        elif role in [AgentRole.AGRONOMIST, AgentRole.MACRO_ANALYST, AgentRole.SUPPLY_CHAIN_ANALYST, AgentRole.GEOPOLITICAL_ANALYST, AgentRole.INVENTORY_ANALYST, AgentRole.VOLATILITY_ANALYST]:
            gem_pro = get_model('gemini', 'pro', 'gemini-1.5-pro-preview-0409')
            oai_pro = get_model('openai', 'pro', 'gpt-4o')
            anth_pro = get_model('anthropic', 'pro', 'claude-3-opus-20240229')

            if primary_provider == ModelProvider.GEMINI:
                return [
                    (ModelProvider.OPENAI, oai_pro),
                    (ModelProvider.ANTHROPIC, anth_pro)
                ]
            else:
                return [
                    (ModelProvider.GEMINI, gem_pro),
                    (ModelProvider.ANTHROPIC, anth_pro)
                ]

        # TIER 1: SENTINELS (Speed Required)
        else:
            # If Gemini Flash fails, try OpenAI Mini/Flash, then xAI.
            oai_flash = get_model('openai', 'flash', 'gpt-4o-mini')
            xai_flash = get_model('xai', 'flash', 'grok-beta')
            gem_flash = get_model('gemini', 'flash', 'gemini-1.5-flash-preview-0514')

            fallbacks = []
            if primary_provider != ModelProvider.OPENAI:
                 fallbacks.append((ModelProvider.OPENAI, oai_flash))
            if primary_provider != ModelProvider.XAI:
                 fallbacks.append((ModelProvider.XAI, xai_flash))
            if primary_provider != ModelProvider.GEMINI:
                 fallbacks.append((ModelProvider.GEMINI, gem_flash))

            return fallbacks

    async def route(
        self,
        role: AgentRole,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_json: bool = False
    ) -> str:
        """
        Route request to appropriate model with robust fallback.

        AMENDED (Protocol 2C): Includes rate limiting before API calls.
        HOTFIX: Corrected RouterMetrics API calls.
        """
        from trading_bot.router_metrics import get_router_metrics
        import time

        metrics = get_router_metrics()

        # Check Cache (Force OFF for Sentinels)
        use_cache = True
        if role in [AgentRole.WEATHER_SENTINEL, AgentRole.LOGISTICS_SENTINEL,
                    AgentRole.NEWS_SENTINEL, AgentRole.PRICE_SENTINEL,
                    AgentRole.MICROSTRUCTURE_SENTINEL]:
            use_cache = False

        full_key_prompt = f"{system_prompt or ''}:{prompt}"

        if use_cache:
            cached_response = self.cache.get(full_key_prompt, role.value)
            if cached_response:
                logger.debug(f"Cache hit for {role.value}")
                return cached_response

        # === SEMANTIC CACHE CHECK ===
        role_key = role.value if hasattr(role, 'value') else str(role)
        cached = self.semantic_cache.get(prompt, role_key)
        if cached is not None:
            logger.info(f"SEMANTIC CACHE HIT for {role_key} — skipping API call")
            return cached

        # 1. Get Primary Assignment
        primary_provider, primary_model = self.assignments.get(
            role,
            (ModelProvider.GEMINI, self.registry.get('gemini', {}).get('flash', 'gemini-1.5-flash'))
        )

        last_exception = None

        # === ATTEMPT PRIMARY (with Rate Limiting) ===
        if primary_provider is not None:
            provider_name = primary_provider.value.lower()

            # Acquire rate limit slot (blocks if at limit)
            slot_acquired = await acquire_api_slot(provider_name, timeout=30.0)

            if slot_acquired:
                start_time = time.time()
                try:
                    client = self._get_client(primary_provider, primary_model)
                    response = await client.generate(prompt, system_prompt, response_json)

                    latency_ms = (time.time() - start_time) * 1000

                    # Success - cache and return
                    if use_cache:
                        self.cache.set(full_key_prompt, role.value, response)

                    self.semantic_cache.set(prompt, role.value, response)

                    # HOTFIX: Use correct API - record_request with success=True
                    metrics.record_request(
                        role=role.value,
                        provider=primary_provider.value,
                        success=True,
                        latency_ms=latency_ms,
                        was_fallback=False
                    )
                    return response

                except RuntimeError as e:
                    if "cannot schedule new futures after shutdown" in str(e):
                        logger.critical(
                            f"EXECUTOR SHUTDOWN detected during {role.value} call to "
                            f"{primary_provider.value}. Process may be restarting."
                        )
                        # All providers share the same broken executor — don't try fallbacks
                        raise CriticalRPCError(
                            f"Executor shutdown during {role.value}: {e}. "
                            f"All providers will fail. Aborting cycle gracefully."
                        ) from e
                    # Re-raise other RuntimeErrors as generic exceptions to be caught below
                    raise e

                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    last_exception = e
                    logger.warning(
                        f"{primary_provider.value}/{primary_model} failed for {role.value}: {e}"
                    )
                    # HOTFIX: Use correct API - record_request with success=False
                    metrics.record_request(
                        role=role.value,
                        provider=primary_provider.value,
                        success=False,
                        latency_ms=latency_ms,
                        was_fallback=False
                    )
            else:
                # Rate limit timeout - treat as failure, move to fallback
                logger.warning(
                    f"Rate limit slot timeout for {provider_name} on role {role.value}. "
                    f"Skipping to fallback."
                )
                last_exception = RuntimeError(f"Rate limit timeout for {provider_name}")

        # === FALLBACK CHAIN (with Rate Limiting per provider) ===
        fallback_chain = self._get_fallback_models(role, primary_provider)

        for fallback_provider, fallback_model in fallback_chain:
            # Skip if provider not available
            if fallback_provider not in self.available_providers:
                continue

            fallback_name = fallback_provider.value.lower()

            # Acquire rate limit slot for fallback provider
            slot_acquired = await acquire_api_slot(fallback_name, timeout=30.0)

            if not slot_acquired:
                logger.warning(f"Rate limit timeout for fallback {fallback_name}, trying next")
                continue

            start_time = time.time()
            try:
                client = self._get_client(fallback_provider, fallback_model)
                response = await client.generate(prompt, system_prompt, response_json)

                latency_ms = (time.time() - start_time) * 1000

                # Fallback succeeded
                logger.warning(
                    f"FALLBACK SUCCESS: Used {fallback_provider.value}/{fallback_model} "
                    f"for {role.value} after primary failure."
                )

                # HOTFIX: Use correct API - record_request for the successful fallback
                metrics.record_request(
                    role=role.value,
                    provider=fallback_provider.value,
                    success=True,
                    latency_ms=latency_ms,
                    was_fallback=True,
                    primary_provider=primary_provider.value if primary_provider else None
                )

                # Record fallback event (this method DOES exist)
                metrics.record_fallback(
                    role.value,
                    primary_provider.value if primary_provider else "none",
                    fallback_provider.value,
                    str(last_exception) if last_exception else "rate_limit"
                )

                # Cache successful response
                if use_cache:
                    self.cache.set(full_key_prompt, role.value, response)

                self.semantic_cache.set(prompt, role.value, response)

                return response

            except RuntimeError as e:
                if "cannot schedule new futures after shutdown" in str(e):
                    logger.critical(
                        f"EXECUTOR SHUTDOWN detected during {role.value} call to "
                        f"{fallback_provider.value}. Process may be restarting."
                    )
                    raise CriticalRPCError(
                        f"Executor shutdown during {role.value}: {e}. "
                        f"Aborting cycle gracefully."
                    ) from e
                # Re-raise other RuntimeErrors as generic exceptions to be caught below
                raise e

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                last_exception = e
                logger.warning(
                    f"Fallback {fallback_provider.value}/{fallback_model} "
                    f"also failed for {role.value}: {e}"
                )
                # HOTFIX: Use correct API - record_request with success=False
                metrics.record_request(
                    role=role.value,
                    provider=fallback_provider.value,
                    success=False,
                    latency_ms=latency_ms,
                    was_fallback=True,
                    primary_provider=primary_provider.value if primary_provider else None
                )
                continue

        # === ALL PROVIDERS EXHAUSTED ===
        error_msg = (
            f"All providers exhausted for {role.value}. "
            f"Primary: {primary_provider.value if primary_provider else 'None'}. "
            f"Fallbacks tried: {len(fallback_chain)}. "
            f"Last error: {last_exception}"
        )
        logger.error(error_msg)

        # Raise CriticalRPCError for Tier 3 roles to abort cycle
        if role in [AgentRole.MASTER_STRATEGIST, AgentRole.COMPLIANCE_OFFICER]:
            raise CriticalRPCError(error_msg) from last_exception

        raise RuntimeError(error_msg) from last_exception

    def get_metrics_summary(self) -> dict:
        """Get routing metrics summary."""
        from trading_bot.router_metrics import get_router_metrics
        return get_router_metrics().get_summary()

    def get_diversity_report(self) -> dict:
        """Report on model diversity."""
        report = {
            "available_providers": [p.value for p in self.available_providers],
            "diversity_score": len(self.available_providers) / len(ModelProvider),
        }
        return report


# Singleton instance
_router: Optional[HeterogeneousRouter] = None

def get_router(config: dict) -> HeterogeneousRouter:
    """Get or create router instance."""
    global _router
    if _router is None:
        _router = HeterogeneousRouter(config)
    return _router
