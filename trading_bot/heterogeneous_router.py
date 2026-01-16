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
from datetime import datetime, timedelta
from functools import wraps
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass

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


class ResponseCache:
    """Simple in-memory cache for LLM responses."""

    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds

    def _hash_key(self, prompt: str, role: str) -> str:
        return hashlib.md5(f"{role}:{prompt}".encode()).hexdigest()

    def get(self, prompt: str, role: str) -> Optional[str]:
        key = self._hash_key(prompt, role)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self.ttl):
                return entry['response']
            del self.cache[key]
        return None

    def set(self, prompt: str, role: str, response: str):
        key = self._hash_key(prompt, role)
        self.cache[key] = {'response': response, 'timestamp': datetime.now()}


def with_retry(max_retries=3, backoff_factor=2):
    """Decorator to retry async methods with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Attempt {attempt+1} failed for {func.__name__}, retrying in {wait_time}s: {e}")
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

        response = await self.client.aio.models.generate_content(
            model=self.config.model_name,
            contents=full_prompt,
            config=gen_config
        )
        return response.text


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
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)
        return response.content[0].text


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
        cache_ttl = config.get('cache_ttl', 300)
        self.cache = ResponseCache(ttl_seconds=cache_ttl)

        # 1. LOAD MODEL KEYS (With Defaults)
        gem_flash = self.registry.get('gemini', {}).get('flash', 'gemini-3-flash-preview')
        gem_pro = self.registry.get('gemini', {}).get('pro', 'gemini-3-pro-preview')

        anth_pro = self.registry.get('anthropic', {}).get('pro', 'claude-opus-4-5-20251101')

        oai_pro = self.registry.get('openai', {}).get('pro', 'gpt-5.2-pro-2025-12-11')
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
        """Route request to appropriate model with robust fallback."""

        # Check Cache (Force OFF for Sentinels)
        use_cache = True
        if role in [AgentRole.WEATHER_SENTINEL, AgentRole.LOGISTICS_SENTINEL, AgentRole.NEWS_SENTINEL, AgentRole.PRICE_SENTINEL, AgentRole.MICROSTRUCTURE_SENTINEL]:
            use_cache = False

        full_key_prompt = f"{system_prompt or ''}:{prompt}"

        if use_cache:
            cached_response = self.cache.get(full_key_prompt, role.value)
            if cached_response:
                logger.debug(f"Cache hit for {role.value}")
                return cached_response

        # 1. Determine Execution Chain (Primary + Fallbacks)
        primary_provider, primary_model = self.assignments.get(
            role,
            (ModelProvider.GEMINI, "gemini-3-flash-preview")
        )

        execution_chain = [(primary_provider, primary_model)]

        # Add fallbacks
        fallbacks = self._get_fallback_models(role, primary_provider)
        execution_chain.extend(fallbacks)

        last_exception = None

        for provider, model_name in execution_chain:
            # Skip if provider strictly unavailable (no API key)
            if provider not in self.available_providers:
                logger.warning(f"Skipping {provider.value} (unavailable) for {role.value}")
                continue

            logger.debug(f"Routing {role.value} to {provider.value}/{model_name}")

            try:
                client = self._get_client(provider, model_name)
                response = await client.generate(prompt, system_prompt, response_json)

                # Success! Cache and return
                self.cache.set(full_key_prompt, role.value, response)

                if provider != primary_provider:
                    logger.warning(f"FALLBACK SUCCESS: Used {provider.value}/{model_name} for {role.value} after primary failure.")

                return response

            except Exception as e:
                logger.warning(f"{provider.value}/{model_name} failed for {role.value}: {e}")
                last_exception = e
                # Continue to next model in chain

        # If we get here, all models in the chain failed or were unavailable
        error_msg = f"ALL models failed for {role.value}. Last error: {last_exception}"
        logger.error(error_msg)

        # Raise CriticalRPCError for Tier 3 roles to abort cycle
        if role in [AgentRole.MASTER_STRATEGIST, AgentRole.COMPLIANCE_OFFICER]:
             raise CriticalRPCError(error_msg) from last_exception

        raise RuntimeError(error_msg) from last_exception

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
