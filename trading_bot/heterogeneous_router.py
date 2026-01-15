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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


# Optimal model assignment based on cognitive profiles
MODEL_ASSIGNMENTS = {
    # Tier 1: Cost-efficient sentinels (Gemini Flash)
    AgentRole.WEATHER_SENTINEL: (ModelProvider.GEMINI, "gemini-1.5-flash"),
    AgentRole.LOGISTICS_SENTINEL: (ModelProvider.GEMINI, "gemini-1.5-flash"),
    AgentRole.NEWS_SENTINEL: (ModelProvider.GEMINI, "gemini-1.5-flash"),
    AgentRole.PRICE_SENTINEL: (ModelProvider.GEMINI, "gemini-1.5-flash"),
    AgentRole.MICROSTRUCTURE_SENTINEL: (ModelProvider.GEMINI, "gemini-1.5-flash"),

    # Tier 2: Domain experts
    AgentRole.AGRONOMIST: (ModelProvider.GEMINI, "gemini-1.5-pro"),
    AgentRole.MACRO_ANALYST: (ModelProvider.OPENAI, "gpt-4o"),
    AgentRole.GEOPOLITICAL_ANALYST: (ModelProvider.OPENAI, "gpt-4o"),
    AgentRole.SENTIMENT_ANALYST: (ModelProvider.XAI, "grok-beta"),
    AgentRole.TECHNICAL_ANALYST: (ModelProvider.OPENAI, "gpt-4o"),
    AgentRole.VOLATILITY_ANALYST: (ModelProvider.GEMINI, "gemini-1.5-pro"),
    AgentRole.INVENTORY_ANALYST: (ModelProvider.GEMINI, "gemini-1.5-pro"),

    # Tier 3: Strategic decision makers
    AgentRole.PERMABEAR: (ModelProvider.ANTHROPIC, "claude-3-5-sonnet-20241022"),
    AgentRole.PERMABULL: (ModelProvider.OPENAI, "gpt-4o"),
    AgentRole.MASTER_STRATEGIST: (ModelProvider.OPENAI, "gpt-4o"),
    AgentRole.COMPLIANCE_OFFICER: (ModelProvider.ANTHROPIC, "claude-3-5-sonnet-20241022"),
}


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

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      response_json: bool = False) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
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
        self.fallback_provider = ModelProvider.GEMINI

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

    def _get_fallback(self, role: AgentRole) -> tuple[ModelProvider, str]:
        """Get fallback when preferred provider unavailable."""
        fallbacks = [
            (ModelProvider.GEMINI, "gemini-1.5-flash"),
            (ModelProvider.OPENAI, "gpt-4o-mini"),
            (ModelProvider.ANTHROPIC, "claude-3-5-sonnet-20241022"),
        ]

        for provider, model in fallbacks:
            if provider in self.available_providers:
                logger.warning(f"Using fallback {provider.value}/{model} for {role.value}")
                return provider, model

        raise RuntimeError("No LLM providers available!")

    async def route(
        self,
        role: AgentRole,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_json: bool = False
    ) -> str:
        """Route request to appropriate model."""
        provider, model_name = MODEL_ASSIGNMENTS.get(
            role,
            (ModelProvider.GEMINI, "gemini-1.5-flash")
        )

        if provider not in self.available_providers:
            provider, model_name = self._get_fallback(role)

        logger.debug(f"Routing {role.value} to {provider.value}/{model_name}")

        try:
            client = self._get_client(provider, model_name)
            return await client.generate(prompt, system_prompt, response_json)
        except Exception as e:
            logger.error(f"{provider.value}/{model_name} failed: {e}")
            fallback_provider, fallback_model = self._get_fallback(role)
            if fallback_provider != provider:
                client = self._get_client(fallback_provider, fallback_model)
                return await client.generate(prompt, system_prompt, response_json)
            raise

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
