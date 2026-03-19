"""
Perplexity Sonar Client — Grounded data gathering via search-native LLM.

Replaces Gemini + Google Search AFC for Phase 1 data collection.
Uses OpenAI-compatible API with Perplexity's base_url.

Cost model: ~$0.006 per request (Sonar, low context).
No AFC multiplier — one API call = one search = deterministic cost.
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

# Per-request fees by search_context_size ($5/$8/$12 per 1K requests)
_REQUEST_FEES = {"low": 0.005, "medium": 0.008, "high": 0.012}

# Lazy-init async client — openai SDK already installed for GPT routing
_async_client = None


def _get_client(api_key: str):
    """Lazy-init async OpenAI client pointed at Perplexity."""
    global _async_client
    if _async_client is None:
        from openai import AsyncOpenAI
        _async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
        )
    return _async_client


def reset_client():
    """Force re-init (e.g., after key rotation)."""
    global _async_client
    _async_client = None


def get_request_fee(search_context_size: str = "low") -> float:
    """Return the per-request fee for a given search context size."""
    return _REQUEST_FEES.get(search_context_size, 0.005)


async def perplexity_grounded_search(
    search_instruction: str,
    api_key: str,
    model: str = "sonar",
    search_context_size: str = "low",
    timeout_seconds: int = 30,
) -> dict:
    """
    Execute a grounded web search via Perplexity Sonar.

    Args:
        search_instruction: The research task (same string passed to Gemini today).
        api_key: Perplexity API key.
        model: "sonar" or "sonar-pro".
        search_context_size: "low", "medium", or "high" (affects cost per request).
        timeout_seconds: Request timeout.

    Returns:
        Dict matching GroundedDataPacket expectations:
        {
            "raw_summary": str,
            "dated_facts": list[dict],
            "search_queries_used": list[str],
            "data_quality": str,
            "data_freshness": str,
            "_usage": {"input_tokens": int, "output_tokens": int, "model": str,
                       "request_fee": float}
        }

    Raises:
        Exception on API failure (caller handles fallback).
    """
    client = _get_client(api_key)

    system_prompt = (
        "You are a Research Data Gatherer. Your ONLY job is to find CURRENT information "
        "from the web and return it as structured JSON.\n\n"
        "RULES:\n"
        "1. Search for information from the LAST 24-48 HOURS.\n"
        "2. Focus on: dates, numbers, quotes from officials, specific events.\n"
        "3. Do NOT analyze or give opinions. Just report factual findings.\n"
        "4. If you cannot find news from the last 72 hours, set raw_summary to 'NO RECENT NEWS FOUND'.\n\n"
        "OUTPUT FORMAT (respond with ONLY this JSON, no markdown fences, no preamble):\n"
        '{\n'
        '  "raw_summary": "Brief summary of what you found (2-3 paragraphs)",\n'
        '  "dated_facts": [\n'
        '    {"date": "YYYY-MM-DD or today/yesterday", "fact": "specific finding", "source": "publication name"}\n'
        '  ],\n'
        '  "search_queries_used": ["urls or source names from your search"],\n'
        '  "data_quality": "HIGH|MEDIUM|LOW",\n'
        '  "data_freshness": "Data appears to be from [timeframe]"\n'
        '}'
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": search_instruction},
                ],
                temperature=0.0,
                web_search_options={"search_context_size": search_context_size},
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Perplexity request timed out after {timeout_seconds}s")

    raw_text = response.choices[0].message.content or ""

    # Extract citations from Perplexity response
    citations = []
    if hasattr(response, 'citations') and response.citations:
        citations = list(response.citations)
    elif hasattr(response.choices[0].message, 'context'):
        ctx = response.choices[0].message.context
        if isinstance(ctx, dict):
            citations = ctx.get('citations', [])

    # Parse JSON response (strip markdown fences if present)
    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        lines = clean_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean_text = "\n".join(lines)

    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError:
        logger.warning("Perplexity returned non-JSON response, wrapping as raw_summary")
        data = {
            "raw_summary": raw_text[:2000],
            "dated_facts": [],
            "search_queries_used": citations,
            "data_quality": "LOW",
            "data_freshness": "Unknown",
        }

    # Type guards
    if isinstance(data, list):
        data = {
            "raw_summary": str(data),
            "dated_facts": [item for item in data if isinstance(item, dict)],
            "data_freshness": "",
            "search_queries_used": citations,
        }
    elif not isinstance(data, dict):
        data = {
            "raw_summary": str(data),
            "dated_facts": [],
            "data_freshness": "",
            "search_queries_used": citations,
        }

    # Merge citations into search_queries_used
    if citations and not data.get("search_queries_used"):
        data["search_queries_used"] = citations
    elif citations:
        existing = set(data.get("search_queries_used", []))
        for c in citations:
            if c not in existing:
                data["search_queries_used"].append(c)

    # Usage for budget tracking
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage') and response.usage:
        input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
        output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0

    data["_usage"] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
        "request_fee": get_request_fee(search_context_size),
    }

    return data
