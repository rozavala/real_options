#!/usr/bin/env python3
"""Weekly LLM Model Availability & Deprecation Checker.

Validates all configured models still exist via each provider's model-listing API,
detects newer models, and sends a Pushover notification with results.

Usage:
    python scripts/check_model_availability.py

Exit codes:
    0 — All models available
    1 — One or more models unavailable or provider unreachable
"""

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config
from notifications import send_pushover_notification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ModelChecker")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    provider: str
    tier: str
    model_id: str
    available: bool
    source: str = "config"  # "config" or "router_fallback"
    error: str | None = None


@dataclass
class ProviderReport:
    provider: str
    results: list[ModelResult] = field(default_factory=list)
    latest_models: list[str] = field(default_factory=list)
    error: str | None = None  # provider-level error (e.g. auth failure)


# ---------------------------------------------------------------------------
# Extract fallback defaults from heterogeneous_router.py
# ---------------------------------------------------------------------------

def extract_router_fallback_models() -> dict[str, dict[str, str]]:
    """Parse heterogeneous_router.py for hardcoded fallback model IDs.

    Returns dict like: {'openai': {'pro': 'gpt-4o'}, 'anthropic': {'pro': 'claude-opus-4-5-20251101'}, ...}
    These are the defaults used when the registry key is missing.
    """
    router_path = Path(__file__).resolve().parent.parent / "trading_bot" / "heterogeneous_router.py"
    if not router_path.exists():
        logger.warning("heterogeneous_router.py not found, skipping fallback extraction")
        return {}

    text = router_path.read_text()

    # Match patterns like: .get('gemini', {}).get('flash', 'gemini-3-flash-preview')
    pattern = r"\.get\(['\"](\w+)['\"],\s*\{\}\)\.get\(['\"](\w+)['\"],\s*['\"]([^'\"]+)['\"]\)"
    matches = re.findall(pattern, text)

    fallbacks: dict[str, dict[str, str]] = {}
    for provider, tier, model_id in matches:
        fallbacks.setdefault(provider, {})[tier] = model_id

    return fallbacks


# ---------------------------------------------------------------------------
# Provider checkers
# ---------------------------------------------------------------------------

async def check_openai(models: dict[str, str]) -> ProviderReport:
    """Check OpenAI models via client.models.list()."""
    report = ProviderReport(provider="openai")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        available = await client.models.list()
        available_ids = {m.id for m in available.data}

        for tier, model_id in models.items():
            exists = model_id in available_ids
            report.results.append(ModelResult(
                provider="openai", tier=tier, model_id=model_id, available=exists,
            ))

        # Latest GPT/o-series models for upgrade awareness
        gpt_models = sorted([m.id for m in available.data
                             if m.id.startswith(("gpt-", "o1", "o3", "o4"))])
        report.latest_models = gpt_models[-10:] if gpt_models else []

    except Exception as e:
        report.error = str(e)
        logger.error(f"OpenAI check failed: {e}")
    return report


async def check_anthropic(models: dict[str, str]) -> ProviderReport:
    """Check Anthropic models via client.models.list()."""
    report = ProviderReport(provider="anthropic")
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()
        available = await client.models.list()
        available_ids = {m.id for m in available.data}

        for tier, model_id in models.items():
            exists = model_id in available_ids
            report.results.append(ModelResult(
                provider="anthropic", tier=tier, model_id=model_id, available=exists,
            ))

        claude_models = sorted([m.id for m in available.data
                                if "claude" in m.id.lower()])
        report.latest_models = claude_models[-10:] if claude_models else []

    except Exception as e:
        report.error = str(e)
        logger.error(f"Anthropic check failed: {e}")
    return report


async def check_gemini(models: dict[str, str]) -> ProviderReport:
    """Check Gemini models via genai.Client().models.list()."""
    report = ProviderReport(provider="gemini")
    try:
        from google import genai
        client = genai.Client()
        available = list(client.models.list())
        # Gemini model names come as "models/gemini-..." — normalize
        available_ids = set()
        for m in available:
            name = m.name if hasattr(m, 'name') else str(m)
            # Strip "models/" prefix if present
            clean = name.replace("models/", "") if name.startswith("models/") else name
            available_ids.add(clean)
            available_ids.add(name)  # keep original too

        for tier, model_id in models.items():
            exists = model_id in available_ids or f"models/{model_id}" in available_ids
            report.results.append(ModelResult(
                provider="gemini", tier=tier, model_id=model_id, available=exists,
            ))

        gemini_models = sorted([n for n in available_ids
                                if "gemini" in n and not n.startswith("models/")])
        report.latest_models = gemini_models[-10:] if gemini_models else []

    except Exception as e:
        report.error = str(e)
        logger.error(f"Gemini check failed: {e}")
    return report


async def check_xai(models: dict[str, str]) -> ProviderReport:
    """Check xAI models via OpenAI-compatible API."""
    report = ProviderReport(provider="xai")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=os.getenv("XAI_API_KEY"),
        )
        available = await client.models.list()
        available_ids = {m.id for m in available.data}

        for tier, model_id in models.items():
            exists = model_id in available_ids
            report.results.append(ModelResult(
                provider="xai", tier=tier, model_id=model_id, available=exists,
            ))

        grok_models = sorted([m.id for m in available.data
                              if "grok" in m.id.lower()])
        report.latest_models = grok_models[-10:] if grok_models else []

    except Exception as e:
        report.error = str(e)
        logger.error(f"xAI check failed: {e}")
    return report


# ---------------------------------------------------------------------------
# Merge config + fallback models into a unified check list
# ---------------------------------------------------------------------------

def build_model_map(config: dict) -> dict[str, dict[str, str]]:
    """Build combined model map from config registry + router fallback defaults.

    Config values take precedence; fallbacks are added only for tiers not in config.
    """
    registry = config.get("model_registry", {})
    fallbacks = extract_router_fallback_models()

    merged: dict[str, dict[str, str]] = {}
    for provider in ("openai", "anthropic", "gemini", "xai"):
        cfg_models = dict(registry.get(provider, {}))
        fb_models = fallbacks.get(provider, {})
        # Add fallback-only models (tiers not already in config)
        for tier, model_id in fb_models.items():
            if tier not in cfg_models and model_id not in cfg_models.values():
                cfg_models[f"{tier}_fallback"] = model_id
        merged[provider] = cfg_models

    return merged


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(reports: list[ProviderReport]) -> tuple[str, str, bool]:
    """Format reports into (title, body, has_problems)."""
    problems = []
    ok_lines = []
    upgrade_lines = []
    has_problems = False

    for report in reports:
        if report.error:
            has_problems = True
            problems.append(f"ERROR: {report.provider} — {report.error}")
            continue

        provider_ok = []
        for r in report.results:
            if r.available:
                provider_ok.append(f"{r.model_id} ({r.tier})")
            else:
                has_problems = True
                source_note = " [fallback]" if "fallback" in r.source else ""
                problems.append(
                    f"UNAVAILABLE: {report.provider}/{r.model_id} ({r.tier}){source_note}"
                )

        if provider_ok:
            ok_lines.append(f"{report.provider}: {', '.join(provider_ok)}")

        if report.latest_models:
            upgrade_lines.append(
                f"{report.provider} latest: {', '.join(report.latest_models[-5:])}"
            )

    if has_problems:
        title = "Weekly Model Check: Action Needed"
        body_parts = []
        if problems:
            body_parts.append("\n".join(problems))
        if ok_lines:
            body_parts.append("OK: " + " | ".join(ok_lines))
        if upgrade_lines:
            body_parts.append("\n".join(upgrade_lines))
        body = "\n\n".join(body_parts)
    else:
        title = "Weekly Model Check: All OK"
        body_parts = ok_lines[:]
        if upgrade_lines:
            body_parts.append("")
            body_parts.extend(upgrade_lines)
        body = "\n".join(body_parts)

    return title, body, has_problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    config = load_config()
    if not config:
        logger.error("Failed to load config")
        return 1

    model_map = build_model_map(config)

    logger.info("Starting model availability checks...")
    for provider, models in model_map.items():
        logger.info(f"  {provider}: {models}")

    # Run all provider checks concurrently
    checker_map = {
        "openai": check_openai,
        "anthropic": check_anthropic,
        "gemini": check_gemini,
        "xai": check_xai,
    }

    tasks = []
    order = []
    for provider, models in model_map.items():
        if models and provider in checker_map:
            tasks.append(checker_map[provider](models))
            order.append(provider)

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    reports: list[ProviderReport] = []
    for provider, result in zip(order, raw_results):
        if isinstance(result, Exception):
            reports.append(ProviderReport(provider=provider, error=str(result)))
        else:
            reports.append(result)

    # Format and display
    title, body, has_problems = format_report(reports)

    icon = "⚠️" if has_problems else "✅"
    print(f"\n{'='*60}")
    print(f"{icon} {title}")
    print(f"{'='*60}")
    print(body)
    print(f"{'='*60}\n")

    # Send Pushover notification
    notif_config = config.get("notifications", {})
    priority = 1 if has_problems else -1  # high priority for problems, quiet for OK
    try:
        send_pushover_notification(
            config=notif_config,
            title=f"{icon} {title}",
            message=body,
            monospace=True,
            priority=priority,
        )
    except Exception as e:
        logger.warning(f"Pushover notification failed: {e}")

    # GitHub Actions: set output for downstream steps
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"has_problems={'true' if has_problems else 'false'}\n")
            # Escape newlines for GH Actions
            safe_body = body.replace("\n", "%0A")
            f.write(f"report={safe_body}\n")

    return 1 if has_problems else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
