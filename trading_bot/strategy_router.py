"""
Strategy Router — Unified routing logic for scheduled and emergency cycles.

Design Principle: LLMs decide WHAT (direction + thesis).
                  Python code decides HOW (strategy type + sizing).

Extracts duplicated logic from signal_generator.py and orchestrator.py
into a single, testable module.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFLICT DETECTION (dual-format adapter)
# =============================================================================

def calculate_agent_conflict(agent_data: dict, mode: str = "scheduled") -> float:
    """
    Calculate directional disagreement among agents.

    Handles both scheduled-cycle format (agent_data with *_sentiment keys)
    and emergency-cycle format (agent_reports with mixed dict/string values).

    Args:
        agent_data: Agent output dictionary
        mode: "scheduled" (signal_generator) or "emergency" (orchestrator)

    Returns:
        0.0 (full agreement) to 1.0 (full disagreement)
    """
    sentiments = []

    if mode == "scheduled":
        # Scheduled path: keys are like 'macro_sentiment', 'technical_sentiment'
        sentiment_keys = [
            'macro_sentiment', 'technical_sentiment', 'geopolitical_sentiment',
            'sentiment_sentiment', 'agronomist_sentiment'
        ]
        for key in sentiment_keys:
            s = agent_data.get(key, 'NEUTRAL')
            if s == 'BULLISH':
                sentiments.append(1)
            elif s == 'BEARISH':
                sentiments.append(-1)
            else:
                sentiments.append(0)

    elif mode == "emergency":
        # Emergency path: values are dicts with 'data' key or raw strings
        for key, report in agent_data.items():
            if key in ('master_decision', 'master'):
                continue
            report_str = str(
                report.get('data', '') if isinstance(report, dict) else report
            ).upper()
            if 'BULLISH' in report_str:
                sentiments.append(1)
            elif 'BEARISH' in report_str:
                sentiments.append(-1)
            # NEUTRAL agents don't contribute to conflict

    if len(sentiments) < 2:
        return 0.0

    avg = sum(sentiments) / len(sentiments)

    if mode == "scheduled":
        # Variance-based (canonical implementation used by signal_generator.py)
        variance = sum((s - avg) ** 2 for s in sentiments) / len(sentiments)
        return min(1.0, variance)
    else:
        # MAD-based (canonical implementation used by orchestrator.py)
        conflict = sum(abs(d - avg) for d in sentiments) / len(sentiments)
        return min(1.0, conflict)


def detect_imminent_catalyst(agent_data: dict, mode: str = "scheduled") -> Optional[str]:
    """
    Detect imminent, unpriced market-moving events.

    Args:
        agent_data: Agent output dictionary
        mode: "scheduled" or "emergency"

    Returns:
        Catalyst description string, or None
    """
    if mode == "scheduled":
        # Full v7.0 compound+conditional keyword logic
        return _detect_catalyst_scheduled(agent_data)
    else:
        # Simplified emergency keyword scan
        return _detect_catalyst_emergency(agent_data)


def _detect_catalyst_scheduled(agent_data: dict) -> Optional[str]:
    """
    v7.0 catalyst detection with compound keywords and urgency co-occurrence.

    Canonical implementation used by signal_generator.py.
    """
    # Compound keywords — trigger directly (high confidence)
    catalyst_keywords = [
        ('frost warning', 'Active frost warning in growing regions'),
        ('freeze warning', 'Active freeze warning'),
        ('frost risk', 'Elevated frost risk detected'),
        ('conab report', 'CONAB report imminent'),
        ('usda report', 'USDA report pending'),
        ('strike action', 'Labor strike in progress or imminent'),
        ('port closure', 'Port closure affecting exports'),
        ('export ban', 'Export restriction announced'),
    ]

    # Single-word triggers — REQUIRE urgency co-occurrence
    conditional_keywords = [
        ('drought', 'Drought conditions'),
        ('frost', 'Frost conditions'),
        ('freeze', 'Freeze conditions'),
        ('strike', 'Labor disruption'),
    ]

    # Urgency markers
    urgency_markers = [
        'imminent', 'warning', 'developing', 'worsening', 'escalating',
        'confirmed', 'breaking', 'emergency', 'unprecedented', 'severe',
        'expected this week', 'forecast for', 'risk elevated',
    ]

    # Resolved markers — suppress false positives
    resolved_markers = [
        'ended', 'resolved', 'passed', 'recovered', 'abated', 'eased',
        'improving', 'relief', 'normalized', 'subsided', 'historical',
        'priced in', 'already reflected',
    ]

    reports_to_check = [
        agent_data.get('agronomist_summary', ''),
        agent_data.get('geopolitical_summary', ''),
        agent_data.get('sentiment_summary', ''),
    ]

    combined_text = ' '.join(str(r).lower() for r in reports_to_check)
    has_resolved = any(marker in combined_text for marker in resolved_markers)

    # Phase 1: Compound keywords
    for keyword, description in catalyst_keywords:
        if keyword in combined_text and not has_resolved:
            return description

    # Phase 2: Single keywords + urgency
    if not has_resolved:
        has_urgency = any(marker in combined_text for marker in urgency_markers)
        if has_urgency:
            for keyword, description in conditional_keywords:
                if keyword in combined_text:
                    return f"{description} (urgent)"

    return None


def _detect_catalyst_emergency(agent_reports: dict) -> str:
    """
    Emergency catalyst detection — simple keyword scan.

    Canonical implementation used by orchestrator.py.
    """
    catalyst_keywords = [
        'USDA report', 'FOMC', 'frost', 'freeze', 'hurricane',
        'strike', 'embargo', 'election', 'earnings', 'inventory report',
        'COT report', 'export ban', 'tariff', 'sanctions'
    ]

    for key, report in agent_reports.items():
        report_text = str(
            report.get('data', '') if isinstance(report, dict) else report
        )
        for keyword in catalyst_keywords:
            if keyword.lower() in report_text.lower():
                return f"{keyword} (detected in {key} report)"

    return ""


# =============================================================================
# STRATEGY ROUTING (unified)
# =============================================================================

def route_strategy(
    direction: str,
    confidence: float,
    vol_sentiment: str,
    regime: str,
    thesis_strength: str,
    conviction_multiplier: float,
    reasoning: str,
    agent_data: dict,
    mode: str = "scheduled",
) -> dict:
    """
    Unified strategy routing for both scheduled and emergency cycles.

    Determines prediction_type (DIRECTIONAL vs VOLATILITY) and vol_level
    based on regime, vol_sentiment, agent conflict, and catalyst detection.

    Args:
        direction: BULLISH, BEARISH, or NEUTRAL
        confidence: 0.0–1.0
        vol_sentiment: BULLISH (cheap), BEARISH (expensive), or N/A
        regime: RANGE_BOUND, TRENDING_UP, TRENDING_DOWN, HIGH_VOLATILITY, etc.
        thesis_strength: PROVEN, PLAUSIBLE, SPECULATIVE
        conviction_multiplier: 0.0–2.0
        reasoning: Council reasoning text
        agent_data: Agent reports dict (format depends on mode)
        mode: "scheduled" or "emergency"

    Returns:
        dict with keys: prediction_type, vol_level, direction, confidence,
                        thesis_strength, conviction_multiplier, volatility_sentiment,
                        regime, reason
    """
    # v7.0 SAFETY: Default to BEARISH (expensive) when vol data missing.
    # Fail-safe, not fail-neutral.
    if not vol_sentiment or vol_sentiment == 'N/A':
        vol_sentiment = 'BEARISH'

    prediction_type = "DIRECTIONAL"
    vol_level = None
    reason = reasoning

    if direction == 'NEUTRAL':
        # === NEUTRAL PATH: Vol trade or No Trade ===
        agent_conflict_score = calculate_agent_conflict(agent_data, mode=mode)
        imminent_catalyst = detect_imminent_catalyst(agent_data, mode=mode)

        # PATH 1: IRON CONDOR — sell premium in range when vol is expensive
        if regime == 'RANGE_BOUND' and vol_sentiment == 'BEARISH':
            prediction_type = "VOLATILITY"
            vol_level = "LOW"
            prefix = "Emergency " if mode == "emergency" else ""
            reason = f"{prefix}Iron Condor: Range-bound + expensive vol (sell premium)"
            logger.info(f"STRATEGY SELECTED ({mode}): IRON_CONDOR | regime={regime}, vol={vol_sentiment}")

        # PATH 2: LONG STRADDLE — expect big move, options not expensive
        elif (imminent_catalyst or agent_conflict_score > 0.6) and vol_sentiment != 'BEARISH':
            prediction_type = "VOLATILITY"
            vol_level = "HIGH"
            prefix = "Emergency " if mode == "emergency" else ""
            reason = f"{prefix}Long Straddle: {imminent_catalyst or f'High conflict ({agent_conflict_score:.2f})'}"
            logger.info(
                f"STRATEGY SELECTED ({mode}): LONG_STRADDLE | "
                f"catalyst={imminent_catalyst}, conflict={agent_conflict_score:.2f}"
            )

        # PATH 3: NO TRADE
        else:
            prediction_type = "DIRECTIONAL"
            vol_level = None
            prefix = "Emergency " if mode == "emergency" else ""
            reason = (
                f"{prefix}NO TRADE: Direction neutral, no positive-EV vol trade. "
                f"(vol={vol_sentiment}, regime={regime}, conflict={agent_conflict_score:.2f})"
            )
            logger.info(f"NO TRADE ({mode}): vol={vol_sentiment}, regime={regime}")

    else:
        # === DIRECTIONAL PATH: Always defined-risk spreads ===
        if vol_sentiment == 'BEARISH':
            reason += f" [VOL WARNING: Options expensive. Spread costs elevated. Thesis: {thesis_strength}]"
        logger.info(f"STRATEGY ({mode}): DIRECTIONAL spread (thesis={thesis_strength}, vol={vol_sentiment})")

    return {
        'prediction_type': prediction_type,
        'vol_level': vol_level,
        'direction': direction if prediction_type != 'VOLATILITY' else 'VOLATILITY',
        'confidence': confidence,
        'thesis_strength': thesis_strength,
        'conviction_multiplier': conviction_multiplier,
        'volatility_sentiment': vol_sentiment,
        'regime': regime,
        'reason': reason,
    }


def extract_agent_prediction(report) -> tuple:
    """
    Extract direction and confidence from an agent report.

    A4 FIX: Single canonical implementation.
    Handles both structured dict reports and legacy string reports.

    Returns: (direction: str, confidence: float)
    """
    direction = 'NEUTRAL'
    confidence = 0.5

    if isinstance(report, dict):
        direction = report.get('sentiment', report.get('direction', 'NEUTRAL'))
        confidence = report.get('confidence', 0.5)

        if not direction or direction in ('N/A', ''):
            report_str = str(report.get('data', '')).upper()
            if 'BULLISH' in report_str:
                direction = 'BULLISH'
            elif 'BEARISH' in report_str:
                direction = 'BEARISH'

    elif isinstance(report, str):
        report_str = report.upper()
        if 'BULLISH' in report_str:
            direction = 'BULLISH'
        elif 'BEARISH' in report_str:
            direction = 'BEARISH'

        import re
        conf_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:conf|certain|sure)', report_str, re.I)
        if conf_match:
            confidence = float(conf_match.group(1)) / 100 if float(conf_match.group(1)) > 1 else float(conf_match.group(1))

    return direction, confidence


def infer_strategy_type(routed: dict) -> str:
    """Infer human-readable strategy type from routed signal."""
    if routed['prediction_type'] == 'VOLATILITY':
        if routed.get('vol_level') == 'HIGH':
            return 'LONG_STRADDLE'
        elif routed.get('vol_level') == 'LOW':
            return 'IRON_CONDOR'
    elif routed['direction'] in ('BULLISH', 'BEARISH'):
        return 'DIRECTIONAL'
    return 'NONE'
