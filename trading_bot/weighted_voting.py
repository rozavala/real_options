"""Confidence-Weighted Voting for Multi-Agent Decisions.

Implements: Decision Score = Σ(Vote × Confidence × DomainWeight)

Domain experts get higher weights on relevant events:
- Weather Analyst gets 3x weight on frost events
- Geopolitical Analyst gets 3x weight on logistics events
"""

import logging
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Classification of decision trigger."""
    WEATHER = "weather"
    LOGISTICS = "logistics"
    PRICE_SPIKE = "price"
    NEWS_SENTIMENT = "news"
    MICROSTRUCTURE = "microstructure"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class Direction(Enum):
    """Trading direction with numeric value."""
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1


@dataclass
class AgentVote:
    """Single agent's vote."""
    agent_name: str
    direction: Direction
    confidence: float
    sentiment_tag: str


# Domain weight matrix: weight by trigger type
DOMAIN_WEIGHTS = {
    TriggerType.WEATHER: {
        "agronomist": 3.0,
        "macro": 1.0,
        "geopolitical": 0.5,
        "sentiment": 1.0,
        "technical": 1.5,
        "volatility": 2.0,
        "inventory": 1.5,
    },
    TriggerType.LOGISTICS: {
        "agronomist": 0.5,
        "macro": 1.5,
        "geopolitical": 3.0,
        "sentiment": 1.0,
        "technical": 1.0,
        "volatility": 1.5,
        "inventory": 2.5,
    },
    TriggerType.PRICE_SPIKE: {
        "agronomist": 1.0,
        "macro": 1.5,
        "geopolitical": 1.0,
        "sentiment": 2.0,
        "technical": 3.0,
        "volatility": 2.5,
        "inventory": 1.0,
    },
    TriggerType.NEWS_SENTIMENT: {
        "agronomist": 1.0,
        "macro": 2.0,
        "geopolitical": 2.0,
        "sentiment": 3.0,
        "technical": 1.5,
        "volatility": 1.5,
        "inventory": 1.0,
    },
    TriggerType.MICROSTRUCTURE: {
        "agronomist": 0.5,
        "macro": 1.0,
        "geopolitical": 0.5,
        "sentiment": 1.5,
        "technical": 2.5,
        "volatility": 3.0,
        "inventory": 1.0,
    },
    TriggerType.SCHEDULED: {
        "agronomist": 1.5,
        "macro": 1.5,
        "geopolitical": 1.5,
        "sentiment": 1.5,
        "technical": 1.5,
        "volatility": 1.5,
        "inventory": 1.5,
    },
    TriggerType.MANUAL: {
        "agronomist": 1.5,
        "macro": 1.5,
        "geopolitical": 1.5,
        "sentiment": 1.5,
        "technical": 1.5,
        "volatility": 1.5,
        "inventory": 1.5,
    },
}


def parse_sentiment_to_direction(sentiment_tag: str) -> Direction:
    """Convert sentiment string to Direction."""
    if not sentiment_tag:
        return Direction.NEUTRAL
    upper = sentiment_tag.upper()
    if "BULLISH" in upper:
        return Direction.BULLISH
    elif "BEARISH" in upper:
        return Direction.BEARISH
    return Direction.NEUTRAL


def extract_sentiment_from_report(report_text: str) -> tuple[str, float]:
    """Extract sentiment and estimate confidence from report."""
    sentiment_match = re.search(r'\[SENTIMENT[:\s]+(\w+)\]', report_text, re.IGNORECASE)
    sentiment_tag = sentiment_match.group(1) if sentiment_match else "NEUTRAL"

    confidence = 0.5
    report_lower = report_text.lower()

    strong_signals = ["strong", "extremely", "high conviction", "overwhelming"]
    weak_signals = ["uncertain", "mixed", "unclear", "conflicting"]

    for phrase in strong_signals:
        if phrase in report_lower:
            confidence = 0.85
            break
    for phrase in weak_signals:
        if phrase in report_lower:
            confidence = 0.3
            break

    return sentiment_tag, confidence


def calculate_weighted_decision(
    agent_reports: dict[str, str],
    trigger_type: TriggerType,
    ml_signal: Optional[dict] = None
) -> dict:
    """
    Calculate weighted decision from all agent reports.

    Returns:
        dict with 'direction', 'confidence', 'weighted_score',
        'vote_breakdown', 'dominant_agent'
    """
    weights = DOMAIN_WEIGHTS.get(trigger_type, DOMAIN_WEIGHTS[TriggerType.SCHEDULED])
    votes: list[AgentVote] = []

    for agent_name, report in agent_reports.items():
        if not report or report in ["Data Unavailable", "N/A"]:
            continue

        report_text = report.get('data', report) if isinstance(report, dict) else str(report)
        sentiment_tag, confidence = extract_sentiment_from_report(report_text)
        direction = parse_sentiment_to_direction(sentiment_tag)

        votes.append(AgentVote(
            agent_name=agent_name,
            direction=direction,
            confidence=confidence,
            sentiment_tag=sentiment_tag
        ))

    if not votes:
        return {
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'weighted_score': 0.0,
            'vote_breakdown': [],
            'dominant_agent': None
        }

    vote_breakdown = []
    total_weighted_score = 0.0
    total_weight = 0.0
    max_contribution = 0.0
    dominant_agent = None

    for vote in votes:
        domain_weight = weights.get(vote.agent_name, 1.0)
        contribution = vote.direction.value * vote.confidence * domain_weight
        total_weighted_score += contribution
        total_weight += domain_weight

        vote_breakdown.append({
            'agent': vote.agent_name,
            'direction': vote.direction.name,
            'confidence': vote.confidence,
            'domain_weight': domain_weight,
            'contribution': contribution,
        })

        if abs(contribution) > max_contribution:
            max_contribution = abs(contribution)
            dominant_agent = vote.agent_name

    normalized_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    # Blend with ML signal (70% agents, 30% ML)
    if ml_signal:
        ml_dir = ml_signal.get('direction', 'NEUTRAL')
        ml_conf = ml_signal.get('confidence', 0.5)
        ml_val = {'BULLISH': 1, 'BEARISH': -1, 'NEUTRAL': 0}.get(ml_dir, 0)
        normalized_score = (0.7 * normalized_score) + (0.3 * ml_val * ml_conf)

    if normalized_score > 0.15:
        final_direction = 'BULLISH'
    elif normalized_score < -0.15:
        final_direction = 'BEARISH'
    else:
        final_direction = 'NEUTRAL'

    # Unanimity-based confidence
    direction_values = [v.direction.value for v in votes]
    avg_dir = sum(direction_values) / len(direction_values)
    unanimity = 1.0 - (sum(abs(d - avg_dir) for d in direction_values) / len(direction_values))
    avg_conf = sum(v.confidence for v in votes) / len(votes)
    final_confidence = (unanimity * 0.4) + (avg_conf * 0.6)

    logger.info(f"Weighted Vote: {final_direction} (score={normalized_score:.3f}, dominant={dominant_agent})")

    return {
        'direction': final_direction,
        'confidence': round(final_confidence, 3),
        'weighted_score': round(normalized_score, 4),
        'vote_breakdown': vote_breakdown,
        'dominant_agent': dominant_agent,
        'trigger_type': trigger_type.value
    }


def determine_trigger_type(trigger_source: Optional[str]) -> TriggerType:
    """Map sentinel source to TriggerType."""
    if not trigger_source:
        return TriggerType.SCHEDULED

    source_lower = trigger_source.lower()
    if "weather" in source_lower:
        return TriggerType.WEATHER
    elif "logistics" in source_lower:
        return TriggerType.LOGISTICS
    elif "price" in source_lower:
        return TriggerType.PRICE_SPIKE
    elif "news" in source_lower:
        return TriggerType.NEWS_SENTIMENT
    elif "microstructure" in source_lower:
        return TriggerType.MICROSTRUCTURE
    return TriggerType.MANUAL
