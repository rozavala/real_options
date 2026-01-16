"""Confidence-Weighted Voting for Multi-Agent Decisions.

Implements: Decision Score = Σ(Vote × Confidence × DomainWeight)

Domain experts get higher weights on relevant events:
- Weather Analyst gets 3x weight on frost events
- Geopolitical Analyst gets 3x weight on logistics events
"""

import logging
import re
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any
from trading_bot.brier_scoring import get_brier_tracker

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
    """Extract sentiment with multiple fallback patterns."""
    if not report_text or not isinstance(report_text, str):
        return "NEUTRAL", 0.3

    # Try multiple patterns
    patterns = [
        r'\[SENTIMENT[:\s]+(\w+)\]',           # Standard format
        r'"sentiment"[:\s]*"?(\w+)"?',          # JSON format
        r'SENTIMENT:\s*(\w+)',                  # Plain text
        r'\*\*Sentiment\*\*[:\s]*(\w+)',        # Markdown bold
    ]

    sentiment_tag = "NEUTRAL"
    for pattern in patterns:
        match = re.search(pattern, report_text, re.IGNORECASE)
        if match:
            sentiment_tag = match.group(1).upper()
            break

    # Fallback: scan for keywords if NEUTRAL and no pattern matched (or matched neutral)
    if sentiment_tag == "NEUTRAL":
        text_lower = report_text.lower()
        bullish_words = ['bullish', 'buy', 'long', 'positive', 'upside', 'support']
        bearish_words = ['bearish', 'sell', 'short', 'negative', 'downside', 'resistance']

        bull_count = sum(1 for w in bullish_words if w in text_lower)
        bear_count = sum(1 for w in bearish_words if w in text_lower)

        if bull_count > bear_count:
            sentiment_tag = "BULLISH"
        elif bear_count > bull_count:
            sentiment_tag = "BEARISH"

    # Confidence estimation
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


class RegimeDetector:
    """Detects market regime for dynamic weight adjustment."""

    @staticmethod
    async def detect_regime(ib, contract) -> str:
        """Returns: 'HIGH_VOL', 'LOW_VOL', 'TRENDING', 'RANGE_BOUND'"""
        if not ib or not contract:
            return 'UNKNOWN'

        try:
            bars = await ib.reqHistoricalDataAsync(
                contract, '', '5 D', '1 day', 'TRADES', True
            )
            if not bars or len(bars) < 2:
                return 'UNKNOWN'

            # Calculate 5-day volatility
            returns = [(bars[i].close - bars[i-1].close) / bars[i-1].close
                      for i in range(1, len(bars))]
            vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5

            # Detect trend
            price_change = (bars[-1].close - bars[0].close) / bars[0].close

            if vol > 0.03:  # >3% daily vol
                return 'HIGH_VOLATILITY'
            elif abs(price_change) > 0.05:  # >5% 5-day move
                return 'TRENDING'
            else:
                return 'RANGE_BOUND'
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return 'UNKNOWN'


def detect_market_regime_simple(volatility_report: str, price_change_pct: float) -> str:
    """Detect current market regime for weight adjustment (Simple Fallback)."""

    # Check volatility agent's report
    if volatility_report and ("HIGH" in volatility_report.upper() or "ELEVATED" in volatility_report.upper()):
        return "HIGH_VOLATILITY"

    # Check price action
    if abs(price_change_pct) > 0.03:  # >3% move
        return "HIGH_VOLATILITY"

    return "RANGE_BOUND"


def get_regime_adjusted_weights(trigger_type: TriggerType, regime: str) -> dict:
    base_weights = DOMAIN_WEIGHTS.get(trigger_type, DOMAIN_WEIGHTS[TriggerType.SCHEDULED])

    regime_multipliers = {
        "HIGH_VOLATILITY": {
            "agronomist": 1.5,
            "volatility": 1.5,
            "technical": 0.7,
        },
        "RANGE_BOUND": {
            "technical": 1.5,
            "volatility": 0.7,
        },
        "TRENDING": {
            "macro": 1.5,
            "geopolitical": 1.5
        }
    }

    multipliers = regime_multipliers.get(regime, {})
    # Apply multipliers to base weights
    return {k: v * multipliers.get(k, 1.0) for k, v in base_weights.items()}


async def calculate_weighted_decision(
    agent_reports: dict[str, str],
    trigger_type: TriggerType,
    ml_signal: Optional[dict] = None,
    ib: Optional[Any] = None,
    contract: Optional[Any] = None,
    regime: str = "UNKNOWN"
) -> dict:
    """
    Calculate weighted decision from all agent reports.

    Returns:
        dict with 'direction', 'confidence', 'weighted_score',
        'vote_breakdown', 'dominant_agent'
    """
    # 1. Detect Regime (if not provided)
    if regime == "UNKNOWN":
        # Try Advanced Detection first
        if ib and contract:
            regime = await RegimeDetector.detect_regime(ib, contract)

    # 2. Fallback to Simple Detection
    if regime == "UNKNOWN":
        vol_report = agent_reports.get('volatility', '')
        if isinstance(vol_report, dict): vol_report = vol_report.get('data', '')

        # Estimate price change from ML signal if available
        price_change = 0.0
        if ml_signal and 'price' in ml_signal and 'expected_price' in ml_signal:
            try:
                curr = ml_signal['price']
                exp = ml_signal['expected_price']
                if curr: price_change = (exp - curr) / curr
            except: pass

        regime = detect_market_regime_simple(str(vol_report), price_change)

    logger.info(f"Market Regime Detected: {regime}")

    weights = get_regime_adjusted_weights(trigger_type, regime)
    votes: list[AgentVote] = []

    for agent_name, report in agent_reports.items():
        if not report or report in ["Data Unavailable", "N/A"]:
            continue

        report_text = report.get('data', report) if isinstance(report, dict) else str(report)
        sentiment_tag, confidence = extract_sentiment_from_report(report_text)
        confidence = max(0.0, min(1.0, float(confidence)))  # Defensive clamp
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

    # Brier Score Integration
    tracker = get_brier_tracker()

    for vote in votes:
        base_domain_weight = weights.get(vote.agent_name, 1.0)

        # Apply Brier Score Multiplier
        reliability_multiplier = tracker.get_agent_weight_multiplier(vote.agent_name)
        final_weight = base_domain_weight * reliability_multiplier

        contribution = vote.direction.value * vote.confidence * final_weight
        total_weighted_score += contribution
        total_weight += final_weight

        vote_breakdown.append({
            'agent': vote.agent_name,
            'direction': vote.direction.name,
            'confidence': vote.confidence,
            'domain_weight': base_domain_weight,
            'reliability_mult': round(reliability_multiplier, 2),
            'final_weight': round(final_weight, 2),
            'contribution': round(contribution, 3),
        })

        if abs(contribution) > max_contribution:
            max_contribution = abs(contribution)
            dominant_agent = vote.agent_name

    normalized_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    # DYNAMIC ML BLENDING (replaces hardcoded 30%)
    if ml_signal:
        ml_dir = ml_signal.get('direction', 'NEUTRAL')
        ml_conf = ml_signal.get('confidence', 0.5)
        ml_val = {'BULLISH': 1, 'BEARISH': -1, 'NEUTRAL': 0}.get(ml_dir, 0)

        # Get ML model's Brier-adjusted weight
        ml_base_weight = 0.30  # Configurable base weight
        ml_reliability = tracker.get_agent_weight_multiplier('ml_model')

        # REGIME OVERRIDE: Zero out ML during crisis (Black Swan protection)
        if regime in ('HIGH_VOLATILITY', 'CRISIS', 'HIGH_VOL'):
            logger.warning(f"Regime={regime}: Zeroing ML weight (structural break risk)")
            ml_dynamic_weight = 0.0
        else:
            # Apply same formula as other agents: base * reliability
            ml_dynamic_weight = ml_base_weight * ml_reliability
            # Cap at 40% max to prevent ML domination even with high accuracy
            ml_dynamic_weight = min(0.40, ml_dynamic_weight)

        # Blend: (1 - ml_weight) * agents + ml_weight * ML
        agent_weight = 1.0 - ml_dynamic_weight
        normalized_score = (agent_weight * normalized_score) + (ml_dynamic_weight * ml_val * ml_conf)

        logger.info(f"ML Blend: Reliability={ml_reliability:.2f}, DynamicWeight={ml_dynamic_weight:.2f}, Regime={regime}")

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
    logger.info(f"Vote Breakdown: {json.dumps(vote_breakdown, indent=2)}")

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
