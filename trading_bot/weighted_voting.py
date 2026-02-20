"""Confidence-Weighted Voting for Multi-Agent Decisions.

Implements: Decision Score = Σ(Vote × Confidence × DomainWeight)

Domain experts get higher weights on relevant events:
- Weather Analyst gets 3x weight on frost events
- Geopolitical Analyst gets 3x weight on logistics events
"""

import asyncio
import logging
import re
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any
from trading_bot.brier_bridge import get_agent_reliability
from trading_bot.strategy_router import extract_agent_prediction
from trading_bot.confidence_utils import CONFIDENCE_BANDS, parse_confidence
from trading_bot.agent_names import DEPRECATED_AGENTS

logger = logging.getLogger(__name__)

# Mutable data directory — set by orchestrator via set_data_dir()
_data_dir: Optional[str] = None


def set_data_dir(data_dir: str):
    """Configure data directory for weight evolution CSV."""
    global _data_dir
    _data_dir = data_dir
    logger.info(f"WeightedVoting data_dir set to: {data_dir}")


class TriggerType(Enum):
    """Classification of decision trigger."""
    WEATHER = "weather"
    LOGISTICS = "logistics"
    PRICE_SPIKE = "price"
    NEWS_SENTIMENT = "news"
    MICROSTRUCTURE = "microstructure"
    PREDICTION_MARKET = "prediction_market"
    MACRO_SHIFT = "macro_shift"
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
    age_hours: float = 0.0  # Populated from StateManager metadata
    is_stale: bool = False
    data_freshness_hours: float = 0.0


def calculate_staleness_weight(age_hours: float, max_useful_age: float = 24.0) -> float:
    """
    Graduated staleness decay — aged data gets reduced weight, never zero.

    Fresh (0-1h):    1.0 (full weight — data is current)
    Aging (1-6h):    0.9 → 0.5 (linear decay — still highly relevant)
    Stale (6-24h):   0.5 → 0.1 (slower decay — context still useful)
    Ancient (>24h):  0.1 (floor — prevents total exclusion)

    The key insight: 4-hour-old macro analysis is MORE useful than
    an abstained vote. The current binary system throws away this signal.

    Edge case: float('inf') (legacy data with no timestamp) falls to
    the >24h branch and returns 0.1 (floor). This is intentional —
    legacy agents participate at minimum weight rather than being excluded.
    """
    if age_hours <= 1.0:
        return 1.0
    elif age_hours <= 6.0:
        # Linear decay: 0.9 at 1h → 0.5 at 6h
        return 1.0 - (age_hours - 1.0) * 0.1
    elif age_hours <= max_useful_age:
        # Slower decay: 0.5 at 6h → 0.1 at 24h
        return max(0.1, 0.5 - (age_hours - 6.0) * 0.022)
    else:
        return 0.1  # Floor — never fully exclude


# Domain weight matrix: weight by trigger type
DOMAIN_WEIGHTS = {
    TriggerType.WEATHER: {
        "agronomist": 3.0,
        "macro": 1.0,
        "geopolitical": 0.5,
        "supply_chain": 1.5,
        "sentiment": 1.0,
        "technical": 1.5,
        "volatility": 2.0,
        "inventory": 1.5,
    },
    TriggerType.LOGISTICS: {
        "agronomist": 0.5,
        "macro": 1.5,
        "geopolitical": 3.0,
        "supply_chain": 3.0,
        "sentiment": 1.0,
        "technical": 1.0,
        "volatility": 1.5,
        "inventory": 2.5,
    },
    TriggerType.PRICE_SPIKE: {
        "agronomist": 1.0,
        "macro": 1.5,
        "geopolitical": 1.0,
        "supply_chain": 1.0,
        "sentiment": 2.0,
        "technical": 3.0,
        "volatility": 2.5,
        "inventory": 1.0,
    },
    TriggerType.NEWS_SENTIMENT: {
        "agronomist": 1.0,
        "macro": 2.0,
        "geopolitical": 2.0,
        "supply_chain": 1.5,
        "sentiment": 3.0,
        "technical": 1.5,
        "volatility": 1.5,
        "inventory": 1.0,
    },
    TriggerType.MICROSTRUCTURE: {
        "agronomist": 0.5,
        "macro": 1.0,
        "geopolitical": 0.5,
        "supply_chain": 0.5,
        "sentiment": 1.5,
        "technical": 2.5,
        "volatility": 3.0,
        "inventory": 1.0,
    },
    TriggerType.PREDICTION_MARKET: {
        "agronomist": 0.5,
        "macro": 3.0,
        "geopolitical": 2.5,
        "supply_chain": 1.0,
        "sentiment": 2.0,
        "technical": 1.0,
        "volatility": 2.0,
        "inventory": 0.5,
    },
    TriggerType.MACRO_SHIFT: {
        "agronomist": 0.5,
        "macro": 3.0,
        "geopolitical": 2.0,
        "supply_chain": 1.0,
        "sentiment": 2.0,
        "technical": 1.5,
        "volatility": 2.0,
        "inventory": 0.5,
    },
    TriggerType.SCHEDULED: {
        "agronomist": 1.5,
        "macro": 1.5,
        "geopolitical": 1.5,
        "supply_chain": 1.5,
        "sentiment": 1.5,
        "technical": 1.5,
        "volatility": 1.5,
        "inventory": 1.5,
    },
    TriggerType.MANUAL: {
        "agronomist": 1.5,
        "macro": 1.5,
        "geopolitical": 1.5,
        "supply_chain": 1.5,
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


def normalize_ml_confidence(raw_confidence: float, calibration_data: dict = None) -> float:
    """
    DEPRECATED: ML pipeline removed in v4.0.

    Retained for backward compatibility with any external callers.
    Will be removed in v5.0.
    """
    raw_confidence = max(0.0, min(1.0, raw_confidence))
    damping_factor = 1.5
    return 0.5 + ((raw_confidence - 0.5) / damping_factor)

def extract_sentiment_from_report(report_text: str) -> tuple[str, float, bool]:
    """Extract sentiment with validated patterns for actual text format.

    Returns:
        (sentiment, confidence, matched) where matched indicates whether
        any pattern or keyword actually matched (False = fell through to defaults).
    """
    if not report_text or not isinstance(report_text, str):
        return "NEUTRAL", 0.5, False

    # Normalize whitespace
    text = ' '.join(report_text.split())

    # VALIDATED PATTERNS - Order matters! Most specific first.
    patterns = [
        # 1. Exact format from research_topic: [SENTIMENT TAG]: [SENTIMENT: BULLISH]
        r'\[SENTIMENT TAG\]:\s*\[SENTIMENT:\s*(\w+)\]',

        # 2. Standard format: [SENTIMENT: BULLISH]
        r'\[SENTIMENT:\s*(\w+)\]',

        # 3. JSON format with BOTH single AND double quotes (FIX ADDENDUM A)
        # Handles: "sentiment": "BULLISH" AND 'sentiment': 'BULLISH'
        r'[\'"]sentiment[\'"]\s*:\s*[\'"]?(\w+)[\'"]?',

        # 4. Loose fallback: SENTIMENT: BULLISH
        r'SENTIMENT:\s*(\w+)',
    ]

    sentiment = "NEUTRAL"
    matched = False
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found = match.group(1).upper()
            if found in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                sentiment = found
                matched = True
                break
            # If matched but invalid value (e.g., 'TAG'), continue to next pattern

    # Fallback: scan for keywords if NEUTRAL and no pattern matched
    if sentiment == "NEUTRAL" and not matched:
        text_lower = text.lower()
        bullish_words = ['bullish', 'buy', 'long', 'positive', 'upside', 'support']
        bearish_words = ['bearish', 'sell', 'short', 'negative', 'downside', 'resistance']

        bull_count = sum(1 for w in bullish_words if w in text_lower)
        bear_count = sum(1 for w in bearish_words if w in text_lower)

        if bull_count > bear_count:
            sentiment = "BULLISH"
            matched = True
        elif bear_count > bull_count:
            sentiment = "BEARISH"
            matched = True

    # CONFIDENCE: Extract [CONFIDENCE]: 0.9
    confidence = 0.5
    conf_match = re.search(r'\[CONFIDENCE\]:\s*([0-9.]+)', text)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            pass
    else:
        # Fallback: strength words
        text_lower = text.lower()
        if any(w in text_lower for w in ["strong", "high conviction", "extremely"]):
            confidence = 0.85
        elif any(w in text_lower for w in ["uncertain", "mixed", "unclear"]):
            confidence = 0.3

    return sentiment, confidence, matched


class RegimeDetector:
    """Detects market regime for dynamic weight adjustment."""

    @staticmethod
    async def detect_regime(ib, contract) -> str:
        """Returns: 'HIGH_VOL', 'LOW_VOL', 'TRENDING', 'RANGE_BOUND'"""
        if not ib or not contract:
            return 'UNKNOWN'

        try:
            bars = await asyncio.wait_for(ib.reqHistoricalDataAsync(
                contract, '', '5 D', '1 day', 'TRADES', True
            ), timeout=10)
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
    market_data: Optional[dict] = None,
    ib: Optional[Any] = None,
    contract: Optional[Any] = None,
    regime: str = "UNKNOWN",
    min_quorum: int = 3,
    config: Optional[dict] = None
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

        # Estimate price change from market data if available
        price_change = 0.0
        if market_data and 'price' in market_data and 'expected_price' in market_data:
            try:
                curr = market_data['price']
                exp = market_data['expected_price']
                if curr: price_change = (exp - curr) / curr
            except (TypeError, ValueError, ZeroDivisionError) as e:
                logger.debug(f"Price change calculation failed: {e}")

        regime = detect_market_regime_simple(str(vol_report), price_change)

    logger.info(f"Market Regime Detected: {regime}")

    weights = get_regime_adjusted_weights(trigger_type, regime)
    votes: list[AgentVote] = []

    for agent_name, report in agent_reports.items():
        # === EARLY SKIP: Deprecated agents (stale state.json entries) ===
        if agent_name in DEPRECATED_AGENTS:
            logger.debug(f"Agent {agent_name}: Skipping — deprecated agent")
            continue

        # === EARLY SKIP: Explicitly unavailable data ===
        if not report or report in ["Data Unavailable", "N/A"]:
            logger.debug(f"Agent {agent_name}: Skipping - Data Unavailable")
            continue

        # === NEW: Track stale/failure state for abstention logic ===
        is_stale = False
        is_failure = False
        age_hours = 0.0

        # === FIX #4: Handle STALE prefix from StateManager ===
        if isinstance(report, str) and report.startswith('STALE'):
            is_stale = True
            # Extract the actual data after the prefix
            # Format: "STALE (Xmin old) - <actual_data>"
            stale_match = re.match(r'STALE \([^)]+\) - (.+)', report, re.DOTALL)
            if stale_match:
                report = stale_match.group(1)
                logger.debug(f"Agent {agent_name}: Extracting sentiment from stale data ({stale_match.group(0)[:50]}...)")
            else:
                logger.warning(f"Agent {agent_name}: Unparseable STALE format, skipping")
                continue

        # === EXTRACT AGE METADATA (from load_state_with_metadata) ===
        if isinstance(report, dict) and '_age_hours' in report:
            age_hours = report.get('_age_hours', 0.0)
            # Remove metadata before processing
            if 'data' in report:
                # Unwrap the data
                report_data = report['data']
                # If data itself is a dict (nested), use it. If string, use string.
                if isinstance(report_data, (dict, str)):
                    report = report_data

        # Extract data freshness (Issue A)
        is_data_stale = False
        data_freshness = 0.0
        if isinstance(report, dict):
            is_data_stale = report.get('is_stale', False)
            data_freshness = report.get('data_freshness_hours', 0.0)

        # === Check for error messages ===
        if isinstance(report, str) and ('Error conducting research' in report or 'All providers exhausted' in report):
            is_failure = True
            logger.warning(f"Agent {agent_name}: Research failed - {report[:100]}")
            continue  # Skip failed agents entirely

        # === PRIORITY 1: Use structured dict fields if available ===
        sentiment_tag = None
        confidence = 0.5
        sentiment_matched = False  # Track whether parsing found a real signal

        if isinstance(report, dict) and report.get('sentiment'):
            sentiment_tag = str(report['sentiment']).upper().strip()
            if sentiment_tag in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                # === v5.3.1: Use shared confidence parser (replaces inline logic) ===
                raw_conf = report.get('confidence', 0.5)
                confidence = parse_confidence(raw_conf)
                sentiment_matched = True
                logger.debug(f"Agent {agent_name}: Using structured data -> {sentiment_tag} ({confidence:.2f})")
            else:
                # Invalid sentiment value, fall through to text parsing
                sentiment_tag = None
                confidence = 0.5

        # === PRIORITY 2: Parse from text (with fixed regex) ===
        if sentiment_tag is None:
            report_text = report.get('data', report) if isinstance(report, dict) else str(report)
            sentiment_tag, confidence, sentiment_matched = extract_sentiment_from_report(report_text)
            logger.debug(f"Agent {agent_name}: Parsed from text -> {sentiment_tag} ({confidence:.2f})")

        confidence = max(0.0, min(1.0, float(confidence)))  # Defensive clamp
        direction = parse_sentiment_to_direction(sentiment_tag)

        # === FIX #7 REVISED: GRADUATED STALENESS (replaces binary abstention) ===
        # Failed agents still abstain. Stale agents vote with reduced weight.
        if is_failure:
            logger.warning(f"Agent {agent_name}: ABSTAINING due to failed research")
            continue  # Skip failed agents entirely

        if direction == Direction.NEUTRAL and confidence == 0.5 and not is_stale:
            # Genuinely neutral with fresh data — this is a legitimate vote
            logger.info(f"Agent {agent_name}: Fresh NEUTRAL vote (not abstention)")

        # Stale agents that produced real sentiment still vote, with reduced weight
        # The staleness_weight will be applied in the weighting section below

        # Log default value occurrences for monitoring — only when parsing truly failed
        if direction == Direction.NEUTRAL and confidence == 0.5 and not sentiment_matched:
            logger.warning(f"Agent {agent_name}: Using DEFAULT values (NEUTRAL/0.5). "
                          f"Report type: {type(report).__name__}, is_stale: {is_stale}, "
                          f"Report preview: {str(report)[:100]}...")

        votes.append(AgentVote(
            agent_name=agent_name,
            direction=direction,
            confidence=confidence,
            sentiment_tag=sentiment_tag,
            age_hours=age_hours,
            is_stale=is_data_stale,
            data_freshness_hours=data_freshness
        ))

    # Quorum Check
    if len(votes) < min_quorum:
        logger.warning(
            f"QUORUM FAILURE: Only {len(votes)} agents voted "
            f"(minimum: {min_quorum}). "
            f"Returning NEUTRAL to prevent low-confidence decisions."
        )
        voting_agents = [v.agent_name for v in votes]
        return {
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'weighted_score': 0.0,
            'vote_breakdown': [],
            'dominant_agent': None,
            'trigger_type': trigger_type.value if hasattr(trigger_type, 'value') else str(trigger_type),
            'quorum_failure': True,
            'voters_present': voting_agents,
        }

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
    # Removed legacy tracker initialization

    for vote in votes:
        base_domain_weight = weights.get(vote.agent_name, 1.0)

        # Apply Brier Score Multiplier (Enhanced + Legacy Blend)
        reliability_multiplier = get_agent_reliability(vote.agent_name, regime=regime)

        # Apply Staleness Weight (graduated, from metadata)
        staleness_weight = calculate_staleness_weight(vote.age_hours)

        # Apply Freshness Penalty (from Issue A)
        freshness_penalty = 1.0
        if vote.is_stale:
            SOFT_FRESHNESS_LIMIT_HOURS = 4.0
            freshness = vote.data_freshness_hours
            # Linear decay: 100% weight at 4h, 50% weight at 12h, 25% at 24h
            decay_factor = max(0.25, 1.0 - (freshness - SOFT_FRESHNESS_LIMIT_HOURS) / 40.0)
            freshness_penalty = decay_factor
            logger.debug(f"Agent {vote.agent_name}: Freshness penalty applied: {freshness:.1f}h -> factor {decay_factor:.2f}")

        final_weight = base_domain_weight * reliability_multiplier * staleness_weight * freshness_penalty

        contribution = vote.direction.value * vote.confidence * final_weight
        total_weighted_score += contribution
        total_weight += final_weight

        vote_breakdown.append({
            'agent': vote.agent_name,
            'direction': vote.direction.name,
            'confidence': vote.confidence,
            'domain_weight': base_domain_weight,
            'reliability_mult': round(reliability_multiplier, 2),
            'staleness_weight': round(staleness_weight, 2),
            'age_hours': round(vote.age_hours, 1),
            'final_weight': round(final_weight, 2),
            'contribution': round(contribution, 3),
        })

        if abs(contribution) > max_contribution:
            max_contribution = abs(contribution)
            dominant_agent = vote.agent_name

    normalized_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    if market_data:
        logger.debug(f"Market context received: "
                     f"regime={market_data.get('regime')}, price={market_data.get('price')}")

    # v7.0: Widen deadlock zone from ±0.15 to ±0.10
    # RATIONALE: With 7 agents designed to disagree (Permabear vs Permabull),
    # the ±0.15 band catches ~85% of votes in the deadlock zone. Downstream
    # gates (Compliance, DA on emergencies) still filter weak signals.
    # A "Weak Bull" at +0.12 should at least reach the Master for consideration.
    if normalized_score > 0.10:
        final_direction = 'BULLISH'
    elif normalized_score < -0.10:
        final_direction = 'BEARISH'
    else:
        final_direction = 'NEUTRAL'

    # Unanimity-based confidence
    direction_values = [v.direction.value for v in votes]
    avg_dir = sum(direction_values) / len(direction_values)
    unanimity = 1.0 - (sum(abs(d - avg_dir) for d in direction_values) / len(direction_values))
    avg_conf = sum(v.confidence for v in votes) / len(votes)
    final_confidence = (unanimity * 0.4) + (avg_conf * 0.6)

    # VaR-aware confidence dampener
    var_dampener = 1.0
    var_utilization = 0.0
    raw_confidence = None
    try:
        if config:
            from trading_bot.var_calculator import get_var_calculator
            var_calc = get_var_calculator(config)
            cached_var = var_calc.get_cached_var() if var_calc else None
            if cached_var:
                var_limit_pct = config.get('compliance', {}).get('var_limit_pct', 0.03)
                var_utilization = cached_var.var_95_pct / var_limit_pct if var_limit_pct > 0 else 0
                if var_utilization > 0.8:
                    # Smooth dampening: 80% util -> 1.0, 100% util -> 0.5
                    var_dampener = max(0.5, 1.0 - (var_utilization - 0.8) * 2.5)
                    raw_confidence = final_confidence
                    final_confidence *= var_dampener
                    logger.info(
                        f"VaR dampener: {var_dampener:.2f} (util: {var_utilization:.0%}), "
                        f"confidence: {raw_confidence:.2f} -> {final_confidence:.2f}"
                    )
    except Exception:
        pass  # Non-fatal

    weight_summary = {
        v['agent']: (
            f"domain={v.get('domain_weight', 1.0):.1f} × "
            f"reliability={v.get('reliability_mult', 1.0):.1f} = "
            f"{v.get('final_weight', 1.0):.2f}"
        )
        for v in vote_breakdown
    }
    logger.info(f"Dynamic Weights: {json.dumps(weight_summary)}")

    # === WEIGHT EVOLUTION PERSISTENCE ===
    # Track weight changes over time for dashboard visualization.
    # Uses asyncio.run_in_executor to avoid blocking the event loop
    # (calculate_weighted_decision is async, so no sync I/O in hot path).
    try:
        import asyncio
        import os

        def _write_weight_csv(vote_breakdown_snapshot, trigger_str, regime_str):
            """Sync CSV writer — runs in thread pool."""
            import csv
            from datetime import datetime, timezone

            weight_csv = os.path.join(
                _data_dir or os.path.join('data', os.environ.get('COMMODITY_TICKER', 'KC')),
                'weight_evolution.csv'
            )
            file_exists = os.path.exists(weight_csv)

            # Ensure directory exists
            os.makedirs(os.path.dirname(weight_csv), exist_ok=True)

            with open(weight_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'cycle_key', 'agent', 'regime',
                        'domain_weight', 'reliability_mult', 'final_weight'
                    ])

                timestamp = datetime.now(timezone.utc).isoformat()
                # Composite key from trigger_type + timestamp
                # (cycle_id is not available in this function's signature)
                cycle_key = f"{trigger_str}_{timestamp[:19]}"

                for vote in vote_breakdown_snapshot:
                    agent = vote.get('agent', 'unknown')
                    writer.writerow([
                        timestamp,
                        cycle_key,
                        agent,
                        regime_str,
                        round(vote.get('domain_weight', 1.0), 3),
                        round(vote.get('reliability_mult', 1.0), 3),
                        round(vote.get('final_weight', 1.0), 3),
                    ])

        # Snapshot vote_breakdown (list of dicts is safe to pass to thread)
        trigger_str = trigger_type.value if hasattr(trigger_type, 'value') else str(trigger_type)

        # Fire-and-forget: don't await — CSV write must never block voting
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,  # Default thread pool
            _write_weight_csv,
            list(vote_breakdown),  # Defensive copy
            trigger_str,
            regime
        )
    except Exception as e:
        logger.warning(f"Weight evolution tracking failed (non-fatal): {e}")

    logger.info(f"Weighted Vote: {final_direction} (score={normalized_score:.3f}, dominant={dominant_agent})")
    logger.info(f"Vote Breakdown: {json.dumps(vote_breakdown, indent=2)}")

    return {
        'direction': final_direction,
        'confidence': round(final_confidence, 3),
        'weighted_score': round(normalized_score, 4),
        'vote_breakdown': vote_breakdown,
        'dominant_agent': dominant_agent,
        'trigger_type': trigger_type.value,
        'var_dampener': round(var_dampener, 3),
        'var_utilization': round(var_utilization, 3),
        'raw_confidence': round(raw_confidence, 3) if raw_confidence is not None else None,
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
    elif "prediction" in source_lower:
        return TriggerType.PREDICTION_MARKET
    elif "macro" in source_lower or "contagion" in source_lower:
        return TriggerType.MACRO_SHIFT
    return TriggerType.MANUAL
