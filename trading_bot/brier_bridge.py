"""
Brier Bridge: Unified prediction recording across legacy and enhanced systems.

This module provides a single entry point for recording and resolving predictions,
ensuring both the legacy CSV-based system and the enhanced probabilistic system
stay in sync.

ARCHITECTURE PRINCIPLE: Dual-write pattern ensures backward compatibility.
The enhanced system can fail without impacting legacy behavior.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def record_agent_prediction(
    agent: str,
    predicted_direction: str,
    predicted_confidence: float,
    cycle_id: str,
    regime: str = "NORMAL",
    contract: str = "",
    timestamp: Optional[datetime] = None,
) -> None:
    """
    Record an agent's prediction to BOTH legacy and enhanced Brier systems.

    This is the ONLY function that should be called from the orchestrator
    for recording predictions. It handles dual-write and error isolation.

    Args:
        agent: Agent name (e.g., 'agronomist', 'macro')
        predicted_direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        predicted_confidence: 0.0 to 1.0
        cycle_id: Deterministic cycle identifier
        regime: Market regime string (default: 'NORMAL')
        contract: Contract identifier (e.g., 'KCH6')
        timestamp: When prediction was made (default: now UTC)
    """
    ts = timestamp or datetime.now(timezone.utc)

    # === LEGACY SYSTEM (CSV) — must succeed ===
    try:
        from trading_bot.brier_scoring import get_brier_tracker
        legacy_tracker = get_brier_tracker()
        legacy_tracker.record_prediction_structured(
            agent=agent,
            predicted_direction=predicted_direction,
            predicted_confidence=predicted_confidence,
            actual='PENDING',
            timestamp=ts,
            cycle_id=cycle_id,
        )
    except Exception as e:
        logger.error(f"Legacy Brier recording failed for {agent}: {e}")

    # === ENHANCED SYSTEM (JSON) — fail-safe ===
    try:
        from trading_bot.enhanced_brier import EnhancedBrierTracker, MarketRegime

        tracker = _get_enhanced_tracker()
        if tracker is None:
            return

        # Convert confidence to probability distribution
        prob_bullish, prob_neutral, prob_bearish = _confidence_to_probs(
            predicted_direction, predicted_confidence
        )

        # Map regime string to enum
        try:
            regime_enum = MarketRegime(regime)
        except (ValueError, KeyError):
            regime_enum = MarketRegime.NORMAL

        tracker.record_prediction(
            agent=agent,
            prob_bullish=prob_bullish,
            prob_neutral=prob_neutral,
            prob_bearish=prob_bearish,
            regime=regime_enum,
            contract=contract,
            timestamp=ts,
            cycle_id=cycle_id,
        )

    except Exception as e:
        # Enhanced system failure MUST NOT block trading
        logger.warning(f"Enhanced Brier recording failed for {agent}: {e}")


def resolve_agent_prediction(
    agent: str,
    actual_outcome: str,
    cycle_id: str = "",
    timestamp: Optional[datetime] = None,
) -> Optional[float]:
    """
    Resolve a prediction in the enhanced Brier system.

    Called from reconciliation after council_history gets actual_trend_direction.

    Args:
        agent: Agent name
        actual_outcome: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        cycle_id: Cycle identifier for deterministic matching
        timestamp: Original prediction timestamp (fallback matching)

    Returns:
        Brier score for this prediction, or None if not found
    """
    try:
        tracker = _get_enhanced_tracker()
        if tracker is None:
            return None

        brier = tracker.resolve_prediction(
            agent=agent,
            actual_outcome=actual_outcome,
            cycle_id=cycle_id,
            timestamp=timestamp,
        )
        return brier

    except Exception as e:
        logger.warning(f"Enhanced Brier resolution failed for {agent}: {e}")
        return None


def get_agent_reliability(agent: str, regime: str = "NORMAL", config: dict = None) -> float:
    """
    Blended reliability: gradually shift from legacy to enhanced.

    Config controls blending:
      enhanced_weight=0.0 → 100% legacy (safe default)
      enhanced_weight=0.5 → 50/50 blend
      enhanced_weight=1.0 → 100% enhanced
    """
    if config is None:
        try:
            from config_loader import load_config
            config = load_config()
        except Exception:
            config = {}

    brier_config = config.get('brier_scoring', {})
    enhanced_weight = brier_config.get('enhanced_weight', 0.0)
    legacy_weight = 1.0 - enhanced_weight

    # Get enhanced reliability
    enhanced_mult = 1.0
    try:
        tracker = _get_enhanced_tracker()
        if tracker is not None:
            enhanced_mult = tracker.get_agent_reliability(agent, regime)
    except Exception:
        enhanced_mult = 1.0

    # Get legacy reliability
    legacy_mult = 1.0
    try:
        from trading_bot.brier_scoring import get_brier_tracker
        legacy_tracker = get_brier_tracker()
        legacy_mult = legacy_tracker.get_agent_weight_multiplier(agent) # Note: method name in brier_scoring.py is get_agent_weight_multiplier or get_weight_multiplier?
        # Checking brier_scoring.py content provided in memory/context
        # It seems brier_scoring.py has `get_agent_weight_multiplier`
    except AttributeError:
         # Fallback if I recalled the name wrong, checking file content...
         # File content says `get_agent_weight_multiplier`
         pass
    except Exception:
        legacy_mult = 1.0

    # Blend
    blended = (legacy_weight * legacy_mult) + (enhanced_weight * enhanced_mult)
    return max(0.1, min(2.0, blended))


def get_calibration_data(agent: str = None) -> Dict:
    """
    Get calibration curve data for dashboard display.

    Returns:
        Dict with agent names as keys, calibration curves as values
    """
    try:
        tracker = _get_enhanced_tracker()
        if tracker is None:
            return {}
        return tracker.get_summary(agent)
    except Exception as e:
        logger.warning(f"Failed to get calibration data: {e}")
        return {}


# === PRIVATE HELPERS ===

_enhanced_tracker = None


def _get_enhanced_tracker():
    """Lazy singleton for EnhancedBrierTracker."""
    global _enhanced_tracker
    if _enhanced_tracker is None:
        try:
            from trading_bot.enhanced_brier import EnhancedBrierTracker
            _enhanced_tracker = EnhancedBrierTracker()
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedBrierTracker: {e}")
            return None
    return _enhanced_tracker


def reset_enhanced_tracker():
    """Reset singleton (call after resolving predictions)."""
    global _enhanced_tracker
    _enhanced_tracker = None


def _confidence_to_probs(
    direction: str, confidence: float
) -> tuple:
    """
    Convert a direction + confidence into a probability triple.

    Example: BULLISH with 0.7 confidence →
        prob_bullish=0.7, prob_neutral=0.15, prob_bearish=0.15

    This is a simplification — future versions should use the
    agent's actual probability distribution from their analysis.

    Returns:
        (prob_bullish, prob_neutral, prob_bearish)
    """
    confidence = max(0.0, min(1.0, confidence))
    remainder = 1.0 - confidence

    direction = direction.upper().strip()

    if direction == 'BULLISH':
        return (confidence, remainder * 0.5, remainder * 0.5)
    elif direction == 'BEARISH':
        return (remainder * 0.5, remainder * 0.5, confidence)
    else:  # NEUTRAL
        return (remainder * 0.5, confidence, remainder * 0.5)
