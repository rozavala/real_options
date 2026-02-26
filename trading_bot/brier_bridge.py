"""
Brier Bridge: Unified prediction recording across legacy and enhanced systems.

This module provides a single entry point for recording and resolving predictions,
ensuring both the legacy CSV-based system and the enhanced probabilistic system
stay in sync.

ARCHITECTURE PRINCIPLE: Dual-write pattern ensures backward compatibility.
The enhanced system can fail without impacting legacy behavior.
"""

import logging
import warnings
from datetime import datetime, timezone
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# B1 FIX: Track legacy usage for migration
_LEGACY_USAGE_COUNT = 0
_LEGACY_DEPRECATION_DATE = datetime(2026, 3, 1, tzinfo=timezone.utc)


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
    Record an agent's prediction with deprecation path for legacy system.

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
    global _LEGACY_USAGE_COUNT

    # === ENHANCED SYSTEM (JSON) — fail-safe ===
    try:
        from trading_bot.enhanced_brier import EnhancedBrierTracker, MarketRegime, normalize_regime

        tracker = _get_enhanced_tracker()
        if tracker is None:
            pass # Fall through to legacy if enabled
        else:
            # Convert confidence to probability distribution
            prob_bullish, prob_neutral, prob_bearish = _confidence_to_probs(
                predicted_direction, predicted_confidence
            )

            # Map regime string to enum using canonical normalizer
            regime_enum = normalize_regime(regime)

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

    # === LEGACY SYSTEM (CSV) — Deprecated ===
    if datetime.now(timezone.utc) < _LEGACY_DEPRECATION_DATE:
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
            _LEGACY_USAGE_COUNT += 1

            if _LEGACY_USAGE_COUNT % 100 == 0:
                logger.warning(
                    f"DEPRECATION: Legacy Brier system used {_LEGACY_USAGE_COUNT} times. "
                    f"Will be removed after {_LEGACY_DEPRECATION_DATE}"
                )
        except Exception as e:
            logger.error(f"Legacy Brier recording failed for {agent}: {e}")


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


def backfill_enhanced_from_csv() -> int:
    """
    Catch-up: resolve Enhanced Brier predictions using already-resolved CSV data.

    Called from orchestrator's run_brier_reconciliation to handle the pipeline
    gap where fix_brier_data.py resolves the CSV without updating the JSON.

    Returns:
        Number of predictions backfilled
    """
    try:
        tracker = _get_enhanced_tracker()
        if tracker is None:
            return 0

        return tracker.backfill_from_resolved_csv()

    except Exception as e:
        logger.warning(f"Enhanced Brier backfill failed (non-fatal): {e}")
        return 0


def auto_orphan_enhanced_brier(max_age_hours: float = 168.0) -> int:
    """
    Orphan stale Enhanced Brier predictions that have been PENDING too long.

    Called from orchestrator's periodic maintenance. Safe to call frequently —
    only affects predictions older than max_age_hours.

    Returns:
        Number of predictions orphaned
    """
    try:
        tracker = _get_enhanced_tracker()
        if tracker is None:
            return 0
        return tracker.auto_orphan_stale_predictions(max_age_hours)
    except Exception as e:
        logger.warning(f"Enhanced Brier auto-orphan failed (non-fatal): {e}")
        return 0


def get_agent_reliability(agent_name: str, regime: str = "NORMAL", window: int = 20) -> float:
    """
    Rolling reliability multiplier from Enhanced Brier scores.

    Delegates to EnhancedBrierTracker.get_agent_reliability() which
    implements the Brier-to-multiplier conversion internally.

    Returns multiplier in [0.1, 2.0]:
    - Brier ~0.0  → 2.0x (excellent calibration)
    - Brier ~0.25 → 1.0x (average / insufficient data)
    - Brier ~0.5  → 0.1x (poor calibration)
    """
    try:
        from trading_bot.agent_names import normalize_agent_name
        agent_name = normalize_agent_name(agent_name)

        tracker = _get_enhanced_tracker()
        if tracker is None:
            return 1.0

        # FIX (P1-B, 2026-02-04): Call the correct method on the tracker.
        # v8.0: Tracker now handles cross-regime fallback internally (4-path).
        # Bridge NORMAL fallback removed — would cause double-fallback confusion.
        from trading_bot.enhanced_brier import normalize_regime
        canonical_regime = normalize_regime(regime).value  # .value → string for dict lookup
        multiplier = tracker.get_agent_reliability(agent_name, canonical_regime)

        logger.debug(
            f"Agent {agent_name} reliability (regime={regime}): "
            f"multiplier={multiplier:.2f}"
        )
        return multiplier

    except Exception as e:
        logger.warning(f"Failed to get reliability for {agent_name}: {e}")
        return 1.0  # Fail-safe: neutral weight


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

# Per-engine tracker registry (keyed by data_dir to prevent cross-contamination)
_enhanced_trackers: dict = {}  # data_dir → EnhancedBrierTracker
_enhanced_tracker_data_dir = None


def set_data_dir(data_dir: str):
    """Set data directory and force tracker recreation on next access."""
    global _enhanced_tracker_data_dir
    _enhanced_tracker_data_dir = data_dir
    _enhanced_trackers.pop(data_dir, None)  # Force recreation for this data_dir
    logger.info(f"BrierBridge data_dir set to: {data_dir}")


def _get_enhanced_tracker(data_dir: str = None):
    """Per-engine EnhancedBrierTracker. Uses data_dir-keyed registry for isolation."""
    global _enhanced_tracker_data_dir
    # ContextVar > explicit arg > module global
    if data_dir is None:
        try:
            from trading_bot.data_dir_context import get_engine_data_dir
            data_dir = get_engine_data_dir()
        except LookupError:
            pass
    effective_dir = data_dir or _enhanced_tracker_data_dir or ""
    if effective_dir in _enhanced_trackers:
        return _enhanced_trackers[effective_dir]
    try:
        from trading_bot.enhanced_brier import EnhancedBrierTracker
        if effective_dir:
            import os
            data_path = os.path.join(effective_dir, "enhanced_brier.json")
            _enhanced_trackers[effective_dir] = EnhancedBrierTracker(data_path=data_path)
        else:
            _enhanced_trackers[effective_dir] = EnhancedBrierTracker()
    except Exception as e:
        logger.error(f"Failed to initialize EnhancedBrierTracker: {e}")
        return None
    return _enhanced_trackers[effective_dir]


def reset_enhanced_tracker():
    """Reset all tracker instances (call after resolving predictions)."""
    _enhanced_trackers.clear()


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
