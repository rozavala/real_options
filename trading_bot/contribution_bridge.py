"""
Bridge between ContributionTracker and the rest of the system.

Mirrors brier_bridge.py pattern: per-engine tracker registry keyed by data_dir.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Per-engine tracker registry (data_dir-keyed for multi-commodity isolation)
_trackers: dict = {}
_data_dir: Optional[str] = None

# Scoring mode flag — set once during orchestrator init, avoids repeated config reads
_use_contribution_scoring: Optional[bool] = None


def set_data_dir(data_dir: str):
    """Set data directory for contribution tracker."""
    global _data_dir
    _data_dir = data_dir
    _trackers.pop(data_dir, None)  # Force re-creation
    logger.info(f"ContributionBridge data_dir set to: {data_dir}")


def set_scoring_mode(use_contribution: bool):
    """
    Set scoring mode flag. Called once during orchestrator init.
    Avoids repeated config.json reads in hot path.
    """
    global _use_contribution_scoring
    _use_contribution_scoring = use_contribution
    logger.info(f"ContributionBridge scoring mode: {'CONTRIBUTION' if use_contribution else 'BRIER'}")


def is_contribution_scoring_enabled() -> bool:
    """Check if contribution scoring is active."""
    return _use_contribution_scoring is True


def _get_tracker(data_dir: str = None):
    """Per-engine ContributionTracker (data_dir-keyed registry)."""
    global _data_dir
    if data_dir is None:
        try:
            from trading_bot.data_dir_context import get_engine_data_dir
            data_dir = get_engine_data_dir()
        except LookupError:
            pass
    effective_dir = data_dir or _data_dir or ""
    if effective_dir in _trackers:
        return _trackers[effective_dir]
    try:
        from trading_bot.contribution_scorer import ContributionTracker
        import os
        if effective_dir:
            data_path = os.path.join(effective_dir, "contribution_scores.json")
            _trackers[effective_dir] = ContributionTracker(data_path=data_path)
        else:
            _trackers[effective_dir] = ContributionTracker()
    except Exception as e:
        logger.error(f"Failed to initialize ContributionTracker: {e}")
        return None
    return _trackers[effective_dir]


def get_contribution_reliability(
    agent_name: str, regime: str = "NORMAL"
) -> float:
    """
    Get reliability multiplier from contribution scores.

    Drop-in replacement for brier_bridge.get_agent_reliability()
    when contribution scoring is enabled.
    """
    try:
        tracker = _get_tracker()
        if tracker is None:
            return 1.0
        return tracker.get_agent_reliability(agent_name, regime)
    except Exception as e:
        logger.warning(f"Contribution reliability failed for {agent_name}: {e}")
        return 1.0


def record_cycle_contributions(
    vote_breakdown: list,
    master_direction: str,
    master_confidence: float,
    actual_outcome: str,
    prediction_type: str = "DIRECTIONAL",
    strategy_type: str = "",
    volatility_outcome: str = "",
    regime: str = "NORMAL",
    cycle_id: str = "",
    contract: str = "",
) -> Dict[str, float]:
    """
    Record contribution scores for all agents in a resolved cycle.

    Called from reconciliation after outcome is determined.

    Args:
        vote_breakdown: List of dicts from weighted voting
            (each has 'agent', 'direction', 'confidence', 'final_weight')
        master_direction: Master's final directional call
        master_confidence: Master's stated confidence
        actual_outcome: Materiality-thresholded outcome (BULLISH/BEARISH/NEUTRAL)
        prediction_type: DIRECTIONAL or VOLATILITY
        strategy_type: LONG_STRADDLE, IRON_CONDOR, etc.
        volatility_outcome: BIG_MOVE/STAYED_FLAT if VOLATILITY cycle, else ""
        regime: Market regime at time of decision
        cycle_id: Council cycle ID
        contract: Contract identifier

    Returns:
        Dict of agent_name → contribution_score
    """
    tracker = _get_tracker()
    if tracker is None:
        return {}

    scores = {}
    for vote in vote_breakdown:
        agent = vote.get("agent", "")
        direction = vote.get("direction", "NEUTRAL")
        confidence = float(vote.get("confidence", 0.5))
        weight = float(vote.get("final_weight", 1.0))

        if not agent:
            continue

        score = tracker.record_contribution(
            agent=agent,
            agent_direction=direction,
            agent_confidence=confidence,
            master_direction=master_direction,
            actual_outcome=actual_outcome,
            prediction_type=prediction_type,
            strategy_type=strategy_type,
            volatility_outcome=volatility_outcome,
            regime=regime,
            influence_weight=weight,
            cycle_id=cycle_id,
            contract=contract,
        )
        scores[agent] = score

    return scores


def reset_trackers():
    """Reset all tracker instances (call after migration)."""
    _trackers.clear()
