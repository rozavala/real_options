"""
Contribution-Based Agent Attribution Scoring.

Replaces Brier-vs-price-direction with "did the agent's vote help or
hurt the council's final decision?"

Three scoring paths:
  PATH 1 (DIRECTIONAL): alignment × outcome_sign × weight × confidence
  PATH 2 (VOLATILITY, non-vol agents): NEUTRAL=support, directional=oppose
  PATH 3 (VOLATILITY, vol agent): regime accuracy vs volatility_outcome

Key properties:
  - NEUTRAL domain votes = zero contribution (no penalty, no reward)
  - Confidence-weighted: high conviction amplifies both reward and penalty
  - Volatility agent scored on IV regime accuracy, not price direction
  - Master NEUTRAL + VOLATILITY = active strategy, scored on strategy outcome
  - Exponential recency decay in rolling average (half-life = 15 scores)
  - Min-sample blending across regimes to prevent wild swings
"""

import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

from trading_bot.enhanced_brier import normalize_regime

logger = logging.getLogger(__name__)

# Configuration constants
MAX_SCORES_PER_REGIME = 200
ROLLING_WINDOW = 30
RECENCY_HALF_LIFE = 15  # scores (~1 week at 3 cycles/day)
RECENCY_LAMBDA = math.log(2) / RECENCY_HALF_LIFE
MIN_SAMPLES_FULL_REGIME = 12   # 100% regime-specific
MIN_SAMPLES_BLEND_REGIME = 5   # 70/30 blend with cross-regime
BLEND_RATIO_REGIME = 0.7       # Weight for regime-specific in blend


def compress_confidence(raw_confidence: float) -> float:
    """
    Compress confidence to prevent multiplier explosion from miscalibrated LLMs.

    Maps [0.0, 1.0] → [0.5, 1.0].
    Even perfect confidence (1.0) only doubles score impact vs zero confidence.
    A typical HIGH (0.80) maps to 0.90 — moderate amplification.
    A typical LOW (0.55) maps to 0.775 — minimal amplification.

    Args:
        raw_confidence: Agent's stated confidence [0.0, 1.0]

    Returns:
        Compressed confidence [0.5, 1.0]
    """
    clamped = max(0.0, min(1.0, raw_confidence))
    return 0.5 + 0.5 * clamped


class ContributionTracker:
    """
    Tracks per-agent contribution scores and computes reliability multipliers.

    Storage: JSON file per commodity (data/{ticker}/contribution_scores.json).
    Structure mirrors EnhancedBrierTracker for consistency.
    """

    def __init__(self, data_path: str = None):
        if data_path is None:
            ticker = os.environ.get("COMMODITY_TICKER", "KC")
            data_path = f"./data/{ticker}/contribution_scores.json"
        self.data_path = data_path
        # Per-agent, per-regime score lists (most recent last)
        self.agent_scores: Dict[str, Dict[str, List[float]]] = {}
        self._load()

    def record_contribution(
        self,
        agent: str,
        agent_direction: str,
        agent_confidence: float,
        master_direction: str,
        actual_outcome: str,
        prediction_type: str = "DIRECTIONAL",
        strategy_type: str = "",
        volatility_outcome: str = "",
        regime: str = "NORMAL",
        influence_weight: float = 1.0,
        cycle_id: str = "",
        contract: str = "",
        override_score: Optional[float] = None,
    ) -> float:
        """
        Record a single agent's contribution score for one council cycle.

        Args:
            agent: Canonical agent name
            agent_direction: BULLISH, BEARISH, or NEUTRAL
            agent_confidence: Agent's stated confidence [0.0, 1.0]
            master_direction: Master's final call (NEUTRAL for vol cycles)
            actual_outcome: Materiality-thresholded outcome
            prediction_type: DIRECTIONAL or VOLATILITY
            strategy_type: LONG_STRADDLE, IRON_CONDOR, etc.
            volatility_outcome: BIG_MOVE or STAYED_FLAT (vol cycles only)
            regime: Market regime string
            influence_weight: From weighted voting final_weight
            cycle_id: Council cycle ID for traceability
            contract: Contract name for traceability
            override_score: If provided, skip computation and use this score directly

        Returns:
            The computed (or overridden) contribution score
        """
        from trading_bot.agent_names import normalize_agent_name
        agent = normalize_agent_name(agent)
        canonical_regime = normalize_regime(regime).value

        if override_score is not None:
            score = override_score
        else:
            score = self._compute_score(
                agent_name=agent,
                agent_direction=agent_direction,
                agent_confidence=agent_confidence,
                master_direction=master_direction,
                actual_outcome=actual_outcome,
                prediction_type=prediction_type,
                strategy_type=strategy_type,
                volatility_outcome=volatility_outcome,
                influence_weight=influence_weight,
            )

        # Store in per-agent, per-regime structure
        if agent not in self.agent_scores:
            self.agent_scores[agent] = {}
        if canonical_regime not in self.agent_scores[agent]:
            self.agent_scores[agent][canonical_regime] = []
        self.agent_scores[agent][canonical_regime].append(score)

        # Trim to prevent unbounded growth
        regime_scores = self.agent_scores[agent][canonical_regime]
        if len(regime_scores) > MAX_SCORES_PER_REGIME:
            self.agent_scores[agent][canonical_regime] = regime_scores[-MAX_SCORES_PER_REGIME:]

        self._save()

        logger.info(
            f"Contribution: {agent} ({agent_direction}, conf={agent_confidence:.2f}) "
            f"| Master={master_direction} | Outcome={actual_outcome} "
            f"| pred_type={prediction_type} "
            f"→ score={score:+.3f} [regime={canonical_regime}]"
        )

        return score

    def _compute_score(
        self,
        agent_name: str,
        agent_direction: str,
        agent_confidence: float,
        master_direction: str,
        actual_outcome: str,
        prediction_type: str,
        strategy_type: str,
        volatility_outcome: str,
        influence_weight: float,
    ) -> float:
        """
        Unified scoring with three paths.

        PATH 1 (DIRECTIONAL): Standard alignment × outcome scoring
        PATH 2 (VOLATILITY, non-vol agents): NEUTRAL=support, directional=oppose
        PATH 3 (VOLATILITY, vol agent): IV regime accuracy vs volatility_outcome
        """
        # Normalize inputs
        pred_type = (prediction_type or "DIRECTIONAL").upper().strip()
        vol_outcome = (volatility_outcome or "").upper().strip()
        strat_type = (strategy_type or "").upper().strip()
        agent_dir = (agent_direction or "NEUTRAL").upper().strip()
        master_dir = (master_direction or "NEUTRAL").upper().strip()
        outcome = (actual_outcome or "NEUTRAL").upper().strip()

        # Common: compress confidence and normalize weight
        eff_conf = compress_confidence(agent_confidence)
        norm_weight = min(1.0, influence_weight / 2.0)

        # ================================================================
        # PATH 3: VOLATILITY cycle — Volatility agent (special track)
        # Scored on IV regime accuracy, not price direction.
        # Prompt says: BULLISH = low IV = cheap options, BEARISH = high IV = expensive
        # ================================================================
        if pred_type == "VOLATILITY" and agent_name == "volatility":
            if agent_dir == "NEUTRAL":
                return 0.0  # No IV opinion → no contribution
            if not vol_outcome:
                return 0.0  # No vol outcome data → can't score

            correct_for_vol = (
                (agent_dir == "BULLISH" and vol_outcome == "BIG_MOVE") or
                (agent_dir == "BEARISH" and vol_outcome == "STAYED_FLAT")
            )
            alignment = 1.0 if correct_for_vol else -1.0

            # Outcome sign: was Master's strategy choice correct?
            master_correct = _is_vol_strategy_correct(strat_type, vol_outcome)
            outcome_sign = 1.0 if master_correct else -1.0

            return alignment * outcome_sign * norm_weight * eff_conf

        # ================================================================
        # PATH 2: VOLATILITY cycle — All other agents
        # NEUTRAL = supported the vol thesis (got out of the way)
        # Directional = opposed the vol thesis (pushed for direction)
        # ================================================================
        if pred_type == "VOLATILITY":
            if agent_dir == "NEUTRAL":
                alignment = 1.0   # Supported vol thesis
            else:
                alignment = -1.0  # Pushed for directional when Master chose vol

            # Outcome: was Master's strategy correct?
            master_correct = _is_vol_strategy_correct(strat_type, vol_outcome)
            outcome_sign = 1.0 if master_correct else -1.0

            return alignment * outcome_sign * norm_weight * eff_conf

        # ================================================================
        # PATH 1: DIRECTIONAL cycle — Standard scoring
        # ================================================================

        # Gate: NEUTRAL agent → 0 (abstention, no contribution)
        if agent_dir == "NEUTRAL":
            return 0.0

        # Gate: NEUTRAL master → 0 (no trade taken, nothing to attribute)
        if master_dir == "NEUTRAL":
            return 0.0

        # Gate: NEUTRAL outcome → 0 (market noise, nothing to learn)
        if outcome == "NEUTRAL":
            return 0.0

        # Alignment: did agent agree with Master?
        alignment = 1.0 if agent_dir == master_dir else -1.0

        # Outcome: was Master correct?
        outcome_sign = 1.0 if master_dir == outcome else -1.0

        return alignment * outcome_sign * norm_weight * eff_conf

    def get_agent_reliability(self, agent: str, regime: str = "NORMAL") -> float:
        """
        Get reliability multiplier from contribution scores.

        Drop-in replacement for EnhancedBrierTracker.get_agent_reliability().

        Uses min-sample blending:
          >= 12 samples: 100% regime-specific
          >= 5 samples: 70% regime + 30% cross-regime
          < 5 samples: cross-regime blend
          No data: 1.0 baseline

        Returns:
            Multiplier in [0.1, 2.0]
        """
        from trading_bot.agent_names import normalize_agent_name
        agent = normalize_agent_name(agent)
        canonical = normalize_regime(regime).value

        agent_regimes = self.agent_scores.get(agent, {})
        regime_scores = agent_regimes.get(canonical, [])

        # Path 1: Full regime-specific (>= 12 samples)
        if len(regime_scores) >= MIN_SAMPLES_FULL_REGIME:
            return self._scores_to_multiplier(regime_scores)

        # Compute cross-regime blend for fallback paths
        all_cross = []
        for r, r_scores in agent_regimes.items():
            if r_scores:
                mult = self._scores_to_multiplier(r_scores)
                all_cross.append((mult, len(r_scores)))
        cross_total = sum(n for _, n in all_cross) if all_cross else 0
        cross_mult = (
            sum(m * n for m, n in all_cross) / cross_total
            if cross_total > 0 else 1.0
        )

        # Path 2: Blended regime (>= 5 samples + cross-regime data)
        if len(regime_scores) >= MIN_SAMPLES_BLEND_REGIME:
            regime_mult = self._scores_to_multiplier(regime_scores)
            blended = BLEND_RATIO_REGIME * regime_mult + (1 - BLEND_RATIO_REGIME) * cross_mult
            return max(0.1, min(2.0, blended))

        # Path 3: Cross-regime only (>= 5 total samples)
        if cross_total >= MIN_SAMPLES_BLEND_REGIME:
            return max(0.1, min(2.0, cross_mult))

        # Path 4: No data — baseline
        return 1.0

    def _scores_to_multiplier(self, scores: list) -> float:
        """
        Convert contribution scores to reliability multiplier.

        Uses exponential recency decay: recent scores matter more.
        Half-life = RECENCY_HALF_LIFE scores.

        Maps weighted average from [-0.5, +0.5] to [0.1, 2.0]:
          avg ~ +0.5 → 2.0x (consistently helped winning decisions)
          avg ~  0.0 → 1.0x (neutral contributor / baseline)
          avg ~ -0.5 → 0.1x (consistently hurt decisions)
        """
        recent = scores[-ROLLING_WINDOW:]
        if not recent:
            return 1.0

        total_weight = 0.0
        weighted_sum = 0.0
        for i, score in enumerate(recent):
            # i=0 is oldest in window, i=len-1 is most recent
            age = len(recent) - 1 - i  # 0 for newest, len-1 for oldest
            decay = math.exp(-RECENCY_LAMBDA * age)
            weighted_sum += score * decay
            total_weight += decay

        avg = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Linear map: -0.5 → 0.1, 0.0 → 1.0, +0.5 → 2.0
        multiplier = 1.0 + (avg * 2.0)
        return max(0.1, min(2.0, multiplier))

    def get_summary(self) -> Dict:
        """Get summary for dashboard display."""
        summary = {}
        for agent, regimes in self.agent_scores.items():
            summary[agent] = {}
            for regime, scores in regimes.items():
                mult = self._scores_to_multiplier(scores)
                recent = scores[-ROLLING_WINDOW:]
                # Simple average for display (recency-weighted is used for multiplier)
                avg = sum(recent) / len(recent) if recent else 0.0
                summary[agent][regime] = {
                    "avg_score": round(avg, 4),
                    "multiplier": round(mult, 3),
                    "total_scores": len(scores),
                    "recent_count": len(recent),
                }
        return summary

    def _load(self):
        """Load from JSON."""
        if not os.path.exists(self.data_path):
            return
        try:
            with open(self.data_path, "r") as f:
                data = json.load(f)
            self.agent_scores = data.get("agent_scores", {})
            logger.info(
                f"Loaded contribution scores for "
                f"{len(self.agent_scores)} agents from {self.data_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load contribution scores: {e}")

    def _save(self):
        """Persist to JSON (atomic write)."""
        data = {
            "agent_scores": self.agent_scores,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        dir_path = os.path.dirname(self.data_path)
        try:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=dir_path or ".", suffix=".tmp"
            )
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.data_path)
        except Exception as e:
            logger.error(f"Failed to save contribution scores: {e}")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ============================================================================
# MODULE-LEVEL HELPERS
# ============================================================================

def _is_vol_strategy_correct(strategy_type: str, volatility_outcome: str) -> bool:
    """
    Was the Master's volatility strategy choice correct?

    IRON_CONDOR + STAYED_FLAT = correct (sold premium, market stayed in range)
    LONG_STRADDLE + BIG_MOVE = correct (bought vol, market moved)
    Anything else = incorrect
    """
    strat = (strategy_type or "").upper().strip()
    vol_out = (volatility_outcome or "").upper().strip()

    if not strat or not vol_out:
        return False  # Can't evaluate without both

    return (
        (strat == "IRON_CONDOR" and vol_out == "STAYED_FLAT") or
        (strat == "LONG_STRADDLE" and vol_out == "BIG_MOVE")
    )
