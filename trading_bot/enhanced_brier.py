"""
Enhanced Brier Scoring with Probabilistic Calibration.

This module extends basic accuracy tracking to include:
- Full probabilistic Brier scores
- Calibration curves per agent
- Regime-aware performance weighting
- Dynamic reliability multipliers
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import os
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification for performance tracking."""
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    WEATHER_EVENT = "WEATHER_EVENT"
    MACRO_SHIFT = "MACRO_SHIFT"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE_BOUND = "RANGE_BOUND"
    UNKNOWN = "UNKNOWN"


# Canonical mapping: any regime string → MarketRegime enum
# This is the SINGLE SOURCE OF TRUTH for regime normalization.
_REGIME_ALIASES: Dict[str, MarketRegime] = {
    # Exact matches (already correct)
    "NORMAL": MarketRegime.NORMAL,
    "HIGH_VOL": MarketRegime.HIGH_VOL,
    "WEATHER_EVENT": MarketRegime.WEATHER_EVENT,
    "MACRO_SHIFT": MarketRegime.MACRO_SHIFT,
    "TRENDING_UP": MarketRegime.TRENDING_UP,
    "TRENDING_DOWN": MarketRegime.TRENDING_DOWN,
    "RANGE_BOUND": MarketRegime.RANGE_BOUND,
    "UNKNOWN": MarketRegime.UNKNOWN,
    # Aliases from detect_market_regime_simple() and RegimeDetector
    "HIGH_VOLATILITY": MarketRegime.HIGH_VOL,
    "TRENDING": MarketRegime.TRENDING_UP,  # Ambiguous → default to UP
    # Edge cases
    "LOW_VOL": MarketRegime.RANGE_BOUND,
    "": MarketRegime.NORMAL,
}


def normalize_regime(regime_str: str) -> MarketRegime:
    """
    Normalize any regime string to a MarketRegime enum value.

    Call this at every system boundary where a regime string enters
    the Brier scoring pipeline.

    Args:
        regime_str: Raw regime string from any producer

    Returns:
        MarketRegime enum value (never raises, defaults to NORMAL)
    """
    if isinstance(regime_str, MarketRegime):
        return regime_str

    cleaned = str(regime_str).strip().upper()
    result = _REGIME_ALIASES.get(cleaned, MarketRegime.NORMAL)

    if cleaned and cleaned not in _REGIME_ALIASES:
        logger.warning(
            f"Unknown regime '{regime_str}' normalized to NORMAL. "
            f"Add it to _REGIME_ALIASES if this is a valid regime."
        )

    return result


@dataclass
class ProbabilisticPrediction:
    """A prediction with full probability distribution."""
    timestamp: datetime
    agent: str

    # Probability distribution (must sum to 1.0)
    prob_bullish: float
    prob_neutral: float
    prob_bearish: float

    # Metadata
    regime: MarketRegime = MarketRegime.NORMAL
    contract: str = ""
    cycle_id: str = ""  # NEW: Deterministic foreign key

    # Outcome (filled after resolution)
    actual_outcome: Optional[str] = None
    resolved_at: Optional[datetime] = None

    def __post_init__(self):
        # Normalize probabilities
        total = self.prob_bullish + self.prob_neutral + self.prob_bearish
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Probabilities don't sum to 1.0 ({total}), normalizing")
            self.prob_bullish /= total
            self.prob_neutral /= total
            self.prob_bearish /= total

    @property
    def predicted_direction(self) -> str:
        """Return the highest probability direction."""
        probs = {
            'BULLISH': self.prob_bullish,
            'NEUTRAL': self.prob_neutral,
            'BEARISH': self.prob_bearish
        }
        return max(probs, key=probs.get)

    @property
    def confidence(self) -> float:
        """Return the confidence (highest probability)."""
        return max(self.prob_bullish, self.prob_neutral, self.prob_bearish)

    def calc_brier_score(self) -> Optional[float]:
        """
        Calculate Brier score for this prediction.

        Brier Score = mean((probability - outcome)^2) across all classes
        Lower is better (0 = perfect, 0.5 = random)
        """
        if self.actual_outcome is None:
            return None

        # Convert outcome to one-hot
        outcome_vec = [0.0, 0.0, 0.0]
        if self.actual_outcome == 'BULLISH':
            outcome_vec[0] = 1.0
        elif self.actual_outcome == 'NEUTRAL':
            outcome_vec[1] = 1.0
        elif self.actual_outcome == 'BEARISH':
            outcome_vec[2] = 1.0
        else:
            return None

        # Calculate Brier score
        pred_vec = [self.prob_bullish, self.prob_neutral, self.prob_bearish]
        brier = sum((p - o) ** 2 for p, o in zip(pred_vec, outcome_vec)) / 3

        return brier


@dataclass
class CalibrationBucket:
    """Bucket for calibration curve calculation."""
    lower_bound: float
    upper_bound: float
    predictions: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> Optional[float]:
        """
        Return accuracy or None if no predictions in this bucket.

        FIX (MECE 1.6): Return None instead of 0.0 for empty buckets.
        This prevents misleading "0% accurate" in calibration curves
        when the truth is "unknown/no data".
        """
        if self.predictions == 0:
            return None  # Unknown, not zero
        return self.correct / self.predictions

    @property
    def midpoint(self) -> float:
        return (self.lower_bound + self.upper_bound) / 2


class EnhancedBrierTracker:
    """
    Enhanced Brier scoring with calibration and regime tracking.

    Key features:
    - Full probabilistic scoring (not just accuracy)
    - Calibration curves per agent
    - Regime-specific performance
    - Dynamic weight multipliers
    """

    def __init__(self, data_path: str = "./data/enhanced_brier.json"):
        self.data_path = data_path
        self.predictions: List[ProbabilisticPrediction] = []

        # Calibration buckets (10 bins: 0-10%, 10-20%, ..., 90-100%)
        self.calibration_buckets: Dict[str, List[CalibrationBucket]] = {}

        # Per-agent, per-regime Brier scores
        self.agent_scores: Dict[str, Dict[str, List[float]]] = {}

        # Load existing data
        self._load()

    def record_prediction(
        self,
        agent: str,
        prob_bullish: float,
        prob_neutral: float,
        prob_bearish: float,
        regime: MarketRegime = MarketRegime.NORMAL,
        contract: str = "",
        timestamp: Optional[datetime] = None,
        cycle_id: str = ""  # B3 FIX: Now REQUIRED
    ) -> str:
        """
        Record a new prediction.

        Returns:
            Prediction ID for later resolution
        """
        if not cycle_id or cycle_id in ("nan", "None", "null"):
            raise ValueError("cycle_id is required for prediction recording (B3 fix)")

        pred = ProbabilisticPrediction(
            timestamp=timestamp or datetime.now(timezone.utc),
            agent=agent,
            prob_bullish=prob_bullish,
            prob_neutral=prob_neutral,
            prob_bearish=prob_bearish,
            regime=regime,
            contract=contract,
            cycle_id=cycle_id
        )

        self.predictions.append(pred)

        # FIX (MECE 1.4): Trim to prevent unbounded memory growth
        # Keep 50% buffer over save limit to reduce trim frequency
        MAX_IN_MEMORY = 1500
        if len(self.predictions) > MAX_IN_MEMORY:
            self.predictions = self.predictions[-1000:]
            logger.debug(f"Brier tracker trimmed to {len(self.predictions)} predictions")

        pred_id = f"{agent}_{pred.timestamp.isoformat()}"

        logger.debug(f"Recorded prediction: {pred_id} -> {pred.predicted_direction} ({pred.confidence:.2f})")

        # FIX: Persist to disk so predictions survive process restarts.
        # Without this, predictions exist only in memory and are lost when the
        # orchestrator restarts, making resolution impossible.
        # Batched save: only save every N predictions to reduce I/O overhead.
        if len(self.predictions) % 8 == 0:
            self._save()

        return pred_id

    def resolve_prediction(
        self,
        agent: str,
        timestamp: datetime = None,
        actual_outcome: str = None,
        cycle_id: str = ""  # NEW: preferred lookup key
    ) -> Optional[float]:
        """
        Resolve a prediction with actual outcome.

        Uses cycle_id for lookup if provided (preferred),
        falls back to agent+timestamp matching (legacy).

        Returns:
            Brier score for this prediction
        """
        # DEFENSIVE GUARD
        if cycle_id in ("nan", "None", "null", None):
            cycle_id = ""

        from trading_bot.cycle_id import is_valid_cycle_id

        # Find matching prediction
        for pred in self.predictions:
            # PRIMARY: match on cycle_id
            is_match = False

            if is_valid_cycle_id(cycle_id) and pred.cycle_id == cycle_id and pred.agent == agent:
                is_match = True
            # FALLBACK: match on agent + timestamp (legacy)
            elif not is_valid_cycle_id(cycle_id) and timestamp:
                if pred.agent == agent and pred.timestamp == timestamp:
                    is_match = True

            if is_match and pred.actual_outcome is None:
                pred.actual_outcome = actual_outcome
                pred.resolved_at = datetime.now(timezone.utc)

                # Calculate Brier score
                brier = pred.calc_brier_score()

                if brier is not None:
                    # Update agent scores
                    self._update_agent_score(pred.agent, pred.regime.value, brier)

                    # Update calibration buckets
                    self._update_calibration(pred)

                    # Persist
                    self._save()

                    logger.info(f"Resolved {agent}: {pred.predicted_direction} vs {actual_outcome}, Brier={brier:.4f}")

                return brier

        logger.warning(f"No matching prediction found for {agent} (cycle={cycle_id})")
        return None

    def get_agent_reliability(self, agent: str, regime: str = "NORMAL") -> float:
        """
        Get reliability multiplier for an agent in a specific regime.

        Returns:
            Multiplier in range [0.1, 2.0]
            - 0.1 = very unreliable (Brier close to 0.5)
            - 1.0 = baseline reliability (Brier ~0.25)
            - 2.0 = very reliable (Brier close to 0)
        """
        # FIX (P1-C, 2026-02-06): Normalize agent name to prevent lookup misses.
        from trading_bot.agent_names import normalize_agent_name
        agent = normalize_agent_name(agent)

        # Normalize regime (P0-REGIME)
        canonical = normalize_regime(regime).value
        scores = self.agent_scores.get(agent, {}).get(canonical, [])

        if len(scores) < 5:
            # Not enough data, return baseline
            return 1.0

        # Use recent scores (last 30)
        recent_scores = scores[-30:]
        avg_brier = np.mean(recent_scores)

        # Convert Brier to multiplier
        # Brier 0.0 -> 2.0x, Brier 0.25 -> 1.0x, Brier 0.5 -> 0.1x
        multiplier = 2.0 - (avg_brier * 4.0)
        multiplier = max(0.1, min(2.0, multiplier))

        return multiplier

    def get_calibration_curve(self, agent: str) -> List[Tuple[float, float]]:
        """
        Get calibration curve for an agent.

        Returns:
            List of (predicted_probability, actual_accuracy) tuples
        """
        buckets = self.calibration_buckets.get(agent, [])

        curve = []
        for bucket in buckets:
            if bucket.predictions > 0:
                curve.append((bucket.midpoint, bucket.accuracy))

        return curve

    def get_summary(self, agent: Optional[str] = None) -> Dict:
        """Get summary statistics."""
        if agent:
            agents = [agent]
        else:
            agents = list(self.agent_scores.keys())

        summary = {}
        for a in agents:
            agent_data = self.agent_scores.get(a, {})
            all_scores = []
            regime_avgs = {}

            for regime, scores in agent_data.items():
                if scores:
                    regime_avgs[regime] = np.mean(scores[-30:])
                    all_scores.extend(scores[-30:])

            summary[a] = {
                'overall_brier': np.mean(all_scores) if all_scores else None,
                'regime_brier': regime_avgs,
                'reliability': self.get_agent_reliability(a),
                'total_predictions': len([p for p in self.predictions if p.agent == a and p.actual_outcome]),
            }

        return summary

    def _update_agent_score(self, agent: str, regime: str, brier: float) -> None:
        """Update agent's Brier score history."""
        if agent not in self.agent_scores:
            self.agent_scores[agent] = {}

        if regime not in self.agent_scores[agent]:
            self.agent_scores[agent][regime] = []

        self.agent_scores[agent][regime].append(brier)

    def _update_calibration(self, pred: ProbabilisticPrediction) -> None:
        """Update calibration buckets for this prediction."""
        agent = pred.agent

        # Initialize buckets if needed
        if agent not in self.calibration_buckets:
            self.calibration_buckets[agent] = [
                CalibrationBucket(lower_bound=i/10, upper_bound=(i+1)/10)
                for i in range(10)
            ]

        # Find the bucket for this prediction's confidence
        confidence = pred.confidence
        bucket_idx = min(int(confidence * 10), 9)

        # Update bucket
        self.calibration_buckets[agent][bucket_idx].predictions += 1
        if pred.predicted_direction == pred.actual_outcome:
            self.calibration_buckets[agent][bucket_idx].correct += 1

    # Schema version for data file (MECE Issue 3.3)
    SCHEMA_VERSION = "1.0"

    def _save(self) -> None:
        """Persist data to disk."""
        data = {
            # FIX (MECE 3.3): Add schema version for forward compatibility
            'schema_version': self.SCHEMA_VERSION,
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'predictions': [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'agent': p.agent,
                    'prob_bullish': p.prob_bullish,
                    'prob_neutral': p.prob_neutral,
                    'prob_bearish': p.prob_bearish,
                    'regime': p.regime.value,
                    'contract': p.contract,
                    'cycle_id': p.cycle_id,  # NEW
                    'actual_outcome': p.actual_outcome,
                    'resolved_at': p.resolved_at.isoformat() if p.resolved_at else None
                }
                for p in self.predictions[-1000:]  # Keep last 1000
            ],
            'agent_scores': {
                agent: {
                    regime: scores[-100:]  # Keep last 100 per regime
                    for regime, scores in regimes.items()
                }
                for agent, regimes in self.agent_scores.items()
            },
            'calibration_buckets': {
                agent: [
                    {'lower': b.lower_bound, 'upper': b.upper_bound,
                     'predictions': b.predictions, 'correct': b.correct}
                    for b in buckets
                ]
                for agent, buckets in self.calibration_buckets.items()
            }
        }

        # FIX: Wrap file I/O in try/except - save failures should not crash trading
        try:
            os.makedirs(os.path.dirname(self.data_path) or '.', exist_ok=True)
            with open(self.data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save enhanced Brier data (non-fatal): {e}")

    def _load(self) -> None:
        """Load data from disk."""
        if not os.path.exists(self.data_path):
            return

        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)

            # Check schema version (MECE Issue 3.3)
            file_version = data.get('schema_version', '0.9')  # Pre-versioning assumed 0.9
            if file_version != self.SCHEMA_VERSION:
                logger.warning(
                    f"Brier data schema mismatch: file={file_version}, expected={self.SCHEMA_VERSION}. "
                    f"Data may need migration."
                )

            # Load predictions
            for p in data.get('predictions', []):
                pred = ProbabilisticPrediction(
                    timestamp=datetime.fromisoformat(p['timestamp']),
                    agent=p['agent'],
                    prob_bullish=p['prob_bullish'],
                    prob_neutral=p['prob_neutral'],
                    prob_bearish=p['prob_bearish'],
                    regime=MarketRegime(p.get('regime', 'NORMAL')),
                    contract=p.get('contract', ''),
                    cycle_id=p.get('cycle_id', ''),  # NEW
                    actual_outcome=p.get('actual_outcome'),
                    resolved_at=datetime.fromisoformat(p['resolved_at']) if p.get('resolved_at') else None
                )
                self.predictions.append(pred)

            # Load agent scores
            self.agent_scores = data.get('agent_scores', {})

            # Load calibration buckets
            for agent, buckets in data.get('calibration_buckets', {}).items():
                self.calibration_buckets[agent] = [
                    CalibrationBucket(
                        lower_bound=b['lower'],
                        upper_bound=b['upper'],
                        predictions=b['predictions'],
                        correct=b['correct']
                    )
                    for b in buckets
                ]

            logger.info(f"Loaded {len(self.predictions)} predictions from {self.data_path}")

        except Exception as e:
            logger.error(f"Failed to load Brier data: {e}")
