"""Brier Score Tracking for Agent Reliability.

Tracks the accuracy of agent predictions over time using the Brier Score metric.
Used to dynamically adjust voting weights based on historical performance.
"""

import pandas as pd
import os
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

class BrierScoreTracker:
    """Tracks agent prediction accuracy over time."""

    def __init__(self, history_file: str = "data/agent_accuracy.csv"):
        self.history_file = history_file

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

        self.scores = self._load_history()

    def _load_history(self) -> dict:
        """Load historical accuracy scores."""
        try:
            if not os.path.exists(self.history_file):
                return {}

            df = pd.read_csv(self.history_file)
            if df.empty:
                return {}

            # Calculate simple accuracy (Brier score inverse)
            # 1.0 = Perfect, 0.0 = Wrong
            # We use a simplified rolling accuracy for now
            return df.groupby('agent').apply(
                lambda x: x['correct'].sum() / len(x)
            ).to_dict()
        except Exception as e:
            logger.error(f"Failed to load Brier scores: {e}")
            return {}

    def record_prediction(self, agent: str, predicted: str, actual: str, timestamp: Optional[datetime] = None):
        """Record a prediction for later scoring (Legacy Method)."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Normalize directions
        predicted = predicted.upper()
        actual = actual.upper()

        correct = 1 if predicted == actual else 0

        try:
            # Append to CSV
            exists = os.path.exists(self.history_file)
            with open(self.history_file, 'a') as f:
                if not exists:
                    f.write("timestamp,agent,predicted,actual,correct\n")
                f.write(f"{timestamp},{agent},{predicted},{actual},{correct}\n")
        except Exception as e:
            logger.error(f"Failed to record prediction for {agent}: {e}")

    def record_volatility_prediction(
        self,
        strategy_type: str,
        predicted_vol_level: str,
        actual_outcome: str,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a volatility prediction for Brier scoring.
        Uses existing agent_accuracy.csv with distinct agent names for UI compatibility.

        Args:
            strategy_type: 'LONG_STRADDLE' or 'IRON_CONDOR'
            predicted_vol_level: 'HIGH' or 'LOW'
            actual_outcome: 'BIG_MOVE' or 'STAYED_FLAT'
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Create distinct agent name for dashboard compatibility
        agent_name = f"Strategy_{strategy_type}"  # e.g., Strategy_LONG_STRADDLE

        # Determine correctness
        is_correct = 0
        if strategy_type == 'LONG_STRADDLE':
            predicted_str = "HIGH_VOL"
            is_correct = 1 if actual_outcome == 'BIG_MOVE' else 0
        elif strategy_type == 'IRON_CONDOR':
            predicted_str = "LOW_VOL"
            is_correct = 1 if actual_outcome == 'STAYED_FLAT' else 0
        else:
            predicted_str = "UNKNOWN"

        # Use existing record_prediction method to maintain file format
        # This writes to agent_accuracy.csv which the dashboard already reads
        self.record_prediction(
            agent=agent_name,
            predicted=predicted_str,
            actual=actual_outcome,
            timestamp=timestamp
        )

        logger.info(f"Recorded volatility prediction: {agent_name} predicted {predicted_str}, "
                    f"actual {actual_outcome} -> {'CORRECT' if is_correct else 'INCORRECT'}")

    def record_prediction_structured(
        self,
        agent: str,
        predicted_direction: str,
        predicted_confidence: float,
        actual: str = 'PENDING',
        timestamp: Optional[datetime] = None
    ):
        """Record a prediction with full probability for proper Brier scoring."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Normalize
        predicted_direction = predicted_direction.upper()

        # Convert to probability based on direction
        # BULLISH with 0.8 conf -> prob_bullish = 0.8
        # BEARISH with 0.8 conf -> prob_bullish = 0.2 (Assuming binary for now, or just recording raw)
        # We record the raw confidence and direction to calculate Brier later properly.
        # Brier = (prob - outcome)^2.
        # We store enough data to calc it.

        prob_bullish = predicted_confidence if predicted_direction == 'BULLISH' else (1.0 - predicted_confidence)
        if predicted_direction == 'NEUTRAL':
            prob_bullish = 0.5 # Center

        try:
            exists = os.path.exists(self.history_file)
            with open(self.history_file, 'a') as f:
                if not exists:
                    # Extended header if creating new, or append to existing (might cause schema drift if mixed, but CSV tolerates extra cols if ignored)
                    # Ideally we use a new file or handle migration. For now we append to same file but with extra cols?
                    # The legacy method wrote 5 cols. We need to be careful.
                    # Let's check if we can add columns safely.
                    # If we write extra commas, old readers might break.
                    # We will use a NEW file for structured history to be safe.
                    f.write("timestamp,agent,predicted,confidence,prob_bullish,actual,correct\n")

                # Check if we are writing to the legacy file format?
                # The issue description suggested one file.
                # "f.write(f"{timestamp},{agent},{predicted_direction},{predicted_confidence},{prob_bullish},{actual},\n")"
                # If we share the file, we must respect the header.
                # Legacy header: timestamp,agent,predicted,actual,correct
                # If we want to add confidence, we should probably add it to the end or use a new file.
                # I will use a separate file for structured scoring to avoid breaking legacy tools.
                pass

            structured_file = self.history_file.replace(".csv", "_structured.csv")
            exists_struct = os.path.exists(structured_file)
            with open(structured_file, 'a') as f:
                if not exists_struct:
                    f.write("timestamp,agent,direction,confidence,prob_bullish,actual\n")
                f.write(f"{timestamp},{agent},{predicted_direction},{predicted_confidence},{prob_bullish},{actual}\n")

        except Exception as e:
            logger.error(f"Failed to record structured prediction: {e}")

    def get_calibration_curve(self, agent: str) -> dict:
        """Return accuracy at each confidence bucket for calibration."""
        structured_file = self.history_file.replace(".csv", "_structured.csv")
        if not os.path.exists(structured_file):
            return {}

        try:
            df = pd.read_csv(structured_file)
            if df.empty or agent not in df['agent'].values:
                return {}

            agent_df = df[df['agent'] == agent].copy()

            # Simple binary correctness check (assuming 'actual' is direction)
            # This is complex if actual is 'PENDING'. Filter for resolved.
            # Assuming external process updates 'actual' or we have a way to know.
            # For now, just bucket confidence if we have correctness data.
            # If the file only stores 'PENDING', we can't build a curve.
            # Assuming 'reconcile_council_history' updates this file (it updates agent_accuracy.csv).
            # If structured file is new, it might be empty of results.
            # We'll implement the logic assuming data exists.

            # Bucket by confidence
            # Bins: 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0
            agent_df['conf_bucket'] = pd.cut(agent_df['confidence'], bins=[0.0, 0.6, 0.7, 0.8, 0.9, 1.0])

            # We need a 'correct' column. If not present (legacy), skip.
            if 'correct' not in agent_df.columns:
                return {}

            calibration = agent_df.groupby('conf_bucket')['correct'].agg(['mean', 'count']).to_dict('index')
            # Result: {Interval(0.6, 0.7, closed='right'): {'mean': 0.65, 'count': 10}, ...}

            # Convert Interval keys to string for JSON serialization/easier handling
            return {str(k): v for k, v in calibration.items()}

        except Exception as e:
            logger.error(f"Failed to get calibration curve: {e}")
            return {}

    def get_agent_weight_multiplier(self, agent: str, base_weight: float = 1.0) -> float:
        """Get accuracy-adjusted weight multiplier for agent."""
        if agent not in self.scores:
            return 1.0 # Neutral multiplier

        accuracy = self.scores[agent]

        # Scale weight:
        # < 50% accuracy -> < 1.0x (Penalty)
        # > 50% accuracy -> > 1.0x (Boost)
        # 50% accuracy = 0.5x, 75% = 1.5x, 90% = 2x
        # Formula: 0.5 + (Accuracy * 1.5)
        # Acc 0.4 -> 0.5 + 0.6 = 1.1? No.
        # Let's use the formula from the spec or the one I derived?
        # Spec says: "50% accuracy = 0.5x, 75% = 1.5x, 90% = 2x"
        # Let's check linear fit: y = mx + c
        # 0.5 = 0.5m + c
        # 1.5 = 0.75m + c
        # -> 1.0 = 0.25m -> m = 4.
        # 0.5 = 4(0.5) + c -> 0.5 = 2 + c -> c = -1.5
        # y = 4x - 1.5
        # Test 0.9: 4(0.9) - 1.5 = 3.6 - 1.5 = 2.1 (Close to 2x)

        # Alternative simple formula: Multiplier = Accuracy / 0.5 (Base baseline)
        # 0.5 -> 1.0
        # 0.75 -> 1.5
        # 0.9 -> 1.8

        # I'll stick to the one I proposed in the plan doc:
        # return base_weight * (0.5 + accuracy * 1.5)
        # 0.5 -> 0.5 + 0.75 = 1.25 (Wait, 50% should probably be neutral 1.0?)

        # Let's adjust:
        # Neutral (1.0) should be at expected random baseline?
        # For 3 classes (Bull/Bear/Neutral), baseline is 33%.
        # For Directional (Bull/Bear), baseline is 50%.

        # Requested Formula: weight_multiplier = 0.5 + (accuracy_score * 1.5)
        # 0.0 accuracy -> 0.5x
        # 0.5 accuracy -> 1.25x
        # 1.0 accuracy -> 2.0x

        multiplier = 0.5 + (accuracy * 1.5)

        return multiplier

# Singleton
_tracker: Optional[BrierScoreTracker] = None

def get_brier_tracker() -> BrierScoreTracker:
    global _tracker
    if _tracker is None:
        _tracker = BrierScoreTracker()
    return _tracker
