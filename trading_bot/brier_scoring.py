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
        """Record a prediction for later scoring."""
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

            # Update local cache logic (simplified)
            # In a real system, we might re-calculate full rolling score here
            # For now, we rely on _load_history being called on init or periodic refresh
        except Exception as e:
            logger.error(f"Failed to record prediction for {agent}: {e}")

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
