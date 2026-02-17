import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ReliabilityScorer:
    """Tracks and scores agent prediction accuracy using Brier scores."""

    def __init__(self, scores_file: str = "./data/KC/agent_scores.json"):
        self.scores_file = Path(scores_file)
        self.scores = self._load_scores()

        # Regime-based multipliers
        self.REGIME_BOOSTS = {
            "HIGH_VOLATILITY": {"agronomist": 2.0, "volatility": 2.0, "sentiment": 1.5},
            "RANGE_BOUND": {"technical": 2.0, "microstructure": 1.5},
            "WEATHER_EVENT": {"agronomist": 3.0, "supply_chain": 1.5},
            "MACRO_SHIFT": {"macro": 2.5, "geopolitical": 2.0},
        }

    def _load_scores(self) -> dict:
        if self.scores_file.exists():
            try:
                return json.loads(self.scores_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load scores file: {e}")
        return {}

    def get_weight(self, agent: str, regime: str = "NORMAL") -> float:
        """Calculate agent weight based on historical accuracy and current regime."""
        base_score = self.scores.get(agent, {}).get("accuracy", 0.5)
        # Normalize/Clip base score
        base_score = max(0.1, min(0.9, base_score))

        regime_boost = self.REGIME_BOOSTS.get(regime, {}).get(agent, 1.0)
        return base_score * regime_boost

    def update_score(self, agent: str, prediction: str, actual_outcome: str):
        """
        Update agent's Brier score after outcome is known.
        prediction: "BULLISH" | "BEARISH"
        actual_outcome: "BULLISH" | "BEARISH"
        """
        if agent not in self.scores:
            self.scores[agent] = {"correct": 0, "total": 0, "accuracy": 0.5}

        self.scores[agent]["total"] += 1
        if prediction == actual_outcome:
            self.scores[agent]["correct"] += 1

        # Simple accuracy for now (Brier score is (prob - outcome)^2, but we simplify to accuracy)
        self.scores[agent]["accuracy"] = (
            self.scores[agent]["correct"] / self.scores[agent]["total"]
        )

        try:
            self.scores_file.write_text(json.dumps(self.scores, indent=2))
        except Exception as e:
            logger.error(f"Failed to save scores: {e}")
