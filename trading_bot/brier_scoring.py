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
        """
        Load historical accuracy scores from agent_accuracy.csv.

        Returns dict of canonical_agent_name -> accuracy (0.0 to 1.0)
        """
        try:
            if not os.path.exists(self.history_file):
                logger.info("No accuracy history file found")
                return {}

            df = pd.read_csv(self.history_file)

            if df.empty:
                logger.info("Accuracy history file is empty")
                return {}

            # Validate expected columns
            expected_cols = ['timestamp', 'agent', 'predicted', 'actual', 'correct']
            if not all(col in df.columns for col in expected_cols):
                logger.warning(f"Accuracy file has unexpected schema: {list(df.columns)}")
                # Try to work with what we have
                if 'agent' not in df.columns or 'correct' not in df.columns:
                    return {}

            # Normalize agent names
            from trading_bot.agent_names import normalize_agent_name
            df['agent_normalized'] = df['agent'].apply(normalize_agent_name)

            # Ensure 'correct' is numeric
            df['correct'] = pd.to_numeric(df['correct'], errors='coerce').fillna(0)

            # Calculate accuracy per agent
            scores = df.groupby('agent_normalized').apply(
                lambda x: x['correct'].sum() / len(x) if len(x) > 0 else 0.5
            ).to_dict()

            logger.info(f"Loaded Brier scores for {len(scores)} agents: {scores}")
            return scores

        except Exception as e:
            logger.error(f"Failed to load Brier scores: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def record_prediction(self, agent: str, predicted: str, actual: str, timestamp: Optional[datetime] = None):
        """
        Record a prediction outcome to agent_accuracy.csv.

        This is the "legacy" method used by reconciliation for master_decision and ml_model.
        Agent names are normalized before writing.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        from trading_bot.agent_names import normalize_agent_name

        # Normalize
        agent = normalize_agent_name(agent)
        predicted = predicted.upper()
        actual = actual.upper()

        correct = 1 if predicted == actual else 0

        try:
            exists = os.path.exists(self.history_file)
            with open(self.history_file, 'a') as f:
                if not exists:
                    f.write("timestamp,agent,predicted,actual,correct\n")
                f.write(f"{timestamp},{agent},{predicted},{actual},{correct}\n")

            logger.debug(f"Recorded prediction: {agent} predicted {predicted}, actual {actual} -> {'CORRECT' if correct else 'INCORRECT'}")

            # Update in-memory scores
            if agent not in self.scores:
                self.scores[agent] = correct
            else:
                # Simple moving average approximation
                self.scores[agent] = (self.scores[agent] * 0.9) + (correct * 0.1)

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

        predicted_direction = predicted_direction.upper()

        prob_bullish = predicted_confidence if predicted_direction == 'BULLISH' else (1.0 - predicted_confidence)
        if predicted_direction == 'NEUTRAL':
            prob_bullish = 0.5

        try:
            # REMOVED: The dead code that writes wrong header to self.history_file
            # We ONLY write to the structured file

            structured_file = self.history_file.replace(".csv", "_structured.csv")
            exists_struct = os.path.exists(structured_file)

            with open(structured_file, 'a') as f:
                if not exists_struct:
                    f.write("timestamp,agent,direction,confidence,prob_bullish,actual\n")
                f.write(f"{timestamp},{agent},{predicted_direction},{predicted_confidence},{prob_bullish},{actual}\n")

            logger.debug(f"Recorded structured prediction: {agent} -> {predicted_direction}")

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

    def get_agent_weight_multiplier(self, agent: str, min_samples: int = 5) -> float:
        """
        Get accuracy-adjusted weight multiplier for an agent.

        Formula: multiplier = 0.5 + (accuracy * 1.5)
        - 0% accuracy -> 0.5x (heavily penalized)
        - 50% accuracy -> 1.25x (slight boost for being better than random)
        - 100% accuracy -> 2.0x (maximum boost)

        Args:
            agent: Agent name (will be normalized)
            min_samples: Minimum samples required before adjusting from baseline

        Returns:
            Weight multiplier (0.5 to 2.0)
        """
        from trading_bot.agent_names import normalize_agent_name

        # Normalize the agent name
        normalized = normalize_agent_name(agent)

        if normalized not in self.scores:
            logger.debug(f"No accuracy data for {agent} (normalized: {normalized}), using 1.0x")
            return 1.0

        accuracy = self.scores[normalized]

        # Apply formula
        multiplier = 0.5 + (accuracy * 1.5)

        # Clamp to reasonable range
        multiplier = max(0.5, min(2.0, multiplier))

        logger.debug(f"Agent {normalized}: accuracy={accuracy:.2%}, multiplier={multiplier:.2f}x")
        return multiplier

    def get_diagnostics(self) -> dict:
        """
        Return diagnostic information about the Brier score system.
        Useful for debugging and dashboard display.
        """
        diagnostics = {
            'history_file': self.history_file,
            'history_file_exists': os.path.exists(self.history_file),
            'structured_file_exists': os.path.exists(self.history_file.replace('.csv', '_structured.csv')),
            'agents_tracked': list(self.scores.keys()),
            'agent_scores': dict(self.scores),
            'total_agents': len(self.scores),
        }

        # Count records in files
        if os.path.exists(self.history_file):
            try:
                df = pd.read_csv(self.history_file)
                diagnostics['legacy_record_count'] = len(df)
                diagnostics['legacy_agents'] = df['agent'].unique().tolist() if 'agent' in df.columns else []
            except:
                diagnostics['legacy_record_count'] = 'ERROR'

        structured_file = self.history_file.replace('.csv', '_structured.csv')
        if os.path.exists(structured_file):
            try:
                df = pd.read_csv(structured_file)
                diagnostics['structured_record_count'] = len(df)
                diagnostics['pending_count'] = len(df[df['actual'] == 'PENDING'])
                diagnostics['resolved_count'] = len(df[df['actual'] != 'PENDING'])
            except:
                diagnostics['structured_record_count'] = 'ERROR'

        return diagnostics

# Singleton
_tracker: Optional[BrierScoreTracker] = None

def get_brier_tracker() -> BrierScoreTracker:
    global _tracker
    if _tracker is None:
        _tracker = BrierScoreTracker()
    return _tracker

def resolve_pending_predictions(council_history_path: str = "data/council_history.csv") -> int:
    """
    Resolve PENDING predictions by cross-referencing with reconciled council_history.

    IDEMPOTENT: Running this multiple times will NOT duplicate data because
    we only process rows where actual == 'PENDING'.

    Args:
        council_history_path: Path to the reconciled council history CSV

    Returns:
        Number of newly resolved predictions
    """
    structured_file = "data/agent_accuracy_structured.csv"

    if not os.path.exists(structured_file):
        logger.info("No structured predictions file found - nothing to resolve")
        return 0

    if not os.path.exists(council_history_path):
        logger.warning(f"Council history not found at {council_history_path}")
        return 0

    try:
        # Load both files
        predictions_df = pd.read_csv(structured_file)
        council_df = pd.read_csv(council_history_path)

        if predictions_df.empty or council_df.empty:
            logger.info("Empty dataframes - nothing to resolve")
            return 0

        # Ensure timestamp columns are datetime with UTC
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'], utc=True)
        council_df['timestamp'] = pd.to_datetime(council_df['timestamp'], utc=True)

        # Filter to ONLY PENDING predictions - this ensures idempotency
        pending_mask = predictions_df['actual'] == 'PENDING'
        pending_count = pending_mask.sum()

        if pending_count == 0:
            logger.info("No pending predictions to resolve")
            return 0

        logger.info(f"Found {pending_count} pending predictions to check")

        # Track which indices we resolve THIS RUN
        newly_resolved_indices = []

        for idx in predictions_df[pending_mask].index:
            pred_row = predictions_df.loc[idx]
            pred_time = pred_row['timestamp']

            # Find matching council decision within 5-minute window
            time_window = pd.Timedelta(minutes=5)
            matches = council_df[
                (council_df['timestamp'] >= pred_time - time_window) &
                (council_df['timestamp'] <= pred_time + time_window) &
                (council_df['actual_trend_direction'].notna()) &
                (council_df['actual_trend_direction'] != '')
            ]

            if not matches.empty:
                actual_direction = str(matches.iloc[0]['actual_trend_direction']).upper().strip()

                # Normalize direction names to match prediction format
                if actual_direction == 'UP':
                    actual_direction = 'BULLISH'
                elif actual_direction == 'DOWN':
                    actual_direction = 'BEARISH'

                # Only resolve if we got a valid direction
                if actual_direction in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                    predictions_df.loc[idx, 'actual'] = actual_direction
                    newly_resolved_indices.append(idx)
                    logger.debug(f"Resolved {pred_row['agent']}: predicted {pred_row['direction']} -> actual {actual_direction}")

        if newly_resolved_indices:
            # 1. Save updated structured file (this is the source of truth)
            predictions_df.to_csv(structured_file, index=False)
            logger.info(f"Updated {len(newly_resolved_indices)} predictions in {structured_file}")

            # 2. Append ONLY newly resolved rows to legacy accuracy file
            newly_resolved_df = predictions_df.loc[newly_resolved_indices].copy()
            _append_to_legacy_accuracy(newly_resolved_df)

            return len(newly_resolved_indices)

        logger.info("No predictions could be resolved this run")
        return 0

    except Exception as e:
        logger.error(f"Failed to resolve pending predictions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


def _append_to_legacy_accuracy(resolved_df: pd.DataFrame):
    """
    Append newly resolved predictions to agent_accuracy.csv.

    Normalizes agent names before writing to ensure consistency.
    """
    accuracy_file = "data/agent_accuracy.csv"

    try:
        from trading_bot.agent_names import normalize_agent_name

        resolved_df = resolved_df.copy()

        # Normalize agent names
        resolved_df['agent'] = resolved_df['agent'].apply(normalize_agent_name)

        # Calculate correctness
        resolved_df['correct'] = (
            resolved_df['direction'].str.upper() == resolved_df['actual'].str.upper()
        ).astype(int)

        # Check if file exists for header
        file_exists = os.path.exists(accuracy_file)

        with open(accuracy_file, 'a') as f:
            if not file_exists:
                f.write("timestamp,agent,predicted,actual,correct\n")

            for _, row in resolved_df.iterrows():
                f.write(f"{row['timestamp']},{row['agent']},{row['direction']},{row['actual']},{row['correct']}\n")

        logger.info(f"Appended {len(resolved_df)} resolved predictions to {accuracy_file}")

    except Exception as e:
        logger.error(f"Failed to append to legacy accuracy file: {e}")
