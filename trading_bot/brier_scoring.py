"""Brier Score Tracking for Agent Reliability.

Tracks the accuracy of agent predictions over time using the Brier Score metric.
Used to dynamically adjust voting weights based on historical performance.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timezone
from typing import Optional, List
from trading_bot.timestamps import parse_ts_column

logger = logging.getLogger(__name__)

class BrierScoreTracker:
    """Tracks agent prediction accuracy over time."""

    def __init__(self, history_file: str = None):
        if history_file is None:
            ticker = os.environ.get("COMMODITY_TICKER", "KC")
            history_file = f"data/{ticker}/agent_accuracy.csv"
        self.history_file = history_file

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

        self.scores = self._load_history()

    def _load_history(self) -> dict:
        """
        Load historical accuracy scores from agent_accuracy.csv.
        Uses exponential decay: recent predictions weighted more heavily.
        Half-life: 14 days.

        Returns dict of canonical_agent_name -> weighted_accuracy (0.0 to 1.0)
        """
        try:
            if not os.path.exists(self.history_file):
                logger.info("No accuracy history file found")
                return {}

            df = pd.read_csv(self.history_file)

            if df.empty:
                return {}

            expected_cols = ['timestamp', 'agent', 'predicted', 'actual', 'correct']
            if not all(col in df.columns for col in expected_cols):
                logger.warning(f"Accuracy file has unexpected schema: {list(df.columns)}")
                if 'agent' not in df.columns or 'correct' not in df.columns:
                    return {}

            # Normalize agent names
            from trading_bot.agent_names import normalize_agent_name
            df['agent'] = df['agent'].apply(normalize_agent_name)

            # Ensure 'correct' is numeric
            df['correct'] = pd.to_numeric(df['correct'], errors='coerce').fillna(0)

            # Parse timestamps for time-weighting (handles mixed formats)
            df['timestamp'] = parse_ts_column(df['timestamp'])
            # Drop rows where timestamp couldn't be parsed (formerly errors='coerce')
            df = df.dropna(subset=['timestamp'])

            now = pd.Timestamp.now(tz='UTC')

            # Exponential decay: half-life = 14 days
            HALF_LIFE_DAYS = 14.0
            decay_rate = 0.693 / HALF_LIFE_DAYS  # ln(2) / half_life

            scores = {}
            for agent, group in df.groupby('agent'):
                if len(group) < 3:
                    # Too few samples — use simple average
                    scores[agent] = group['correct'].mean()
                    continue

                # Calculate age-based weights
                ages_days = (now - group['timestamp']).dt.total_seconds() / 86400.0
                ages_days = ages_days.clip(lower=0)  # No negative ages
                weights = np.exp(-decay_rate * ages_days)

                # Weighted average
                weighted_correct = (group['correct'] * weights).sum()
                total_weight = weights.sum()

                if total_weight > 0:
                    scores[agent] = weighted_correct / total_weight
                else:
                    scores[agent] = group['correct'].mean()

            # Filter deprecated for display
            from trading_bot.agent_names import DEPRECATED_AGENTS
            displayable_scores = {k: v for k, v in scores.items() if k not in DEPRECATED_AGENTS}

            logger.info(f"Loaded time-weighted Brier scores for {len(displayable_scores)} agents: "
                       f"{{{', '.join(f'{k}: {v:.2f}' for k, v in displayable_scores.items())}}}")
            return scores

        except Exception as e:
            logger.exception(f"Failed to load Brier scores: {e}")
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
        timestamp: Optional[datetime] = None,
        cycle_id: str = None
    ):
        """Record a prediction with full probability for proper Brier scoring.

        Args:
            agent: Agent name (will be normalized)
            predicted_direction: BULLISH, BEARISH, or NEUTRAL
            predicted_confidence: 0.0 to 1.0
            actual: Outcome (PENDING until resolved)
            timestamp: When prediction was made (defaults to now UTC)
            cycle_id: Deterministic foreign key linking to council_history.csv
                      Format: "KC-a1b2c3d4" (commodity-namespaced)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        predicted_direction = predicted_direction.upper()

        # Normalize agent name
        from trading_bot.agent_names import normalize_agent_name
        agent = normalize_agent_name(agent)

        prob_bullish = predicted_confidence if predicted_direction == 'BULLISH' else (1.0 - predicted_confidence)
        if predicted_direction == 'NEUTRAL':
            prob_bullish = 0.5

        # Confidence floor: never record 0.0 (causes Brier score issues)
        if predicted_direction == 'NEUTRAL':
            predicted_confidence = max(0.5, predicted_confidence)
        else:
            predicted_confidence = max(0.1, predicted_confidence)

        # Issue 6: Skip default-value predictions (NEUTRAL/0.5 carries no information)
        if predicted_direction == 'NEUTRAL' and abs(predicted_confidence - 0.5) < 0.01:
            logger.debug(f"Skipping default-value prediction for {agent} (NEUTRAL/0.5 carries no information)")
            return  # Don't record uninformative predictions

        try:
            structured_file = self.history_file.replace(".csv", "_structured.csv")
            exists_struct = os.path.exists(structured_file)

            # === DEDUPLICATION: Skip if this cycle_id + agent already recorded ===
            if cycle_id and exists_struct:
                try:
                    existing = pd.read_csv(structured_file)
                    if not existing.empty and 'cycle_id' in existing.columns:
                        if ((existing['cycle_id'] == cycle_id) &
                            (existing['agent'] == agent)).any():
                            logger.debug(f"Skipping duplicate: {agent} already recorded for cycle {cycle_id}")
                            return
                except Exception:
                    pass  # If dedup check fails, record anyway (safe to have dupes)

            CANONICAL_HEADER = "cycle_id,timestamp,agent,direction,confidence,prob_bullish,actual"

            # === Append the new prediction (clean file handle) ===
            with open(structured_file, 'a') as f:
                if not exists_struct:
                    f.write(CANONICAL_HEADER + "\n")
                f.write(f"{cycle_id or ''},{timestamp},{agent},{predicted_direction},{predicted_confidence},{prob_bullish},{actual}\n")

            logger.debug(f"Recorded structured prediction: {agent} -> {predicted_direction} (cycle={cycle_id})")

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

        Uses TIME-WEIGHTED scoring: recent predictions matter more.
        Exponential decay with half-life of 14 days.

        Returns:
            0.5 to 2.0 multiplier (1.0 = baseline/unknown)
        """
        # Skip deprecated agents
        from trading_bot.agent_names import DEPRECATED_AGENTS
        if agent in DEPRECATED_AGENTS:
            return 1.0  # Neutral multiplier — no influence

        from trading_bot.agent_names import normalize_agent_name
        agent = normalize_agent_name(agent)

        if agent not in self.scores or self.scores[agent] is None:
            return 1.0

        accuracy = self.scores.get(agent, 0.5)

        # Need minimum samples for statistical significance
        # (This check is approximate — _load_history counts are not tracked per-agent)
        # For now, use the score directly if available

        # Linear mapping: accuracy 0.0 → 0.5x, accuracy 0.5 → 1.25x, accuracy 1.0 → 2.0x
        multiplier = 0.5 + (accuracy * 1.5)
        multiplier = max(0.5, min(2.0, multiplier))

        return round(multiplier, 3)

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
            except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError, ValueError):
                diagnostics['legacy_record_count'] = 'ERROR'

        structured_file = self.history_file.replace('.csv', '_structured.csv')
        if os.path.exists(structured_file):
            try:
                df = pd.read_csv(structured_file)
                diagnostics['structured_record_count'] = len(df)
                diagnostics['pending_count'] = len(df[df['actual'] == 'PENDING'])
                diagnostics['resolved_count'] = len(df[df['actual'] != 'PENDING'])
            except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError, ValueError):
                diagnostics['structured_record_count'] = 'ERROR'

        return diagnostics

# Singleton
_tracker: Optional[BrierScoreTracker] = None
_data_dir: Optional[str] = None


def set_data_dir(data_dir: str):
    """Set data directory for the Brier tracker singleton."""
    global _data_dir, _tracker
    _data_dir = data_dir
    _tracker = None  # Force recreation with new path
    logger.info(f"BrierScoring data_dir set to: {data_dir}")


def get_brier_tracker(data_dir: str = None) -> BrierScoreTracker:
    global _tracker
    if _tracker is None:
        # ContextVar > explicit arg > module global
        if data_dir is None:
            try:
                from trading_bot.data_dir_context import get_engine_data_dir
                data_dir = get_engine_data_dir()
            except LookupError:
                pass
        effective_dir = data_dir or _data_dir
        if effective_dir:
            _tracker = BrierScoreTracker(history_file=os.path.join(effective_dir, "agent_accuracy.csv"))
        else:
            _tracker = BrierScoreTracker()
    return _tracker

def resolve_pending_predictions(council_history_path: str = None, data_dir: str = None) -> List[int]:
    """
    Resolve PENDING predictions by cross-referencing with reconciled council_history.

    STRATEGY:
    1. PRIMARY: JOIN on cycle_id (deterministic, 100% reliable)
    2. FALLBACK: Nearest-match algorithm for legacy data without cycle_id

    IDEMPOTENT: Only processes rows where actual == 'PENDING'.

    Returns:
        List of indices of newly resolved predictions
    """
    effective_dir = data_dir or "data"
    if council_history_path is None:
        council_history_path = os.path.join(effective_dir, "council_history.csv")
    structured_file = os.path.join(effective_dir, "agent_accuracy_structured.csv")

    if not os.path.exists(structured_file):
        logger.info("No structured predictions file found — nothing to resolve")
        return []

    if not os.path.exists(council_history_path):
        logger.warning(f"Council history not found at {council_history_path}")
        return []

    try:
        predictions_df = pd.read_csv(structured_file)
        council_df = pd.read_csv(council_history_path)

        if predictions_df.empty or council_df.empty:
            logger.info("Empty dataframes — nothing to resolve")
            return []

        # Ensure timestamp columns are datetime with UTC (handles mixed formats)
        # Use coerce mode to handle any residual data corruption gracefully
        predictions_df['timestamp'] = parse_ts_column(predictions_df['timestamp'], errors='coerce')
        council_df['timestamp'] = parse_ts_column(council_df['timestamp'], errors='coerce')

        # Drop rows with unparseable timestamps (column-alignment corruption, etc.)
        pred_nat_mask = predictions_df['timestamp'].isna()
        if pred_nat_mask.any():
            nat_count = pred_nat_mask.sum()
            logger.warning(
                f"Dropping {nat_count} predictions with unparseable timestamps "
                f"(likely column-alignment corruption — run fix_brier_data.py to repair)"
            )
            predictions_df = predictions_df[~pred_nat_mask].copy().reset_index(drop=True)

        # Ensure cycle_id column exists (backward compat)
        if 'cycle_id' not in predictions_df.columns:
            predictions_df['cycle_id'] = ''
        if 'cycle_id' not in council_df.columns:
            council_df['cycle_id'] = ''

        # Filter to PENDING predictions only
        pending_mask = predictions_df['actual'] == 'PENDING'
        pending_count = pending_mask.sum()

        if pending_count == 0:
            logger.info("No pending predictions to resolve")
            return 0

        logger.info(f"Found {pending_count} pending predictions to check")

        # --- BUILD RESOLUTION LOOKUP ---
        # Sort ALL council decisions (not just reconciled) for cycle matching
        all_decisions_sorted = council_df.sort_values('timestamp').reset_index(drop=True)

        # Build reconciled mask
        reconciled_mask = (
            all_decisions_sorted['actual_trend_direction'].notna() &
            (all_decisions_sorted['actual_trend_direction'] != '') &
            (all_decisions_sorted['actual_trend_direction'].astype(str).str.strip() != '')
        )

        logger.info(f"Council decisions: {len(all_decisions_sorted)} total, "
                    f"{reconciled_mask.sum()} reconciled")

        # Build cycle_id → actual_direction lookup (PRIMARY strategy)
        from trading_bot.cycle_id import is_valid_cycle_id
        cycle_id_lookup = {}
        for i, row in all_decisions_sorted[reconciled_mask].iterrows():
            cid = str(row.get('cycle_id', '')).strip()
            if is_valid_cycle_id(cid):
                direction = _normalize_direction(str(row['actual_trend_direction']))
                if direction:
                    cycle_id_lookup[cid] = direction

        logger.info(f"Built cycle_id lookup with {len(cycle_id_lookup)} entries")

        # Diagnostics stats
        stats = {'cycle_id_match': 0, 'nearest_match': 0, 'no_council_within_window': 0,
                 'council_not_reconciled': 0, 'no_valid_direction': 0, 'orphaned': 0}

        # --- RESOLVE PREDICTIONS ---
        newly_resolved_indices = []

        # Calculate age once
        now_utc = datetime.now(timezone.utc)

        for idx in predictions_df[pending_mask].index:
            pred_row = predictions_df.loc[idx]
            pred_cycle_id = str(pred_row.get('cycle_id', '')).strip()

            # Calculate age in hours
            pred_age_hours = (now_utc - pred_row['timestamp']).total_seconds() / 3600

            actual_direction = None

            # STRATEGY 1: cycle_id JOIN (deterministic)
            if is_valid_cycle_id(pred_cycle_id) and pred_cycle_id in cycle_id_lookup:
                actual_direction = cycle_id_lookup[pred_cycle_id]
                stats['cycle_id_match'] += 1

            # STRATEGY 2: Cycle-aware match for legacy data (no cycle_id)
            elif not is_valid_cycle_id(pred_cycle_id):
                actual_direction = _cycle_aware_resolve(
                    pred_row['timestamp'],
                    all_decisions_sorted,       # Searches ALL decisions
                    reconciled_mask,            # Then checks if reconciled
                    max_distance_minutes=120
                )
                if actual_direction:
                    stats['nearest_match'] += 1
                elif pred_age_hours > 48:
                    # Entry is old enough but no match
                    stats['no_council_within_window'] += 1
                    logger.debug(
                        f"ORPHAN candidate: {pred_row.get('agent', '?')} at {pred_row['timestamp']} "
                        f"(age: {pred_age_hours:.0f}h, no cycle_id, no council match within 120min)"
                    )

            # Auto-ORPHAN stale legacy entries
            orphan_threshold = _get_orphan_window_hours(pred_row['timestamp'])

            if not actual_direction and pred_age_hours > orphan_threshold and not is_valid_cycle_id(pred_cycle_id):
                actual_direction = 'ORPHANED'
                stats['orphaned'] += 1
                logger.info(
                    f"Auto-ORPHANED: {pred_row.get('agent', '?')} at {pred_row['timestamp']} "
                    f"(age: {pred_age_hours:.0f}h > {orphan_threshold}h, no cycle_id)"
                )

            # Apply resolution
            if actual_direction:
                predictions_df.loc[idx, 'actual'] = actual_direction
                newly_resolved_indices.append(idx)

        # Log summary stats
        logger.info(
            f"Brier Resolution stats: "
            f"cycle_id={stats['cycle_id_match']}, "
            f"nearest={stats['nearest_match']}, "
            f"orphaned={stats['orphaned']}, "
            f"no_council_window={stats['no_council_within_window']}"
        )

        if newly_resolved_indices:
            # Save updated structured file
            predictions_df.to_csv(structured_file, index=False)
            logger.info(
                f"Resolved {len(newly_resolved_indices)} predictions "
                f"(cycle_id: {stats['cycle_id_match']}, nearest: {stats['nearest_match']}, orphans: {stats['orphaned']})"
            )

            # Sync to legacy accuracy file
            newly_resolved_df = predictions_df.loc[newly_resolved_indices].copy()
            _append_to_legacy_accuracy(newly_resolved_df, data_dir=effective_dir)

            # Reset singleton tracker so weighted voting picks up new scores
            _reset_tracker_singleton()

            return newly_resolved_indices

        logger.info("No predictions could be resolved this run")
        return []

    except Exception as e:
        logger.exception(f"Failed to resolve pending predictions: {e}")
        return []


def _normalize_direction(raw: str) -> str:
    """Normalize direction values to BULLISH/BEARISH/NEUTRAL."""
    raw = raw.upper().strip()
    mapping = {
        'UP': 'BULLISH',
        'DOWN': 'BEARISH',
        'BULLISH': 'BULLISH',
        'BEARISH': 'BEARISH',
        'NEUTRAL': 'NEUTRAL',
        'FLAT': 'NEUTRAL',
    }
    return mapping.get(raw, None)


def _cycle_aware_resolve(
    pred_timestamp,
    all_decisions_sorted_df: pd.DataFrame,
    reconciled_mask: pd.Series,
    max_distance_minutes: int = 120
) -> Optional[str]:
    """
    Cycle-aware resolution for legacy data without cycle_id.

    Unlike _nearest_match_resolve which searches only reconciled decisions
    (risking cross-cycle contamination), this function:
    1. Finds the nearest council decision (any status) — this is the prediction's own cycle
    2. Checks if that cycle's decision is reconciled
    3. Returns the direction only if reconciled, None otherwise

    This prevents predictions from being graded against the wrong cycle's outcome.
    """
    if all_decisions_sorted_df.empty:
        return None

    try:
        time_diffs = abs(all_decisions_sorted_df['timestamp'] - pred_timestamp)
        nearest_idx = time_diffs.idxmin()
        nearest_gap = time_diffs[nearest_idx]

        # Safety cap
        if nearest_gap > pd.Timedelta(minutes=max_distance_minutes):
            return None

        # Check if THIS cycle's decision is reconciled
        if not reconciled_mask.iloc[nearest_idx]:
            return None  # Own cycle not yet reconciled — don't guess

        raw_direction = str(all_decisions_sorted_df.loc[nearest_idx, 'actual_trend_direction'])
        return _normalize_direction(raw_direction)

    except Exception:
        return None


def _reset_tracker_singleton():
    """Reset the global BrierScoreTracker so it reloads from disk."""
    global _tracker
    try:
        _tracker = None
        logger.info("Reset BrierScoreTracker singleton — will reload on next access")
    except Exception:
        pass


def _get_orphan_window_hours(timestamp: datetime) -> int:
    """
    Calculate orphan window accounting for weekends/holidays.

    B4 FIX: 72 base hours + adjustment for non-trading days.
    """
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        import pandas as pd
        from datetime import timedelta

        BASE_HOURS = 72
        cal = USFederalHolidayCalendar()

        # Count non-trading days in window
        start = timestamp
        end = timestamp + timedelta(hours=BASE_HOURS)

        # Ensure datetimes are naive or normalized for pandas
        if start.tzinfo: start = start.replace(tzinfo=None)
        if end.tzinfo: end = end.replace(tzinfo=None)

        holidays = cal.holidays(start=start.date(), end=end.date())
        weekend_days = sum(1 for d in pd.date_range(start, end) if d.weekday() >= 5)
        non_trading_days = len(holidays) + weekend_days

        # Add 24 hours per non-trading day
        adjusted_hours = BASE_HOURS + (non_trading_days * 24)

        return min(adjusted_hours, 168)  # Cap at 1 week
    except Exception:
        return 72  # Fallback


def _append_to_legacy_accuracy(resolved_df: pd.DataFrame, data_dir: str = None):
    """
    Append newly resolved predictions to agent_accuracy.csv.

    Normalizes agent names before writing to ensure consistency.
    """
    effective_dir = data_dir or "data"
    accuracy_file = os.path.join(effective_dir, "agent_accuracy.csv")

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
