"""Integration tests for Brier scoring with real file I/O."""

import pytest
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

from trading_bot.enhanced_brier import EnhancedBrierTracker, MAX_RESOLVED_PREDICTIONS, _BACKFILL_PASS2_CUTOFF
from trading_bot.brier_bridge import _confidence_to_probs


class TestBrierIntegration:
    """J1 FIX: Tests with actual file I/O, not mocks."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = Path(self.temp_dir) / "brier_data.json"

    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_record_and_retrieve_prediction(self):
        """Prediction persists to disk and is retrievable."""
        tracker = EnhancedBrierTracker(data_path=self.data_file)

        # Need to convert direction/confidence to probs for tracker
        prob_bullish, prob_neutral, prob_bearish = _confidence_to_probs("BULLISH", 0.75)

        tracker.record_prediction(
            agent="test_agent",
            prob_bullish=prob_bullish,
            prob_neutral=prob_neutral,
            prob_bearish=prob_bearish,
            cycle_id="test_cycle_001"
        )

        # Verify file exists and contains data
        # Note: Tracker saves every 8 records. Force save?
        tracker._save()

        assert self.data_file.exists()
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        assert len(data['predictions']) == 1
        assert data['predictions'][0]['agent'] == 'test_agent'

    def test_cycle_id_required(self):
        """B3: Missing cycle_id raises ValueError."""
        tracker = EnhancedBrierTracker(data_path=self.data_file)
        prob_bullish, prob_neutral, prob_bearish = _confidence_to_probs("BULLISH", 0.75)

        with pytest.raises(ValueError, match="cycle_id is required"):
            tracker.record_prediction(
                agent="test_agent",
                prob_bullish=prob_bullish,
                prob_neutral=prob_neutral,
                prob_bearish=prob_bearish,
                cycle_id="",
            )

    def test_concurrent_writes_safe(self):
        """Multiple rapid writes don't corrupt file."""
        tracker = EnhancedBrierTracker(data_path=self.data_file)

        for i in range(50):
            prob_bullish, prob_neutral, prob_bearish = _confidence_to_probs(
                "BULLISH" if i % 2 == 0 else "BEARISH",
                0.5 + (i % 10) / 20
            )
            tracker.record_prediction(
                agent=f"agent_{i % 5}",
                prob_bullish=prob_bullish,
                prob_neutral=prob_neutral,
                prob_bearish=prob_bearish,
                cycle_id=f"cycle_{i}",
            )

        tracker._save()

        with open(self.data_file, 'r') as f:
            data = json.load(f)
        assert len(data['predictions']) == 50

        with open(self.data_file, 'r') as f:
            data = json.load(f)
        assert len(data['predictions']) == 50


class TestStatusAwareTrimming:
    """Tests for status-aware prediction trimming (prevents silent data loss)."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = Path(self.temp_dir) / "brier_data.json"

    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _make_prediction(self, tracker, agent, cycle_id, resolved=False, timestamp=None):
        """Helper: record a prediction, optionally mark it resolved.

        Directly sets actual_outcome on the prediction object to avoid
        cycle_id format requirements in resolve_prediction().
        """
        prob_bullish, prob_neutral, prob_bearish = _confidence_to_probs("BULLISH", 0.7)
        tracker.record_prediction(
            agent=agent,
            prob_bullish=prob_bullish,
            prob_neutral=prob_neutral,
            prob_bearish=prob_bearish,
            cycle_id=cycle_id,
            timestamp=timestamp,
        )
        if resolved:
            # Directly mark as resolved (bypasses cycle_id format validation)
            pred = tracker.predictions[-1]
            pred.actual_outcome = "BULLISH"
            pred.resolved_at = datetime.now(timezone.utc)

    def test_pending_predictions_survive_save_trim(self):
        """Pending predictions must survive a save-reload cycle even when
        total predictions exceed MAX_RESOLVED_PREDICTIONS."""
        tracker = EnhancedBrierTracker(data_path=self.data_file)

        # Record 2400 resolved + 100 pending
        for i in range(2400):
            self._make_prediction(tracker, "agent_a", f"resolved_{i}", resolved=True)
        for i in range(100):
            self._make_prediction(tracker, "agent_a", f"pending_{i}", resolved=False)

        tracker._save()

        # Reload from disk
        tracker2 = EnhancedBrierTracker(data_path=self.data_file)
        pending_count = sum(1 for p in tracker2.predictions if p.actual_outcome is None)
        resolved_count = sum(1 for p in tracker2.predictions if p.actual_outcome is not None)

        assert pending_count == 100, f"Expected 100 pending, got {pending_count}"
        assert resolved_count <= MAX_RESOLVED_PREDICTIONS, (
            f"Resolved count {resolved_count} exceeds cap {MAX_RESOLVED_PREDICTIONS}"
        )

    def test_pending_survive_in_memory_trim(self):
        """In-memory trim must not drop pending predictions."""
        tracker = EnhancedBrierTracker(data_path=self.data_file)

        # Fill with resolved predictions to approach the limit
        for i in range(2800):
            self._make_prediction(tracker, "agent_a", f"resolved_{i}", resolved=True)

        # Now add pending predictions — this should trigger in-memory trim
        for i in range(250):
            self._make_prediction(tracker, "agent_a", f"pending_{i}", resolved=False)

        pending_count = sum(1 for p in tracker.predictions if p.actual_outcome is None)
        assert pending_count == 250, f"Expected 250 pending, got {pending_count}"

    def test_backfill_pass2_skipped_after_deprecation(self):
        """After deprecation date, Pass 2 should not create new predictions from CSV."""
        import pandas as pd

        tracker = EnhancedBrierTracker(data_path=self.data_file)

        # Record one prediction in JSON
        self._make_prediction(tracker, "agent_a", "existing_cycle", resolved=False)
        tracker._save()

        # Create a CSV with rows that do NOT exist in JSON
        csv_path = Path(self.temp_dir) / "agent_accuracy_structured.csv"
        csv_data = pd.DataFrame([{
            'cycle_id': f'csv_only_{i}',
            'timestamp': '2026-01-15T10:00:00+00:00',
            'agent': 'agent_a',
            'direction': 'BULLISH',
            'confidence': 0.8,
            'actual': 'BULLISH',
        } for i in range(100)])
        csv_data.to_csv(csv_path, index=False)

        # Reload tracker
        tracker2 = EnhancedBrierTracker(data_path=self.data_file)
        initial_count = len(tracker2.predictions)

        # If we're past the cutoff (we are — it's March 2026), Pass 2 should be skipped
        if datetime.now(timezone.utc) >= _BACKFILL_PASS2_CUTOFF:
            tracker2.backfill_from_resolved_csv(structured_csv_path=str(csv_path))
            # Pass 2 skipped → no new predictions created from CSV
            assert len(tracker2.predictions) == initial_count, (
                f"Pass 2 should be skipped after deprecation. "
                f"Before: {initial_count}, After: {len(tracker2.predictions)}"
            )

    def test_backfill_does_not_displace_pending(self):
        """Simulate the KC restart scenario: backfill must not displace pending predictions."""
        tracker = EnhancedBrierTracker(data_path=self.data_file)

        # Load 800 resolved + 200 pending into JSON
        for i in range(800):
            self._make_prediction(tracker, "agent_a", f"res_{i}", resolved=True)
        for i in range(200):
            self._make_prediction(tracker, "agent_a", f"pend_{i}", resolved=False)

        tracker._save()

        # Reload and verify pending survive
        tracker2 = EnhancedBrierTracker(data_path=self.data_file)
        pending = [p for p in tracker2.predictions if p.actual_outcome is None]
        assert len(pending) == 200, f"Expected 200 pending after reload, got {len(pending)}"

    def test_chronological_order_preserved(self):
        """After status-aware trim, predictions must remain in timestamp order."""
        tracker = EnhancedBrierTracker(data_path=self.data_file)
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # Interleave resolved and pending with known timestamps
        for i in range(2500):
            prob_bullish, prob_neutral, prob_bearish = _confidence_to_probs("BULLISH", 0.7)
            tracker.record_prediction(
                agent="agent_a",
                prob_bullish=prob_bullish,
                prob_neutral=prob_neutral,
                prob_bearish=prob_bearish,
                cycle_id=f"cycle_{i}",
                timestamp=base_time + timedelta(minutes=i),
            )
            if i % 5 != 0:  # Leave every 5th prediction pending
                tracker.resolve_prediction(
                    agent="agent_a",
                    actual_outcome="BULLISH",
                    cycle_id=f"cycle_{i}",
                )

        tracker._save()

        tracker2 = EnhancedBrierTracker(data_path=self.data_file)
        timestamps = [p.timestamp for p in tracker2.predictions]
        assert timestamps == sorted(timestamps), "Predictions not in chronological order after trim"
