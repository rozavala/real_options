"""Integration tests for Brier scoring with real file I/O."""

import pytest
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from trading_bot.enhanced_brier import EnhancedBrierTracker
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
