import unittest
import os
import shutil
import pandas as pd
from datetime import datetime, timezone
from trading_bot.brier_scoring import BrierScoreTracker, resolve_pending_predictions

class TestBrierScoring(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/data_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.history_file = os.path.join(self.test_dir, "agent_accuracy.csv")
        self.tracker = BrierScoreTracker(history_file=self.history_file)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_record_and_load_prediction(self):
        # Record legacy style
        self.tracker.record_prediction("agronomist", "BULLISH", "BULLISH")
        self.tracker.record_prediction("macro", "BULLISH", "BEARISH")

        # Reload
        tracker2 = BrierScoreTracker(history_file=self.history_file)
        scores = tracker2.scores

        self.assertIn("agronomist", scores)
        self.assertIn("macro", scores)

        # agronomist: 1 correct -> 1.0 (or smoothed)
        # macro: 0 correct -> 0.0 (or smoothed)
        # logic: if agent not in scores: score = correct.
        # So agronomist=1, macro=0.

        self.assertEqual(scores["agronomist"], 1.0)
        self.assertEqual(scores["macro"], 0.0)

    def test_normalization(self):
        # Record with different casing/aliases
        self.tracker.record_prediction("Weather", "BULLISH", "BULLISH") # -> agronomist
        self.tracker.record_prediction("MACRO_economist", "BEARISH", "BEARISH") # -> macro

        tracker2 = BrierScoreTracker(history_file=self.history_file)
        scores = tracker2.scores

        self.assertIn("agronomist", scores)
        self.assertIn("macro", scores)
        self.assertNotIn("Weather", scores)

        self.assertEqual(scores["agronomist"], 1.0)
        self.assertEqual(scores["macro"], 1.0)

    def test_weight_multiplier(self):
        # 100% accuracy
        self.tracker.scores['agronomist'] = 1.0
        mult = self.tracker.get_agent_weight_multiplier('agronomist')
        # 0.5 + (1.0 * 1.5) = 2.0
        self.assertEqual(mult, 2.0)

        # 0% accuracy
        self.tracker.scores['bad_agent'] = 0.0
        mult = self.tracker.get_agent_weight_multiplier('bad_agent')
        # 0.5 + 0 = 0.5
        self.assertEqual(mult, 0.5)

        # 50% accuracy
        self.tracker.scores['avg_agent'] = 0.5
        mult = self.tracker.get_agent_weight_multiplier('avg_agent')
        # 0.5 + (0.5 * 1.5) = 0.5 + 0.75 = 1.25
        self.assertEqual(mult, 1.25)

        # Unknown agent -> 1.0
        self.assertEqual(self.tracker.get_agent_weight_multiplier('unknown'), 1.0)

    def test_record_structured_schema(self):
        # Verify it writes to _structured.csv without error and has correct schema
        self.tracker.record_prediction_structured(
            "agronomist", "BULLISH", 0.9, actual="PENDING"
        )

        struct_file = self.history_file.replace(".csv", "_structured.csv")
        self.assertTrue(os.path.exists(struct_file))

        df = pd.read_csv(struct_file)
        expected_cols = ['timestamp','agent','direction','confidence','prob_bullish','actual']
        self.assertTrue(all(col in df.columns for col in expected_cols))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['actual'], 'PENDING')

if __name__ == '__main__':
    unittest.main()
