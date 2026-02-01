
import unittest
import pandas as pd
import os
import tempfile
import shutil
import unittest.mock as mock
from datetime import datetime, timezone

# We need to add the parent directory to sys.path to import trading_bot
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot import brier_scoring
from trading_bot.cycle_id import generate_cycle_id, parse_cycle_id, is_valid_cycle_id

class TestBrierScoringNew(unittest.TestCase):

    def test_generate_cycle_id_format(self):
        """Cycle IDs should follow commodity-namespaced format."""
        cid = generate_cycle_id("KC")
        self.assertTrue(cid.startswith("KC-"))
        self.assertEqual(len(cid), 11)  # "KC-" + 8 hex chars
        self.assertTrue(is_valid_cycle_id(cid))

        parsed = parse_cycle_id(cid)
        self.assertEqual(parsed['commodity'], 'KC')
        self.assertTrue(parsed['valid'])

        # Multi-commodity
        ct_cid = generate_cycle_id("CT")
        self.assertTrue(ct_cid.startswith("CT-"))
        self.assertNotEqual(cid, ct_cid)

    def test_cycle_id_validation(self):
        """Invalid cycle_ids should be rejected."""
        self.assertFalse(is_valid_cycle_id(None))
        self.assertFalse(is_valid_cycle_id(''))
        self.assertFalse(is_valid_cycle_id('nan'))
        self.assertFalse(is_valid_cycle_id('invalid'))
        self.assertFalse(is_valid_cycle_id('K'))
        self.assertTrue(is_valid_cycle_id('KC-abcd1234'))

    def test_resolve_by_cycle_id(self):
        """Predictions should resolve via cycle_id JOIN."""

        # Create a temporary directory for test files
        with tempfile.TemporaryDirectory() as tmpdir:
            struct_file = os.path.join(tmpdir, "agent_accuracy_structured.csv")
            council_file = os.path.join(tmpdir, "council_history.csv")

            # 1. Create PREDICTIONS with cycle_id
            pd.DataFrame([
                {'cycle_id': 'KC-test1234', 'timestamp': '2026-01-20 10:00:00+00:00',
                 'agent': 'agronomist', 'direction': 'BULLISH', 'confidence': 0.9,
                 'prob_bullish': 0.9, 'actual': 'PENDING'},
                {'cycle_id': 'KC-test1234', 'timestamp': '2026-01-20 10:00:01+00:00',
                 'agent': 'macro', 'direction': 'NEUTRAL', 'confidence': 0.5,
                 'prob_bullish': 0.5, 'actual': 'PENDING'},
                {'cycle_id': 'KC-test5678', 'timestamp': '2026-01-21 14:00:00+00:00',
                 'agent': 'agronomist', 'direction': 'BEARISH', 'confidence': 0.8,
                 'prob_bullish': 0.2, 'actual': 'PENDING'},
            ]).to_csv(struct_file, index=False)

            # 2. Create COUNCIL HISTORY with one resolved cycle
            pd.DataFrame([
                {'cycle_id': 'KC-test1234', 'timestamp': '2026-01-20 09:55:00+00:00',
                 'contract': 'KCH6', 'master_decision': 'BULLISH',
                 'actual_trend_direction': 'BULLISH', 'exit_price': 350.0},
                {'cycle_id': 'KC-test5678', 'timestamp': '2026-01-21 13:50:00+00:00',
                 'contract': 'KCH6', 'master_decision': 'BEARISH',
                 'actual_trend_direction': '',  # NOT YET RECONCILED
                 'exit_price': None},
            ]).to_csv(council_file, index=False)

            # 3. Patch pandas.read_csv to return DFs from temp files AND monkey-patch to_csv on the result

            original_read_csv = pd.read_csv

            def side_effect_read_csv(filepath, **kwargs):
                real_path = filepath
                if "agent_accuracy_structured.csv" in str(filepath):
                    real_path = struct_file
                elif "council_history.csv" in str(filepath):
                    real_path = council_file

                # Read dataframe from temp location
                df = original_read_csv(real_path, **kwargs)

                # If this is the structured file, we need to intercept writes to it
                if "agent_accuracy_structured.csv" in str(filepath):
                    original_to_csv = df.to_csv

                    def custom_to_csv(path_or_buf=None, **csv_kwargs):
                        # If writing to the hardcoded location, redirect to temp file
                        target = path_or_buf
                        if target and "agent_accuracy_structured.csv" in str(target):
                            target = struct_file
                        return original_to_csv(target, **csv_kwargs)

                    # Monkey-patch the instance method
                    df.to_csv = custom_to_csv

                return df

            with mock.patch('pandas.read_csv', side_effect=side_effect_read_csv):
                with mock.patch('os.path.exists', return_value=True):
                    # Mock _append_to_legacy_accuracy to avoid writing to real files
                    with mock.patch('trading_bot.brier_scoring._append_to_legacy_accuracy') as mock_append:

                        # Run resolution
                        resolved_count = brier_scoring.resolve_pending_predictions(council_file)

                        # Check results
                        self.assertEqual(resolved_count, 2) # 2 predictions for KC-test1234

                        # Verify structured file update by reading it back
                        # (We use original_read_csv to bypass our mock logic for verification)
                        df_new = original_read_csv(struct_file)

                        # KC-test1234 should be resolved to BULLISH
                        resolved_rows = df_new[df_new['cycle_id'] == 'KC-test1234']
                        self.assertTrue(all(resolved_rows['actual'] == 'BULLISH'))

                        # KC-test5678 should still be PENDING
                        pending_rows = df_new[df_new['cycle_id'] == 'KC-test5678']
                        self.assertTrue(all(pending_rows['actual'] == 'PENDING'))

    def test_cycle_aware_legacy_resolution(self):
        """
        New Cycle-Aware algorithm (v5) should ensure predictions match to their OWN cycle,
        even if that cycle is unreconciled and there is a reconciled decision nearby.
        This prevents cross-cycle contamination.
        """
        # Create a temporary directory for test files
        with tempfile.TemporaryDirectory() as tmpdir:
            struct_file = os.path.join(tmpdir, "agent_accuracy_structured.csv")
            council_file = os.path.join(tmpdir, "council_history.csv")

            # Timestamps
            time_cycle_a = '2026-01-20 10:00:00+00:00'  # Prediction time for Cycle A
            time_council_a = '2026-01-20 10:05:00+00:00'  # Council A decision (5 min later) - UNRECONCILED
            time_council_b = '2026-01-20 11:30:00+00:00'  # Council B decision (90 min later) - RECONCILED

            # 1. Create LEGACY PREDICTION (no cycle_id)
            pd.DataFrame([
                # Prediction happened at 10:00. Nearest decision is A (10:05). Next nearest is B (11:30).
                # Under OLD logic (nearest RECONCILED), it would match B (gap=90m < 2h).
                # Under NEW logic (nearest ANY), it matches A (gap=5m). A is unreconciled -> PENDING.
                {'cycle_id': '', 'timestamp': time_cycle_a,
                 'agent': 'agronomist', 'direction': 'BULLISH', 'confidence': 0.9,
                 'prob_bullish': 0.9, 'actual': 'PENDING'},
            ]).to_csv(struct_file, index=False)

            # 2. Create COUNCIL HISTORY
            pd.DataFrame([
                # Cycle A: Nearest to prediction, but NOT reconciled
                {'cycle_id': 'KC-cycleA', 'timestamp': time_council_a,
                 'contract': 'KCH6', 'master_decision': 'BULLISH',
                 'actual_trend_direction': '', 'exit_price': None},

                # Cycle B: Further away, but RECONCILED
                {'cycle_id': 'KC-cycleB', 'timestamp': time_council_b,
                 'contract': 'KCH6', 'master_decision': 'BEARISH',
                 'actual_trend_direction': 'BEARISH', 'exit_price': 340.0},
            ]).to_csv(council_file, index=False)

            # 3. Setup Mocking (same as above)
            original_read_csv = pd.read_csv

            def side_effect_read_csv(filepath, **kwargs):
                real_path = filepath
                if "agent_accuracy_structured.csv" in str(filepath):
                    real_path = struct_file
                elif "council_history.csv" in str(filepath):
                    real_path = council_file
                df = original_read_csv(real_path, **kwargs)
                if "agent_accuracy_structured.csv" in str(filepath):
                    original_to_csv = df.to_csv
                    def custom_to_csv(path_or_buf=None, **csv_kwargs):
                        target = path_or_buf
                        if target and "agent_accuracy_structured.csv" in str(target):
                            target = struct_file
                        return original_to_csv(target, **csv_kwargs)
                    df.to_csv = custom_to_csv
                return df

            with mock.patch('pandas.read_csv', side_effect=side_effect_read_csv):
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('trading_bot.brier_scoring._append_to_legacy_accuracy'):

                        # Run resolution
                        resolved_count = brier_scoring.resolve_pending_predictions(council_file)

                        # EXPECTATION: 0 resolved.
                        # It should match to Cycle A (nearest), see it's unreconciled, and stay PENDING.
                        # It should NOT match to Cycle B (which is reconciled but wrong cycle).
                        self.assertEqual(resolved_count, 0)

                        df_new = original_read_csv(struct_file)
                        self.assertEqual(df_new.iloc[0]['actual'], 'PENDING')

    def test_cycle_aware_legacy_resolution_success(self):
        """
        Verify it DOES resolve when the correct cycle IS reconciled.
        """
        # Create a temporary directory for test files
        with tempfile.TemporaryDirectory() as tmpdir:
            struct_file = os.path.join(tmpdir, "agent_accuracy_structured.csv")
            council_file = os.path.join(tmpdir, "council_history.csv")

            time_cycle_a = '2026-01-20 10:00:00+00:00'
            time_council_a = '2026-01-20 10:05:00+00:00'

            # 1. Prediction
            pd.DataFrame([
                {'cycle_id': '', 'timestamp': time_cycle_a,
                 'agent': 'agronomist', 'direction': 'BULLISH', 'confidence': 0.9,
                 'prob_bullish': 0.9, 'actual': 'PENDING'},
            ]).to_csv(struct_file, index=False)

            # 2. Council History (Cycle A is now RECONCILED)
            pd.DataFrame([
                {'cycle_id': 'KC-cycleA', 'timestamp': time_council_a,
                 'contract': 'KCH6', 'master_decision': 'BULLISH',
                 'actual_trend_direction': 'BULLISH', 'exit_price': 360.0},
            ]).to_csv(council_file, index=False)

            # 3. Setup Mocking
            original_read_csv = pd.read_csv

            def side_effect_read_csv(filepath, **kwargs):
                real_path = filepath
                if "agent_accuracy_structured.csv" in str(filepath):
                    real_path = struct_file
                elif "council_history.csv" in str(filepath):
                    real_path = council_file
                df = original_read_csv(real_path, **kwargs)
                if "agent_accuracy_structured.csv" in str(filepath):
                    original_to_csv = df.to_csv
                    def custom_to_csv(path_or_buf=None, **csv_kwargs):
                        target = path_or_buf
                        if target and "agent_accuracy_structured.csv" in str(target):
                            target = struct_file
                        return original_to_csv(target, **csv_kwargs)
                    df.to_csv = custom_to_csv
                return df

            with mock.patch('pandas.read_csv', side_effect=side_effect_read_csv):
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('trading_bot.brier_scoring._append_to_legacy_accuracy'):

                        # Run resolution
                        resolved_count = brier_scoring.resolve_pending_predictions(council_file)

                        # EXPECTATION: 1 resolved. Matches A (nearest), sees reconciled, resolves.
                        self.assertEqual(resolved_count, 1)

                        df_new = original_read_csv(struct_file)
                        self.assertEqual(df_new.iloc[0]['actual'], 'BULLISH')

if __name__ == '__main__':
    unittest.main()
