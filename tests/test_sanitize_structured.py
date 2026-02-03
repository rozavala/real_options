"""
Tests for the structured CSV sanitization logic in fix_brier_data.py.
"""
import unittest
import tempfile
import os
import pandas as pd

# Adjust import path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')


class TestSanitizeStructuredCSV(unittest.TestCase):
    """Test detection and recovery of column-misaligned rows."""

    def _write_csv(self, path, header, rows):
        """Helper to write test CSV files."""
        with open(path, 'w') as f:
            f.write(header + '\n')
            for row in rows:
                f.write(','.join(str(v) for v in row) + '\n')

    def test_clean_file_unchanged(self):
        """A file with no corruption should pass through unchanged."""
        from scripts.fix_brier_data import sanitize_structured_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'agent_accuracy_structured.csv')
            # Mock the path in the module
            import scripts.fix_brier_data as module_under_test
            original_path = module_under_test.STRUCTURED_FILE
            module_under_test.STRUCTURED_FILE = path

            try:
                self._write_csv(path,
                    'cycle_id,timestamp,agent,direction,confidence,prob_bullish,actual',
                    [
                        ['KC-abc12345', '2026-01-30 09:00:00+00:00', 'agronomist', 'BULLISH', '0.9', '0.9', 'PENDING'],
                        ['KC-def67890', '2026-01-30 10:00:00+00:00', 'macro', 'BEARISH', '0.8', '0.2', 'PENDING'],
                    ]
                )

                sanitize_structured_csv()

                df = pd.read_csv(path)
                # Should detect 0 misaligned rows and leave file as is
                # We check by seeing if it's still loadable and data is same
                self.assertEqual(len(df), 2)
                self.assertEqual(df.iloc[0]['cycle_id'], 'KC-abc12345')
            finally:
                module_under_test.STRUCTURED_FILE = original_path

    def test_detect_misaligned_rows(self):
        """Rows where timestamp contains a cycle_id should be detected and recovered."""
        from scripts.fix_brier_data import sanitize_structured_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'agent_accuracy_structured.csv')
            import scripts.fix_brier_data as module_under_test
            original_path = module_under_test.STRUCTURED_FILE
            module_under_test.STRUCTURED_FILE = path

            try:
                # Simulate: header has timestamp first (or loaded that way), but data was written with cycle_id first
                # NOTE: The sanitization function reads the file. If we write it with misaligned data relative to header:
                # Header: cycle_id,timestamp,agent,direction,confidence,prob_bullish,actual
                # Row: KC-c0f3b12c,2026-01-30 09:53:45+00:00,agronomist,BULLISH,0.9,0.9,PENDING
                # Wait, the corruption is:
                # File Header: timestamp,agent,direction,confidence,prob_bullish,actual,cycle_id
                # Data Written: cycle_id,timestamp,agent,direction,confidence,prob_bullish,actual
                # So when pandas reads it:
                # timestamp column gets cycle_id

                self._write_csv(path,
                    'timestamp,agent,direction,confidence,prob_bullish,actual,cycle_id',
                    [
                        # Good row (old format, no cycle_id or cycle_id at end empty)
                        ['2026-01-20 09:00:00+00:00', 'agronomist', 'BULLISH', '0.9', '0.9', 'BEARISH', ''],
                        # Bad row (cycle_id written first, shifts everything right)
                        ['KC-c0f3b12c', '2026-01-30 09:53:45+00:00', 'agronomist', 'BULLISH', '0.9', '0.9', 'PENDING'],
                    ]
                )

                sanitize_structured_csv()

                df = pd.read_csv(path)

                # The sanitization function should have fixed the bad row
                # And saved it.
                # Let's check the row that was bad.
                # Note: The sanitize function rewrites the file.
                # Since the header in the file was 'timestamp,...', pandas reads it with that header.
                # The sanitize function fixes the values in the DataFrame.
                # Then it writes it back. `to_csv(index=False)` writes the header based on the DataFrame columns.
                # The DataFrame columns came from the original read, which were 'timestamp', 'agent', etc.
                # So the corrected file will have the original header 'timestamp,agent,...' but with correct values?
                # The function assigns:
                # df.loc[idx, 'cycle_id'] = real_cycle_id
                # df.loc[idx, 'timestamp'] = real_timestamp
                # ...
                # If the original header didn't have 'cycle_id' as the first column, it might still be 'timestamp' first.
                # Let's check what the function does.
                # It reads the file. If 'timestamp' has cycle_id pattern, it fixes it.
                # It assigns values to columns 'cycle_id', 'timestamp', etc.
                # Then `df.to_csv`.

                # Check the bad row (index 1)
                row = df.iloc[1]
                self.assertEqual(row['cycle_id'], 'KC-c0f3b12c')
                self.assertEqual(row['timestamp'], '2026-01-30 09:53:45+00:00')
                self.assertEqual(row['agent'], 'agronomist')
                self.assertEqual(row['direction'], 'BULLISH')
                self.assertEqual(row['actual'], 'PENDING')

            finally:
                module_under_test.STRUCTURED_FILE = original_path

    def test_unrecoverable_row_dropped(self):
        """Rows that fail validation should be dropped, not recovered."""
        from scripts.fix_brier_data import sanitize_structured_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'agent_accuracy_structured.csv')
            import scripts.fix_brier_data as module_under_test
            original_path = module_under_test.STRUCTURED_FILE
            module_under_test.STRUCTURED_FILE = path

            try:
                 # Header: timestamp,agent,direction,confidence,prob_bullish,actual,cycle_id
                self._write_csv(path,
                    'timestamp,agent,direction,confidence,prob_bullish,actual,cycle_id',
                    [
                        # Garbage row
                        # timestamp gets 'KC-baddata1' (looks like cycle_id)
                        # agent gets 'not_a_timestamp' -> fails datetime parse
                        ['KC-a1b2c3d4', 'not_a_timestamp', 'not_an_agent', '0.0', '0.0', 'PENDING', '']
                    ]
                )

                sanitize_structured_csv()

                df = pd.read_csv(path)
                # Should have dropped the row
                self.assertEqual(len(df), 0)

            finally:
                module_under_test.STRUCTURED_FILE = original_path


class TestParseTimestampCoerce(unittest.TestCase):
    """Test that parse_ts_column with errors='coerce' handles bad data."""

    def test_coerce_returns_nat_for_bad_values(self):
        """Bad timestamp values should become NaT instead of crashing."""
        from trading_bot.timestamps import parse_ts_column

        series = pd.Series([
            '2026-01-30 09:00:00+00:00',
            'KC-c0f3b12c',  # Bad value
            '2026-01-30 10:00:00+00:00',
        ])

        result = parse_ts_column(series, errors='coerce')

        # First and third should parse fine
        self.assertFalse(pd.isna(result.iloc[0]))
        self.assertFalse(pd.isna(result.iloc[2]))

        # Second should be NaT
        self.assertTrue(pd.isna(result.iloc[1]))

    def test_raise_mode_still_crashes(self):
        """Default mode should still raise on bad values (backward compat)."""
        from trading_bot.timestamps import parse_ts_column

        series = pd.Series(['KC-c0f3b12c', '2026-01-30 09:00:00+00:00'])

        with self.assertRaises(Exception):
            parse_ts_column(series)  # default errors='raise'


if __name__ == '__main__':
    unittest.main()
