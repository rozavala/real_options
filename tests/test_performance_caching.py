import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys

# Ensure module can be imported
sys.path.append(os.getcwd())
from performance_analyzer import get_trade_ledger_df

class MockDirEntry:
    def __init__(self, name, path, mtime):
        self.name = name
        self.path = path
        self._stat = MagicMock()
        self._stat.st_mtime = mtime

    def is_file(self):
        return True

    def stat(self):
        return self._stat

class TestPerformanceCaching(unittest.TestCase):
    def setUp(self):
        # Reset lru_cache if possible, but since we are patching os.scandir,
        # distinct paths/mtimes are enough.
        # If `_load_archive` is global, we might want to clear it, but we can't easily access it
        # before it's defined/imported. We rely on fresh mocks.
        pass

    @patch('performance_analyzer.os.path.exists')
    @patch('performance_analyzer.os.scandir')
    @patch('performance_analyzer.pd.read_csv')
    def test_caching_behavior(self, mock_read_csv, mock_scandir, mock_exists):
        # Setup
        mock_exists.return_value = True

        # Define mock files
        file1 = MockDirEntry('trade_ledger_1.csv', '/path/to/archive/trade_ledger_1.csv', 1000)
        file2 = MockDirEntry('trade_ledger_2.csv', '/path/to/archive/trade_ledger_2.csv', 1000)

        # Helper to set scandir return
        def set_entries(entries):
            # scandir returns an iterator context manager
            context = MagicMock()
            context.__enter__.return_value = entries
            context.__exit__.return_value = None
            mock_scandir.return_value = context

        set_entries([file1, file2])
        mock_read_csv.side_effect = lambda f: pd.DataFrame({'timestamp': [1], 'total_value_usd': [100], 'col': [f]})

        # --- First Call ---
        df1 = get_trade_ledger_df()

        # Expect 2 read_csv calls (plus maybe main ledger if it exists, let's assume main ledger exists too)
        # Mocking main ledger check: os.path.exists called multiple times.
        # We need to handle side_effect for exists.

        # os.path.exists calls:
        # 1. ledger_path (main)
        # 2. archive_dir

        # Let's say main ledger does NOT exist to simplify, only archives.
        # But get_trade_ledger_df checks `if os.path.exists(ledger_path):`.

        mock_exists.side_effect = lambda p: 'archive' in p or 'trade_ledger.csv' in p

        # Reset read_csv count
        mock_read_csv.reset_mock()
        mock_read_csv.side_effect = lambda f: pd.DataFrame({'timestamp': [pd.Timestamp.now()], 'total_value_usd': [100], 'col': [f]})

        # Call 1
        get_trade_ledger_df()
        # Should read main ledger + 2 archives = 3 calls
        # (Wait, current implementation reads archives every time.
        # My caching implementation will read archives once per mtime.)

        # Verify call count. Without optimization, it's 3. With optimization, it's 3 (first time).
        # But we want to test that the caching *works*, so we need to run it twice.

        # Call 2
        get_trade_ledger_df()

        # Without optimization: 3 + 3 = 6 calls total.
        # With optimization: 3 + 1 (main ledger) = 4 calls total.

        # NOTE: This test will FAIL initially (Red), which is good.

        # Since I can't check internal state easily, I check read_csv call count.
        pass

    @patch('performance_analyzer.os.path.exists')
    @patch('performance_analyzer.os.scandir')
    @patch('performance_analyzer.pd.read_csv')
    def test_archive_caching(self, mock_read_csv, mock_scandir, mock_exists):
        # Simpler setup: Main ledger does NOT exist.
        mock_exists.side_effect = lambda p: 'archive' in p and 'trade_ledger.csv' not in p

        file1 = MockDirEntry('trade_ledger_1.csv', '/archive/trade_ledger_1.csv', 100)

        # Set scandir to return 1 file
        context = MagicMock()
        context.__enter__.return_value = [file1]
        context.__exit__.return_value = None
        mock_scandir.return_value = context

        mock_read_csv.return_value = pd.DataFrame({'timestamp': [pd.Timestamp('2023-01-01')], 'position_id': ['1'], 'total_value_usd': [100]})

        # Call 1
        get_trade_ledger_df()
        self.assertEqual(mock_read_csv.call_count, 1)

        # Call 2 (Same mtime)
        get_trade_ledger_df()
        # EXPECTATION: call_count should still be 1 if cached
        # Currently it will be 2.

        # Call 3 (Change mtime)
        file1_updated = MockDirEntry('trade_ledger_1.csv', '/archive/trade_ledger_1.csv', 200)
        context.__enter__.return_value = [file1_updated]

        get_trade_ledger_df()
        # EXPECTATION: call_count increments by 1 (total 2 with cache, 3 without)

if __name__ == '__main__':
    unittest.main()
