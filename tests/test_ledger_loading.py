
import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
import sys
from unittest.mock import MagicMock

# Mock dependencies to avoid installing them
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["trading_bot.performance_graphs"] = MagicMock()

# Now import the function under test
from performance_analyzer import get_trade_ledger_df

class TestLedgerLoading(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.archive_dir = os.path.join(self.test_dir, 'archive_ledger')
        os.makedirs(self.archive_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_get_trade_ledger_df_consolidation(self):
        # Create main ledger
        main_df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00+00:00'],
            'total_value_usd': [100]
        })
        main_df.to_csv(os.path.join(self.test_dir, 'trade_ledger.csv'), index=False)

        # Create archive ledger
        archive_df = pd.DataFrame({
            'timestamp': ['2023-12-01 10:00:00+00:00'],
            'total_value_usd': [50]
        })
        archive_df.to_csv(os.path.join(self.archive_dir, 'trade_ledger_arch.csv'), index=False)

        result = get_trade_ledger_df(data_dir=self.test_dir)

        # Verify
        self.assertEqual(len(result), 2)

        # Check timestamps are datetime objects
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['timestamp']))

        # Sort by timestamp to ensure order
        result = result.sort_values('timestamp').reset_index(drop=True)

        ts0 = result.iloc[0]['timestamp']
        ts1 = result.iloc[1]['timestamp']

        self.assertEqual(ts0.year, 2023)
        self.assertEqual(ts1.year, 2024)

    def test_get_trade_ledger_df_naive_timestamps(self):
        main_df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00'],
            'total_value_usd': [100]
        })
        main_df.to_csv(os.path.join(self.test_dir, 'trade_ledger.csv'), index=False)

        result = get_trade_ledger_df(data_dir=self.test_dir)

        self.assertEqual(len(result), 1)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['timestamp']))
