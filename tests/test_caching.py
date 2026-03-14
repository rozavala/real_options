"""
Tests for get_market_data_cached functionality.
"""
import os
import shutil
import time
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from trading_bot.utils import get_market_data_cached, set_data_dir

class TestCaching(unittest.TestCase):
    def setUp(self):
        # Setup a temporary data directory for testing
        self.test_data_dir = "data/test_caching"
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
        os.makedirs(self.test_data_dir)
        set_data_dir(self.test_data_dir)

    def tearDown(self):
        # Clean up
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
        set_data_dir(None)  # Reset

    @patch('yfinance.download')
    def test_caching_behavior(self, mock_download):
        # Create a mock DataFrame simulating YFinance response
        mock_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1000, 2000]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        mock_download.return_value = mock_df

        ticker = "KC=F"
        period = "1d"

        # 1. First call - should trigger download
        df1 = get_market_data_cached([ticker], period=period)
        self.assertEqual(mock_download.call_count, 1)
        self.assertFalse(df1.empty)

        # Verify cache file was created
        cache_dir = os.path.join(self.test_data_dir, "yf_cache")
        cache_file = os.path.join(cache_dir, f"{ticker.replace('=', '').replace('^', '')}_{period}.csv")
        self.assertTrue(os.path.exists(cache_file))

        # 2. Second call immediately - should NOT trigger download (cache hit)
        df2 = get_market_data_cached([ticker], period=period)
        self.assertEqual(mock_download.call_count, 1)  # Count should stay 1
        pd.testing.assert_frame_equal(df1, df2)

    @patch('yfinance.download')
    def test_cache_expiry(self, mock_download):
        # Mock DF
        mock_df = pd.DataFrame({'Close': [100.0]})
        mock_download.return_value = mock_df

        ticker = "KC=F"
        period = "1d"

        # 1. First call
        get_market_data_cached([ticker], period=period)
        self.assertEqual(mock_download.call_count, 1)

        # Manually age the file (set mtime to 25 hours ago)
        cache_file = os.path.join(self.test_data_dir, "yf_cache", f"{ticker.replace('=', '').replace('^', '')}_{period}.csv")
        old_time = time.time() - (25 * 3600)
        os.utime(cache_file, (old_time, old_time))

        # 2. Call again - should trigger download due to expiry (default TTL is likely < 25h)
        get_market_data_cached([ticker], period=period)
        self.assertEqual(mock_download.call_count, 2)

if __name__ == '__main__':
    unittest.main()
