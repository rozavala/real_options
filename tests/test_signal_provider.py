import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import pandas as pd

# --- Add project root to path to allow imports ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---

from signal_provider import get_trading_signals

class TestSignalProvider(unittest.TestCase):

    def setUp(self):
        """Set up a mock config and a dummy dataframe for all tests."""
        self.config = {
            "api_base_url": "http://fake-api.com",
            # Add other necessary config keys if the functions need them
            "fred_api_key": "fake",
            "nasdaq_api_key": "fake",
            "final_column_order": ["date", "H_price", "K_price"],
            "notifications": {} # Mock notifications to avoid errors
        }
        self.dummy_df = pd.DataFrame({
            "date": ["10/01/25"],
            "H_price": [200.50],
            "K_price": [202.30]
        })

    @patch('signal_provider.send_pushover_notification')
    @patch('signal_provider.requests')
    @patch('signal_provider._fetch_market_data')
    @patch('signal_provider.os.remove')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_signals_success_flow(self, mock_file, mock_os_remove, mock_fetch_data, mock_requests, mock_notification):
        """
        Test the complete successful pipeline: data fetch -> API post -> poll -> complete -> cleanup.
        """
        # --- Mocks Setup ---
        # 1. _fetch_market_data returns a dummy file path
        mock_fetch_data.return_value = "dummy_data.csv"

        # 2. Mock the API responses
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"id": "job123"}
        mock_post_response.raise_for_status.return_value = None

        mock_get_pending_response = MagicMock()
        mock_get_pending_response.json.return_value = {"status": "pending"}
        mock_get_pending_response.raise_for_status.return_value = None

        mock_get_completed_response = MagicMock()
        expected_signals = [{"contract_month": "202512", "direction": "BULLISH"}]
        mock_get_completed_response.json.return_value = {"status": "completed", "result": expected_signals}
        mock_get_completed_response.raise_for_status.return_value = None

        # Set up the sequence of responses for requests.get
        mock_requests.post.return_value = mock_post_response
        mock_requests.get.side_effect = [mock_get_pending_response, mock_get_completed_response]

        # 3. Mock pandas reading the CSV
        with patch('signal_provider.pd.read_csv', return_value=self.dummy_df):
            # --- Run Test ---
            signals = get_trading_signals(self.config)

        # --- Assertions ---
        # Check that the orchestrator function was called correctly
        mock_fetch_data.assert_called_once_with(self.config)

        # Check that the final signals are what we expect
        self.assertEqual(signals, expected_signals)

        # Check that the temporary file was removed
        mock_os_remove.assert_called_once_with("dummy_data.csv")


    @patch('signal_provider.send_pushover_notification')
    @patch('signal_provider.requests')
    @patch('signal_provider._fetch_market_data')
    @patch('signal_provider.os.remove')
    def test_get_signals_api_job_fails(self, mock_os_remove, mock_fetch_data, mock_requests, mock_notification):
        """
        Test the flow where the API job fails.
        """
        # --- Mocks Setup ---
        mock_fetch_data.return_value = "dummy_data.csv"

        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"id": "job123"}

        mock_get_failed_response = MagicMock()
        mock_get_failed_response.json.return_value = {"status": "failed", "error": "Model exploded"}

        mock_requests.post.return_value = mock_post_response
        mock_requests.get.return_value = mock_get_failed_response

        # --- Run Test ---
        with patch('signal_provider.pd.read_csv', return_value=self.dummy_df):
            signals = get_trading_signals(self.config)

        # --- Assertions ---
        # The function should return None on failure
        self.assertIsNone(signals)
        # The temporary file should NOT be removed on failure for debugging
        mock_os_remove.assert_not_called()


    @patch('signal_provider.send_pushover_notification')
    @patch('signal_provider._fetch_market_data')
    def test_get_signals_data_fetch_fails(self, mock_fetch_data, mock_notification):
        """
        Test the flow where the initial data fetch fails.
        """
        # --- Mocks Setup ---
        # _fetch_market_data returns None, simulating a failure
        mock_fetch_data.return_value = None

        # --- Run Test ---
        signals = get_trading_signals(self.config)

        # --- Assertions ---
        self.assertIsNone(signals)


if __name__ == '__main__':
    unittest.main()