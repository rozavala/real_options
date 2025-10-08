import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from datetime import datetime
import os

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coffee_factors_data_pull_new import main as run_data_pull

class TestDataPull(unittest.TestCase):

    @patch('coffee_factors_data_pull_new.get_active_coffee_tickers')
    @patch('coffee_factors_data_pull_new.datetime')
    @patch('coffee_factors_data_pull_new.yf.download')
    @patch('coffee_factors_data_pull_new.Fred')
    @patch('coffee_factors_data_pull_new.ndl.get')
    @patch('coffee_factors_data_pull_new.requests.get')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    @patch('builtins.open', new_callable=mock_open, read_data='{"fred_api_key": "test", "nasdaq_api_key": "test", "weather_stations": {"loc1": [0,0]}, "fred_series": {}, "yf_series_map": {}, "final_column_order": ["H_price"], "validation_thresholds": {"price_spike_pct": 0.1}}')
    @patch('pandas.DataFrame.to_csv')
    def test_data_pull_success(self, mock_to_csv, mock_file_open, mock_send_notification, mock_requests_get, mock_ndl_get, mock_fred, mock_yf_download, mock_datetime, mock_get_tickers):
        # --- Mock API responses ---

        # Mock yfinance for coffee prices
        mock_coffee_data = pd.DataFrame({
            ('Close', 'KCH25.NYB'): [100, 101],
            ('Close', 'KCK25.NYB'): [102, 103],
        }, index=pd.to_datetime(['2025-01-01', '2025-01-02']))
        mock_coffee_data.columns = pd.MultiIndex.from_tuples(mock_coffee_data.columns)

        # Mock yfinance for other series
        mock_other_yf_data = pd.DataFrame({
             'Close': {'ticker1': [1,2]}
        })

        mock_yf_download.side_effect = [mock_coffee_data, mock_other_yf_data]

        # Mock datetime to control the recency check
        mock_datetime.now.return_value = datetime(2025, 1, 3)
        # Mock the ticker generation to match our mock data
        mock_get_tickers.return_value = ['KCH25.NYB', 'KCK25.NYB']

        # Mock FRED
        mock_fred_instance = mock_fred.return_value
        mock_fred_instance.get_series.return_value = pd.Series([1, 2], name="fred_series")

        # Mock weather API
        mock_weather_response = MagicMock()
        mock_weather_response.json.return_value = [{
            'daily': {
                'time': ['2025-01-01', '2025-01-02'],
                'temperature_2m_mean': [10, 11],
                'precipitation_sum': [1, 2]
            }
        }]
        mock_requests_get.return_value = mock_weather_response

        # --- Run the script ---
        success = run_data_pull()

        # --- Assertions ---
        self.assertTrue(success)
        # Check that a CSV was written
        mock_to_csv.assert_called_once()
        # Check that a success notification was sent
        mock_send_notification.assert_called()
        self.assertIn("SUCCESS", mock_send_notification.call_args[0][1])


    @patch('coffee_factors_data_pull_new.yf.download')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    @patch('builtins.open', new_callable=mock_open, read_data='{"fred_api_key": "test", "nasdaq_api_key": "test", "weather_stations": {}, "fred_series": {}, "yf_series_map": {}, "validation_thresholds": {}}')
    def test_data_pull_failure(self, mock_file_open, mock_send_notification, mock_yf_download):
        # Mock a failure from yfinance
        mock_yf_download.return_value = pd.DataFrame() # Empty dataframe signifies failure

        success = run_data_pull()

        self.assertFalse(success)
        # Check that a failure notification was sent
        mock_send_notification.assert_called()
        self.assertIn("FAILURE", mock_send_notification.call_args[0][1])


if __name__ == '__main__':
    unittest.main()