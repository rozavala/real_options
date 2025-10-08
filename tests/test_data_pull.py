import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

from coffee_factors_data_pull_new import main as run_data_pull

class TestDataPull(unittest.TestCase):

    def setUp(self):
        """Set up a mock config for all tests in this class."""
        self.mock_config = {
            "fred_api_key": "test",
            "nasdaq_api_key": "test",
            "weather_stations": {"loc1": [0, 0]},
            "fred_series": {},
            "yf_series_map": {},
            "final_column_order": ["H_price"],
            "validation_thresholds": {"price_spike_pct": 0.1},
            "notifications": {"enabled": True}
        }

    @patch('coffee_factors_data_pull_new.get_active_coffee_tickers')
    @patch('coffee_factors_data_pull_new.datetime')
    @patch('coffee_factors_data_pull_new.yf.download')
    @patch('coffee_factors_data_pull_new.Fred')
    @patch('coffee_factors_data_pull_new.ndl.get')
    @patch('coffee_factors_data_pull_new.requests.get')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    @patch('pandas.DataFrame.to_csv')
    def test_data_pull_success(self, mock_to_csv, mock_send_notification, mock_requests_get, mock_ndl_get, mock_fred, mock_yf_download, mock_datetime, mock_get_tickers):
        # --- Mock API responses ---
        mock_coffee_data = pd.DataFrame({
            ('Close', 'KCH25.NYB'): [100, 101],
            ('Close', 'KCK25.NYB'): [102, 103],
        }, index=pd.to_datetime(['2025-01-01', '2025-01-02']))
        mock_coffee_data.columns = pd.MultiIndex.from_tuples(mock_coffee_data.columns)

        mock_other_yf_data = pd.DataFrame({'Close': {'ticker1': [1,2]}})
        mock_yf_download.side_effect = [mock_coffee_data, mock_other_yf_data]

        mock_datetime.now.return_value = datetime(2025, 1, 3)
        mock_get_tickers.return_value = ['KCH25.NYB', 'KCK25.NYB']

        mock_fred_instance = mock_fred.return_value
        mock_fred_instance.get_series.return_value = pd.Series([1, 2], name="fred_series")

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
        success = run_data_pull(self.mock_config)

        # --- Assertions ---
        self.assertTrue(success)
        mock_to_csv.assert_called_once()
        mock_send_notification.assert_called()
        self.assertIn("SUCCESS", mock_send_notification.call_args[0][1])

    @patch('coffee_factors_data_pull_new.yf.download')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    def test_data_pull_failure(self, mock_send_notification, mock_yf_download):
        # Mock a failure from yfinance
        mock_yf_download.return_value = pd.DataFrame() # Empty dataframe signifies failure

        success = run_data_pull(self.mock_config)

        self.assertFalse(success)
        mock_send_notification.assert_called()
        self.assertIn("FAILURE", mock_send_notification.call_args[0][1])

if __name__ == '__main__':
    unittest.main()