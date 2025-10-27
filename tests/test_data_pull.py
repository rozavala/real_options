import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import io
import zipfile

from coffee_factors_data_pull_new import main as run_data_pull

class TestDataPull(unittest.TestCase):

    def setUp(self):
        """Set up a mock config for all tests in this class."""
        self.mock_config = {
            "databento_api_key": "test_api_key",
            "fred_api_key": "test",
            "weather_stations": {"loc1": [0, 0]},
            "fred_series": {"DCOILWTICO": "oil_price"},
            "yf_series_map": {"USDBRL=X": "usd_brl_fx"},
            "final_column_order": ["c1_price"],
            "validation_thresholds": {"price_spike_pct": 0.15},
            "notifications": {"enabled": True},
        }

    @patch('coffee_factors_data_pull_new.db.Historical')
    @patch('coffee_factors_data_pull_new.datetime')
    @patch('coffee_factors_data_pull_new.yf.download')
    @patch('coffee_factors_data_pull_new.Fred')
    @patch('coffee_factors_data_pull_new.requests.get')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    @patch('pandas.DataFrame.to_csv')
    def test_data_pull_success(self, mock_to_csv, mock_send_notification, mock_requests_get, mock_fred, mock_yf_download, mock_datetime, mock_databento):
        # --- Mock Databento ---
        mock_db_client = mock_databento.return_value

        # Mock the DBNStore object and its to_df method
        mock_dbn_store = MagicMock()
        mock_df = pd.DataFrame({
            'ts_event': [pd.to_datetime('2025-01-01T00:00:00Z')],
            'open': [101], 'high': [106], 'low': [99],
            'close': [103], 'volume': [1000],
        })
        mock_dbn_store.to_df.return_value = mock_df

        # Make get_range return a new mock store for each call
        def mock_get_range(*args, **kwargs):
            mock_dbn_store = MagicMock()
            mock_df = pd.DataFrame({
                'open': [101], 'high': [106], 'low': [99],
                'close': [103], 'volume': [1000],
            }, index=pd.to_datetime(['2025-01-01T00:00:00Z']))
            mock_dbn_store.to_df.return_value = mock_df
            return mock_dbn_store
        mock_db_client.timeseries.get_range.side_effect = mock_get_range

        # --- Mock other APIs ---
        mock_other_yf_data = pd.DataFrame({'Close': {'USDBRL=X': [5.0]}}, index=pd.to_datetime(['2025-01-01']))
        mock_yf_download.return_value = mock_other_yf_data

        mock_datetime.now.return_value = datetime(2025, 1, 2)
        mock_fred_instance = mock_fred.return_value
        mock_fred_instance.get_series.return_value = pd.Series([80], name="oil_price", index=pd.to_datetime(['2025-01-01']))

        mock_weather_response = MagicMock(status_code=200)
        mock_weather_response.json.return_value = [{'daily': {'time': ['2025-01-01'], 'temperature_2m_mean': [10], 'precipitation_sum': [1]}}]

        cot_csv_data = "Market and Exchange Names,As of Date in Form YYYY-MM-DD,Noncommercial Positions-Long (All),Noncommercial Positions-Short (All)\nCOFFEE C - NEW YORK BOARD OF TRADE,2025-01-01,1000,500\n"
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('annual.txt', cot_csv_data)
        zip_buffer.seek(0)
        mock_cot_response = MagicMock(status_code=200, content=zip_buffer.read())

        def requests_get_side_effect(url, **kwargs):
            if "open-meteo.com" in url: return mock_weather_response
            if "cftc.gov" in url: return mock_cot_response
            return MagicMock(status_code=404)
        mock_requests_get.side_effect = requests_get_side_effect

        # --- Run the script ---
        success = run_data_pull(self.mock_config)

        # --- Assertions ---
        self.assertTrue(success, "The data pull script should return True on success.")
        mock_to_csv.assert_called_once()
        self.assertEqual(mock_db_client.timeseries.get_range.call_count, 5)
        self.assertIn("SUCCESS", mock_send_notification.call_args[0][1])

    @patch('coffee_factors_data_pull_new.db.Historical')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    def test_data_pull_failure_on_databento_fetch(self, mock_send_notification, mock_databento):
        mock_db_client = mock_databento.return_value
        mock_db_client.timeseries.get_range.side_effect = Exception("Databento unavailable")

        success = run_data_pull(self.mock_config)

        self.assertFalse(success, "The data pull script should return False on failure.")
        self.assertIn("FAILURE", mock_send_notification.call_args[0][1])

if __name__ == '__main__':
    unittest.main()
