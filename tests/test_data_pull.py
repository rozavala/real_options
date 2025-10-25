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
            "fred_api_key": "test",
            "nasdaq_api_key": "test",
            "weather_stations": {"loc1": [0, 0]},
            "fred_series": {"DCOILWTICO": "oil_price"},
            "yf_series_map": {"USDBRL=X": "usd_brl_fx"},
            "final_column_order": ["H_price"],
            "validation_thresholds": {"price_spike_pct": 0.1},
            "notifications": {"enabled": True}
        }

    @patch('coffee_factors_data_pull_new.get_historical_coffee_tickers')
    @patch('coffee_factors_data_pull_new.datetime')
    @patch('coffee_factors_data_pull_new.yf.download')
    @patch('coffee_factors_data_pull_new.Fred')
    @patch('coffee_factors_data_pull_new.requests.get')
    @patch('coffee_factors_data_pull_new.sync_playwright')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    @patch('pandas.DataFrame.to_csv')
    def test_data_pull_success(self, mock_to_csv, mock_send_notification, mock_sync_playwright, mock_requests_get, mock_fred, mock_yf_download, mock_datetime, mock_get_tickers):
        # --- Mock yfinance data for historical contracts ---
        mock_coffee_data = {
            ('Open', 'KCH25.NYB'): {'2025-01-01': 99, '2025-01-02': 100},
            ('High', 'KCH25.NYB'): {'2025-01-01': 101, '2025-01-02': 102},
            ('Low', 'KCH25.NYB'): {'2025-01-01': 98, '2025-01-02': 99},
            ('Close', 'KCH25.NYB'): {'2025-01-01': 100, '2025-01-02': 101},
            ('Volume', 'KCH25.NYB'): {'2025-01-01': 1000, '2025-01-02': 1100},
            ('Open', 'KCK25.NYB'): {'2025-01-01': 101, '2025-01-02': 102},
            ('High', 'KCK25.NYB'): {'2025-01-01': 103, '2025-01-02': 104},
            ('Low', 'KCK25.NYB'): {'2025-01-01': 100, '2025-01-02': 101},
            ('Close', 'KCK25.NYB'): {'2025-01-01': 102, '2025-01-02': 103},
            ('Volume', 'KCK25.NYB'): {'2025-01-01': 1200, '2025-01-02': 1300},
        }
        mock_coffee_df = pd.DataFrame(mock_coffee_data)
        mock_coffee_df.index = pd.to_datetime(mock_coffee_df.index)

        mock_other_yf_data = pd.DataFrame({'Close': {'USDBRL=X': [5.0, 5.1]}}, index=pd.to_datetime(['2025-01-01', '2025-01-02']))

        # The first call to yf.download is for coffee, the second is for other market data.
        mock_yf_download.side_effect = [mock_coffee_df, mock_other_yf_data]

        mock_datetime.now.return_value = datetime(2025, 1, 3)
        mock_get_tickers.return_value = ['KCH25.NYB', 'KCK25.NYB']

        # --- Mock FRED data ---
        mock_fred_instance = mock_fred.return_value
        mock_fred_instance.get_series.return_value = pd.Series([80, 81], name="oil_price", index=pd.to_datetime(['2025-01-01', '2025-01-02']))

        # --- Mock requests.get with side_effect for different APIs ---
        mock_weather_response = MagicMock()
        mock_weather_response.status_code = 200
        mock_weather_response.json.return_value = [{'daily': {'time': ['2025-01-01', '2025-01-02'], 'temperature_2m_mean': [10, 11], 'precipitation_sum': [1, 2]}}]

        cot_csv_data = ("Market and Exchange Names,As of Date in Form YYYY-MM-DD,Noncommercial Positions-Long (All),Noncommercial Positions-Short (All)\n"
                        "COFFEE C - NEW YORK BOARD OF TRADE,2025-01-01,1000,500\n")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('annual.txt', cot_csv_data)
        zip_buffer.seek(0)
        mock_cot_response = MagicMock(status_code=200, content=zip_buffer.read())

        mock_ice_response = MagicMock()
        mock_ice_response.status_code = 200
        mock_ice_response.content = b"""
            <html>
                <body>
                    <script>
                        var chart_config = {"series":[{"data":[[1672531200000, 100], [1672617600000, 101]]}]};
                    </script>
                </body>
            </html>
        """

        # --- Mock Playwright ---
        mock_playwright = MagicMock()
        mock_sync_playwright.return_value.__enter__.return_value = mock_playwright
        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_page.return_value = mock_page
        mock_page.content.return_value = mock_ice_response.content
        # Ensure the wait_for_selector call in the script does not hang
        mock_page.wait_for_selector.return_value = None

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
        self.assertIn("SUCCESS", mock_send_notification.call_args[0][1])

    @patch('coffee_factors_data_pull_new.yf.download')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    def test_data_pull_failure(self, mock_send_notification, mock_yf_download):
        mock_yf_download.return_value = pd.DataFrame()
        success = run_data_pull(self.mock_config)
        self.assertFalse(success)
        self.assertIn("FAILURE", mock_send_notification.call_args[0][1])

if __name__ == '__main__':
    unittest.main()