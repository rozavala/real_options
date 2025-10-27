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
            "weather_stations": {"loc1": [0, 0]},
            "fred_series": {"DCOILWTICO": "oil_price"},
            "yf_series_map": {"USDBRL=X": "usd_brl_fx"},
            "final_column_order": ["c1_price"],
            "validation_thresholds": {"price_spike_pct": 0.15},
            "notifications": {"enabled": True},
            "tradingview": {"username": "testuser", "password": "testpassword"}
        }

    @patch('coffee_factors_data_pull_new.TvDatafeed')
    @patch('coffee_factors_data_pull_new.datetime')
    @patch('coffee_factors_data_pull_new.yf.download')
    @patch('coffee_factors_data_pull_new.Fred')
    @patch('coffee_factors_data_pull_new.requests.get')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    @patch('pandas.DataFrame.to_csv')
    def test_data_pull_success(self, mock_to_csv, mock_send_notification, mock_requests_get, mock_fred, mock_yf_download, mock_datetime, mock_tvdatafeed):
        # --- Mock TvDatafeed ---
        mock_tv_instance = mock_tvdatafeed.return_value
        mock_tv_instance.search_symbol.return_value = [{'symbol': 'KC', 'exchange': 'ICE', 'type': 'futures'}]

        mock_contracts = []
        for i in range(1, 6):
            exp_date = pd.to_datetime('2025-03-20')
            mock_df = pd.DataFrame({
                'open': [100 + i], 'high': [105 + i], 'low': [98 + i],
                'close': [102 + i], 'volume': [1000 * i],
                'expiration': [exp_date]
            }, index=[pd.to_datetime('2025-01-01')])
            mock_contracts.append(mock_df)

        mock_tv_instance.get_hist.side_effect = mock_contracts

        # --- Mock other APIs ---
        mock_other_yf_data = pd.DataFrame({'Close': {'USDBRL=X': [5.0]}}, index=pd.to_datetime(['2025-01-01']))
        mock_yf_download.return_value = mock_other_yf_data

        # Mock datetime to be close to the mock data's date
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

        mock_ice_response = MagicMock(status_code=200, content=b'<html><body><script>var chart_config = {"series":[{"data":[[1672531200000, 100]]}]};</script></body></html>')

        def requests_get_side_effect(url, **kwargs):
            if "open-meteo.com" in url: return mock_weather_response
            if "cftc.gov" in url: return mock_cot_response
            if "macromicro.me" in url: return mock_ice_response
            return MagicMock(status_code=404)
        mock_requests_get.side_effect = requests_get_side_effect

        # --- Run the script ---
        success = run_data_pull(self.mock_config)

        # --- Assertions ---
        self.assertTrue(success, "The data pull script should return True on success.")
        mock_to_csv.assert_called_once()
        self.assertEqual(mock_tv_instance.get_hist.call_count, 5)
        self.assertIn("SUCCESS", mock_send_notification.call_args[0][1])

    @patch('coffee_factors_data_pull_new.TvDatafeed')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    def test_data_pull_failure_on_tv_fetch(self, mock_send_notification, mock_tvdatafeed):
        mock_tv_instance = mock_tvdatafeed.return_value
        mock_tv_instance.search_symbol.side_effect = Exception("TradingView unavailable")

        success = run_data_pull(self.mock_config)

        self.assertFalse(success, "The data pull script should return False on failure.")
        self.assertIn("FAILURE", mock_send_notification.call_args[0][1])
        # The exact error message is in the validation report, which is good enough.
        # Just checking for the overall failure is sufficient here.

    @patch('coffee_factors_data_pull_new.TvDatafeed')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    @patch('pandas.DataFrame.to_csv')
    def test_data_pull_fallback_to_nologin(self, mock_to_csv, mock_send_notification, mock_tvdatafeed):
        def tv_side_effect(*args, **kwargs):
            if args:
                raise Exception("Login failed")
            mock_no_login = MagicMock()
            mock_no_login.search_symbol.return_value = []
            return mock_no_login

        mock_tvdatafeed.side_effect = tv_side_effect

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            run_data_pull(self.mock_config)
            output = mock_stdout.getvalue()

        self.assertEqual(mock_tvdatafeed.call_count, 2)
        self.assertIn("Warning: Could not log in", output)

    @patch('coffee_factors_data_pull_new.TvDatafeed')
    @patch('coffee_factors_data_pull_new.send_pushover_notification')
    def test_data_pull_no_credentials(self, mock_send_notification, mock_tvdatafeed):
        no_cred_config = self.mock_config.copy()
        del no_cred_config['tradingview']

        mock_tv_instance = MagicMock()
        mock_tv_instance.search_symbol.return_value = []
        mock_tvdatafeed.return_value = mock_tv_instance

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            run_data_pull(no_cred_config)
            output = mock_stdout.getvalue()

        mock_tvdatafeed.assert_called_once_with()
        self.assertIn("No TradingView credentials found", output)

if __name__ == '__main__':
    unittest.main()
