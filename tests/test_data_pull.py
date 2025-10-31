
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import io
import zipfile
import pytest

from coffee_factors_data_pull_new import main as run_data_pull

@pytest.fixture
def mock_config():
    """Provides a mock config dictionary for tests."""
    return {
        "fred_api_key": "test",
        "nasdaq_api_key": "test",
        "weather_stations": {"loc1": [0, 0]},
        "fred_series": {"DCOILWTICO": "oil_price"},
        "yf_series_map": {"USDBRL=X": "usd_brl_fx"},
        "final_column_order": [
            "date", "front_month_price", "front_month_open", "front_month_high",
            "front_month_low", "front_month_volume", "front_month_dte",
            "second_month_price", "second_month_open", "second_month_high",
            "second_month_low", "second_month_volume", "second_month_dte"
        ],
        "validation_thresholds": {"price_spike_pct": 0.1},
        "notifications": {"enabled": True}
    }

@patch('coffee_factors_data_pull_new.get_active_coffee_tickers')
@patch('coffee_factors_data_pull_new.datetime')
@patch('coffee_factors_data_pull_new.yf.download')
@patch('coffee_factors_data_pull_new.Fred')
@patch('coffee_factors_data_pull_new.requests.get')
@patch('coffee_factors_data_pull_new.send_pushover_notification')
@patch('pandas.DataFrame.to_csv', MagicMock())
def test_data_pull_success(mock_send_notification, mock_requests_get, mock_fred, mock_yf_download, mock_datetime, mock_get_tickers, mock_config):
    # --- Mock comprehensive yfinance data ---
    mock_coffee_data = pd.DataFrame({
        ('Close', 'KCH25.NYB'): [100, 101], ('Open', 'KCH25.NYB'): [99, 100],
        ('High', 'KCH25.NYB'): [101, 102], ('Low', 'KCH25.NYB'): [98, 99],
        ('Volume', 'KCH25.NYB'): [1000, 1100],
        ('Close', 'KCK25.NYB'): [102, 103], ('Open', 'KCK25.NYB'): [101, 102],
        ('High', 'KCK25.NYB'): [103, 104], ('Low', 'KCK25.NYB'): [100, 101],
        ('Volume', 'KCK25.NYB'): [1200, 1300],
    }, index=pd.to_datetime(['2025-01-01', '2025-01-02']))
    mock_coffee_data.columns = pd.MultiIndex.from_tuples(mock_coffee_data.columns)

    mock_other_yf_data = pd.DataFrame({'Close': {'USDBRL=X': [5.0, 5.1]}}, index=pd.to_datetime(['2025-01-01', '2025-01-02']))
    mock_yf_download.side_effect = [mock_coffee_data, mock_other_yf_data]

    mock_datetime.now.return_value = datetime(2025, 1, 3)
    mock_get_tickers.return_value = ['KCH25.NYB', 'KCK25.NYB']

    # --- Mock FRED data ---
    mock_fred_instance = mock_fred.return_value
    mock_fred_instance.get_series.return_value = pd.Series([80, 81], name="oil_price", index=pd.to_datetime(['2025-01-01', '2025-01-02']))

    # --- Mock requests.get for weather and COT ---
    mock_weather_response = MagicMock(status_code=200)
    mock_weather_response.json.return_value = [{'daily': {'time': ['2025-01-01', '2025-01-02'], 'temperature_2m_mean': [10, 11], 'precipitation_sum': [1, 2]}}]

    cot_csv_data = ("Market and Exchange Names,As of Date in Form YYYY-MM-DD,Noncommercial Positions-Long (All),Noncommercial Positions-Short (All)\n"
                    "COFFEE C - NEW YORK BOARD OF TRADE,2025-01-01,1000,500\n")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('annual.txt', cot_csv_data)
    zip_buffer.seek(0)
    mock_cot_response = MagicMock(status_code=200, content=zip_buffer.read())

    def requests_get_side_effect(url, **kwargs):
        if "open-meteo.com" in url: return mock_weather_response
        if "cftc.gov" in url: return mock_cot_response
        return MagicMock(status_code=444)

    mock_requests_get.side_effect = requests_get_side_effect

    # --- Run the script ---
    success, df_result = run_data_pull(mock_config)

    # --- Assertions ---
    assert success, "The data pull script should return True on success."
    assert df_result is not None

    # Add specific assertions about the content of the CSV
    assert "front_month_price" in df_result.columns
    assert "second_month_open" in df_result.columns
    assert df_result["front_month_price"].iloc[0] == 100
    assert df_result["second_month_open"].iloc[0] == 101

    assert "SUCCESS" in mock_send_notification.call_args[0][1]

@patch('coffee_factors_data_pull_new.yf.download')
@patch('coffee_factors_data_pull_new.send_pushover_notification')
def test_data_pull_failure(mock_send_notification, mock_yf_download, mock_config):
    mock_yf_download.return_value = pd.DataFrame() # Simulate yfinance failure
    success, df_result = run_data_pull(mock_config)
    assert not success, "The data pull script should return False on failure."
    assert df_result is None
    assert "FAILURE" in mock_send_notification.call_args[0][1]
