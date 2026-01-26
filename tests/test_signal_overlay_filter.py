import unittest
from unittest.mock import MagicMock
import sys
import pandas as pd
import pytz
from datetime import datetime, timedelta
import importlib.util
import os

# Mock dependencies to allow importing the page script
sys.modules['streamlit'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()

# Robust mock for ib_insync
mock_ib = MagicMock()
mock_ib.IB = MagicMock
mock_ib.Trade = MagicMock
mock_ib.Contract = MagicMock
mock_ib.Bag = MagicMock
mock_ib.Position = MagicMock
mock_ib.Fill = MagicMock
mock_ib.util = MagicMock
mock_ib.FuturesOption = MagicMock
sys.modules['ib_insync'] = mock_ib

# Configure streamlit mock to return valid values
mock_st = sys.modules['streamlit']
# For the timeframe selectbox
mock_st.selectbox.side_effect = [
    '5m', # timeframe
    'FRONT_MONTH', # selected_contract
    "👑 Master Decision" # selected_agent_label
]
mock_st.slider.return_value = 3

# Import the module
file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pages', '6_Signal_Overlay.py')
spec = importlib.util.spec_from_file_location("signal_overlay", file_path)
signal_overlay = importlib.util.module_from_spec(spec)
sys.modules["signal_overlay"] = signal_overlay
spec.loader.exec_module(signal_overlay)

class TestSignalOverlayFilter(unittest.TestCase):
    def setUp(self):
        self.filter_non_trading_days = signal_overlay.filter_non_trading_days
        self.ny_tz = pytz.timezone('America/New_York')

    def test_filter_weekends(self):
        # Create a range including a weekend
        # Friday Jan 5, 2024 to Monday Jan 8, 2024
        dates = pd.date_range(start='2024-01-05', end='2024-01-08', freq='D', tz=self.ny_tz)
        df = pd.DataFrame(index=dates, data={'price': [1, 2, 3, 4]})

        # Expectation: 05 (Fri) and 08 (Mon) should be kept. 06 (Sat) and 07 (Sun) removed.
        filtered = self.filter_non_trading_days(df)

        self.assertEqual(len(filtered), 2)
        self.assertIn(pd.Timestamp('2024-01-05', tz=self.ny_tz), filtered.index)
        self.assertIn(pd.Timestamp('2024-01-08', tz=self.ny_tz), filtered.index)
        self.assertNotIn(pd.Timestamp('2024-01-06', tz=self.ny_tz), filtered.index)
        self.assertNotIn(pd.Timestamp('2024-01-07', tz=self.ny_tz), filtered.index)

    def test_filter_holidays(self):
        # Create a range including a holiday
        # Dec 25, 2023 (Christmas) - Monday
        dates = pd.date_range(start='2023-12-22', end='2023-12-26', freq='D', tz=self.ny_tz)
        # 22 (Fri), 23 (Sat), 24 (Sun), 25 (Mon - Holiday), 26 (Tue)
        df = pd.DataFrame(index=dates, data={'price': range(5)})

        filtered = self.filter_non_trading_days(df)

        # Expected: 22 (Fri) and 26 (Tue) only.
        # 23, 24 are weekend. 25 is holiday.
        self.assertEqual(len(filtered), 2)
        self.assertIn(pd.Timestamp('2023-12-22', tz=self.ny_tz), filtered.index)
        self.assertIn(pd.Timestamp('2023-12-26', tz=self.ny_tz), filtered.index)
        self.assertNotIn(pd.Timestamp('2023-12-25', tz=self.ny_tz), filtered.index)

    def test_empty_df(self):
        df = pd.DataFrame()
        filtered = self.filter_non_trading_days(df)
        self.assertTrue(filtered.empty)

    def test_none_df(self):
        filtered = self.filter_non_trading_days(None)
        self.assertIsNone(filtered)

if __name__ == '__main__':
    unittest.main()
