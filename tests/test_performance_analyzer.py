import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from datetime import datetime
import os
import io

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from performance_analyzer import analyze_performance

class TestPerformanceAnalyzer(unittest.TestCase):

    @patch('performance_analyzer.send_pushover_notification')
    @patch('performance_analyzer.load_config')
    @patch('os.path.exists', return_value=True)
    def test_analyze_performance(self, mock_exists, mock_load_config, mock_send_notification):
        # --- Setup Mocks ---
        mock_load_config.return_value = {'notifications': {}}

        today_str = datetime.now().strftime('%Y-%m-%d')
        yesterday_str = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        # P&L for combo 123 = -18750 + 7500 + 30000 - 3750 = 15000
        csv_data = f"""timestamp,combo_id,local_symbol,action,quantity,avg_fill_price,strike,right,total_value_usd,reason
{today_str} 10:00:00,123,KCH6 C3.5,BUY,1,0.5,3.5,C,-18750.0,Strategy Execution
{today_str} 10:00:00,123,KCH6 C3.6,SELL,1,0.2,3.6,C,7500.0,Strategy Execution
{today_str} 14:00:00,123,KCH6 C3.5,SELL,1,0.8,3.5,C,30000.0,Take-Profit
{today_str} 14:00:00,123,KCH6 C3.6,BUY,1,0.1,3.6,C,-3750.0,Take-Profit
{yesterday_str} 11:00:00,456,KCK6 P3.4,BUY,1,0.4,3.4,P,-15000.0,Strategy Execution
"""

        # Use mock_open to simulate the file
        with patch('builtins.open', mock_open(read_data=csv_data)):
            analyze_performance(config=mock_load_config.return_value)

        # --- Assertions ---
        mock_send_notification.assert_called_once()

        args, kwargs = mock_send_notification.call_args
        title = kwargs.get('title')
        message = kwargs.get('message')

        self.assertIn("Daily Report: P&L $15,000.00", title)
        self.assertIn("<b>Daily Net P&L: $15,000.00</b>", message)
        self.assertIn("Combo 123: Net P&L = $15,000.00", message)
        self.assertIn("<b>Currently Open Positions:</b>", message)
        self.assertIn("BUY 1 KCK6 P3.4", message)


if __name__ == '__main__':
    unittest.main()