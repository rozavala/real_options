import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from datetime import datetime

from performance_analyzer import analyze_performance

class TestPerformanceAnalyzer(unittest.TestCase):

    @patch('performance_analyzer.send_pushover_notification')
    @patch('os.path.exists', return_value=True)
    def test_analyze_performance(self, mock_exists, mock_send_notification):
        # --- Setup Mocks ---
        mock_config = {
            'notifications': {
                'enabled': True,
                'pushover_user_key': 'test_user',
                'pushover_api_token': 'test_token'
            }
        }

        today_str = datetime.now().strftime('%Y-%m-%d')
        yesterday_str = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        # P&L for combo 123 = -18750 (BUY) + 7500 (SELL) + 30000 (SELL) - 3750 (BUY) = 15000
        # Note: The calculation in the original script sums total_value_usd.
        # A BUY action results in a negative value (cash outflow), and a SELL is positive (cash inflow).
        # The total P&L is the sum of these values for a closed trade.
        csv_data = f"""timestamp,combo_id,local_symbol,action,quantity,avg_fill_price,strike,right,total_value_usd,reason
{today_str} 10:00:00,123,KCH6 C3.5,BUY,1,0.5,3.5,C,-18750.0,Strategy Execution
{today_str} 10:00:00,123,KCH6 C3.6,SELL,1,0.2,3.6,C,7500.0,Strategy Execution
{today_str} 14:00:00,123,KCH6 C3.5,SELL,1,0.8,3.5,C,30000.0,Take-Profit
{today_str} 14:00:00,123,KCH6 C3.6,BUY,1,0.1,3.6,C,-3750.0,Take-Profit
{yesterday_str} 11:00:00,456,KCK6 P3.4,BUY,1,0.4,3.4,P,-15000.0,Strategy Execution
"""

        # Use mock_open to simulate the file
        with patch('builtins.open', mock_open(read_data=csv_data)):
            analyze_performance(config=mock_config)

        # --- Assertions ---
        mock_send_notification.assert_called_once()

        # Extract arguments from the mock call
        call_args = mock_send_notification.call_args[0]
        call_kwargs = mock_send_notification.call_args[1]

        # The first argument is the config part, second is title, third is message
        # Let's check kwargs as they are more explicit
        title = call_kwargs.get('title')
        message = call_kwargs.get('message')

        self.assertIn("Daily Report: P&L $15,000.00", title)
        self.assertIn("<b>Daily Net P&L: $15,000.00</b>", message)
        self.assertIn("Combo 123: Net P&L = $15,000.00", message)
        self.assertIn("<b>Currently Open Positions:</b>", message)
        self.assertIn("BUY 1 KCK6 P3.4", message)
        self.assertIn("(Entry Cost: $15,000.00)", message)

if __name__ == '__main__':
    unittest.main()