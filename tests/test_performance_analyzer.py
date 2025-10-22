import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from datetime import datetime

from performance_analyzer import analyze_performance

class TestPerformanceAnalyzer(unittest.TestCase):

    @patch('performance_analyzer.generate_performance_chart')
    @patch('performance_analyzer.get_trade_ledger_df')
    def test_analyze_performance(self, mock_get_ledger, mock_generate_chart):
        # --- Setup Mocks ---
        mock_config = {} # Config is not used for notification anymore
        mock_generate_chart.return_value = "/path/to/fake_chart.png"
        today_str = datetime.now().strftime('%Y-%m-%d')
        yesterday_str = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        # P&L for combo 'KC-123' = -18750 + 7500 + 30000 - 3750 = 15000
        csv_data = f"""timestamp,combo_id,local_symbol,action,quantity,avg_fill_price,strike,right,total_value_usd,reason
{today_str} 10:00:00,KC-123,KCH6 C3.5,BUY,1,0.5,3.5,C,-18750.0,Strategy Execution
{today_str} 10:00:00,KC-123,KCH6 C3.6,SELL,1,0.2,3.6,C,7500.0,Strategy Execution
{today_str} 14:00:00,KC-123,KCH6 C3.5,SELL,1,0.8,3.5,C,30000.0,Take-Profit
{today_str} 14:00:00,KC-123,KCH6 C3.6,BUY,1,0.1,3.6,C,-3750.0,Take-Profit
{yesterday_str} 11:00:00,KC-456,KCK6 P3.4,BUY,1,0.4,3.4,P,-15000.0,Strategy Execution
"""
        mock_get_ledger.return_value = pd.read_csv(pd.io.common.StringIO(csv_data))

        # --- Act ---
        result = analyze_performance(config=mock_config)
        self.assertIsNotNone(result)
        report, total_pnl, chart_path = result

        # --- Assertions ---
        self.assertAlmostEqual(total_pnl, 15000.00)
        self.assertIn("<b>Daily Net P&L: $15,000.00</b>", report)
        self.assertIn("KC (KC-123): Net P&L = $15,000.00", report)
        self.assertIn("<b>Currently Open Positions:</b>", report)
        self.assertIn("BUY 1 KCK6 P3.4", report)
        self.assertIn("(Net Debit: $15,000.00)", report)

if __name__ == '__main__':
    unittest.main()