import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from datetime import datetime
import pytest
import asyncio

from performance_analyzer import analyze_performance

class TestPerformanceAnalyzer:

    @pytest.mark.asyncio
    async def test_analyze_performance(self):
        # --- Setup Mocks ---
        with patch('performance_analyzer.generate_performance_charts') as mock_generate_charts, \
             patch('performance_analyzer.get_model_signals_df') as mock_get_signals, \
             patch('performance_analyzer.get_trade_ledger_df') as mock_get_ledger, \
             patch('performance_analyzer.get_account_pnl_and_positions') as mock_get_account_summary:

            mock_config = {}
            mock_generate_charts.return_value = ["/path/to/fake_chart.png"]
            today_str = datetime.now().strftime('%Y-%m-%d')

            # Mock the live data fetch
            mock_get_account_summary.return_value = {
                'daily_pnl': 155.25, # A slightly different value to confirm it's used
                'positions': []
            }

            # Trade data uses 'KCH6' -> March 2026 contract
            csv_data = f"""timestamp,combo_id,local_symbol,action,quantity,avg_fill_price,strike,right,total_value_usd,reason
    {today_str} 10:00:00,123,KCH6 C3.5,BUY,1,0.5,3.5,C,-18750.0,Strategy Execution
    {today_str} 10:00:00,123,KCH6 C3.6,SELL,1,0.2,3.6,C,7500.0,Strategy Execution
    {today_str} 14:00:00,123,KCH6 C3.5,SELL,1,0.8,3.5,C,30000.0,Take-Profit
    {today_str} 14:00:00,123,KCH6 C3.6,BUY,1,0.1,3.6,C,-3750.0,Take-Profit
    """
            trade_df = pd.read_csv(pd.io.common.StringIO(csv_data))
            trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])

            # Manually apply the scaling as the real function would
            trade_df['total_value_usd'] = trade_df['total_value_usd'] / 100.0

            mock_get_ledger.return_value = trade_df

            # Signal data must use YYYYMM format, corresponding to KCH6
            signals_csv_data = f"""timestamp,contract,signal
    {today_str} 09:00:00,202603,BULLISH
    """
            signals_df = pd.read_csv(pd.io.common.StringIO(signals_csv_data))
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            mock_get_signals.return_value = signals_df

            # --- Act ---
            result = await analyze_performance(config=mock_config)

            # --- Assertions ---
            assert result is not None
            assert "title" in result
            assert "reports" in result
            assert "charts" in result

            # Check that the title uses the live P&L from the mock
            assert "$155.25" in result['title']

            # Check that all the report sections were generated
            reports = result['reports']
            assert "Exec. Summary" in reports
            assert "Morning Signals" in reports
            assert "Open Positions" in reports
            assert "Closed Positions" in reports

            # Spot check some content
            assert "Net P&L" in reports['Exec. Summary']
            assert "$155.25" in reports['Exec. Summary'] # Live P&L in the summary table
            assert "BULLISH" in reports['Morning Signals']
            assert "No open positions" in reports['Open Positions']
            assert "KCH6 C3.5 / KCH6 C3.6" in reports['Closed Positions']
            assert "$150.00" in reports['Closed Positions'] # Ledger P&L for closed positions

            assert result['charts'] == ["/path/to/fake_chart.png"]
