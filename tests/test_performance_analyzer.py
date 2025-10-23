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
             patch('performance_analyzer.get_trade_ledger_df') as mock_get_ledger:

            mock_config = {}
            mock_generate_charts.return_value = ["/path/to/fake_chart.png"]
            today_str = datetime.now().strftime('%Y-%m-%d')

            # Trade data uses 'KCH6' -> March 2026 contract
            csv_data = f"""timestamp,combo_id,local_symbol,action,quantity,avg_fill_price,strike,right,total_value_usd,reason
    {today_str} 10:00:00,123,KCH6 C3.5,BUY,1,0.5,3.5,C,-18750.0,Strategy Execution
    {today_str} 10:00:00,123,KCH6 C3.6,SELL,1,0.2,3.6,C,7500.0,Strategy Execution
    {today_str} 14:00:00,123,KCH6 C3.5,SELL,1,0.8,3.5,C,30000.0,Take-Profit
    {today_str} 14:00:00,123,KCH6 C3.6,BUY,1,0.1,3.6,C,-3750.0,Take-Profit
    """
            trade_df = pd.read_csv(pd.io.common.StringIO(csv_data))
            trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])

            # Manually apply the same scaling that the real get_trade_ledger_df does
            trade_df['total_value_usd'] = trade_df['total_value_usd'] / 100.0

            mock_get_ledger.return_value = trade_df

            # Signal data must use YYYYMM format, corresponding to KCH6
            signals_csv_data = f"""timestamp,contract,signal
    {today_str} 09:00:00,202603,BULLISH
    """
            signals_df = pd.read_csv(pd.io.common.StringIO(signals_csv_data))
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            mock_get_signals.return_value = signals_df

            with patch('performance_analyzer.check_for_open_orders', return_value=[]):
                # --- Act ---
                result = await analyze_performance(config=mock_config)
                assert result is not None
                report, total_pnl, chart_paths = result

            # --- Assertions ---
            # Corrected P&L assertion to reflect the scaling change (15000.00 -> 150.00)
            assert total_pnl == pytest.approx(150.00)
            assert "Section 1: Exec. Summary" in report
            assert "Net P&L" in report
            # Corrected P&L string check in the report
            assert "$150.00" in report
            assert "Section 2: Model Performance" in report
            assert "Section 3: System Status" in report
            assert "Data Integrity Check: PASS" in report
