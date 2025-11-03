import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd
from datetime import datetime
import pytest

# Mock objects to simulate ib_insync classes without needing the library
class MockContract:
    def __init__(self, localSymbol=""):
        self.localSymbol = localSymbol

class MockPortfolioItem:
    def __init__(self, contract, position, averageCost, unrealizedPNL):
        self.contract = contract
        self.position = position
        self.averageCost = averageCost
        self.unrealizedPNL = unrealizedPNL

class MockAccountValue:
    def __init__(self, tag, value, account):
        self.tag = tag
        self.value = value
        self.account = account

from performance_analyzer import analyze_performance

class TestPerformanceAnalyzer:

    @pytest.mark.asyncio
    async def test_analyze_performance_with_live_data(self):
        # --- Test Setup ---
        # Hardcode the date for deterministic test results
        test_date = datetime(2025, 10, 31)
        test_date_str = test_date.strftime('%Y-%m-%d')

        # --- Setup Mocks ---
        with patch('performance_analyzer.generate_performance_charts') as mock_generate_charts, \
             patch('performance_analyzer.get_model_signals_df') as mock_get_signals, \
             patch('performance_analyzer.get_trade_ledger_df') as mock_get_ledger, \
             patch('performance_analyzer.datetime') as mock_datetime, \
             patch('performance_analyzer.IB') as mock_ib_class:

            # 1. Mock IB Class and its methods
            mock_ib_instance = AsyncMock()
            # Configure sync methods with MagicMock to avoid RuntimeWarning
            mock_ib_instance.isConnected = MagicMock(return_value=True)
            mock_ib_instance.disconnect = MagicMock()
            mock_ib_class.return_value = mock_ib_instance

            # Configure sync methods with MagicMock to avoid RuntimeWarning
            mock_ib_instance.isConnected = MagicMock(return_value=True)
            mock_ib_instance.disconnect = MagicMock()
            mock_ib_instance.reqAccountSummary = MagicMock()
            mock_ib_instance.cancelAccountSummary = MagicMock()

            # Mock the polling behavior for accountValues
            mock_pnl_summary = MockAccountValue(tag='DailyPnL', value='155.25', account='U12345')
            # First call returns empty, second call returns the value
            mock_ib_instance.accountValues = MagicMock(side_effect=[[], [mock_pnl_summary]])

            mock_open_contract = MockContract(localSymbol="KOZ5 P4.5")
            mock_open_position = MockPortfolioItem(
                contract=mock_open_contract, position=2.0,
                averageCost=0.75, unrealizedPNL=-150.0
            )
            # Configure the portfolio() method, which is synchronous, using a MagicMock
            mock_ib_instance.portfolio = MagicMock(return_value=[mock_open_position])

            # 2. Mock other dependencies
            mock_config = {'connection': {'account_number': 'U12345'}}
            mock_generate_charts.return_value = ["/path/to/fake_chart.png"]
            mock_datetime.now.return_value = test_date

            # 3. Mock Trade Ledger Data (for closed positions)
            csv_data = f"""timestamp,combo_id,local_symbol,action,quantity,avg_fill_price,strike,right,total_value_usd,reason
    {test_date_str} 10:00:00,123,KCH6 C3.5,BUY,1,0.5,3.5,C,-18750.0,Strategy Execution
    {test_date_str} 10:00:00,123,KCH6 C3.6,SELL,1,0.2,3.6,C,7500.0,Strategy Execution
    {test_date_str} 14:00:00,123,KCH6 C3.5,SELL,1,0.8,3.5,C,30000.0,Take-Profit
    {test_date_str} 14:00:00,123,KCH6 C3.6,BUY,1,0.1,3.6,C,-3750.0,Take-Profit
    """
            trade_df = pd.read_csv(pd.io.common.StringIO(csv_data))
            trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
            # Convert cents to dollars, as the analysis functions expect
            trade_df['total_value_usd'] = trade_df['total_value_usd'] / 100.0
            mock_get_ledger.return_value = trade_df

            # 4. Mock Signals Data
            signals_csv_data = f"""timestamp,contract,signal
    {test_date_str} 09:00:00,202603,BULLISH
    """
            signals_df = pd.read_csv(pd.io.common.StringIO(signals_csv_data))
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            mock_get_signals.return_value = signals_df

            # --- Act ---
            result = await analyze_performance(config=mock_config)

            # --- Assertions ---
            assert result is not None

            # Check title uses live P&L
            assert "$155.25" in result['title']

            reports = result['reports']

            # Assert Exec Summary uses live P&L for "Today"
            assert "Exec. Summary" in reports
            assert "$155.25" in reports['Exec. Summary']

            # Assert Morning Signals are present
            assert "Morning Signals" in reports
            assert "BULLISH" in reports['Morning Signals']

            # Assert Open Positions report is correctly formatted from live data
            assert "Open Positions" in reports
            assert "KOZ5 P4.5" in reports['Open Positions']
            assert "2" in reports['Open Positions']      # Quantity
            assert "$-150.00" in reports['Open Positions'] # Unrealized P&L

            # Assert Closed Positions report is correctly calculated from ledger
            assert "Closed Positions" in reports
            assert "KCH6 C3.5 / KCH6 C3.6" in reports['Closed Positions']
            assert "$150.00" in reports['Closed Positions'] # P&L from ledger

            assert result['charts'] == ["/path/to/fake_chart.png"]

            # Verify IB methods were called
            mock_ib_instance.connectAsync.assert_awaited_once()
            mock_ib_instance.portfolio.assert_called_once()
            mock_ib_instance.disconnect.assert_called_once()
