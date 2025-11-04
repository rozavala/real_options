import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd
from datetime import datetime
import pytest

# Mock objects to simulate ib_insync classes without needing the library
class MockContract:
    def __init__(self, localSymbol="", conId=0):
        self.localSymbol = localSymbol
        self.conId = conId

class MockPortfolioItem:
    def __init__(self, contract, position, averageCost, unrealizedPNL):
        self.contract = contract
        self.position = position
        self.averageCost = averageCost
        self.unrealizedPNL = unrealizedPNL

class MockPnL:
    def __init__(self, dailyPnL):
        self.dailyPnL = dailyPnL

class MockAccountValue:
    def __init__(self, tag, value, account):
        self.tag = tag
        self.value = value
        self.account = account

from performance_analyzer import analyze_performance, generate_closed_positions_report

# Additional Mock classes for Executions
class MockExecution:
    def __init__(self, side, shares, price):
        self.side = side
        self.shares = shares
        self.price = price

class MockFill:
    def __init__(self, contract, execution, time):
        self.contract = contract
        self.execution = execution
        self.time = time

class TestPerformanceAnalyzer:

    @pytest.mark.asyncio
    async def test_analyze_performance_hybrid_approach(self):
        # --- Test Setup ---
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
            mock_ib_class.return_value = mock_ib_instance
            mock_ib_instance.isConnected = MagicMock(return_value=True)
            mock_ib_instance.disconnect = MagicMock()
            mock_ib_instance.managedAccounts = MagicMock(return_value=['U54321'])
            mock_ib_instance.cancelPnL = MagicMock()

            # Mock P&L and Portfolio data
            mock_account_summary = [
                MockAccountValue(tag='EquityWithLoanValue', value='100200.00', account='U12345'),
                MockAccountValue(tag='PreviousDayEquityWithLoanValue', value='100000.00', account='U12345')
            ]
            mock_ib_instance.accountSummaryAsync = AsyncMock(return_value=mock_account_summary)

            mock_open_position = MockPortfolioItem(
                contract=MockContract(localSymbol="KCK6 C3.0"), position=1.0,
                averageCost=0.5, unrealizedPNL=50.0
            )
            mock_ib_instance.portfolio = MagicMock(return_value=[mock_open_position])

            # Mock Executions for TODAY's closed positions
            fills_today = [
                MockFill(MockContract(conId=1, localSymbol="KOZ5 P4.5"), MockExecution('BOT', 1.0, 75.0), test_date),
                MockFill(MockContract(conId=1, localSymbol="KOZ5 P4.5"), MockExecution('SLD', 1.0, 225.0), test_date)
            ]
            mock_ib_instance.reqExecutionsAsync = AsyncMock(return_value=fills_today)

            # 2. Mock other dependencies
            mock_config = {'connection': {'account_number': 'U12345'}}
            mock_generate_charts.return_value = ["/path/to/chart.png"]
            mock_datetime.now.return_value = test_date

            # 3. Mock Trade Ledger Data (for LTD stats)
            ltd_csv_data = f"""timestamp,combo_id,local_symbol,action,total_value_usd,reason
    2025-10-30 10:00:00,10,KCH6 C3.5,BUY,-100.0,Strategy Execution
    2025-10-30 14:00:00,10,KCH6 C3.5,SELL,120.0,Take-Profit
    """
            trade_df = pd.read_csv(pd.io.common.StringIO(ltd_csv_data))
            trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
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
            # Title P&L = Realized (150) + Unrealized (50) = 200
            assert "Total P&L $200.00" in result['title']

            summary = result['reports']['Exec. Summary']
            assert "Total P&L" in summary and "$200.00" in summary
            assert "Realized P&L" in summary and "$150.00" in summary # From live fills
            assert "Unrealized P&L" in summary and "$50.00" in summary # From live portfolio
            assert "LTD" in summary and "$20.00" in summary # From historical ledger

            # Assert Closed Positions report is from live data
            closed_report = result['reports']['Closed Positions']
            assert "KOZ5 P4.5" in closed_report
            assert "$150.00" in closed_report

            # Verify correct IB methods were called
            mock_ib_instance.connectAsync.assert_awaited_once()
            mock_ib_instance.reqExecutionsAsync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_analyze_performance_no_account_in_config(self):
        # --- Test Setup ---
        test_date = datetime(2025, 10, 31)

        # --- Setup Mocks ---
        with patch('performance_analyzer.generate_performance_charts'), \
             patch('performance_analyzer.get_model_signals_df') as mock_get_signals, \
             patch('performance_analyzer.get_trade_ledger_df') as mock_get_ledger, \
             patch('performance_analyzer.datetime') as mock_datetime, \
             patch('performance_analyzer.IB') as mock_ib_class:

            # 1. Mock IB Class and its methods
            mock_ib_instance = AsyncMock()
            mock_ib_class.return_value = mock_ib_instance

            mock_ib_instance.isConnected = MagicMock(return_value=True)
            mock_ib_instance.disconnect = MagicMock()
            mock_ib_instance.managedAccounts = MagicMock(return_value=['U54321'])

            # Mock accountSummaryAsync to return the required P&L values for the fallback account
            mock_account_summary = [
                MockAccountValue(tag='EquityWithLoanValue', value='100155.25', account='U54321'),
                MockAccountValue(tag='PreviousDayEquityWithLoanValue', value='100000.00', account='U54321')
            ]
            mock_ib_instance.accountSummaryAsync = AsyncMock(return_value=mock_account_summary)

            mock_ib_instance.portfolio = MagicMock(return_value=[])

            # 2. Mock other dependencies
            mock_config = {'connection': {'account_number': ''}} # Empty account number
            mock_datetime.now.return_value = test_date
            mock_get_ledger.return_value = pd.DataFrame({'timestamp': [test_date]})
            mock_get_signals.return_value = pd.DataFrame({'timestamp': [test_date]})

            # --- Act ---
            await analyze_performance(config=mock_config)

            # --- Assertions ---
            # Verify that the fallback account is used
            mock_ib_instance.managedAccounts.assert_called_once()
            # Verify that accountSummaryAsync is called, as that's where the P&L is derived from
            mock_ib_instance.accountSummaryAsync.assert_awaited_once()
