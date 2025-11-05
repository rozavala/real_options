import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd
from datetime import datetime
import pytest

# Mock objects to simulate ib_insync classes without needing the library
class MockContract:
    def __init__(self, localSymbol="", conId=0, multiplier='1'):
        self.localSymbol = localSymbol
        self.conId = conId
        self.multiplier = multiplier

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
class MockCommissionReport:
    def __init__(self, realizedPNL):
        self.realizedPNL = realizedPNL

class MockExecution:
    def __init__(self, side, shares, price, permId):
        self.side = side
        self.shares = shares
        self.price = price
        self.permId = permId

class MockFill:
    def __init__(self, contract, execution, commissionReport, time):
        self.contract = contract
        self.execution = execution
        self.commissionReport = commissionReport
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

            # Mock Portfolio data
            mock_open_positions = [
                MockPortfolioItem(contract=MockContract(localSymbol="KOZ5 C4.0"), position=1.0, averageCost=100, unrealizedPNL=50.0),
                MockPortfolioItem(contract=MockContract(localSymbol="KOZ5 C4.1"), position=-1.0, averageCost=120, unrealizedPNL=-20.0),
                MockPortfolioItem(contract=MockContract(localSymbol="KOH6 C3.8"), position=1.0, averageCost=200, unrealizedPNL=100.0),
            ]
            mock_ib_instance.portfolio = MagicMock(return_value=mock_open_positions)

            # Mock Executions for TODAY's closed positions
            fills_today = [
                MockFill(
                    MockContract(localSymbol="KON6 C3.5", multiplier='37500'),
                    MockExecution('BOT', 1.0, 0.1, permId=123),
                    MockCommissionReport(realizedPNL=0),
                    test_date
                ),
                MockFill(
                    MockContract(localSymbol="KON6 C3.5", multiplier='37500'),
                    MockExecution('SLD', 1.0, 0.15, permId=123),
                    MockCommissionReport(realizedPNL=1875.00),
                    test_date
                ),
            ]
            mock_ib_instance.reqExecutionsAsync = AsyncMock(return_value=fills_today)

            # 2. Mock other dependencies
            mock_config = {'connection': {'account_number': 'U12345'}}
            mock_generate_charts.return_value = ["/path/to/chart.png"]
            mock_datetime.now.return_value = test_date

            # 3. Mock Trade Ledger and Signals Data (for LTD stats)
            mock_ledger_data = {
                'timestamp': [test_date, test_date, test_date, test_date],
                'action': ['BUY', 'SELL', 'BUY', 'SELL'],
                'combo_id': ['123,456', '123,456', 999888, 999888], # Test string and int combo_ids
                'total_value_usd': [-100, 150, -200, 180]
            }
            mock_get_ledger.return_value = pd.DataFrame(mock_ledger_data)
            signals_df = pd.DataFrame({
                'timestamp': [test_date, test_date],
                'contract': ['202603', '202604'],
                'signal': ['BULLISH', 'BEARISH']
            })
            mock_get_signals.return_value = signals_df

            # --- Act ---
            result = await analyze_performance(config=mock_config)

            # --- Assertions ---
            assert result is not None
            # --- Assertions ---
            # --- Daily Assertions ---
            # Total PNL = Realized (1875.00) + Unrealized (50 - 20 + 100) = 1875 + 130 = 2005.00
            assert "Total P&L $2,005.00" in result['title']

            summary = result['reports']['Exec. Summary']
            assert "Realized P&L" in summary and "$1,875.00" in summary
            assert "Unrealized P&L" in summary and "$130.00" in summary
            assert "Trades Executed" in summary and "      1" in summary # Today's trades

            # --- LTD Assertions (verifies the integer combo_id fix) ---
            # LTD PNL = (150-100) + (180-200) = 50 - 20 = 30
            assert "Total P&L" in summary and "$30.00" in summary
            assert "Trades Executed" in summary and "      2" in summary # LTD trades
            assert "Win Rate" in summary and "50.0%" in summary
            assert "Unrealized P&L" in summary and "$130.00" in summary
            assert "Trades Executed" in summary and "1" in summary

            # Assert Open Positions report is grouped and formatted correctly
            open_report = result['reports']['Open Positions']
            assert "Subtotal for KOZ5" in open_report
            assert "$30.00" in open_report # 50 - 20
            assert "Subtotal for KOH6" in open_report
            assert "$100.00" in open_report
            assert "GRAND TOTAL" in open_report and "$130.00" in open_report

            # Assert Closed Positions report is from live commission data
            closed_report = result['reports']['Closed Positions']
            assert "KON6 C3.5" in closed_report
            assert "$1,875.00" in closed_report

            # Verify correct IB methods were called
            mock_ib_instance.connectAsync.assert_awaited_once()
            mock_ib_instance.reqExecutionsAsync.assert_awaited_once()
            mock_ib_instance.portfolio.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_performance_runs_without_error(self):
        # A simple test to ensure the main function runs without crashing.
        # --- Test Setup ---
        test_date = datetime(2025, 10, 31)

        with patch('performance_analyzer.get_live_account_data') as mock_get_live_data, \
             patch('performance_analyzer.get_trade_ledger_df') as mock_get_ledger, \
             patch('performance_analyzer.get_model_signals_df') as mock_get_signals, \
             patch('performance_analyzer.generate_performance_charts'):

            # Mock the data fetching functions to return empty or minimal data
            mock_get_live_data.return_value = {
                'portfolio': [],
                'executions': [],
            }
            mock_get_ledger.return_value = pd.DataFrame()
            mock_get_signals.return_value = pd.DataFrame(columns=['timestamp', 'contract', 'signal'])

            # --- Act & Assert ---
            # The main assertion is that this runs without throwing an exception
            await analyze_performance(config={})
