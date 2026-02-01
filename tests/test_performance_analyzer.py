import unittest
from unittest.mock import patch, AsyncMock, MagicMock, mock_open
import pandas as pd
from datetime import datetime
import pytest
import os

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

class MockAccountValue:
    def __init__(self, tag, value, account=""):
        self.tag = tag
        self.value = value
        self.account = account

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

from performance_analyzer import analyze_performance

class TestPerformanceAnalyzer:

    @pytest.mark.asyncio
    async def test_analyze_performance_dynamic_starting_capital(self):
        # --- Test Setup ---
        test_date = datetime(2025, 10, 31)

        # We will mock the daily_equity.csv to have a starting value of 300,000
        # If the code works, the LTD P&L should be based on this 300,000, not 250,000.

        # Realized P&L from ledger will be small (e.g. 30), but Equity P&L will be NetLiq - 300,000.

        with patch('performance_analyzer.generate_performance_charts') as mock_generate_charts, \
             patch('performance_analyzer.get_trade_ledger_df') as mock_get_ledger, \
             patch('performance_analyzer.datetime') as mock_datetime, \
             patch('performance_analyzer.IB') as mock_ib_class, \
             patch('performance_analyzer.pd.read_csv') as mock_read_csv, \
             patch('performance_analyzer.os.path.exists') as mock_exists:

            # 1. Mock File Exists for daily_equity.csv and trade_ledger.csv
            mock_exists.return_value = True

            # 2. Mock Dataframes
            # Trade Ledger
            mock_ledger_df = pd.DataFrame({
                'timestamp': [test_date, test_date],
                'action': ['BUY', 'SELL'],
                'combo_id': ['123', '123'],
                'total_value_usd': [-100, 150] # Realized = 50
            })

            # Daily Equity (Mocking the read_csv call for this)
            mock_equity_df = pd.DataFrame({
                'timestamp': [pd.Timestamp('2024-01-01'), pd.Timestamp('2025-10-31')],
                'total_value_usd': [300000.0, 300500.0]
            })

            # So mock_read_csv will be called ONLY for equity_df inside analyze_performance
            mock_read_csv.return_value = mock_equity_df

            mock_get_ledger.return_value = mock_ledger_df

            # 3. Mock IB
            mock_ib_instance = AsyncMock()
            mock_ib_class.return_value = mock_ib_instance
            mock_ib_instance.isConnected = MagicMock(return_value=True)
            mock_ib_instance.disconnect = MagicMock()

            # Mock Net Liquidation Value
            # Let's say current NetLiq is 301,000.
            # Starting Capital (from mocked equity_df first row) is 300,000.
            # Expected LTD Total P&L = 301,000 - 300,000 = 1,000.
            mock_net_liq = MockAccountValue(tag='NetLiquidation', value='301000.00')
            mock_ib_instance.accountSummaryAsync = AsyncMock(return_value=[mock_net_liq])

            # Mock Portfolio & Fills (Empty for simplicity)
            mock_ib_instance.portfolio = MagicMock(return_value=[])
            mock_ib_instance.reqExecutionsAsync = AsyncMock(return_value=[])

            # 4. Other Mocks
            mock_datetime.now.return_value = test_date

            # --- Act ---
            result = await analyze_performance(config={})

            # --- Assertions ---
            assert result is not None
            summary = result['reports']['Exec. Summary']

            # Verify Total P&L is 1,000 (301k - 300k), NOT 51,000 (301k - 250k)
            # Format is $1,000.00
            assert "$1,000.00" in summary
            assert "$51,000.00" not in summary

            # Verify Starting Capital passed to charts was 300,000
            args, _ = mock_generate_charts.call_args
            # Args: trade_df, signals_df, equity_df, starting_capital
            assert args[3] == 300000.0
