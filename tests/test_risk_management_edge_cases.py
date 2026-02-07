import unittest
import asyncio
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock heavy dependencies to avoid installation requirement
sys.modules['chromadb'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['xgboost'] = MagicMock()

from ib_insync import IB, Contract, Future, Bag, ComboLeg, Position, OrderStatus, Trade, FuturesOption, PnLSingle
import pandas as pd

from trading_bot.risk_management import _check_risk_once
from trading_bot.risk_management import _filled_order_ids as filled_set_module

class TestRiskManagementEdgeCases(unittest.TestCase):

    def setUp(self):
        filled_set_module.clear()

    @patch('trading_bot.risk_management.get_dollar_multiplier')
    @patch('trading_bot.risk_management.get_trade_ledger_df')
    def test_single_leg_debit_magic_number(self, mock_get_ledger, mock_get_mult):
        """Verify fallback logic for single leg debit (uses magic number 2)."""
        async def run_test():
            # Arrange
            ib = AsyncMock(spec=IB)
            ib.isConnected.return_value = True
            ib.managedAccounts.return_value = ['DU12345']

            mock_get_mult.return_value = 37500.0

            # Mock single leg LONG call (Debit)
            leg_pos = Position('Test', FuturesOption('KC', localSymbol='KCH5', conId=123, multiplier='37500', right='C', strike=3.5), 1, 1000.0)
            ib.reqPositionsAsync = AsyncMock(return_value=[leg_pos])

            # Mock Contract Details to ensure grouping works
            ib.reqContractDetailsAsync = AsyncMock(side_effect=[[MagicMock(underConId=999)]])

            # PnL: Profit 1000
            pnl_valid = PnLSingle(conId=123, unrealizedPnL=1000.0)
            ib.pnlSingle.side_effect = lambda *args: pnl_valid if args else []

            # Config
            config = {
                'symbol': 'KC',
                'notifications': {},
                'risk_management': {
                    'stop_loss_max_risk_pct': 0.50,
                    'take_profit_capture_pct': 0.40,
                    'monitoring_grace_period_seconds': 0
                }
            }

            mock_df = pd.DataFrame()
            mock_get_ledger.return_value = mock_df

            # Place Order Mock
            mock_trade = MagicMock(spec=Trade)
            mock_trade.isDone.return_value = True
            ib.placeOrder.return_value = mock_trade

            # Act
            await _check_risk_once(ib, config, set(), 0.50, 0.40)

            # Assert
            print(f"DEBUG: PlaceOrder call count: {ib.placeOrder.call_count}")

            if ib.placeOrder.call_count == 0:
                self.fail("Take Profit not triggered.")

            self.assertEqual(ib.placeOrder.call_count, 1)

        asyncio.run(run_test())

    @patch('trading_bot.risk_management.get_dollar_multiplier')
    @patch('trading_bot.risk_management.get_trade_ledger_df')
    def test_single_leg_credit_negative_max_loss(self, mock_get_ledger, mock_get_mult):
        """Verify single leg credit position results in negative max loss (bug)."""
        async def run_test():
            # Arrange
            ib = AsyncMock(spec=IB)
            ib.isConnected.return_value = True
            ib.managedAccounts.return_value = ['DU12345']

            mock_get_mult.return_value = 37500.0

            # Mock single leg SHORT call (Credit)
            # Position -1, Cost 500 (Credit received = 500)
            leg_pos = Position('Test', FuturesOption('KC', localSymbol='KCH5', conId=123, multiplier='37500', right='C', strike=3.5), -1, 500.0)

            ib.reqPositionsAsync = AsyncMock(return_value=[leg_pos])
            ib.reqContractDetailsAsync = AsyncMock(side_effect=[[MagicMock(underConId=999)]])

            # PnL: Loss -1000.
            pnl_valid = PnLSingle(conId=123, unrealizedPnL=-1000.0)
            ib.pnlSingle.side_effect = lambda *args: pnl_valid if args else []

            config = {
                'symbol': 'KC',
                'notifications': {},
                'risk_management': {
                    'stop_loss_pct': 0.50,
                    'take_profit_pct': 0.80,
                    'monitoring_grace_period_seconds': 0
                }
            }

            mock_df = pd.DataFrame()
            mock_get_ledger.return_value = mock_df

            mock_trade = MagicMock(spec=Trade)
            ib.placeOrder.return_value = mock_trade

            # Act
            await _check_risk_once(ib, config, set(), 0.50, 0.80)

            # Assert
            if ib.placeOrder.call_count > 0:
                self.fail("Stop Loss triggered unexpectedly! Bug might be fixed?")

            self.assertEqual(ib.placeOrder.call_count, 0)

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
