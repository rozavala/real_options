import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
import pytz
from datetime import datetime, timedelta

from ib_insync import IB, Contract, Future, Bag, ComboLeg, Position, OrderStatus, Trade, FuturesOption, PnL, PnLSingle

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.risk_management import manage_existing_positions, _check_risk_once, _on_order_status
from trading_bot.risk_management import _filled_order_ids as filled_set_module

class TestRiskManagementNewLogic(unittest.TestCase):

    def setUp(self):
        """Set up mock trade objects for reuse."""
        self.mock_trade = MagicMock(spec=Trade)
        self.mock_trade.isDone.return_value = True
        self.mock_trade.orderStatus = MagicMock(spec=OrderStatus)
        self.mock_trade.orderStatus.status = OrderStatus.Filled
        # Reset the global set of filled orders before each test
        filled_set_module.clear()

    @patch('trading_bot.risk_management.get_trade_ledger_df')
    def test_check_risk_once_stop_loss_max_risk(self, mock_get_ledger):
        async def run_test():
            # Arrange
            ib = AsyncMock(spec=IB)
            ib.isConnected.return_value = True
            ib.managedAccounts.return_value = ['DU12345']
            ib.placeOrder.return_value = self.mock_trade

            config = {
                'notifications': {},
                'risk_management': {
                    'take_profit_capture_pct': 0.80,
                    'stop_loss_max_risk_pct': 0.50, # Max Risk 50%
                    'monitoring_grace_period_seconds': 60
                },
                'exchange': 'NYBOT'
            }
            # Mock the ledger to show the position was opened a while ago
            # Use naive timestamp as get_trade_ledger_df typically returns naive timestamps
            mock_df = pd.DataFrame({
                'timestamp': [pd.Timestamp.now() - pd.Timedelta(minutes=5)],
                'position_id': ['KCH5-KCJ5'], # Example ID
                'quantity': [1]
            })
            mock_get_ledger.return_value = mock_df

            # Setup a Debit Spread (Long Call Spread)
            # Long 3.5 Call (Cost 100), Short 3.6 Call (Credit 50) -> Net Debit 50.
            # Max Risk = 50 (Entry Cost).
            # 50% of Max Risk = 25 loss.

            # Leg 1: Long Call
            leg1_pos = Position('Test', FuturesOption('KC', localSymbol='KCH5', conId=123, multiplier='37500', right='C', strike=3.5), 1, 100.0)
            # Leg 2: Short Call
            leg2_pos = Position('Test', FuturesOption('KC', localSymbol='KCJ5', conId=124, multiplier='37500', right='C', strike=3.6), -1, 50.0)

            ib.reqPositionsAsync = AsyncMock(return_value=[leg1_pos, leg2_pos])
            ib.reqContractDetailsAsync = AsyncMock(side_effect=[[MagicMock(underConId=999)], [MagicMock(underConId=999)]])

            # PnL:
            # Current Value needs to be such that we lost > 25.
            # Entry Cost = 1 * 100 + (-1) * 50 = 50 (Net Debit).
            # If PnL is -30, we lost 30. 30/50 = 60% loss. > 50% threshold. Trigger Stop.

            pnl1_valid = PnLSingle(conId=123, unrealizedPnL=-20.0)
            pnl2_valid = PnLSingle(conId=124, unrealizedPnL=-10.0)
            # Total PnL = -30.

            pnl_side_effects = {
                123: [pnl1_valid, pnl1_valid],
                124: [pnl2_valid, pnl2_valid]
            }
            pnl_effects_copy = {k: v[:] for k, v in pnl_side_effects.items()}

            def pnl_handler(*args):
                if not args:
                    return []
                else:
                    con_id = args[0]
                    if con_id in pnl_effects_copy and pnl_effects_copy[con_id]:
                        return pnl_effects_copy[con_id].pop(0)
                    return None

            ib.pnlSingle.side_effect = pnl_handler

            # We also need to handle pnlSingle being called inside `get_unrealized_pnl` via ib.portfolio check first
            # `get_unrealized_pnl` checks ib.portfolio() first. Let's mock it to return nothing so it falls back to pnlSingle polling
            ib.portfolio.return_value = []

            # Act
            # Arguments stop_loss_pct and take_profit_pct are ignored but required
            await _check_risk_once(ib, config, set(), 0, 0)

            # Assert
            # Should trigger stop loss and place closing orders for both legs
            self.assertEqual(ib.placeOrder.call_count, 2)

            # Verify arguments of placeOrder
            # Leg 1 was Position 1 (Long), so Closing Order should be SELL 1
            args1, _ = ib.placeOrder.call_args_list[0]
            contract1, order1 = args1
            self.assertEqual(contract1.conId, 123)
            self.assertEqual(order1.action, 'SELL')
            self.assertEqual(order1.totalQuantity, 1)
            self.assertTrue(order1.outsideRth)

            # Leg 2 was Position -1 (Short), so Closing Order should be BUY 1
            args2, _ = ib.placeOrder.call_args_list[1]
            contract2, order2 = args2
            self.assertEqual(contract2.conId, 124)
            self.assertEqual(order2.action, 'BUY')
            self.assertEqual(order2.totalQuantity, 1)

        asyncio.run(run_test())

    @patch('trading_bot.risk_management.get_trade_ledger_df')
    def test_check_risk_once_take_profit_capture(self, mock_get_ledger):
        async def run_test():
            # Arrange
            ib = AsyncMock(spec=IB)
            ib.isConnected.return_value = True
            ib.managedAccounts.return_value = ['DU12345']
            ib.placeOrder.return_value = self.mock_trade

            config = {
                'notifications': {},
                'risk_management': {
                    'take_profit_capture_pct': 0.80, # Capture 80% of Max Profit
                    'stop_loss_max_risk_pct': 0.50,
                    'monitoring_grace_period_seconds': 60
                },
                'exchange': 'NYBOT'
            }
            # Mock the ledger
            mock_df = pd.DataFrame({
                'timestamp': [pd.Timestamp.now() - pd.Timedelta(minutes=5)],
                'position_id': ['KCH5-KCJ5'],
                'quantity': [1]
            })
            mock_get_ledger.return_value = mock_df

            # Setup a Debit Spread again.
            # Long 3.5 Call (Cost 100), Short 3.6 Call (Credit 50) -> Net Debit 50.
            # Spread Width = 3.6 - 3.5 = 0.1.
            # Max Profit (Approx) = Width * Multiplier - Cost = 0.1 * 37500 - 50 = 3750 - 50 = 3700.
            # Target Capture = 80% of 3700 = 2960.

            # Leg 1: Long Call
            leg1_pos = Position('Test', FuturesOption('KC', localSymbol='KCH5', conId=123, multiplier='37500', right='C', strike=3.5), 1, 100.0)
            # Leg 2: Short Call
            leg2_pos = Position('Test', FuturesOption('KC', localSymbol='KCJ5', conId=124, multiplier='37500', right='C', strike=3.6), -1, 50.0)

            ib.reqPositionsAsync = AsyncMock(return_value=[leg1_pos, leg2_pos])
            ib.reqContractDetailsAsync = AsyncMock(side_effect=[[MagicMock(underConId=999)], [MagicMock(underConId=999)]])

            # PnL: We need > 2960.
            # Let's say PnL is +3000.

            pnl1_valid = PnLSingle(conId=123, unrealizedPnL=2000.0)
            pnl2_valid = PnLSingle(conId=124, unrealizedPnL=1000.0)
            # Total PnL = 3000.

            pnl_side_effects = {
                123: [pnl1_valid, pnl1_valid],
                124: [pnl2_valid, pnl2_valid]
            }
            pnl_effects_copy = {k: v[:] for k, v in pnl_side_effects.items()}

            def pnl_handler(*args):
                if not args: return []
                con_id = args[0]
                return pnl_effects_copy.get(con_id, []).pop(0) if pnl_effects_copy.get(con_id) else None

            ib.pnlSingle.side_effect = pnl_handler
            ib.portfolio.return_value = []

            # Act
            await _check_risk_once(ib, config, set(), 0, 0)

            # Assert
            self.assertEqual(ib.placeOrder.call_count, 2) # Should trigger take profit

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
