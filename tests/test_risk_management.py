import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from ib_insync import Contract, Future, Bag, ComboLeg, Position, OrderStatus, Trade, FuturesOption, PnL, PnLSingle

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.risk_management import manage_existing_positions, _check_risk_once, _on_order_status
from trading_bot.risk_management import _filled_order_ids as filled_set_module

class TestRiskManagement(unittest.TestCase):

    def setUp(self):
        """Set up mock trade objects for reuse."""
        self.mock_trade = MagicMock(spec=Trade)
        self.mock_trade.isDone.return_value = True
        self.mock_trade.orderStatus = MagicMock(spec=OrderStatus)
        self.mock_trade.orderStatus.status = OrderStatus.Filled
        # Reset the global set of filled orders before each test
        filled_set_module.clear()

    def test_manage_misaligned_positions(self):
        async def run_test():
            ib = AsyncMock()
            config = {'symbol': 'KC', 'strategy': {}}
            signal = {'prediction_type': 'DIRECTIONAL', 'direction': 'BULLISH'}
            future_contract = Future(conId=100, symbol='KC', lastTradeDateOrContractMonth='202512')
            leg1 = ComboLeg(conId=1, ratio=1, action='BUY', exchange='NYBOT')
            leg2 = ComboLeg(conId=2, ratio=1, action='SELL', exchange='NYBOT')
            bear_spread_contract = Bag(symbol='KC', comboLegs=[leg1, leg2], conId=99)
            position = Position(account='TestAccount', contract=bear_spread_contract, position=1, avgCost=0)
            mock_cd_pos = MagicMock(underConId=100)
            mock_cd_leg1 = MagicMock(contract=FuturesOption(conId=1, right='P', strike=3.5))
            mock_cd_leg2 = MagicMock(contract=FuturesOption(conId=2, right='P', strike=3.4))

            ib.reqPositionsAsync.return_value = [position]
            ib.reqContractDetailsAsync.side_effect = [[mock_cd_pos], [mock_cd_leg1], [mock_cd_leg2]]
            ib.placeOrder.return_value = self.mock_trade
            ib.qualifyContractsAsync = AsyncMock()

            should_trade = await manage_existing_positions(ib, config, signal, 100.0, future_contract)

            self.assertTrue(should_trade)
            ib.placeOrder.assert_called_once()
            self.assertEqual(ib.placeOrder.call_args[0][1].action, 'SELL')

        asyncio.run(run_test())

    def test_manage_aligned_positions(self):
        async def run_test():
            ib = AsyncMock()
            config = {'symbol': 'KC'}
            signal = {'prediction_type': 'DIRECTIONAL', 'direction': 'BULLISH'}
            future_contract = Future(conId=100, symbol='KC', lastTradeDateOrContractMonth='202512')
            leg1 = ComboLeg(conId=1, ratio=1, action='BUY', exchange='NYBOT')
            leg2 = ComboLeg(conId=2, ratio=1, action='SELL', exchange='NYBOT')
            bull_spread_contract = Bag(symbol='KC', comboLegs=[leg1, leg2], conId=98)
            position = Position(account='TestAccount', contract=bull_spread_contract, position=1, avgCost=0)
            mock_cd_pos = MagicMock(underConId=100)
            mock_cd_leg1 = MagicMock(contract=FuturesOption(conId=1, right='C', strike=3.5))
            mock_cd_leg2 = MagicMock(contract=FuturesOption(conId=2, right='C', strike=3.6))

            ib.reqPositionsAsync.return_value = [position]
            ib.reqContractDetailsAsync.side_effect = [[mock_cd_pos], [mock_cd_leg1], [mock_cd_leg2]]

            should_trade = await manage_existing_positions(ib, config, signal, 100.0, future_contract)

            self.assertFalse(should_trade)
            ib.placeOrder.assert_not_called()

        asyncio.run(run_test())

    def test_check_risk_once_stop_loss(self):
        async def run_test():
            ib = AsyncMock(isConnected=MagicMock(return_value=True), managedAccounts=MagicMock(return_value=['DU12345']))
            ib.placeOrder.return_value = self.mock_trade
            config = {'notifications': {}, 'risk_management': {'stop_loss_pct': 0.20, 'take_profit_pct': 3.00}}
            leg1_pos = Position('Test', FuturesOption('KC', conId=123, multiplier='37500', right='C', strike=3.5), 1, 1.0)
            leg2_pos = Position('Test', FuturesOption('KC', conId=124, multiplier='37500', right='C', strike=3.6), -1, 0.5)
            ib.reqPositionsAsync.return_value = [leg1_pos, leg2_pos]
            ib.reqContractDetailsAsync.side_effect = [[MagicMock(underConId=999)], [MagicMock(underConId=999)]]

            pnl1 = PnLSingle(unrealizedPnL=-10000.0)
            pnl2 = PnLSingle(unrealizedPnL=-2000.0)
            ib.reqPnLSingleAsync.side_effect = [pnl1, pnl2]

            await _check_risk_once(ib, config, set(), 0.20, 3.00)
            self.assertEqual(ib.placeOrder.call_count, 2)

        asyncio.run(run_test())

    @patch('trading_bot.risk_management.log_trade_to_ledger')
    def test_on_order_status_handles_new_fill(self, mock_log_trade):
        """Verify that a new fill is logged correctly and the ID is tracked."""
        # Arrange: Create a mock trade representing a new fill
        newly_filled_trade = MagicMock(
            spec=Trade,
            order=MagicMock(orderId=102, action='BUY'),
            contract=MagicMock(localSymbol='KCH5'),
            orderStatus=MagicMock(status=OrderStatus.Filled, filled=1, avgFillPrice=3.50)
        )
        # Ensure the orderId is not already in the tracked set
        self.assertNotIn(102, filled_set_module)

        # Act: Call the event handler
        _on_order_status(newly_filled_trade)

        # Assert: Check that the trade was logged and the ID is now tracked
        mock_log_trade.assert_called_once_with(newly_filled_trade, "Daily Strategy Fill")
        self.assertIn(102, filled_set_module)

    @patch('trading_bot.risk_management.log_trade_to_ledger')
    def test_on_order_status_ignores_duplicate_fill(self, mock_log_trade):
        """Verify that a fill for an already logged order is ignored."""
        # Arrange: Add an order ID to the set of filled orders
        filled_set_module.add(101)
        duplicate_fill_trade = MagicMock(
            spec=Trade,
            order=MagicMock(orderId=101),
            orderStatus=MagicMock(status=OrderStatus.Filled)
        )

        # Act: Call the event handler with the duplicate fill
        _on_order_status(duplicate_fill_trade)

        # Assert: Ensure the logging function was not called again
        mock_log_trade.assert_not_called()

if __name__ == '__main__':
    unittest.main()