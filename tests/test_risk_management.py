import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from ib_insync import Contract, Future, Bag, ComboLeg, Position, OrderStatus, Trade, FuturesOption, PnL, PnLSingle

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.risk_management import manage_existing_positions, monitor_positions_for_risk

class TestRiskManagement(unittest.TestCase):

    def test_manage_misaligned_positions(self):
        async def run_test():
            ib = AsyncMock()
            config = {'symbol': 'KC', 'strategy': {}}
            signal = {'prediction_type': 'DIRECTIONAL', 'direction': 'BULLISH'}
            future_contract = Future(conId=100, symbol='KC', lastTradeDateOrContractMonth='202512')

            # --- Mock existing position (Bear Put Spread) ---
            leg1 = ComboLeg(conId=1, ratio=1, action='BUY', exchange='NYBOT')
            leg2 = ComboLeg(conId=2, ratio=1, action='SELL', exchange='NYBOT')
            bear_spread_contract = Bag(symbol='KC', comboLegs=[leg1, leg2], conId=99)

            position = Position(account='TestAccount', contract=bear_spread_contract, position=1, avgCost=0)

            # Mock contract details to link the position to the future
            mock_cd_pos = MagicMock()
            mock_cd_pos.underConId = 100

            # Mock contract details for the legs that will be resolved inside get_position_details
            mock_cd_leg1 = MagicMock()
            mock_cd_leg1.contract = FuturesOption(conId=1, right='P', strike=3.5)
            mock_cd_leg2 = MagicMock()
            mock_cd_leg2.contract = FuturesOption(conId=2, right='P', strike=3.4)

            ib.reqPositionsAsync = AsyncMock(return_value=[position])
            # The side_effect will serve the call for the main position, then the two legs.
            ib.reqContractDetailsAsync = AsyncMock(side_effect=[[mock_cd_pos], [mock_cd_leg1], [mock_cd_leg2]])
            ib.placeOrder = MagicMock()
            ib.qualifyContractsAsync = AsyncMock() # Ensure this is awaitable

            # Mock the wait_for_fill to avoid actual waiting
            with patch('trading_bot.risk_management.wait_for_fill', new_callable=AsyncMock) as mock_wait:
                should_trade = await manage_existing_positions(ib, config, signal, 100.0, future_contract)

            # --- Assertions ---
            # It should decide to trade because it needs to close the misaligned position
            self.assertTrue(should_trade)
            # It should place an order to close the position
            ib.placeOrder.assert_called_once()
            # The order should be a SELL order to close the long position
            self.assertEqual(ib.placeOrder.call_args[0][1].action, 'SELL')

        asyncio.run(run_test())

    def test_manage_aligned_positions(self):
        async def run_test():
            ib = AsyncMock()
            config = {'symbol': 'KC'}
            signal = {'prediction_type': 'DIRECTIONAL', 'direction': 'BULLISH'}
            future_contract = Future(conId=100, symbol='KC', lastTradeDateOrContractMonth='202512')

            # --- Mock existing position (Bull Call Spread) ---
            leg1 = ComboLeg(conId=1, ratio=1, action='BUY', exchange='NYBOT')
            leg2 = ComboLeg(conId=2, ratio=1, action='SELL', exchange='NYBOT')
            bull_spread_contract = Bag(symbol='KC', comboLegs=[leg1, leg2], conId=98)

            position = Position(account='TestAccount', contract=bull_spread_contract, position=1, avgCost=0)

            mock_cd_pos = MagicMock()
            mock_cd_pos.underConId = 100

            # Mock contract details for the legs
            mock_cd_leg1 = MagicMock()
            mock_cd_leg1.contract = FuturesOption(conId=1, right='C', strike=3.5)
            mock_cd_leg2 = MagicMock()
            mock_cd_leg2.contract = FuturesOption(conId=2, right='C', strike=3.6)

            ib.reqPositionsAsync = AsyncMock(return_value=[position])
            ib.reqContractDetailsAsync = AsyncMock(side_effect=[[mock_cd_pos], [mock_cd_leg1], [mock_cd_leg2]])
            ib.placeOrder = MagicMock()

            should_trade = await manage_existing_positions(ib, config, signal, 100.0, future_contract)

            # It should not trade because an aligned position exists
            self.assertFalse(should_trade)
            # No orders should be placed
            ib.placeOrder.assert_not_called()

        asyncio.run(run_test())

    @unittest.skip("Skipping until PnL subscription can be tested in a live environment.")
    @patch('trading_bot.risk_management.asyncio.sleep', new_callable=AsyncMock)
    def test_monitor_positions_stop_loss(self, mock_sleep):
        async def run_test():
            ib = AsyncMock()
            ib.isConnected.return_value = True
            ib.managedAccounts.return_value = ['DU12345']
            ib.placeOrder = MagicMock()

            config = {
                'notifications': {},
                'risk_management': {
                    'stop_loss_per_contract_usd': 50,
                    'check_interval_seconds': 0  # Run the loop without real sleep
                }
            }

            # --- Mock position and PnL ---
            contract = Contract(conId=123, symbol='KC')
            position = Position(account='TestAccount', contract=contract, position=1, avgCost=0)

            mock_pnl = PnLSingle(account='TestAccount', modelCode='', conId=123, position=1, value=0.0, dailyPnL=0, unrealizedPnL=-60, realizedPnL=0)

            ib.reqPositionsAsync = AsyncMock(side_effect=[[position], []]) # First check finds position, second finds none
            ib.reqPnLSingle = MagicMock(return_value=mock_pnl)
            ib.qualifyContractsAsync = AsyncMock() # Ensure this is awaitable

            with patch('trading_bot.risk_management.wait_for_fill', new_callable=AsyncMock) as mock_wait, \
                 patch('trading_bot.risk_management.send_pushover_notification') as mock_notification:
                # We run the monitor as a task and cancel it after one loop
                monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
                await asyncio.sleep(0.01) # Give the loop time to run once
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            # An order should have been placed to close the position
            ib.placeOrder.assert_called_once()
            self.assertEqual(ib.placeOrder.call_args[0][1].action, 'SELL')

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()