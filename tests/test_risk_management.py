import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from ib_insync import Contract, Future, Bag, ComboLeg, Position, OrderStatus, Trade, FuturesOption, PnL, PnLSingle

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.risk_management import manage_existing_positions, _check_risk_once

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

    def test_check_risk_once_stop_loss(self):
        async def run_test():
            ib = AsyncMock()
            ib.isConnected = MagicMock(return_value=True)
            ib.managedAccounts = MagicMock(return_value=['DU12345'])
            ib.placeOrder = MagicMock()
            ib.qualifyContractsAsync = AsyncMock()
            ib.cancelPnLSingle = MagicMock()

            config = {
                'notifications': {},
                'risk_management': { 'stop_loss_pct': 0.20, 'take_profit_pct': 3.00 }
            }

            # --- Mock Combo Position ---
            # Total entry cost = abs(1*1.0*37500) + abs(-1*0.5*37500) = 37500 + 18750 = 56250
            # Stop loss trigger = -0.20 * 56250 = -11250
            leg1_contract = FuturesOption(conId=123, symbol='KC', multiplier='37500', right='C', strike=3.5)
            leg1_pos = Position(account='TestAccount', contract=leg1_contract, position=1, avgCost=1.0)
            leg1_pnl = MagicMock(spec=PnLSingle, unrealizedPnL=-10000.0)
            leg1_details = MagicMock(underConId=999)

            leg2_contract = FuturesOption(conId=124, symbol='KC', multiplier='37500', right='C', strike=3.6)
            leg2_pos = Position(account='TestAccount', contract=leg2_contract, position=-1, avgCost=0.5)
            leg2_pnl = MagicMock(spec=PnLSingle, unrealizedPnL=-2000.0) # Total PnL = -12000, which is > 11250 loss
            leg2_details = MagicMock(underConId=999)

            ib.reqPositionsAsync.return_value = [leg1_pos, leg2_pos]
            ib.reqContractDetailsAsync.side_effect = [[leg1_details], [leg2_details]]
            ib.reqPnLSingle = MagicMock(side_effect=[leg1_pnl, leg2_pnl])

            closed_ids = set()
            stop_loss_pct = config['risk_management']['stop_loss_pct']
            take_profit_pct = config['risk_management']['take_profit_pct']

            with patch('trading_bot.risk_management.wait_for_fill', new_callable=AsyncMock):
                await _check_risk_once(ib, config, closed_ids, stop_loss_pct, take_profit_pct)

            # Should place two orders to close both legs
            self.assertEqual(ib.placeOrder.call_count, 2)

        asyncio.run(run_test())

    def test_check_risk_once_take_profit(self):
        async def run_test():
            ib = AsyncMock()
            ib.isConnected = MagicMock(return_value=True)
            ib.managedAccounts = MagicMock(return_value=['DU12345'])
            ib.placeOrder = MagicMock()
            ib.qualifyContractsAsync = AsyncMock()
            ib.cancelPnLSingle = MagicMock()

            config = {
                'notifications': {},
                'risk_management': { 'stop_loss_pct': 0.20, 'take_profit_pct': 0.50 } # 50% TP
            }

            # Total entry cost = 56250
            # Take profit trigger = 0.50 * 56250 = 28125
            leg1_contract = FuturesOption(conId=123, symbol='KC', multiplier='37500', right='C', strike=3.5)
            leg1_pos = Position(account='TestAccount', contract=leg1_contract, position=1, avgCost=1.0)
            leg1_pnl = MagicMock(spec=PnLSingle, unrealizedPnL=20000.0)

            leg2_contract = FuturesOption(conId=124, symbol='KC', multiplier='37500', right='C', strike=3.6)
            leg2_pos = Position(account='TestAccount', contract=leg2_contract, position=-1, avgCost=0.5)
            leg2_pnl = MagicMock(spec=PnLSingle, unrealizedPnL=10000.0) # Total PnL = 30000, which is > 28125 profit

            ib.reqPositionsAsync.return_value = [leg1_pos, leg2_pos]
            ib.reqContractDetailsAsync.side_effect = [[MagicMock(underConId=999)], [MagicMock(underConId=999)]]
            ib.reqPnLSingle = MagicMock(side_effect=[leg1_pnl, leg2_pnl])

            closed_ids = set()
            stop_loss_pct = config['risk_management']['stop_loss_pct']
            take_profit_pct = config['risk_management']['take_profit_pct']

            with patch('trading_bot.risk_management.wait_for_fill', new_callable=AsyncMock):
                await _check_risk_once(ib, config, closed_ids, stop_loss_pct, take_profit_pct)

            self.assertEqual(ib.placeOrder.call_count, 2)

        asyncio.run(run_test())

    def test_check_risk_once_no_trigger(self):
        async def run_test():
            ib = AsyncMock()
            ib.isConnected = MagicMock(return_value=True)
            ib.managedAccounts = MagicMock(return_value=['DU12345'])
            ib.placeOrder = MagicMock()

            config = {
                'notifications': {},
                'risk_management': { 'stop_loss_pct': 0.20, 'take_profit_pct': 3.00 }
            }

            leg1_contract = FuturesOption(conId=123, symbol='KC', multiplier='37500', right='C', strike=3.5)
            leg1_pos = Position(account='TestAccount', contract=leg1_contract, position=1, avgCost=1.0)
            leg1_pnl = MagicMock(spec=PnLSingle, unrealizedPnL=1000.0) # Total PnL = 1500, no trigger

            leg2_contract = FuturesOption(conId=124, symbol='KC', multiplier='37500', right='C', strike=3.6)
            leg2_pos = Position(account='TestAccount', contract=leg2_contract, position=-1, avgCost=0.5)
            leg2_pnl = MagicMock(spec=PnLSingle, unrealizedPnL=500.0)

            ib.reqPositionsAsync.return_value = [leg1_pos, leg2_pos]
            ib.reqContractDetailsAsync.side_effect = [[MagicMock(underConId=999)], [MagicMock(underConId=999)]]
            ib.reqPnLSingle = MagicMock(side_effect=[leg1_pnl, leg2_pnl])

            closed_ids = set()
            stop_loss_pct = config['risk_management']['stop_loss_pct']
            take_profit_pct = config['risk_management']['take_profit_pct']

            await _check_risk_once(ib, config, closed_ids, stop_loss_pct, take_profit_pct)

            ib.placeOrder.assert_not_called()

        asyncio.run(run_test())

    def test_manage_orphaned_single_leg(self):
        async def run_test():
            ib = AsyncMock()
            config = {'symbol': 'KC'}
            signal = {'prediction_type': 'DIRECTIONAL', 'direction': 'BULLISH'}
            future_contract = Future(conId=100, symbol='KC', lastTradeDateOrContractMonth='202512')

            # --- Mock existing orphaned single leg position ---
            single_leg_contract = FuturesOption(conId=99, symbol='KC', right='P', strike=3.5)
            position = Position(account='TestAccount', contract=single_leg_contract, position=1, avgCost=0.5)

            mock_cd = MagicMock()
            mock_cd.underConId = 100

            ib.reqPositionsAsync.return_value = [position]
            ib.reqContractDetailsAsync.return_value = [mock_cd]
            ib.placeOrder = MagicMock()

            with patch('trading_bot.risk_management.wait_for_fill', new_callable=AsyncMock):
                should_trade = await manage_existing_positions(ib, config, signal, 100.0, future_contract)

            # It should decide to trade to close the orphaned leg
            self.assertTrue(should_trade)
            # It should place an order to close the position
            ib.placeOrder.assert_called_once()
            self.assertEqual(ib.placeOrder.call_args[0][1].action, 'SELL')

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()