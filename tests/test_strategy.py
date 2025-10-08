import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from ib_insync import Contract, Future

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.strategy import execute_directional_strategy, execute_volatility_strategy


class TestStrategy(unittest.TestCase):

    @patch('trading_bot.strategy.place_combo_order', new_callable=AsyncMock)
    def test_execute_bullish_strategy(self, mock_place_combo):
        async def run_test():
            ib, config = MagicMock(), {'strategy': {}, 'strategy_tuning': {'spread_width_usd': 0.05}}
            signal = {'direction': 'BULLISH'}
            future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512')

            # A simplified chain with enough strikes
            chain = {'expirations': ['20251120'], 'strikes_by_expiration': {'20251120': [3.4, 3.45, 3.5, 3.55, 3.6]}}

            await execute_directional_strategy(ib, config, signal, chain, 3.51, future_contract)

            # Verify that place_combo_order was called
            mock_place_combo.assert_called_once()

            # Extract arguments to check the legs
            args, kwargs = mock_place_combo.call_args
            legs_def = args[3] # legs_def is the 4th positional argument

            # Bull Call Spread: BUY ATM call, SELL OTM call
            # ATM strike for 3.51 is 3.5
            # Target short strike is 3.5 + 0.05 = 3.55
            self.assertEqual(len(legs_def), 2)
            self.assertEqual(legs_def[0], ('C', 'BUY', 3.5))
            self.assertEqual(legs_def[1], ('C', 'SELL', 3.55))

        asyncio.run(run_test())

    @patch('trading_bot.strategy.place_combo_order', new_callable=AsyncMock)
    def test_execute_bearish_strategy(self, mock_place_combo):
        async def run_test():
            ib, config = MagicMock(), {'strategy': {}, 'strategy_tuning': {'spread_width_usd': 0.05}}
            signal = {'direction': 'BEARISH'}
            future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512')

            chain = {'expirations': ['20251120'], 'strikes_by_expiration': {'20251120': [3.4, 3.45, 3.5, 3.55, 3.6]}}

            await execute_directional_strategy(ib, config, signal, chain, 3.51, future_contract)

            mock_place_combo.assert_called_once()
            args, kwargs = mock_place_combo.call_args
            legs_def = args[3]

            # Bear Put Spread: BUY ATM put, SELL OTM put
            # ATM strike for 3.51 is 3.5
            # Target short strike is 3.5 - 0.05 = 3.45
            self.assertEqual(len(legs_def), 2)
            self.assertEqual(legs_def[0], ('P', 'BUY', 3.5))
            self.assertEqual(legs_def[1], ('P', 'SELL', 3.45))

        asyncio.run(run_test())

    @patch('trading_bot.strategy.place_combo_order', new_callable=AsyncMock)
    def test_execute_high_vol_strategy(self, mock_place_combo):
        async def run_test():
            ib, config = MagicMock(), {'strategy': {}, 'strategy_tuning': {}}
            signal = {'level': 'HIGH'}
            future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512')

            chain = {'expirations': ['20251120'], 'strikes_by_expiration': {'20251120': [98, 99, 100, 101, 102]}}

            await execute_volatility_strategy(ib, config, signal, chain, 100.0, future_contract)

            mock_place_combo.assert_called_once()
            args, kwargs = mock_place_combo.call_args
            legs_def = args[3]
            action = args[2]

            # Long Straddle: BUY ATM call, BUY ATM put
            self.assertEqual(action, 'BUY')
            self.assertEqual(len(legs_def), 2)
            self.assertIn(('C', 'BUY', 100), legs_def)
            self.assertIn(('P', 'BUY', 100), legs_def)

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()