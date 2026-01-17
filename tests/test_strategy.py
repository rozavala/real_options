import unittest
from ib_insync import Future

from trading_bot.strategy import define_directional_strategy, define_volatility_strategy


class TestStrategy(unittest.TestCase):

    def test_define_bullish_strategy(self):
        """Tests that the correct legs are defined for a BULLISH signal."""
        # spread_width_points = 3.51 * 0.01425 ~= 0.05
        config = {'strategy': {}, 'strategy_tuning': {'spread_width_percentage': 0.01425}}
        signal = {'direction': 'BULLISH', 'prediction_type': 'DIRECTIONAL'}
        future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512')
        chain = {'expirations': ['20251120'], 'strikes_by_expiration': {'20251120': [3.4, 3.45, 3.5, 3.55, 3.6]}}

        strategy_def = define_directional_strategy(config, signal, chain, 3.51, future_contract)

        self.assertIsNotNone(strategy_def)
        # Bull Call Spread: BUY ATM call, SELL OTM call
        # ATM strike for 3.51 is 3.5
        # Target short strike is 3.5 + 0.05 = 3.55
        self.assertEqual(strategy_def['action'], 'BUY')
        self.assertEqual(len(strategy_def['legs_def']), 2)
        self.assertEqual(strategy_def['legs_def'][0], ('C', 'BUY', 3.5))
        self.assertEqual(strategy_def['legs_def'][1], ('C', 'SELL', 3.55))

    def test_define_bearish_strategy(self):
        """Tests that the correct legs are defined for a BEARISH signal."""
        # spread_width_points = 3.51 * 0.01425 ~= 0.05
        config = {'strategy': {}, 'strategy_tuning': {'spread_width_percentage': 0.01425}}
        signal = {'direction': 'BEARISH', 'prediction_type': 'DIRECTIONAL'}
        future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512')
        chain = {'expirations': ['20251120'], 'strikes_by_expiration': {'20251120': [3.4, 3.45, 3.5, 3.55, 3.6]}}

        strategy_def = define_directional_strategy(config, signal, chain, 3.51, future_contract)

        self.assertIsNotNone(strategy_def)
        # Bear Put Spread: BUY ATM put, SELL OTM put
        # ATM strike for 3.51 is 3.5
        # Target short strike is 3.5 - 0.05 = 3.45
        self.assertEqual(strategy_def['action'], 'BUY')
        self.assertEqual(len(strategy_def['legs_def']), 2)
        self.assertEqual(strategy_def['legs_def'][0], ('P', 'BUY', 3.5))
        self.assertEqual(strategy_def['legs_def'][1], ('P', 'SELL', 3.45))

    def test_define_high_vol_strategy(self):
        """Tests that the correct legs are defined for a HIGH volatility signal."""
        config = {'strategy': {}, 'strategy_tuning': {}}
        signal = {'level': 'HIGH', 'prediction_type': 'VOLATILITY'}
        future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512')
        chain = {'expirations': ['20251120'], 'strikes_by_expiration': {'20251120': [98, 99, 100, 101, 102]}}

        strategy_def = define_volatility_strategy(config, signal, chain, 100.0, future_contract)

        self.assertIsNotNone(strategy_def)
        # Long Straddle: BUY ATM call, BUY ATM put
        self.assertEqual(strategy_def['action'], 'BUY')
        self.assertEqual(len(strategy_def['legs_def']), 2)
        self.assertIn(('C', 'BUY', 100), strategy_def['legs_def'])
        self.assertIn(('P', 'BUY', 100), strategy_def['legs_def'])

    def test_define_low_vol_strategy_iron_condor(self):
        """Tests that the correct legs are defined for a LOW volatility signal (Iron Condor)."""
        config = {
            'strategy': {},
            'strategy_tuning': {
                'iron_condor_short_strikes_from_atm': 2,
                'iron_condor_wing_strikes_apart': 2
            }
        }
        signal = {'level': 'LOW', 'prediction_type': 'VOLATILITY'}
        future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512')
        # Strikes: 96, 97, 98, 99, 100(ATM), 101, 102, 103, 104
        chain = {
            'expirations': ['20251120'],
            'strikes_by_expiration': {'20251120': [96, 97, 98, 99, 100, 101, 102, 103, 104]}
        }

        strategy_def = define_volatility_strategy(config, signal, chain, 100.0, future_contract)

        self.assertIsNotNone(strategy_def)
        # Iron Condor: ATM=100, short_dist=2, wing_width=2
        # Short Put @ 98, Long Put @ 96
        # Short Call @ 102, Long Call @ 104
        self.assertEqual(strategy_def['action'], 'SELL')  # Credit spread
        self.assertEqual(len(strategy_def['legs_def']), 4)

        # Verify all 4 legs
        self.assertIn(('P', 'BUY', 96), strategy_def['legs_def'])   # Long Put (protection)
        self.assertIn(('P', 'SELL', 98), strategy_def['legs_def'])  # Short Put
        self.assertIn(('C', 'SELL', 102), strategy_def['legs_def']) # Short Call
        self.assertIn(('C', 'BUY', 104), strategy_def['legs_def'])  # Long Call (protection)


if __name__ == '__main__':
    unittest.main()