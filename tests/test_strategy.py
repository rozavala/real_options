import unittest
from ib_insync import Future, Ticker, Contract
from unittest.mock import Mock, patch

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.strategy import define_directional_strategy, define_volatility_strategy

def create_mock_option_chain(strikes, rights, underlying_price, vol=0.2):
    """Creates a mock option chain with tickers for testing."""
    chain = {}
    for strike in strikes:
        for right in rights:
            # Simulate some realistic bid/ask prices based on strike and underlying
            price = max(0.01, underlying_price - strike if right == 'C' else strike - underlying_price)
            ticker = Ticker(
                contract=Contract(strike=strike, right=right),
                bid=price - 0.02,
                ask=price + 0.02,
                last=price,
                impliedVolatility=vol
            )
            option_details = Mock()
            option_details.ticker = ticker
            chain[f"{strike}:{right}"] = option_details
    return chain

class TestStrategy(unittest.TestCase):

    @patch('trading_bot.strategy.price_option_black_scholes')
    def test_define_bullish_strategy_passes_profit_check(self, mock_bs):
        """Tests that a profitable bullish strategy is defined correctly."""
        # Mock B-S to return values that will pass the profit check
        mock_bs.side_effect = [
            {'price': 0.12}, # Price of the BUY leg
            {'price': 0.05}  # Price of the SELL leg
        ]

        config = {
            'strategy': {},
            'strategy_tuning': {
                'spread_width_points': 5.0,
                'minimum_profit_threshold': 0.01,
                'risk_free_rate': 0.02,
                'default_volatility': 0.20,
                'slippage_spread_percentage': 0.1
            }
        }
        signal = {'direction': 'BULLISH', 'prediction_type': 'DIRECTIONAL'}
        future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512')
        strikes = [100.0, 105.0, 110.0]
        chain = create_mock_option_chain(strikes, ['C'], 102.0)
        chain['expirations'] = ['20251120']
        chain['strikes_by_expiration'] = {'20251120': strikes}

        with patch('trading_bot.strategy.get_expiration_details') as mock_get_exp:
            mock_get_exp.return_value = {'days_to_exp': 30, 'strikes': strikes}
            strategy_def = define_directional_strategy(config, signal, chain, 102.0, future_contract)

        self.assertIsNotNone(strategy_def)
        self.assertEqual(strategy_def['legs_def'][0], ('C', 'BUY', 100.0))
        self.assertEqual(strategy_def['legs_def'][1], ('C', 'SELL', 105.0))

    @patch('trading_bot.strategy.price_option_black_scholes')
    def test_define_bearish_strategy_fails_profit_check(self, mock_bs):
        """Tests that an unprofitable bearish strategy is rejected."""
        # Mock B-S to return values that will fail the profit check
        mock_bs.side_effect = [
            {'price': 0.10}, # Price of the BUY leg
            {'price': 0.09}  # Price of the SELL leg (not enough credit)
        ]

        config = {
            'strategy': {},
            'strategy_tuning': {
                'spread_width_points': 5.0,
                'minimum_profit_threshold': 0.05,
                'risk_free_rate': 0.02,
                'default_volatility': 0.20,
                'slippage_spread_percentage': 0.5
            }
        }
        signal = {'direction': 'BEARISH', 'prediction_type': 'DIRECTIONAL'}
        future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512')
        strikes = [95.0, 100.0, 105.0]
        chain = create_mock_option_chain(strikes, ['P'], 100.0)
        chain['expirations'] = ['20251120']
        chain['strikes_by_expiration'] = {'20251120': strikes}

        with patch('trading_bot.strategy.get_expiration_details') as mock_get_exp:
            mock_get_exp.return_value = {'days_to_exp': 30, 'strikes': strikes}
            strategy_def = define_directional_strategy(config, signal, chain, 100.0, future_contract)

        self.assertIsNone(strategy_def)


if __name__ == '__main__':
    unittest.main()