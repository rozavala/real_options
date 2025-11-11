import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from ib_insync import Contract, Future, FuturesOption, Bag, ComboLeg, Trade, Order, OrderStatus, LimitOrder

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.ib_interface import (
    get_active_futures,
    build_option_chain,
    create_combo_order_object,
    place_order,
)


class TestIbInterface(unittest.TestCase):

    def test_get_active_futures(self):
        async def run_test():
            ib = MagicMock()
            mock_cd1 = MagicMock(contract=Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='202512'))
            mock_cd2 = MagicMock(contract=Future(conId=2, symbol='KC', lastTradeDateOrContractMonth='202603'))
            ib.reqContractDetailsAsync = AsyncMock(return_value=[mock_cd1, mock_cd2])
            futures = await get_active_futures(ib, 'KC', 'NYBOT')
            self.assertEqual(len(futures), 2)
            self.assertEqual(futures[0].lastTradeDateOrContractMonth, '202512')

        asyncio.run(run_test())

    def test_build_option_chain(self):
        async def run_test():
            ib = MagicMock()
            future_contract = Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='202512', exchange='NYBOT')
            mock_chain = MagicMock(exchange='NYBOT', tradingClass='KCO', expirations=['20251120'], strikes=[340.0, 350.0])
            ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[mock_chain])

            chain = await build_option_chain(ib, future_contract)

            self.assertIsNotNone(chain)
            self.assertEqual(chain['exchange'], 'NYBOT')
            # Assert that the strikes are NOT normalized
            self.assertEqual(chain['strikes_by_expiration']['20251120'], [340.0, 350.0])

        asyncio.run(run_test())

    @patch('trading_bot.ib_interface.price_option_black_scholes')
    @patch('trading_bot.ib_interface.get_option_market_data')
    def test_create_combo_order_object_market_order(self, mock_get_market_data, mock_price_bs):
        """
        Tests that a MarketOrder is created when the config is set to 'MKT'.
        """
        async def run_test():
            # 1. Setup Mocks
            ib = AsyncMock()
            q_leg1 = FuturesOption(conId=101, symbol='KC', lastTradeDateOrContractMonth='20251220', strike=3.5, right='C', exchange='NYBOT')
            q_leg2 = FuturesOption(conId=102, symbol='KC', lastTradeDateOrContractMonth='20251220', strike=3.6, right='C', exchange='NYBOT')
            ib.qualifyContractsAsync.return_value = [q_leg1, q_leg2]
            mock_get_market_data.side_effect = [
                {'bid': 0.9, 'ask': 1.1, 'implied_volatility': 0.2, 'risk_free_rate': 0.05},
                {'bid': 0.4, 'ask': 0.6, 'implied_volatility': 0.18, 'risk_free_rate': 0.05}
            ]
            mock_price_bs.side_effect = [{'price': 1.0}, {'price': 0.5}]

            # 2. Setup Inputs - with order_type set to MKT
            config = {
                'symbol': 'KC',
                'strategy': {'quantity': 1},
                'strategy_tuning': {
                    'order_type': 'MKT',
                    'max_liquidity_spread_percentage': 0.9, # 90%
                }
            }
            strategy_def = {
                "action": "BUY", "legs_def": [('C', 'BUY', 3.5), ('C', 'SELL', 3.6)],
                "exp_details": {'exp_date': '20251220', 'days_to_exp': 30},
                "chain": {'exchange': 'NYBOT', 'tradingClass': 'KCO'},
                "underlying_price": 100.0,
                "future_contract": Future(conId=1, symbol='KC')
            }

            # 3. Execute
            result = await create_combo_order_object(ib, config, strategy_def)

            # 4. Assertions
            self.assertIsNotNone(result)
            _, market_order = result
            self.assertIsInstance(market_order, Order)
            self.assertEqual(market_order.orderType, "MKT")
            self.assertEqual(market_order.action, "BUY")
            self.assertEqual(market_order.totalQuantity, 1)

        asyncio.run(run_test())

    @patch('trading_bot.ib_interface.price_option_black_scholes')
    @patch('trading_bot.ib_interface.get_option_market_data')
    def test_create_combo_order_object(self, mock_get_market_data, mock_price_bs):
        """
        Tests that the combo order price is calculated correctly by combining
        the theoretical price with a dynamic, spread-based slippage.
        """
        async def run_test():
            # 1. Setup Mocks
            ib = AsyncMock()

            # Mock for qualification
            q_leg1 = FuturesOption(conId=101, symbol='KC', lastTradeDateOrContractMonth='20251220', strike=3.5, right='C', exchange='NYBOT')
            q_leg2 = FuturesOption(conId=102, symbol='KC', lastTradeDateOrContractMonth='20251220', strike=3.6, right='C', exchange='NYBOT')
            ib.qualifyContractsAsync.return_value = [q_leg1, q_leg2]

            # Mock for market data (bid, ask, IV)
            mock_get_market_data.side_effect = [
                {'bid': 0.9, 'ask': 1.1, 'implied_volatility': 0.2, 'risk_free_rate': 0.05},
                {'bid': 0.4, 'ask': 0.6, 'implied_volatility': 0.18, 'risk_free_rate': 0.05}
            ]

            # Mock for theoretical pricing
            mock_price_bs.side_effect = [{'price': 1.0}, {'price': 0.5}]

            # 2. Setup Inputs
            config = {
                'symbol': 'KC',
                'strategy': {'quantity': 1},
                'strategy_tuning': {
                    'max_liquidity_spread_percentage': 0.9, # 90%
                    'fixed_slippage_cents': 0.2
                }
            }
            strategy_def = {
                "action": "BUY", "legs_def": [('C', 'BUY', 3.5), ('C', 'SELL', 3.6)],
                "exp_details": {'exp_date': '20251220', 'days_to_exp': 30},
                "chain": {'exchange': 'NYBOT', 'tradingClass': 'KCO'},
                "underlying_price": 100.0,
                "future_contract": Future(conId=1, symbol='KC')
            }

            # 3. Execute
            result = await create_combo_order_object(ib, config, strategy_def)

            # 4. Assertions
            self.assertIsNotNone(result)
            _, limit_order = result

            # Theoretical Price: 1.0 (buy) - 0.5 (sell) = 0.5
            # Fixed Slippage: 0.2
            # Limit Price (BUY) = Theoretical Price + Fixed Slippage = 0.5 + 0.2 = 0.7
            expected_price = 0.70
            self.assertAlmostEqual(limit_order.lmtPrice, expected_price, places=2)

        asyncio.run(run_test())

    @patch('trading_bot.ib_interface.price_option_black_scholes')
    @patch('trading_bot.ib_interface.get_option_market_data')
    def test_create_combo_order_object_handles_qualification_failure(self, mock_get_market_data, mock_price_bs):
        """
        Tests that if a leg fails to qualify (conId=0), the function aborts
        and returns None.
        """
        async def run_test():
            # 1. Setup Mocks
            ib = AsyncMock()

            # Mock for the qualification result - one leg is invalid
            q_leg1 = FuturesOption(conId=101, symbol='KC', strike=3.5)
            q_leg2 = FuturesOption(conId=0, symbol='KC', strike=3.6) # Invalid leg
            ib.qualifyContractsAsync.return_value = [q_leg1, q_leg2]

            # 2. Setup Inputs
            config = {'symbol': 'KC', 'strategy': {'quantity': 1}}
            strategy_def = {
                "action": "BUY", "legs_def": [('C', 'BUY', 3.5), ('C', 'SELL', 3.6)],
                "exp_details": {'exp_date': '20251220', 'days_to_exp': 30},
                "chain": {'exchange': 'NYBOT', 'tradingClass': 'KCO'}, "underlying_price": 100.0,
            }

            # 3. Execute the function
            result = await create_combo_order_object(ib, config, strategy_def)

            # 4. Assertions
            self.assertIsNone(result)

            # Assert that pricing functions were NOT called because validation failed first
            mock_get_market_data.assert_not_called()
            mock_price_bs.assert_not_called()

        asyncio.run(run_test())

    def test_place_order(self):
        """Tests that place_order calls ib.placeOrder with the correct arguments."""
        ib = MagicMock()
        mock_contract = Bag(symbol='KC')
        mock_order = LimitOrder('BUY', 1, 1.23)

        place_order(ib, mock_contract, mock_order)
        ib.placeOrder.assert_called_once_with(mock_contract, mock_order)


if __name__ == '__main__':
    unittest.main()