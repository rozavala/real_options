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
    def test_create_combo_order_object(self, mock_get_market_data, mock_price_bs):
        """
        Tests the new logic flow:
        1. Create leg contracts.
        2. Qualify all legs together.
        3. Price each qualified leg.
        4. Build the final combo with qualified conIds.
        """
        async def run_test():
            # 1. Setup Mocks
            ib = AsyncMock()

            # Mock for the qualification result
            q_leg1 = FuturesOption(conId=101, symbol='KC', lastTradeDateOrContractMonth='20251220', strike=3.5, right='C', exchange='NYBOT', multiplier="37500")
            q_leg2 = FuturesOption(conId=102, symbol='KC', lastTradeDateOrContractMonth='20251220', strike=3.6, right='C', exchange='NYBOT', multiplier="37500")
            ib.qualifyContractsAsync.return_value = [q_leg1, q_leg2]

            # Mocks for pricing functions
            mock_get_market_data.return_value = {'implied_volatility': 0.2, 'risk_free_rate': 0.05}
            mock_price_bs.side_effect = [{'price': 1.0}, {'price': 0.5}] # Price in cents/lb

            # 2. Setup Inputs
            config = {
                'symbol': 'KC',
                'multiplier': "37500",
                'strategy': {'quantity': 1},
                'strategy_tuning': {'slippage_usd_per_contract': 5}
            }
            strategy_def = {
                "action": "BUY",
                "legs_def": [('C', 'BUY', 3.5), ('C', 'SELL', 3.6)],
                "exp_details": {'exp_date': '20251220', 'days_to_exp': 30},
                "chain": {'exchange': 'NYBOT', 'tradingClass': 'KCO'},
                "underlying_price": 100.0,
            }

            # 3. Execute the function
            result = await create_combo_order_object(ib, config, strategy_def)

            # 4. Assertions
            self.assertIsNotNone(result)
            combo_contract, limit_order = result

            # Assert qualification was called once, with the unqualified contracts
            ib.qualifyContractsAsync.assert_called_once()
            unqualified_args = ib.qualifyContractsAsync.call_args[0]
            self.assertEqual(len(unqualified_args), 2)
            self.assertEqual(unqualified_args[0].strike, 3.5)
            self.assertEqual(unqualified_args[0].multiplier, "37500")
            self.assertEqual(unqualified_args[1].strike, 3.6)
            self.assertEqual(unqualified_args[1].multiplier, "37500")

            # Assert pricing was called with the QUALIFIED contracts
            mock_get_market_data.assert_any_call(ib, q_leg1)
            mock_get_market_data.assert_any_call(ib, q_leg2)

            # Assert combo legs have the correct, qualified conIds
            self.assertIsInstance(combo_contract, Bag)
            self.assertEqual(len(combo_contract.comboLegs), 2)
            self.assertEqual(combo_contract.comboLegs[0].conId, 101)
            self.assertEqual(combo_contract.comboLegs[1].conId, 102)

            # Assert order details are correct
            self.assertIsInstance(limit_order, LimitOrder)
            self.assertEqual(limit_order.action, 'BUY')
            # Check price: net theoretical (1.0 - 0.5 = 0.5) + slippage ($5 -> 1.33 cents/lb)
            # $5 * 100 cents/$ / 37500 lbs = 0.01333 cents/lb
            expected_price = (1.0 - 0.5) + ((5 * 100) / 37500)
            self.assertAlmostEqual(limit_order.lmtPrice, round(expected_price, 2))

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