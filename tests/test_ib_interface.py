import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from ib_insync import Contract, Future, FuturesOption, Bag, ComboLeg, Trade, Order, OrderStatus

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.ib_interface import (
    get_option_market_data,
    wait_for_fill,
    get_active_futures,
    build_option_chain,
    place_combo_order,
)


class TestIbInterface(unittest.TestCase):

    def test_get_active_futures(self):
        async def run_test():
            ib = MagicMock()

            # Mock the contract details response
            mock_cd1 = MagicMock()
            mock_cd1.contract = Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='202512')
            mock_cd2 = MagicMock()
            mock_cd2.contract = Future(conId=2, symbol='KC', lastTradeDateOrContractMonth='202603')

            ib.reqContractDetailsAsync = AsyncMock(return_value=[mock_cd1, mock_cd2])

            futures = await get_active_futures(ib, 'KC', 'NYBOT')

            self.assertEqual(len(futures), 2)
            self.assertEqual(futures[0].lastTradeDateOrContractMonth, '202512')

        asyncio.run(run_test())

    def test_build_option_chain(self):
        async def run_test():
            ib = MagicMock()
            future_contract = Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='202512', exchange='NYBOT')

            # Mock the option chain response with magnified strikes
            mock_chain = MagicMock()
            mock_chain.exchange = 'NYBOT'
            mock_chain.tradingClass = 'KCO'
            mock_chain.expirations = ['20251120', '20251220']
            mock_chain.strikes = [340.0, 350.0, 360.0] # Magnified strikes

            ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[mock_chain])

            chain = await build_option_chain(ib, future_contract)

            self.assertIsNotNone(chain)
            self.assertEqual(chain['exchange'], 'NYBOT')
            self.assertIn('20251120', chain['expirations'])
            # Assert that the strikes have been normalized
            self.assertEqual(chain['strikes_by_expiration']['20251120'], [3.4, 3.5, 3.6])

        asyncio.run(run_test())

    @patch('trading_bot.ib_interface.price_option_black_scholes')
    @patch('trading_bot.ib_interface.get_option_market_data')
    @patch('trading_bot.ib_interface.wait_for_fill', new_callable=AsyncMock)
    def test_place_combo_order(self, mock_wait_for_fill, mock_get_market_data, mock_price_bs):
        async def run_test():
            ib = MagicMock()
            ib.qualifyContractsAsync = AsyncMock()

            mock_trade = Trade(
                contract=Bag(symbol='KC'),
                order=Order(action='BUY', totalQuantity=1),
                orderStatus=OrderStatus(status=OrderStatus.Filled)
            )
            ib.placeOrder = MagicMock(return_value=mock_trade)

            mock_get_market_data.return_value = {'implied_volatility': 0.2, 'risk_free_rate': 0.05}
            mock_price_bs.return_value = {'price': 0.1}

            config = {'symbol': 'KC', 'strategy': {'quantity': 1}, 'strategy_tuning': {}}
            legs_def = [('C', 'BUY', 3.5), ('C', 'SELL', 3.6)]
            exp_details = {'exp_date': '20251220', 'days_to_exp': 30}
            chain = {'exchange': 'NYBOT', 'tradingClass': 'KCO'}

            trade = await place_combo_order(ib, config, 'BUY', legs_def, exp_details, chain, 100.0)

            self.assertIsNotNone(trade)
            ib.placeOrder.assert_called_once()
            mock_wait_for_fill.assert_called_once()
            self.assertEqual(mock_get_market_data.call_count, 2)
            self.assertEqual(mock_price_bs.call_count, 2)

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()