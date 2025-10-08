import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import pandas as pd
import pytz

from ib_insync import FuturesOption, Bag, ComboLeg, OrderStatus, Trade

from trading_bot.utils import (
    price_option_black_scholes,
    get_position_details,
    is_market_open,
    calculate_wait_until_market_open,
    get_expiration_details,
    log_trade_to_ledger,
)


class TestUtils(unittest.TestCase):

    def test_price_option_black_scholes(self):
        # Test with known values
        price = price_option_black_scholes(100, 100, 1, 0.05, 0.2, 'C')
        self.assertAlmostEqual(price['price'], 10.45, delta=0.01)

        # Test edge cases
        price = price_option_black_scholes(100, 100, 0, 0.05, 0.2, 'C')
        self.assertIsNone(price)

    def test_get_position_details(self):
        async def run_test():
            ib = Mock()

            # --- Test with a single leg option ---
            position_single = Mock()
            position_single.contract = FuturesOption(symbol='KC', lastTradeDateOrContractMonth='202512', strike=3.5, right='C', exchange='NYBOT')
            details_single = await get_position_details(ib, position_single)
            self.assertEqual(details_single['type'], 'SINGLE_LEG')
            self.assertEqual(details_single['key_strikes'], [3.5])

            # --- Test with a bag option ---
            position_bag = Mock()
            leg1 = ComboLeg(conId=1, ratio=1, action='BUY', exchange='NYBOT')
            leg2 = ComboLeg(conId=2, ratio=1, action='SELL', exchange='NYBOT')
            position_bag.contract = Bag(symbol='KC', comboLegs=[leg1, leg2])

            # Mock the contract details resolution for the legs
            mock_cd1 = Mock()
            mock_cd1.contract = FuturesOption(conId=1, right='C', strike=3.5)
            mock_cd2 = Mock()
            mock_cd2.contract = FuturesOption(conId=2, right='C', strike=3.6)

            ib.reqContractDetailsAsync = AsyncMock(side_effect=[[mock_cd1], [mock_cd2]])

            details_bag = await get_position_details(ib, position_bag)
            self.assertEqual(details_bag['type'], 'BULL_CALL_SPREAD')

        asyncio.run(run_test())

    def test_is_market_open(self):
        contract_details = Mock()
        test_date = datetime(2025, 10, 7)
        contract_details.liquidHours = f'{test_date.strftime("%Y%m%d")}:0800-{test_date.strftime("%Y%m%d")}:1600'
        tz = pytz.timezone('America/New_York')

        with patch('trading_bot.utils.datetime') as mock_datetime:
            # We need to mock the whole class, but restore the strptime method
            mock_datetime.strptime.side_effect = lambda *args, **kwargs: datetime.strptime(*args, **kwargs)

            # Simulate market is open
            mock_datetime.now.return_value = tz.localize(test_date.replace(hour=10, minute=0))
            self.assertTrue(is_market_open(contract_details, 'America/New_York'))

            # Simulate market is closed
            mock_datetime.now.return_value = tz.localize(test_date.replace(hour=18, minute=0))
            self.assertFalse(is_market_open(contract_details, 'America/New_York'))

    def test_calculate_wait_until_market_open(self):
        contract_details = Mock()
        contract_details.liquidHours = '20251008:0800-20251008:1600'
        with patch('trading_bot.utils.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2025, 10, 7, 18, 0)
            wait_time = calculate_wait_until_market_open(contract_details, 'America/New_York')
            self.assertGreater(wait_time, 0)

    def test_get_expiration_details(self):
        chain = {
            'expirations': ['20251120', '20251220'],
            'strikes_by_expiration': {
                '20251120': [3.4, 3.5, 3.6],
                '20251220': [3.4, 3.5, 3.6],
            }
        }
        details = get_expiration_details(chain, '202512')
        self.assertEqual(details['exp_date'], '20251220')

    @patch('builtins.open')
    def test_log_trade_to_ledger(self, mock_open):
        trade = Mock()
        trade.contract = Mock(spec=Bag) # Mock the contract as a Bag
        trade.order = Mock()
        trade.orderStatus = Mock()

        trade.contract.localSymbol = 'KCH6'
        trade.order.action = 'BUY'
        trade.orderStatus.status = OrderStatus.Filled
        trade.orderStatus.filled = 1
        trade.orderStatus.avgFillPrice = 100.0
        trade.order.orderId = 123

        # Ensure isinstance check works
        trade.contract.__class__ = Bag

        log_trade_to_ledger(trade)
        mock_open.assert_called_once()


if __name__ == '__main__':
    unittest.main()