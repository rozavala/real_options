import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, mock_open
from datetime import datetime
import pandas as pd
import pytz
import os

from ib_insync import FuturesOption, Bag, ComboLeg, OrderStatus, Trade, Order, Fill, Execution

from trading_bot.utils import (
    normalize_strike,
    price_option_black_scholes,
    get_position_details,
    is_market_open,
    calculate_wait_until_market_open,
    get_expiration_details,
    log_trade_to_ledger,
)


class TestUtils(unittest.TestCase):

    def test_normalize_strike(self):
        self.assertEqual(normalize_strike(350.0), 3.5)
        self.assertEqual(normalize_strike(3.5), 3.5)
        self.assertEqual(normalize_strike(99), 99)

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
            # Mock a magnified strike price
            position_single.contract = FuturesOption(symbol='KC', lastTradeDateOrContractMonth='202512', strike=350.0, right='C', exchange='NYBOT')
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
            mock_cd1.contract = FuturesOption(conId=1, right='C', strike=350.0) # Magnified
            mock_cd2 = Mock()
            mock_cd2.contract = FuturesOption(conId=2, right='C', strike=360.0) # Magnified

            ib.reqContractDetailsAsync = AsyncMock(side_effect=[[mock_cd1], [mock_cd2]])

            details_bag = await get_position_details(ib, position_bag)
            self.assertEqual(details_bag['type'], 'BULL_CALL_SPREAD')
            self.assertEqual(details_bag['key_strikes'], [3.5, 3.6])

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

    @patch('csv.DictWriter')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.isfile', return_value=True)
    def test_log_trade_to_ledger(self, mock_isfile, mock_open, mock_csv_writer):
        # --- Setup Mocks ---
        trade = Mock(spec=Trade)
        trade.orderStatus = Mock(spec=OrderStatus)
        trade.order = Mock(spec=Order)
        trade.orderStatus.status = OrderStatus.Filled
        trade.order.permId = 123456  # This will be the combo_id

        # Mock Fill 1 (Leg 1)
        fill1 = Mock(spec=Fill)
        fill1.contract = FuturesOption(symbol='KC', localSymbol='KCH6 C3.5', strike=3.5, right='C', multiplier='37500')
        fill1.execution = Execution(execId='E1', time=datetime.now(), side='BOT', shares=1, price=0.5)

        # Mock Fill 2 (Leg 2)
        fill2 = Mock(spec=Fill)
        fill2.contract = FuturesOption(symbol='KC', localSymbol='KCH6 C3.6', strike=3.6, right='C', multiplier='37500')
        fill2.execution = Execution(execId='E2', time=datetime.now(), side='SLD', shares=1, price=0.2)

        trade.fills = [fill1, fill2]

        mock_writer_instance = Mock()
        mock_csv_writer.return_value = mock_writer_instance

        # --- Call the function ---
        log_trade_to_ledger(trade, "Test Combo")

        # --- Assertions ---
        # Check that the file was opened correctly
        mock_open.assert_called_once_with('trade_ledger.csv', 'a', newline='')

        # Check that DictWriter was called with correct fieldnames
        expected_fieldnames = [
            'timestamp', 'combo_id', 'local_symbol', 'action', 'quantity',
            'avg_fill_price', 'strike', 'right', 'total_value_usd', 'reason'
        ]
        mock_csv_writer.assert_called_once_with(mock_open.return_value, fieldnames=expected_fieldnames)

        # Check that writerows was called once
        mock_writer_instance.writerows.assert_called_once()

        # Check the content of the written rows
        written_rows = mock_writer_instance.writerows.call_args[0][0]
        self.assertEqual(len(written_rows), 2)

        self.assertEqual(written_rows[0]['combo_id'], 123456)
        self.assertEqual(written_rows[0]['action'], 'BUY')
        self.assertEqual(written_rows[0]['strike'], 3.5)

        self.assertEqual(written_rows[1]['combo_id'], 123456)
        self.assertEqual(written_rows[1]['action'], 'SELL')
        self.assertEqual(written_rows[1]['strike'], 3.6)


if __name__ == '__main__':
    unittest.main()