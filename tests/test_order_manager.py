import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from ib_insync import Bag, ComboLeg, Contract, Order, Trade

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.order_manager import generate_and_queue_orders, place_queued_orders, ORDER_QUEUE

class TestOrderManager(unittest.TestCase):

    @patch('trading_bot.order_manager.run_data_pull')
    @patch('trading_bot.order_manager.send_data_and_get_prediction')
    @patch('trading_bot.order_manager.IB')
    def test_generate_orders_uses_fallback_on_data_pull_failure(self, mock_ib, mock_send_data, mock_run_data_pull):
        """
        Verify that if run_data_pull fails, the process doesn't abort and
        instead proceeds to the next step (fetching predictions).
        """
        async def run_test():
            # Arrange: Simulate a data pull failure
            mock_run_data_pull.return_value = False

            # Arrange: Mock the subsequent functions to prevent them from running their full logic
            mock_send_data.return_value = {'price_changes': [1.0]} # Needs to return something to proceed
            mock_ib_instance = AsyncMock()
            mock_ib.return_value = mock_ib_instance

            config = {} # Dummy config

            # Act: Run the function
            await generate_and_queue_orders(config)

            # Assert: Check that the data pull was called
            mock_run_data_pull.assert_called_once()

            # Assert: Check that despite the failure, the process continued to the next step
            mock_send_data.assert_called_once()

            # Assert: Check that it tried to connect to IB, which is after the prediction step
            mock_ib_instance.connectAsync.assert_called_once()

        asyncio.run(run_test())

    @patch('trading_bot.order_manager.IB')
    def test_place_queued_orders_primes_cache(self, mock_ib):
        """
        Verify that place_queued_orders correctly qualifies leg contracts
        to prime the cache before placing any orders.
        """
        async def run_test():
            # Arrange: Mock the IB instance and clear the global queue
            mock_ib_instance = AsyncMock()
            mock_ib.return_value = mock_ib_instance
            ORDER_QUEUE.clear()

            # Arrange: Create a mock combo order and add it to the queue
            leg1 = ComboLeg(conId=101, ratio=1, action='BUY', exchange='NYBOT')
            leg2 = ComboLeg(conId=102, ratio=1, action='SELL', exchange='NYBOT')
            combo_contract = Bag(symbol='KC', comboLegs=[leg1, leg2])
            order = Order(action='BUY', totalQuantity=1, orderType='LMT', lmtPrice=5.0)
            ORDER_QUEUE.append((combo_contract, order))

            # Arrange: Configure reqMktData to return a mock Ticker object
            mock_ticker = MagicMock()
            mock_ticker.bid = 5.1
            mock_ticker.ask = 5.2
            mock_ticker.volume = 10
            mock_ticker.last = 5.15
            mock_ticker.time = MagicMock()
            # Configure the synchronous reqMktData method as a MagicMock
            mock_ib_instance.reqMktData = MagicMock(return_value=mock_ticker)

            # Arrange: Configure contract details for leg data calls
            mock_cd = MagicMock()
            mock_cd.contract.localSymbol = 'MOCK_LEG'
            mock_ib_instance.reqContractDetailsAsync.return_value = [mock_cd]

            # Arrange: Mock the placeOrder method to return a mock trade
            mock_trade = AsyncMock(spec=Trade)
            mock_ib_instance.placeOrder.return_value = mock_trade

            config = {'connection': {}} # Dummy config

            # Act: Run the function
            await place_queued_orders(config)

            # Assert: Verify that qualifyContractsAsync was called to prime the cache
            mock_ib_instance.qualifyContractsAsync.assert_called_once()

            # Assert: Check the contracts that were passed to be qualified
            qualified_contracts_call = mock_ib_instance.qualifyContractsAsync.call_args[0]
            self.assertEqual(len(qualified_contracts_call), 2)
            self.assertIn(Contract(conId=101), qualified_contracts_call)
            self.assertIn(Contract(conId=102), qualified_contracts_call)

            # Assert: Verify that the order was actually placed
            mock_ib_instance.placeOrder.assert_called_once_with(combo_contract, order)

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()