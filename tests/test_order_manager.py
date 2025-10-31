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
    @patch('trading_bot.order_manager.get_model_predictions')
    @patch('trading_bot.order_manager.IB')
    def test_generate_orders_aborts_on_data_pull_failure(self, mock_ib, mock_get_predictions, mock_run_data_pull):
        """
        Verify that if run_data_pull fails (returns None), the process aborts
        and does not proceed to model inference.
        """
        async def run_test():
            # Arrange: Simulate a data pull failure
            mock_run_data_pull.return_value = None

            # Arrange: Mock the subsequent functions to prevent them from running their full logic
            mock_get_predictions.return_value = {'price_changes': [1.0]} # Needs to return something to proceed
            mock_ib_instance = AsyncMock()
            mock_ib.return_value = mock_ib_instance

            config = {} # Dummy config

            # Act: Run the function
            await generate_and_queue_orders(config)

            # Assert: Check that the data pull was called
            mock_run_data_pull.assert_called_once()

            # Assert: Check that because of the failure, the process did NOT continue
            mock_get_predictions.assert_not_called()
            mock_ib_instance.connectAsync.assert_not_called()

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()