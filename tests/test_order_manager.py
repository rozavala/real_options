import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.order_manager import generate_and_queue_orders

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

if __name__ == '__main__':
    unittest.main()