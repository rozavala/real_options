import asyncio
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import json
import os

from trading_bot.order_manager import generate_and_queue_orders

class TestOrderManager(unittest.TestCase):

    def setUp(self):
        # Ensure the test queue directory is clean before each test
        self.queue_dir = "order_queue"
        if os.path.exists(self.queue_dir):
            for f in os.listdir(self.queue_dir):
                os.remove(os.path.join(self.queue_dir, f))
        else:
            os.makedirs(self.queue_dir)

    def tearDown(self):
        # Clean up the queue directory after tests
        if os.path.exists(self.queue_dir):
            for f in os.listdir(self.queue_dir):
                os.remove(os.path.join(self.queue_dir, f))
            os.rmdir(self.queue_dir)

    @patch('trading_bot.order_manager.run_data_pull')
    @patch('trading_bot.order_manager.send_data_and_get_prediction')
    @patch('trading_bot.order_manager.generate_signals', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.IB')
    def test_generate_and_queue_signals_successfully(self, mock_ib, mock_generate_signals, mock_send_data, mock_run_data_pull):
        """
        Verify that when valid predictions are received, signals are generated
        and correctly queued as individual JSON files.
        """
        async def run_test():
            # Arrange: Simulate successful data pull and prediction fetching
            mock_run_data_pull.return_value = True
            mock_send_data.return_value = {'some_prediction_data': True}

            # Arrange: Mock the signal generator to return a list of signals
            mock_signals = [
                {'contract_month': '202512', 'prediction_type': 'DIRECTIONAL', 'direction': 'BEARISH'},
                {'contract_month': '202603', 'prediction_type': 'DIRECTIONAL', 'direction': 'BULLISH'}
            ]
            mock_generate_signals.return_value = mock_signals

            # Arrange: Mock the IB connection
            mock_ib_instance = AsyncMock()
            mock_ib.return_value = mock_ib_instance

            config = {'notifications': {}} # Dummy config

            # Act: Run the function
            await generate_and_queue_orders(config)

            # Assert: Check that the core functions were called
            mock_run_data_pull.assert_called_once()
            mock_send_data.assert_called_once()
            mock_ib_instance.connectAsync.assert_called_once()
            mock_generate_signals.assert_called_once()

            # Assert: Check that two signal files were created in the queue
            self.assertEqual(len(os.listdir(self.queue_dir)), 2)

            # Assert: Check the content of one of the files to ensure it's correct
            # Note: The filename includes a timestamp, so we find it by listing the dir
            filename = os.listdir(self.queue_dir)[0]
            filepath = os.path.join(self.queue_dir, filename)
            with open(filepath, 'r') as f:
                queued_signal = json.load(f)

            # The signal should match one of the ones we mocked
            self.assertIn(queued_signal, mock_signals)

        asyncio.run(run_test())

    @patch('trading_bot.order_manager.run_data_pull')
    @patch('trading_bot.order_manager.send_data_and_get_prediction')
    @patch('trading_bot.order_manager.IB')
    def test_generate_signals_uses_fallback_on_data_pull_failure(self, mock_ib, mock_send_data, mock_run_data_pull):
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
            # We need to mock isConnected and disconnect to avoid RuntimeWarnings with async mocks
            mock_ib_instance.isConnected.return_value = True
            mock_ib_instance.disconnect = MagicMock()
            mock_ib.return_value = mock_ib_instance


            config = {'notifications': {}} # Dummy config

            # Act: Run the function
            await generate_and_queue_orders(config)

            # Assert: Check that the data pull was called
            mock_run_data_pull.assert_called_once()

            # Assert: Check that despite the failure, the process continued to the next step
            mock_send_data.assert_called_once()

            # Assert: Check that it tried to connect to IB, which is after the prediction step
            mock_ib_instance.connectAsync.assert_called_once()

            # Assert that disconnect was called
            mock_ib_instance.disconnect.assert_called_once()


        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()