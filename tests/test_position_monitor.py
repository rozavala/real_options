import asyncio
import signal
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from position_monitor import main as position_monitor_main


class TestPositionMonitor(unittest.TestCase):

    @patch('position_monitor.load_config')
    @patch('position_monitor.IB')
    @patch('position_monitor.send_pushover_notification')
    @patch('position_monitor.monitor_positions_for_risk', new_callable=AsyncMock)
    @patch('random.randint', return_value=1)
    def test_monitor_startup_and_shutdown(self, mock_randint, mock_monitor_positions, mock_send_notification, mock_ib_class, mock_load_config):
        async def run_test():
            # --- Mocks ---
            mock_load_config.return_value = {
                'connection': {'host': '127.0.0.1', 'port': 7497, 'clientId': 10},
                'notifications': {}
            }
            ib_instance = AsyncMock()
            ib_instance.isConnected = MagicMock(return_value=True)
            ib_instance.disconnect = MagicMock()
            mock_ib_class.return_value = ib_instance

            # Define an async side_effect that waits indefinitely, simulating the real function
            async def dummy_wait(_ib, _config):
                await asyncio.Event().wait()

            mock_monitor_positions.side_effect = dummy_wait

            # --- Run Test ---
            # We run the main function but cancel it shortly after to simulate a shutdown
            main_task = asyncio.create_task(position_monitor_main())

            # Allow some time for the connection and monitoring to start
            await asyncio.sleep(0.1)

            # --- Assertions for startup ---
            # clientId is base (10) + randint (mocked to 1) = 11
            ib_instance.connectAsync.assert_awaited_once_with('127.0.0.1', 7497, clientId=11)
            mock_send_notification.assert_called_once_with(
                {}, "Position Monitor Started", "The position monitoring service has started and is now watching open positions."
            )
            mock_monitor_positions.assert_awaited_once()

            # --- Simulate shutdown and assert ---
            main_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await main_task

            ib_instance.disconnect.assert_called_once()

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()