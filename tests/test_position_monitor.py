import asyncio
import signal
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from position_monitor import main as position_monitor_main


class TestPositionMonitor(unittest.TestCase):

    @patch('position_monitor.load_config')
    @patch('trading_bot.connection_pool.IBConnectionPool')
    @patch('position_monitor.send_pushover_notification')
    @patch('position_monitor.monitor_positions_for_risk', new_callable=AsyncMock)
    def test_monitor_startup_and_shutdown(self, mock_monitor_positions, mock_send_notification, mock_pool, mock_load_config):
        async def run_test():
            # --- Mocks ---
            mock_load_config.return_value = {
                'connection': {'host': '127.0.0.1', 'port': 7497},
                'notifications': {}
            }
            ib_instance = MagicMock()
            ib_instance.isConnected = MagicMock(return_value=True)
            ib_instance.disconnect = MagicMock()
            mock_pool.get_connection = AsyncMock(return_value=ib_instance)
            mock_pool.release_connection = AsyncMock()

            # Define an async side_effect that waits indefinitely, simulating the real function
            async def dummy_wait(_ib, _config):
                await asyncio.Event().wait()

            mock_monitor_positions.side_effect = dummy_wait

            # --- Run Test ---
            main_task = asyncio.create_task(position_monitor_main())

            # Allow some time for the connection and monitoring to start
            await asyncio.sleep(0.1)

            # --- Assertions for startup ---
            mock_pool.get_connection.assert_awaited_once_with("monitor", mock_load_config.return_value)
            mock_monitor_positions.assert_awaited_once()

            # --- Simulate shutdown and assert ---
            main_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await main_task

            # Pool release should be called in finally block
            mock_pool.release_connection.assert_awaited_once_with("monitor")

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()