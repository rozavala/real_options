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
    @patch('position_monitor.poll_order_queue_and_place', new_callable=AsyncMock)
    def test_monitor_startup_and_shutdown(self, mock_poll, mock_monitor_risk, mock_send_notification, mock_ib_class, mock_load_config):
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

            # Define an async side_effect that waits indefinitely
            async def dummy_wait(*args, **kwargs):
                await asyncio.Event().wait()

            mock_monitor_risk.side_effect = dummy_wait
            mock_poll.side_effect = dummy_wait

            # --- Run Test ---
            main_task = asyncio.create_task(position_monitor_main())
            await asyncio.sleep(0.1) # Allow time for setup

            # --- Assertions for startup ---
            ib_instance.connectAsync.assert_awaited_once_with(
                host='127.0.0.1', port=7497, clientId=10, timeout=30
            )
            mock_send_notification.assert_called_once_with(
                {}, "Central Monitor Online", "The central monitoring and execution service is now online."
            )
            mock_monitor_risk.assert_awaited_once()
            mock_poll.assert_awaited_once()

            # --- Simulate shutdown and assert ---
            main_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await main_task

            ib_instance.disconnect.assert_called_once()

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()