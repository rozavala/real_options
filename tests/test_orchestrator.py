import asyncio
import sys
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from orchestrator import start_monitoring, stop_monitoring


class TestOrchestrator(unittest.TestCase):

    @patch('orchestrator.send_pushover_notification')
    @patch('asyncio.create_subprocess_exec')
    def test_start_monitoring(self, mock_create_subprocess, mock_send_notification):
        async def run_test():
            config = {'notifications': {}}
            mock_process = AsyncMock()
            mock_process.pid = 1234
            mock_process.returncode = None
            mock_create_subprocess.return_value = mock_process

            # Test starting the process
            with patch('orchestrator.monitor_process', None):
                await start_monitoring(config)
                mock_create_subprocess.assert_awaited_once_with(
                    sys.executable, 'position_monitor.py',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                mock_send_notification.assert_called_once_with(config['notifications'], "Orchestrator", "Started position monitoring service.")

            # Test that it doesn't start if already running
            mock_create_subprocess.reset_mock()
            with patch('orchestrator.monitor_process', mock_process):
                await start_monitoring(config)
                mock_create_subprocess.assert_not_awaited()

        asyncio.run(run_test())

    @patch('orchestrator.send_pushover_notification')
    def test_stop_monitoring(self, mock_send_notification):
        async def run_test():
            config = {'notifications': {}}
            mock_process = AsyncMock()
            mock_process.returncode = None  # Simulates a running process
            mock_process.terminate = MagicMock()

            # Test stopping a running process
            with patch('orchestrator.monitor_process', mock_process):
                await stop_monitoring(config)
                mock_process.terminate.assert_called_once()
                mock_process.wait.assert_awaited_once()
                mock_send_notification.assert_called_once_with(config['notifications'], "Orchestrator", "Stopped position monitoring service.")

            # Test that it doesn't try to stop a non-running process
            mock_process.reset_mock()
            with patch('orchestrator.monitor_process', None):
                await stop_monitoring(config)
                mock_process.terminate.assert_not_called()

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()