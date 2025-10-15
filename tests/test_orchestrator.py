import asyncio
import sys
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from orchestrator import start_monitoring, stop_monitoring

class TestOrchestrator(unittest.TestCase):

    @patch('orchestrator.send_pushover_notification')
    @patch('asyncio.create_subprocess_exec')
    def test_start_monitoring_waits_for_readiness(self, mock_create_subprocess, mock_send_notification):
        """
        Verify that start_monitoring correctly launches the subprocess, waits for the
        'MONITOR_READY' signal, and then completes.
        """
        async def run_test():
            config = {'notifications': {}}
            mock_process = AsyncMock()
            mock_process.pid = 1234
            mock_process.returncode = None

            # Simulate the stdout stream of the subprocess
            mock_stream_reader = asyncio.StreamReader()
            mock_stream_reader.feed_data(b'Some initial output\n')
            mock_stream_reader.feed_data(b'MONITOR_READY\n')

            mock_process.stdout = mock_stream_reader
            # Provide an empty stream for stderr
            mock_process.stderr = asyncio.StreamReader()

            mock_create_subprocess.return_value = mock_process

            # Act: Run the start_monitoring function
            with patch('orchestrator.monitor_process', None):
                await start_monitoring(config)

            # Assert: Check that the process was started
            mock_create_subprocess.assert_awaited_once_with(
                sys.executable, 'position_monitor.py',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Assert: Check that the notification was sent AFTER readiness
            mock_send_notification.assert_called_once_with(
                config['notifications'], "Orchestrator", "Position monitoring service is online and ready."
            )

        asyncio.run(run_test())

    @patch('asyncio.create_subprocess_exec')
    def test_start_monitoring_timeout(self, mock_create_subprocess):
        """
        Verify that start_monitoring raises a TimeoutError if the readiness
        signal is not received within the timeout period.
        """
        async def run_test():
            config = {'notifications': {}}
            mock_process = AsyncMock()
            mock_process.pid = 1234
            mock_process.returncode = None
            # Simulate a stream that never sends the ready signal
            mock_process.stdout = asyncio.StreamReader()
            mock_process.stderr = asyncio.StreamReader()
            mock_create_subprocess.return_value = mock_process

            # Act & Assert: Expect a TimeoutError
            with self.assertRaises(asyncio.TimeoutError):
                with patch('orchestrator.monitor_process', None):
                    # Patch timeout to be very short for the test
                    with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
                         await start_monitoring(config)

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