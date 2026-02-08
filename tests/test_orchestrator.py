import asyncio
import sys
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from orchestrator import start_monitoring, stop_monitoring


class TestOrchestrator(unittest.TestCase):

    @patch('asyncio.create_subprocess_exec')
    @patch('orchestrator.is_market_open', return_value=True)
    def test_start_monitoring(self, mock_is_market_open, mock_create_subprocess):
        async def run_test():
            config = {'notifications': {}}
            mock_process = AsyncMock()
            mock_process.pid = 1234
            mock_process.returncode = None

            # Mock stdout/stderr with proper async readline that returns empty bytes
            mock_stdout = AsyncMock()
            mock_stdout.readline = AsyncMock(return_value=b'')
            mock_stderr = AsyncMock()
            mock_stderr.readline = AsyncMock(return_value=b'')
            mock_process.stdout = mock_stdout
            mock_process.stderr = mock_stderr

            mock_create_subprocess.return_value = mock_process

            # Test starting the process
            with patch('orchestrator.monitor_process', None):
                await start_monitoring(config)
                mock_create_subprocess.assert_awaited_once_with(
                    sys.executable, 'position_monitor.py',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

            # Test that it doesn't start if already running
            mock_create_subprocess.reset_mock()
            with patch('orchestrator.monitor_process', mock_process):
                await start_monitoring(config)
                mock_create_subprocess.assert_not_awaited()

        asyncio.run(run_test())

    def test_stop_monitoring(self):
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

            # Test that it doesn't try to stop a non-running process
            mock_process.reset_mock()
            with patch('orchestrator.monitor_process', None):
                await stop_monitoring(config)
                mock_process.terminate.assert_not_called()

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()