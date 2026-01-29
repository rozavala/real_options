import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from trading_bot.connection_pool import IBConnectionPool

@pytest.mark.asyncio
async def test_get_connection_disconnects_on_failure():
    """
    Test that IBConnectionPool.get_connection calls disconnect() on the IB instance
    when connectAsync raises an exception (e.g. TimeoutError).
    """
    # Arrange
    purpose = "test_cleanup"
    config = {"connection": {"host": "127.0.0.1", "port": 7497}}

    # Reset pool state to ensure clean slate
    IBConnectionPool._instances.clear()
    IBConnectionPool._locks.clear()
    IBConnectionPool._reconnect_backoff.clear()
    IBConnectionPool._consecutive_failures.clear()

    # Patch the IB class used in connection_pool
    with patch('trading_bot.connection_pool.IB') as MockIB, \
         patch('trading_bot.connection_pool.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

        # Setup the mock instance that will be returned by IB()
        mock_ib_instance = MockIB.return_value
        # Simulate connection timeout
        mock_ib_instance.connectAsync = AsyncMock(side_effect=TimeoutError("Connection timed out"))
        mock_ib_instance.disconnect = MagicMock()

        # Act
        # We expect the exception to propagate
        with pytest.raises(TimeoutError):
            await IBConnectionPool.get_connection(purpose, config)

        # Assert
        # This assertion verifies that disconnect() was called.
        # Without the fix, this should fail.
        mock_ib_instance.disconnect.assert_called()

        # Verify that it waited for cleanup (the 0.5s sleep)
        mock_sleep.assert_called_with(0.5)
