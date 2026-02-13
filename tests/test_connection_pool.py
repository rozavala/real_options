import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from trading_bot.connection_pool import (
    IBConnectionPool,
    DEV_CLIENT_ID_BASE,
    DEV_CLIENT_ID_JITTER,
    DEV_CLIENT_ID_DEFAULT,
    _is_remote_gateway,
)

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


def test_is_remote_gateway():
    """_is_remote_gateway returns True for non-localhost hosts."""
    assert _is_remote_gateway({"connection": {"host": "100.64.0.1"}}) is True
    assert _is_remote_gateway({"connection": {"host": "10.0.0.5"}}) is True
    assert _is_remote_gateway({"connection": {"host": "127.0.0.1"}}) is False
    assert _is_remote_gateway({"connection": {"host": "localhost"}}) is False
    assert _is_remote_gateway({"connection": {"host": "::1"}}) is False
    assert _is_remote_gateway({}) is False  # defaults to 127.0.0.1


@pytest.mark.asyncio
async def test_dev_client_id_range_for_remote_host():
    """When config host is a Tailscale IP, client ID should be in 10-79 range."""
    purpose = "test_dev_range"
    config = {"connection": {"host": "100.64.0.1", "port": 4001}}

    IBConnectionPool._instances.clear()
    IBConnectionPool._locks.clear()
    IBConnectionPool._reconnect_backoff.clear()
    IBConnectionPool._consecutive_failures.clear()

    with patch('trading_bot.connection_pool.IB') as MockIB, \
         patch('trading_bot.connection_pool.asyncio.sleep', new_callable=AsyncMock):

        mock_ib = MockIB.return_value
        mock_ib.connectAsync = AsyncMock()
        mock_ib.isConnected.return_value = False
        mock_ib.disconnect = MagicMock()

        await IBConnectionPool.get_connection(purpose, config)

        # Verify connectAsync was called with a dev-range client ID (75-79 for unknown purpose)
        call_kwargs = mock_ib.connectAsync.call_args
        client_id = call_kwargs.kwargs.get('clientId') or call_kwargs[1].get('clientId')
        assert DEV_CLIENT_ID_DEFAULT <= client_id <= DEV_CLIENT_ID_DEFAULT + DEV_CLIENT_ID_JITTER, \
            f"Unknown purpose should use dev default range, got {client_id}"


@pytest.mark.asyncio
async def test_dev_client_id_known_purpose():
    """Known purposes on remote host should use their specific dev ID base."""
    purpose = "emergency"
    config = {"connection": {"host": "100.64.0.1", "port": 4001}}

    IBConnectionPool._instances.clear()
    IBConnectionPool._locks.clear()
    IBConnectionPool._reconnect_backoff.clear()
    IBConnectionPool._consecutive_failures.clear()

    with patch('trading_bot.connection_pool.IB') as MockIB, \
         patch('trading_bot.connection_pool.asyncio.sleep', new_callable=AsyncMock):

        mock_ib = MockIB.return_value
        mock_ib.connectAsync = AsyncMock()
        mock_ib.isConnected.return_value = False
        mock_ib.disconnect = MagicMock()

        await IBConnectionPool.get_connection(purpose, config)

        call_kwargs = mock_ib.connectAsync.call_args
        client_id = call_kwargs.kwargs.get('clientId') or call_kwargs[1].get('clientId')
        expected_base = DEV_CLIENT_ID_BASE["emergency"]  # 20
        assert expected_base <= client_id <= expected_base + DEV_CLIENT_ID_JITTER, \
            f"Emergency purpose should use dev base {expected_base}, got {client_id}"


@pytest.mark.asyncio
async def test_remote_paper_tag_in_log(caplog):
    """When config is remote + paper, log should show [REMOTE/PAPER]."""
    purpose = "test_paper_tag"
    config = {"connection": {"host": "100.64.0.1", "port": 4002, "paper": True}}

    IBConnectionPool._instances.clear()
    IBConnectionPool._locks.clear()
    IBConnectionPool._reconnect_backoff.clear()
    IBConnectionPool._consecutive_failures.clear()

    with patch('trading_bot.connection_pool.IB') as MockIB, \
         patch('trading_bot.connection_pool.asyncio.sleep', new_callable=AsyncMock):

        mock_ib = MockIB.return_value
        mock_ib.connectAsync = AsyncMock()
        mock_ib.isConnected.return_value = False
        mock_ib.disconnect = MagicMock()

        import logging
        with caplog.at_level(logging.INFO, logger="trading_bot.connection_pool"):
            await IBConnectionPool.get_connection(purpose, config)

        assert any("[REMOTE/PAPER]" in record.message for record in caplog.records), \
            "Expected [REMOTE/PAPER] tag in log output"


@pytest.mark.asyncio
async def test_prod_client_id_range_for_localhost():
    """When config host is 127.0.0.1, client ID should be in production range (100-279)."""
    purpose = "test_prod_range"
    config = {"connection": {"host": "127.0.0.1", "port": 7497}}

    IBConnectionPool._instances.clear()
    IBConnectionPool._locks.clear()
    IBConnectionPool._reconnect_backoff.clear()
    IBConnectionPool._consecutive_failures.clear()

    with patch('trading_bot.connection_pool.IB') as MockIB, \
         patch('trading_bot.connection_pool.asyncio.sleep', new_callable=AsyncMock):

        mock_ib = MockIB.return_value
        mock_ib.connectAsync = AsyncMock()
        mock_ib.isConnected.return_value = False
        mock_ib.disconnect = MagicMock()

        await IBConnectionPool.get_connection(purpose, config)

        call_kwargs = mock_ib.connectAsync.call_args
        client_id = call_kwargs.kwargs.get('clientId') or call_kwargs[1].get('clientId')
        # Default prod base is 280 + random(0,9)
        assert 280 <= client_id <= 289, \
            f"Unknown purpose on localhost should use prod default range, got {client_id}"


def test_no_prod_client_id_range_overlaps():
    """Verify no overlapping ranges in CLIENT_ID_BASE (prod)."""
    jitter = 9  # prod uses random(0, 9)
    entries = list(IBConnectionPool.CLIENT_ID_BASE.items())
    for i, (name_a, base_a) in enumerate(entries):
        range_a = range(base_a, base_a + jitter + 1)
        for name_b, base_b in entries[i + 1:]:
            range_b = range(base_b, base_b + jitter + 1)
            overlap = set(range_a) & set(range_b)
            assert not overlap, (
                f"PROD overlap: {name_a}({base_a}-{base_a+jitter}) "
                f"and {name_b}({base_b}-{base_b+jitter}) share IDs {overlap}"
            )


def test_no_dev_client_id_range_overlaps():
    """Verify no overlapping ranges in DEV_CLIENT_ID_BASE."""
    entries = list(DEV_CLIENT_ID_BASE.items())
    for i, (name_a, base_a) in enumerate(entries):
        range_a = range(base_a, base_a + DEV_CLIENT_ID_JITTER + 1)
        for name_b, base_b in entries[i + 1:]:
            range_b = range(base_b, base_b + DEV_CLIENT_ID_JITTER + 1)
            overlap = set(range_a) & set(range_b)
            assert not overlap, (
                f"DEV overlap: {name_a}({base_a}-{base_a+DEV_CLIENT_ID_JITTER}) "
                f"and {name_b}({base_b}-{base_b+DEV_CLIENT_ID_JITTER}) share IDs {overlap}"
            )


def test_drawdown_check_has_explicit_entry():
    """drawdown_check must have explicit entries in both ID maps."""
    assert "drawdown_check" in IBConnectionPool.CLIENT_ID_BASE, \
        "drawdown_check missing from CLIENT_ID_BASE"
    assert "drawdown_check" in DEV_CLIENT_ID_BASE, \
        "drawdown_check missing from DEV_CLIENT_ID_BASE"
