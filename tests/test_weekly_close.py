import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, date, timedelta
import pandas as pd
from trading_bot.order_manager import close_stale_positions

# Mock config
MOCK_CONFIG = {
    'connection': {'host': '127.0.0.1', 'port': 7497},
    'risk_management': {'max_holding_days': 2},
    'notifications': {},
    'symbol': 'KC',
    'exchange': 'NYBOT'
}

@pytest.fixture
def mock_ib():
    with patch('trading_bot.order_manager.IB') as MockIB:
        ib_instance = MockIB.return_value
        ib_instance.connectAsync = AsyncMock()
        ib_instance.disconnect = MagicMock()
        ib_instance.reqPositionsAsync = AsyncMock()
        ib_instance.reqMktData = MagicMock()
        ib_instance.cancelMktData = MagicMock()
        ib_instance.placeOrder = MagicMock()
        ib_instance.sleep = AsyncMock() # Use asyncio.sleep in code, but IB object might have sleep mocked just in case

        # Mock ticker for price discovery
        mock_ticker = MagicMock()
        mock_ticker.bid = 100.0
        mock_ticker.ask = 101.0
        mock_ticker.last = 100.5
        mock_ticker.close = 100.0
        ib_instance.reqMktData.return_value = mock_ticker

        yield ib_instance

@pytest.fixture
def mock_ledger():
    with patch('trading_bot.order_manager.get_trade_ledger_df') as mock_get_ledger:
        yield mock_get_ledger

@pytest.fixture
def mock_notification():
    with patch('trading_bot.order_manager.send_pushover_notification') as mock_notify:
        yield mock_notify

@pytest.fixture
def mock_utils_log():
    with patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock) as mock_log:
        yield mock_log

@pytest.mark.asyncio
async def test_friday_weekly_close(mock_ib, mock_ledger, mock_notification, mock_utils_log):
    """Test that ALL positions are closed on Friday, regardless of age."""

    # Mock Date: Friday, Oct 27, 2023
    mock_friday = datetime(2023, 10, 27, 17, 20)

    # Live Position: Opened today (Age 0 days)
    mock_pos = MagicMock()
    mock_pos.contract.localSymbol = 'KCZ23'
    mock_pos.contract.conId = 12345
    mock_pos.position = 1
    mock_ib.reqPositionsAsync.return_value = [mock_pos]

    # Ledger: Shows opened today
    mock_ledger.return_value = pd.DataFrame({
        'timestamp': [mock_friday],
        'local_symbol': ['KCZ23'],
        'action': ['BUY'],
        'quantity': [1],
        'position_id': ['pos_1']
    })

    with patch('trading_bot.order_manager.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_friday
        # USFederalHolidayCalendar needs date() to work
        mock_datetime.date.return_value = mock_friday.date()

        await close_stale_positions(MOCK_CONFIG)

    # Assertions
    # Notification should mention "Weekly Market Close" (or we check the logic triggered)
    # Since we can't easily check internal vars, we check if an order was placed for this 0-day old position

    assert mock_ib.placeOrder.called
    args, _ = mock_ib.placeOrder.call_args
    contract, order = args
    assert contract.conId == 12345
    assert order.action == 'SELL' # Closing a BUY

    # Check notification title
    assert mock_notification.called
    args, _ = mock_notification.call_args
    title = args[1]
    # We expect the code change to update the title
    # For now, if the code ISN'T changed, this test might fail or pass depending on current logic (it should fail to close)
    # Wait, current logic won't close it (age 0 < 2). So verifying placeOrder.called is enough to fail the test.

@pytest.mark.asyncio
async def test_thursday_holiday_close(mock_ib, mock_ledger, mock_notification, mock_utils_log):
    """Test closure on Thursday if Friday is a holiday."""

    # Mock Date: Thursday, Nov 23, 2023 is Thanksgiving (Holiday)
    # So Wednesday, Nov 22, 2023 would be the day to close?
    # No, Thanksgiving is Thursday.
    # Let's pick a Friday holiday. Good Friday is not federal usually.
    # Christmas 2020: Friday Dec 25.

    mock_thursday = datetime(2020, 12, 24, 17, 20) # Thursday

    # Live Position: Opened today (Age 0)
    mock_pos = MagicMock()
    mock_pos.contract.localSymbol = 'KCZ20'
    mock_pos.contract.conId = 67890
    mock_pos.position = 1
    mock_ib.reqPositionsAsync.return_value = [mock_pos]

    # Ledger
    mock_ledger.return_value = pd.DataFrame({
        'timestamp': [mock_thursday],
        'local_symbol': ['KCZ20'],
        'action': ['BUY'],
        'quantity': [1],
        'position_id': ['pos_2']
    })

    with patch('trading_bot.order_manager.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_thursday
        # We need the class method 'now' and 'date' for the class 'datetime'
        # But mocking datetime.datetime is tricky because it's a type.
        # usually patch('module.datetime') works.

        await close_stale_positions(MOCK_CONFIG)

    assert mock_ib.placeOrder.called
    args, _ = mock_ib.placeOrder.call_args
    contract, _ = args
    assert contract.conId == 67890

@pytest.mark.asyncio
async def test_standard_thursday_no_close(mock_ib, mock_ledger, mock_notification, mock_utils_log):
    """Test that on a normal Thursday, young positions are NOT closed."""

    # Mock Date: Thursday, Oct 26, 2023 (Friday is NOT a holiday)
    mock_thursday = datetime(2023, 10, 26, 17, 20)

    # Live Position: Opened today (Age 0)
    mock_pos = MagicMock()
    mock_pos.contract.localSymbol = 'KCZ23'
    mock_pos.contract.conId = 11111
    mock_pos.position = 1
    mock_ib.reqPositionsAsync.return_value = [mock_pos]

    mock_ledger.return_value = pd.DataFrame({
        'timestamp': [mock_thursday],
        'local_symbol': ['KCZ23'],
        'action': ['BUY'],
        'quantity': [1],
        'position_id': ['pos_3']
    })

    with patch('trading_bot.order_manager.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_thursday

        await close_stale_positions(MOCK_CONFIG)

    # Should NOT close
    assert not mock_ib.placeOrder.called
