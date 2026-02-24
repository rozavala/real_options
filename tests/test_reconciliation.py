
import pytest
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta, timezone, date
import os
from trading_bot.reconciliation import reconcile_council_history
from ib_insync import Contract

# Use deterministic dates (Monday 2026-01-05 14:00 UTC)
ENTRY_TIME = datetime(2026, 1, 5, 14, 0, 0, tzinfo=timezone.utc)
# Target exit: Tuesday 2026-01-06 18:25 UTC (1:25 PM ET)
TARGET_EXIT = datetime(2026, 1, 6, 18, 25, 0, tzinfo=timezone.utc)
# "Now" is well past exit time
NOW = ENTRY_TIME + timedelta(hours=72)


@pytest.fixture
def reconciliation_patches():
    """Common patches needed for reconciliation tests."""
    patchers = []

    for target in [
        'trading_bot.reconciliation.TransactiveMemory',
        'trading_bot.reconciliation.get_router',
        'trading_bot.reconciliation.TradeJournal',
        'trading_bot.reconciliation.store_reflexion_lesson',
    ]:
        p = patch(target)
        p.start()
        patchers.append(p)

    # Mock _calculate_actual_exit_time to return deterministic date
    p = patch('trading_bot.reconciliation._calculate_actual_exit_time', return_value=TARGET_EXIT)
    p.start()
    patchers.append(p)

    # Mock datetime.now to return deterministic NOW
    p = patch('trading_bot.reconciliation.datetime')
    mock_dt = p.start()
    mock_dt.now.return_value = NOW
    mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
    mock_dt.combine = datetime.combine
    mock_dt.min = datetime.min
    mock_dt.fromisoformat = datetime.fromisoformat
    patchers.append(p)

    yield

    for p in patchers:
        p.stop()


@pytest.mark.asyncio
async def test_reconcile_council_history_success(reconciliation_patches):
    """Test successful reconciliation of a past trade."""

    mock_csv_data = pd.DataFrame({
        'timestamp': [ENTRY_TIME.isoformat()],
        'contract': ['KCH4 (202403)'],
        'entry_price': [150.0],
        'master_decision': ['BULLISH'],
        'exit_price': [None],
        'exit_timestamp': [None],
        'pnl_realized': [None],
        'actual_trend_direction': [None]
    })

    mock_ib = MagicMock()
    mock_ib.reqContractDetailsAsync = AsyncMock()
    mock_details = MagicMock()
    mock_details.contract = Contract(symbol='KC', lastTradeDateOrContractMonth='202403')
    mock_ib.reqContractDetailsAsync.return_value = [mock_details]

    mock_ib.reqHistoricalDataAsync = AsyncMock()
    mock_bar = MagicMock()
    mock_bar.date = TARGET_EXIT.date()
    mock_bar.close = 155.0
    mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

    config = {'symbol': 'KC', 'exchange': 'NYBOT'}

    original_exists = os.path.exists
    def exists_side_effect(path):
        if 'council_history.csv' in str(path):
            return True
        return original_exists(path)

    with patch('pandas.read_csv', return_value=mock_csv_data), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('os.path.exists', side_effect=exists_side_effect):

        await reconcile_council_history(config, ib=mock_ib)

        mock_to_csv.assert_called_once()
        mock_ib.reqHistoricalDataAsync.assert_called()

@pytest.mark.asyncio
async def test_reconcile_council_history_skip_recent():
    """Test that recent trades are skipped."""

    entry_time = datetime.now() - timedelta(hours=1)
    mock_csv_data = pd.DataFrame({
        'timestamp': [entry_time],
        'contract': ['KCH4 (202403)'],
        'entry_price': [150.0],
        'master_decision': ['BULLISH'],
        'exit_price': [None]
    })

    mock_ib = MagicMock()

    original_exists = os.path.exists
    def exists_side_effect(path):
        if 'council_history.csv' in str(path):
            return True
        return original_exists(path)

    with patch('pandas.read_csv', return_value=mock_csv_data), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('os.path.exists', side_effect=exists_side_effect):

        await reconcile_council_history({}, ib=mock_ib)

        # Should NOT save because no updates made
        mock_to_csv.assert_not_called()

@pytest.mark.asyncio
async def test_reconcile_council_history_pnl_calc(reconciliation_patches):
    """Test P&L and Trend Logic."""

    mock_csv_data = pd.DataFrame({
        'timestamp': [ENTRY_TIME.isoformat()],
        'contract': ['KCH4 (202403)'],
        'entry_price': [150.0],
        'master_decision': ['BEARISH'],
        'exit_price': [None],
        'exit_timestamp': [None],
        'pnl_realized': [None],
        'actual_trend_direction': [None]
    })

    mock_ib = MagicMock()
    mock_ib.reqContractDetailsAsync = AsyncMock(return_value=[MagicMock(contract=Contract())])

    mock_bar = MagicMock()
    mock_bar.date = TARGET_EXIT.date()
    mock_bar.close = 145.0
    mock_ib.reqHistoricalDataAsync = AsyncMock(return_value=[mock_bar])

    original_exists = os.path.exists
    def exists_side_effect(path):
        if 'council_history.csv' in str(path):
            return True
        return original_exists(path)

    with patch('pandas.read_csv', return_value=mock_csv_data), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('os.path.exists', side_effect=exists_side_effect):

        await reconcile_council_history({'symbol': 'KC', 'exchange': 'NYBOT'}, ib=mock_ib)

        assert mock_csv_data.iloc[0]['exit_price'] == 145.0
        assert mock_csv_data.iloc[0]['pnl_realized'] == 5.0  # (150 - 145) for Short
        assert mock_csv_data.iloc[0]['actual_trend_direction'] == 'BEARISH'
