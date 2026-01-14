
import pytest
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
import os
from trading_bot.reconciliation import reconcile_council_history
from ib_insync import Contract

@pytest.mark.asyncio
async def test_reconcile_council_history_success():
    """Test successful reconciliation of a past trade."""

    # Mock Data
    entry_time = datetime.now() - timedelta(hours=48)
    mock_csv_data = pd.DataFrame({
        'timestamp': [entry_time],
        'contract': ['KCH4 (202403)'],
        'entry_price': [150.0],
        'master_decision': ['BULLISH'],
        'exit_price': [None],
        'exit_timestamp': [None],
        'pnl_realized': [None],
        'actual_trend_direction': [None]
    })

    # Mock IB
    mock_ib = MagicMock()
    mock_ib.reqContractDetailsAsync = AsyncMock()
    mock_details = MagicMock()
    mock_details.contract = Contract(symbol='KC', lastTradeDateOrContractMonth='202403')
    mock_ib.reqContractDetailsAsync.return_value = [mock_details]

    mock_ib.reqHistoricalDataAsync = AsyncMock()
    mock_bar = MagicMock()
    mock_bar.date = (entry_time + timedelta(hours=27)).date()
    mock_bar.close = 155.0 # Price went UP
    mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

    # Mock Config
    config = {'symbol': 'KC', 'exchange': 'NYBOT'}

    # Run Function with patched pandas IO and os.path.exists
    with patch('pandas.read_csv', return_value=mock_csv_data), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('os.path.exists', return_value=True):

        await reconcile_council_history(config, ib=mock_ib)

        # Verify call to save
        mock_to_csv.assert_called_once()

        # Verify logic
        args, _ = mock_to_csv.call_args
        saved_path = args[0]
        # We can't easily inspect the dataframe passed to to_csv as it's a bound method call on the object
        # but we can check if reqHistoricalDataAsync was called
        mock_ib.reqHistoricalDataAsync.assert_called()

@pytest.mark.asyncio
async def test_reconcile_council_history_skip_recent():
    """Test that recent trades are skipped."""

    # Mock Data (Only 1 hour old)
    entry_time = datetime.now() - timedelta(hours=1)
    mock_csv_data = pd.DataFrame({
        'timestamp': [entry_time],
        'contract': ['KCH4 (202403)'],
        'entry_price': [150.0],
        'master_decision': ['BULLISH'],
        'exit_price': [None]
    })

    mock_ib = MagicMock()

    with patch('pandas.read_csv', return_value=mock_csv_data), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('os.path.exists', return_value=True):

        await reconcile_council_history({}, ib=mock_ib)

        # Should NOT save because no updates made
        mock_to_csv.assert_not_called()

@pytest.mark.asyncio
async def test_reconcile_council_history_pnl_calc():
    """Test P&L and Trend Logic."""

    # Mock Data
    entry_time = datetime.now() - timedelta(hours=48)
    mock_csv_data = pd.DataFrame({
        'timestamp': [entry_time],
        'contract': ['KCH4 (202403)'],
        'entry_price': [150.0],
        'master_decision': ['BEARISH'], # Short
        'exit_price': [None]
    })

    mock_ib = MagicMock()
    mock_ib.reqContractDetailsAsync = AsyncMock(return_value=[MagicMock(contract=Contract())])

    # Price went DOWN (Win for Short)
    mock_bar = MagicMock()
    mock_bar.date = (entry_time + timedelta(hours=27)).date()
    mock_bar.close = 145.0
    mock_ib.reqHistoricalDataAsync = AsyncMock(return_value=[mock_bar])

    with patch('pandas.read_csv', return_value=mock_csv_data), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('os.path.exists', return_value=True):

        await reconcile_council_history({'symbol': 'KC'}, ib=mock_ib)

        assert mock_csv_data.iloc[0]['exit_price'] == 145.0
        assert mock_csv_data.iloc[0]['pnl_realized'] == 5.0 # (150 - 145) for Short
        assert mock_csv_data.iloc[0]['actual_trend_direction'] == 'BEARISH'
