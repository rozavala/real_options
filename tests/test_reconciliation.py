
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


# ---------------------------------------------------------------------------
# get_local_active_positions: synthetic dedup tests
# ---------------------------------------------------------------------------

class TestSyntheticDedup:
    """Tests for dropping superseded synthetics in get_local_active_positions."""

    def _make_ledger_row(self, symbol, action, qty, reason, ts='2026-03-05 12:00:00'):
        return {
            'timestamp': ts, 'position_id': 'test', 'combo_id': '',
            'local_symbol': symbol, 'action': action, 'quantity': qty,
            'avg_fill_price': 0.0, 'strike': '', 'right': '',
            'total_value_usd': 0.0, 'reason': reason,
        }

    def test_synthetic_dropped_when_recon_missing_exists(self):
        """Synthetic BUY is dropped when RECONCILIATION_MISSING BUY exists
        for the same symbol — net position should be flat."""
        from reconcile_trades import get_local_active_positions

        rows = [
            self._make_ledger_row('KO N25 C2.85', 'SELL', 1,
                                  'Strategy Execution',
                                  ts='2026-03-04 10:00:00'),
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'Ledger reconciliation - closed manually',
                                  ts='2026-03-05 12:00:00'),
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'RECONCILIATION_MISSING',
                                  ts='2026-03-05 15:10:00'),
        ]
        df = pd.DataFrame(rows)
        positions = get_local_active_positions(df)

        # SELL 1 + BUY 1 (recon, kept) = 0; synthetic BUY dropped
        c285 = positions[positions['Symbol'] == 'KO N25 C2.85']
        assert c285.empty, f"Position should be flat, got: {c285}"

    def test_no_drop_when_no_overlap(self):
        """Synthetics for different symbols than RECONCILIATION_MISSING are kept."""
        from reconcile_trades import get_local_active_positions

        rows = [
            self._make_ledger_row('KO N25 C2.85', 'SELL', 1,
                                  'Strategy Execution'),
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'Ledger reconciliation - closed manually'),
            # RECON_MISSING is for a DIFFERENT symbol
            self._make_ledger_row('KO N25 P2.70', 'SELL', 1,
                                  'RECONCILIATION_MISSING'),
        ]
        df = pd.DataFrame(rows)
        positions = get_local_active_positions(df)

        # C2.85: SELL 1 + BUY 1 (synthetic kept, no overlap) = 0
        # P2.70: SELL 1 (RECON_MISSING) = -1
        c285 = positions[positions['Symbol'] == 'KO N25 C2.85']
        assert c285.empty, "C2.85 should be flat"
        p270 = positions[positions['Symbol'] == 'KO N25 P2.70']
        assert not p270.empty, "P2.70 should show position"

    def test_multiple_symbols(self):
        """Handles multiple symbols with mixed actions correctly."""
        from reconcile_trades import get_local_active_positions

        rows = [
            # C2.85: open SELL + synthetic BUY + RECON_MISSING BUY
            self._make_ledger_row('KO N25 C2.85', 'SELL', 1, 'Strategy Execution'),
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'Ledger reconciliation - closed manually'),
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1, 'RECONCILIATION_MISSING'),
            # C2.8: open BUY + synthetic SELL + RECON_MISSING SELL
            self._make_ledger_row('KO N25 C2.8', 'BUY', 1, 'Strategy Execution'),
            self._make_ledger_row('KO N25 C2.8', 'SELL', 1,
                                  'PHANTOM_RECONCILIATION: synthetic close'),
            self._make_ledger_row('KO N25 C2.8', 'SELL', 1, 'RECONCILIATION_MISSING'),
        ]
        df = pd.DataFrame(rows)
        positions = get_local_active_positions(df)

        assert positions.empty, f"All positions should be flat, got:\n{positions}"

    def test_cross_direction_phantom_dropped(self):
        """Phantom entries in BOTH directions are dropped when RECONCILIATION_MISSING
        exists for that symbol — reproduces the NG bear put spread phantom bug."""
        from reconcile_trades import get_local_active_positions

        rows = [
            # Original open: SELL P2800 (short leg of bear put spread)
            self._make_ledger_row('LNEK6 P2800', 'SELL', 1,
                                  'Strategy Execution', ts='2026-02-26 10:00:00'),
            # Flex recon found the close: BUY P2800
            self._make_ledger_row('LNEK6 P2800', 'BUY', 1,
                                  'RECONCILIATION_MISSING', ts='2026-02-26 15:00:00'),
            # Phantom recon also created a BUY close (same position_id)
            self._make_ledger_row('LNEK6 P2800', 'BUY', 1,
                                  'PHANTOM_RECONCILIATION', ts='2026-02-27 19:00:00'),
            # Phantom recon ALSO created a SELL for the recon position_id
            self._make_ledger_row('LNEK6 P2800', 'SELL', 1,
                                  'PHANTOM_RECONCILIATION', ts='2026-02-27 19:00:00'),
        ]
        df = pd.DataFrame(rows)
        positions = get_local_active_positions(df)

        # All phantoms (both BUY and SELL) should be dropped.
        # Net: SELL 1 (original) + BUY 1 (RECON_MISSING) = 0
        p2800 = positions[positions['Symbol'] == 'LNEK6 P2800']
        assert p2800.empty, (
            f"LNEK6 P2800 should be flat after dropping both-direction phantoms, "
            f"got: {p2800}"
        )

    def test_no_reason_column(self):
        """Works without a reason column (legacy ledgers)."""
        from reconcile_trades import get_local_active_positions

        rows = [
            {'timestamp': '2026-03-04', 'position_id': 'x', 'combo_id': '',
             'local_symbol': 'KO N25 C2.85', 'action': 'BUY', 'quantity': 1,
             'avg_fill_price': 0.5},
        ]
        df = pd.DataFrame(rows)
        positions = get_local_active_positions(df)

        assert len(positions) == 1
        assert positions.iloc[0]['Quantity'] == 1
