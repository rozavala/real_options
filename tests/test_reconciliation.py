
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
# invalidate_superseded_synthetics tests
# ---------------------------------------------------------------------------

class TestInvalidateSupersededSynthetics:
    """Tests for the double-counting fix in reconcile_trades.py."""

    def _make_ledger_row(self, symbol, action, qty, reason, ts='2026-03-05 12:00:00'):
        return {
            'timestamp': ts, 'position_id': 'test', 'combo_id': '',
            'local_symbol': symbol, 'action': action, 'quantity': qty,
            'avg_fill_price': 0.0, 'strike': '', 'right': '',
            'total_value_usd': 0.0, 'reason': reason,
        }

    def test_invalidates_superseded_synthetic(self, tmp_path):
        """When both synthetic and RECONCILIATION_MISSING exist for same
        (symbol, action), a reversal entry should be written."""
        from reconcile_trades import invalidate_superseded_synthetics

        archive_dir = tmp_path / 'archive_ledger'
        archive_dir.mkdir()

        # Create a ledger with a synthetic BUY and a RECONCILIATION_MISSING BUY
        rows = [
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'Ledger reconciliation - closed manually in TWS'),
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'RECONCILIATION_MISSING',
                                  ts='2026-03-05 15:10:00'),
        ]
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / 'trade_ledger.csv', index=False)

        n = invalidate_superseded_synthetics(str(tmp_path))
        assert n == 1

        # Verify the invalidation file was written
        inv_path = archive_dir / 'trade_ledger_synthetic_invalidations.csv'
        assert inv_path.exists()
        inv_df = pd.read_csv(inv_path)
        assert len(inv_df) == 1
        assert inv_df.iloc[0]['local_symbol'] == 'KO N25 C2.85'
        assert inv_df.iloc[0]['action'] == 'SELL'  # opposite of BUY
        assert inv_df.iloc[0]['quantity'] == 1

    def test_no_overlap_no_invalidation(self, tmp_path):
        """No invalidation when synthetics and RECONCILIATION_MISSING are
        for different symbols."""
        from reconcile_trades import invalidate_superseded_synthetics

        rows = [
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'Ledger reconciliation - closed manually'),
            self._make_ledger_row('KO N25 P2.70', 'SELL', 1,
                                  'RECONCILIATION_MISSING'),
        ]
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / 'trade_ledger.csv', index=False)

        n = invalidate_superseded_synthetics(str(tmp_path))
        assert n == 0

    def test_idempotent_reruns(self, tmp_path):
        """Running twice should produce the same result (idempotent)."""
        from reconcile_trades import invalidate_superseded_synthetics

        archive_dir = tmp_path / 'archive_ledger'
        archive_dir.mkdir()

        rows = [
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'Ledger reconciliation - closed manually'),
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'RECONCILIATION_MISSING'),
        ]
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / 'trade_ledger.csv', index=False)

        n1 = invalidate_superseded_synthetics(str(tmp_path))
        assert n1 == 1

        # Second run should still write 1 (it excludes existing invalidations
        # from analysis and rewrites the file)
        n2 = invalidate_superseded_synthetics(str(tmp_path))
        assert n2 == 1

    def test_multiple_symbols_and_actions(self, tmp_path):
        """Handles multiple symbols with different actions correctly."""
        from reconcile_trades import invalidate_superseded_synthetics

        archive_dir = tmp_path / 'archive_ledger'
        archive_dir.mkdir()

        rows = [
            # C2.85: synthetic BUY + RECON_MISSING BUY → invalidate
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'Ledger reconciliation - closed manually'),
            self._make_ledger_row('KO N25 C2.85', 'BUY', 1,
                                  'RECONCILIATION_MISSING'),
            # C2.8: synthetic SELL + RECON_MISSING SELL → invalidate
            self._make_ledger_row('KO N25 C2.8', 'SELL', 1,
                                  'PHANTOM_RECONCILIATION - closed externally'),
            self._make_ledger_row('KO N25 C2.8', 'SELL', 1,
                                  'RECONCILIATION_MISSING'),
            # P2.8: synthetic SELL 2 + RECON_MISSING SELL 2 → invalidate
            self._make_ledger_row('KO N25 P2.8', 'SELL', 2,
                                  'Ledger reconciliation - closed externally'),
            self._make_ledger_row('KO N25 P2.8', 'SELL', 2,
                                  'RECONCILIATION_MISSING'),
        ]
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / 'trade_ledger.csv', index=False)

        n = invalidate_superseded_synthetics(str(tmp_path))
        assert n == 3

        inv_df = pd.read_csv(archive_dir / 'trade_ledger_synthetic_invalidations.csv')
        assert len(inv_df) == 3

    def test_partial_overlap(self, tmp_path):
        """When synthetic qty > RECON_MISSING qty, only overlap is invalidated."""
        from reconcile_trades import invalidate_superseded_synthetics

        archive_dir = tmp_path / 'archive_ledger'
        archive_dir.mkdir()

        rows = [
            self._make_ledger_row('KO N25 P2.8', 'SELL', 3,
                                  'Ledger reconciliation - closed externally'),
            self._make_ledger_row('KO N25 P2.8', 'SELL', 2,
                                  'RECONCILIATION_MISSING'),
        ]
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / 'trade_ledger.csv', index=False)

        n = invalidate_superseded_synthetics(str(tmp_path))
        assert n == 1

        inv_df = pd.read_csv(archive_dir / 'trade_ledger_synthetic_invalidations.csv')
        assert inv_df.iloc[0]['quantity'] == 2  # min(3, 2)

    def test_net_position_corrected(self, tmp_path):
        """After invalidation, get_local_active_positions should show
        correct net positions."""
        from reconcile_trades import invalidate_superseded_synthetics, get_local_active_positions

        archive_dir = tmp_path / 'archive_ledger'
        archive_dir.mkdir()

        # Simulate the real scenario: original SELL, synthetic BUY (close),
        # RECONCILIATION_MISSING BUY (real close)
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
        df.to_csv(tmp_path / 'trade_ledger.csv', index=False)

        # Before fix: net = -1 + 1 + 1 = +1 (wrong)
        positions_before = get_local_active_positions(
            pd.read_csv(tmp_path / 'trade_ledger.csv')
        )
        assert not positions_before.empty, "Should show phantom position before fix"

        # Apply fix
        invalidate_superseded_synthetics(str(tmp_path))

        # After fix: load full ledger including invalidation file
        from reconcile_trades import get_trade_ledger_df
        full_ledger = get_trade_ledger_df(str(tmp_path))
        positions_after = get_local_active_positions(full_ledger)

        # Net should be 0 (SELL 1 + BUY 1 synthetic + BUY 1 recon
        #                    + SELL 1 invalidation = 0)
        c285 = positions_after[
            positions_after['Symbol'] == 'KO N25 C2.85'
        ]
        assert c285.empty, f"Position should be flat, got: {c285}"
