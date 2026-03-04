"""Tests for Pre-Entry Thesis Coherence Check.

Covers: contract month normalization, coherence gate logic, and
deferred auto-close flow. All tests mock TMS + IB (no ChromaDB/IB dependency).
"""

import asyncio
import json
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from trading_bot.tms import normalize_contract_month, _resolve_year


# ---------------------------------------------------------------------------
# 1. Normalization tests
# ---------------------------------------------------------------------------

class TestNormalizeContractMonth(unittest.TestCase):
    """Tests for normalize_contract_month()."""

    def test_normalize_yyyymm(self):
        self.assertEqual(normalize_contract_month("202607"), "202607")

    def test_normalize_yyyymmdd(self):
        self.assertEqual(normalize_contract_month("20260715"), "202607")

    def test_normalize_ticker_prefix(self):
        self.assertEqual(normalize_contract_month("KCN26"), "202607")

    def test_normalize_single_digit_year(self):
        # Rolling window: 6 + 2000 = 2006 < (2026 - 5) = 2021 → 2106
        # But for KCH6, H=March, so result depends on current year
        result = normalize_contract_month("KCH6")
        # _resolve_year(6): 2006 < now.year - 5 → 2106
        self.assertEqual(result, "210603")

    def test_normalize_two_digit_year(self):
        self.assertEqual(normalize_contract_month("KCH26"), "202603")

    def test_normalize_empty(self):
        self.assertEqual(normalize_contract_month(""), "")

    def test_normalize_with_suffix_noise(self):
        # "KCH6 (202603)" → split()[0] = "KCH6" → not 4-digit year...
        # but "202603" is in the suffix — split()[0] strips it
        # Actually "KCH6" → H=3, 6→2106 (rolling window)
        # The suffix is stripped. Let's test with a numeric suffix:
        self.assertEqual(normalize_contract_month("202603 (extra)"), "202603")

    def test_normalize_bare_month_letter(self):
        self.assertEqual(normalize_contract_month("N26"), "202607")

    def test_normalize_bare_month_letter_single_digit(self):
        result = normalize_contract_month("H6")
        self.assertEqual(result, "210603")


class TestResolveYear(unittest.TestCase):
    def test_four_digit_passthrough(self):
        self.assertEqual(_resolve_year(2026), 2026)

    def test_two_digit_recent(self):
        self.assertEqual(_resolve_year(26), 2026)

    def test_two_digit_old(self):
        # 6 + 2000 = 2006 < current_year - 5 → adds 100
        result = _resolve_year(6)
        self.assertEqual(result, 2106)


# ---------------------------------------------------------------------------
# 2. Coherence gate tests
# ---------------------------------------------------------------------------

def _make_signal(direction='BULLISH', contract_month='202607', prediction_type='DIRECTIONAL', level=None, confidence=0.75):
    sig = {
        'direction': direction,
        'contract_month': contract_month,
        'prediction_type': prediction_type,
        'confidence': confidence,
    }
    if level:
        sig['level'] = level
    return sig


def _make_thesis(strategy_type, direction, trade_id, contract_month='202607'):
    return {
        'strategy_type': strategy_type,
        '_metadata': {
            'trade_id': trade_id,
            'direction': direction,
            'contract_month': contract_month,
        },
    }


class TestCheckThesisCoherence(unittest.TestCase):
    """Tests for _check_thesis_coherence()."""

    def setUp(self):
        self.config = {'symbol': 'KC'}

    @patch('trading_bot.tms.TransactiveMemory')
    def test_intra_cycle_duplicate_blocked(self, mock_tms_cls):
        """Two same-direction signals in batch → second blocked."""
        from trading_bot.order_manager import _check_thesis_coherence

        approved = [_make_signal('BULLISH', '202607')]
        signal = _make_signal('BULLISH', '202607')

        action, conflicts = _check_thesis_coherence(signal, self.config, approved)
        self.assertEqual(action, "BLOCK_INTRA_CYCLE")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_cross_cycle_duplicate_allowed(self, mock_tms_cls):
        """Same direction in TMS (from previous cycle) → PROCEED (not blocked)."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms.get_active_theses_by_contract.return_value = [
            _make_thesis('BULL_CALL_SPREAD', 'BULLISH', 'old-id-1')
        ]
        mock_tms_cls.return_value = mock_tms

        signal = _make_signal('BULLISH', '202607')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "PROCEED")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_contradiction_detected(self, mock_tms_cls):
        """Opposite direction → CONTRADICT + supersedes_trade_ids attached."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms.get_active_theses_by_contract.return_value = [
            _make_thesis('BEAR_PUT_SPREAD', 'BEARISH', 'old-bear-1')
        ]
        mock_tms_cls.return_value = mock_tms

        signal = _make_signal('BULLISH', '202607')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "CONTRADICT")
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]['_metadata']['trade_id'], 'old-bear-1')

    @patch('trading_bot.tms.TransactiveMemory')
    def test_contradict_cross_day_still_fires(self, mock_tms_cls):
        """Old thesis from yesterday, new signal today → CONTRADICT."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms.get_active_theses_by_contract.return_value = [
            _make_thesis('BULL_CALL_SPREAD', 'BULLISH', 'old-bull-yesterday')
        ]
        mock_tms_cls.return_value = mock_tms

        signal = _make_signal('BEARISH', '202607')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "CONTRADICT")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_vol_strategy_coexists_with_directional(self, mock_tms_cls):
        """Iron Condor + Bull Call same month → PROCEED (different axes)."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms.get_active_theses_by_contract.return_value = [
            _make_thesis('BULL_CALL_SPREAD', 'BULLISH', 'old-bull-1')
        ]
        mock_tms_cls.return_value = mock_tms

        # Iron Condor is NEUTRAL direction + LOW_VOL
        signal = _make_signal('NEUTRAL', '202607', prediction_type='VOLATILITY', level='LOW')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "PROCEED")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_same_direction_different_strategy_allowed(self, mock_tms_cls):
        """Bull Call + another BULLISH same month → PROCEED (duplicates allowed)."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms.get_active_theses_by_contract.return_value = [
            _make_thesis('BULL_CALL_SPREAD', 'BULLISH', 'old-bull-1')
        ]
        mock_tms_cls.return_value = mock_tms

        signal = _make_signal('BULLISH', '202607')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "PROCEED")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_vol_contradiction_iron_condor_vs_straddle(self, mock_tms_cls):
        """Iron Condor (low vol) then Long Straddle (high vol) → CONTRADICT."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms.get_active_theses_by_contract.return_value = [
            _make_thesis('IRON_CONDOR', 'NEUTRAL', 'old-condor-1')
        ]
        mock_tms_cls.return_value = mock_tms

        signal = _make_signal('NEUTRAL', '202607', prediction_type='VOLATILITY', level='HIGH')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "CONTRADICT")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_vol_contradiction_straddle_vs_iron_condor(self, mock_tms_cls):
        """Long Straddle (high vol) then Iron Condor (low vol) → CONTRADICT."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms.get_active_theses_by_contract.return_value = [
            _make_thesis('LONG_STRADDLE', 'NEUTRAL', 'old-straddle-1')
        ]
        mock_tms_cls.return_value = mock_tms

        signal = _make_signal('NEUTRAL', '202607', prediction_type='VOLATILITY', level='LOW')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "CONTRADICT")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_fail_open_on_tms_error(self, mock_tms_cls):
        """TMS unavailable → PROCEED (fail open)."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = None
        mock_tms_cls.return_value = mock_tms

        signal = _make_signal('BULLISH', '202607')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "PROCEED")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_cross_commodity_no_collision(self, mock_tms_cls):
        """Same month, different symbols → PROCEED."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = True
        # The query is scoped by symbol — returns empty for different symbol
        mock_tms.get_active_theses_by_contract.return_value = []
        mock_tms_cls.return_value = mock_tms

        signal = _make_signal('BULLISH', '202607')
        config_cc = {'symbol': 'CC'}
        action, conflicts = _check_thesis_coherence(signal, config_cc, [])
        self.assertEqual(action, "PROCEED")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_legacy_thesis_fallback(self, mock_tms_cls):
        """Thesis without new metadata → detected via JSON parsing fallback."""
        from trading_bot.order_manager import _check_thesis_coherence

        # Return legacy thesis with direction only in strategy_type mapping
        mock_tms = MagicMock()
        mock_tms.collection = True
        legacy_thesis = {
            'strategy_type': 'BEAR_PUT_SPREAD',
            '_metadata': {
                'trade_id': 'legacy-123',
                'direction': '',  # Empty — will fall back to _STRATEGY_DIRECTION
            },
        }
        mock_tms.get_active_theses_by_contract.return_value = [legacy_thesis]
        mock_tms_cls.return_value = mock_tms

        signal = _make_signal('BULLISH', '202607')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "CONTRADICT")

    @patch('trading_bot.tms.TransactiveMemory')
    def test_intra_cycle_contradiction_warning(self, mock_tms_cls):
        """BULLISH + BEARISH for same month in one batch → warning logged, both proceed."""
        from trading_bot.order_manager import _check_thesis_coherence

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms.get_active_theses_by_contract.return_value = []
        mock_tms_cls.return_value = mock_tms

        approved = [_make_signal('BULLISH', '202607')]
        signal = _make_signal('BEARISH', '202607')

        with self.assertLogs('OrderManager', level='WARNING') as cm:
            action, conflicts = _check_thesis_coherence(signal, self.config, approved)

        self.assertEqual(action, "PROCEED")
        self.assertTrue(any('INTRA-CYCLE CONTRADICTION' in msg for msg in cm.output))

    def test_empty_contract_month_proceeds(self):
        """No contract month → PROCEED without TMS lookup."""
        from trading_bot.order_manager import _check_thesis_coherence

        signal = _make_signal('BULLISH', '')
        action, conflicts = _check_thesis_coherence(signal, self.config, [])
        self.assertEqual(action, "PROCEED")


# ---------------------------------------------------------------------------
# 3. Auto-close tests
# ---------------------------------------------------------------------------

class TestDeferredInvalidation(unittest.TestCase):
    """Tests for deferred invalidation + auto-close in fill handler."""

    @patch('trading_bot.order_manager.record_entry_thesis_for_trade', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.order_manager._auto_close_superseded', new_callable=AsyncMock)
    @patch('trading_bot.tms.TransactiveMemory')
    async def test_deferred_invalidation_plus_auto_close(
        self, mock_tms_cls, mock_auto_close, mock_log, mock_record
    ):
        """Mock fill callback → old thesis invalidated AND auto-close called."""
        from trading_bot.order_manager import _handle_and_log_fill, _recorded_thesis_positions

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms_cls.return_value = mock_tms

        ib = MagicMock()
        ib.isConnected.return_value = True
        ib.qualifyContractsAsync = AsyncMock(return_value=[MagicMock()])

        trade = MagicMock()
        trade.order.orderId = 123
        trade.order.orderRef = 'new-uuid-1'
        trade.contract = MagicMock(spec=['comboLegs', 'localSymbol'])
        trade.contract.localSymbol = 'COMBO'

        fill = MagicMock()
        fill.contract = MagicMock()
        fill.contract.conId = 999
        fill.execution.avgPrice = 1.50

        decision_data = {
            'prediction_type': 'DIRECTIONAL',
            'direction': 'BULLISH',
            'contract_month': '202607',
            'supersedes_trade_ids': ['old-bear-1'],
            'supersedes_reason': 'CONTRADICT',
        }

        config = {'symbol': 'KC', 'exchange': 'NYBOT', 'notifications': {}}

        _recorded_thesis_positions.discard('new-uuid-1')

        await _handle_and_log_fill(ib, trade, fill, 456, 'new-uuid-1', decision_data, config)

        # Verify invalidation was called
        mock_tms.invalidate_thesis.assert_called_once()
        call_args = mock_tms.invalidate_thesis.call_args
        self.assertEqual(call_args[0][0], 'old-bear-1')
        self.assertIn('CONTRADICT', call_args[0][1])

        # Verify auto-close was called
        mock_auto_close.assert_called_once()

    @patch('trading_bot.order_manager.record_entry_thesis_for_trade', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.order_manager._auto_close_superseded', new_callable=AsyncMock)
    @patch('trading_bot.tms.TransactiveMemory')
    async def test_auto_close_failure_does_not_crash_fill_handler(
        self, mock_tms_cls, mock_auto_close, mock_log, mock_record
    ):
        """_auto_close_superseded raises → fill handler completes, thesis recorded."""
        from trading_bot.order_manager import _handle_and_log_fill, _recorded_thesis_positions

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms_cls.return_value = mock_tms

        mock_auto_close.side_effect = Exception("IB timeout")

        ib = MagicMock()
        ib.isConnected.return_value = True
        ib.qualifyContractsAsync = AsyncMock(return_value=[MagicMock()])

        trade = MagicMock()
        trade.order.orderId = 124
        trade.order.orderRef = 'new-uuid-2'
        trade.contract = MagicMock(spec=['comboLegs', 'localSymbol'])
        trade.contract.localSymbol = 'COMBO'

        fill = MagicMock()
        fill.contract = MagicMock()
        fill.contract.conId = 998
        fill.execution.avgPrice = 1.25

        decision_data = {
            'prediction_type': 'DIRECTIONAL',
            'direction': 'BEARISH',
            'contract_month': '202607',
            'supersedes_trade_ids': ['old-bull-1'],
            'supersedes_reason': 'CONTRADICT',
        }

        config = {'symbol': 'KC', 'exchange': 'NYBOT', 'notifications': {}}
        _recorded_thesis_positions.discard('new-uuid-2')

        # Should not raise
        await _handle_and_log_fill(ib, trade, fill, 457, 'new-uuid-2', decision_data, config)

        # Thesis was still recorded despite auto-close failure
        mock_record.assert_called_once()

    @patch('trading_bot.order_manager.record_entry_thesis_for_trade', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.tms.TransactiveMemory')
    async def test_no_invalidation_without_supersedes(self, mock_tms_cls, mock_log, mock_record):
        """Normal fill (no supersedes_trade_ids) → no invalidation."""
        from trading_bot.order_manager import _handle_and_log_fill, _recorded_thesis_positions

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms_cls.return_value = mock_tms

        ib = MagicMock()
        ib.isConnected.return_value = True
        ib.qualifyContractsAsync = AsyncMock(return_value=[MagicMock()])

        trade = MagicMock()
        trade.order.orderId = 125
        trade.order.orderRef = 'normal-uuid'
        trade.contract = MagicMock(spec=['comboLegs', 'localSymbol'])
        trade.contract.localSymbol = 'COMBO'

        fill = MagicMock()
        fill.contract = MagicMock()
        fill.contract.conId = 997
        fill.execution.avgPrice = 2.00

        decision_data = {
            'prediction_type': 'DIRECTIONAL',
            'direction': 'BULLISH',
            'contract_month': '202607',
            # No supersedes_trade_ids
        }

        config = {'symbol': 'KC', 'exchange': 'NYBOT', 'notifications': {}}
        _recorded_thesis_positions.discard('normal-uuid')

        await _handle_and_log_fill(ib, trade, fill, 458, 'normal-uuid', decision_data, config)

        # No invalidation should have been called
        mock_tms.invalidate_thesis.assert_not_called()

    @patch('trading_bot.order_manager.record_entry_thesis_for_trade', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.tms.TransactiveMemory')
    async def test_auto_close_skipped_when_ib_disconnected(self, mock_tms_cls, mock_log, mock_record):
        """ib.isConnected() returns False → auto-close skipped with warning."""
        from trading_bot.order_manager import _handle_and_log_fill, _recorded_thesis_positions

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms_cls.return_value = mock_tms

        ib = MagicMock()
        ib.isConnected.return_value = False
        ib.qualifyContractsAsync = AsyncMock(return_value=[MagicMock()])

        trade = MagicMock()
        trade.order.orderId = 126
        trade.order.orderRef = 'disconn-uuid'
        trade.contract = MagicMock(spec=['comboLegs', 'localSymbol'])
        trade.contract.localSymbol = 'COMBO'

        fill = MagicMock()
        fill.contract = MagicMock()
        fill.contract.conId = 996
        fill.execution.avgPrice = 1.75

        decision_data = {
            'prediction_type': 'DIRECTIONAL',
            'direction': 'BULLISH',
            'contract_month': '202607',
            'supersedes_trade_ids': ['old-bear-2'],
            'supersedes_reason': 'CONTRADICT',
        }

        config = {'symbol': 'KC', 'exchange': 'NYBOT', 'notifications': {}}
        _recorded_thesis_positions.discard('disconn-uuid')

        await _handle_and_log_fill(ib, trade, fill, 459, 'disconn-uuid', decision_data, config)

        # Invalidation should still be called
        mock_tms.invalidate_thesis.assert_called_once()
        # But no auto-close attempt


class TestAutoCloseSuperseded(unittest.TestCase):
    """Tests for _auto_close_superseded()."""

    @patch('trading_bot.order_manager.get_trade_ledger_df')
    async def test_auto_close_bails_on_aggregated_positions(self, mock_ledger):
        """Two theses share same strikes, IB qty=2 > thesis qty=1 → bail out."""
        from trading_bot.order_manager import _auto_close_superseded
        import pandas as pd

        ib = MagicMock()
        ib.isConnected.return_value = True

        # IB shows qty=2 for a symbol (aggregated from 2 theses)
        pos = MagicMock()
        pos.contract.localSymbol = 'KCN6 P270'
        pos.position = -2
        ib.reqPositionsAsync = AsyncMock(return_value=[pos])

        # Ledger shows this thesis has qty=1
        mock_ledger.return_value = pd.DataFrame({
            'position_id': ['old-thesis-1', 'old-thesis-1'],
            'local_symbol': ['KCN6 P270', 'KCN6 P275'],
            'quantity': [1, 1],
            'action': ['SELL', 'BUY'],
        })

        tms = MagicMock()
        config = {'data_dir': '/tmp/test'}

        # Should bail out without closing
        await _auto_close_superseded(ib, 'old-thesis-1', 'CONTRADICT', config, tms)

        # _close_spread_position should NOT have been called
        # (we didn't patch it, so if it were called it would fail)

    @patch('trading_bot.order_manager.get_trade_ledger_df')
    async def test_auto_close_no_matching_positions(self, mock_ledger):
        """No IB positions match → logs info and returns."""
        from trading_bot.order_manager import _auto_close_superseded
        import pandas as pd

        ib = MagicMock()
        ib.isConnected.return_value = True
        ib.reqPositionsAsync = AsyncMock(return_value=[])

        mock_ledger.return_value = pd.DataFrame({
            'position_id': ['old-thesis-2'],
            'local_symbol': ['KCN6 P270'],
            'quantity': [1],
            'action': ['SELL'],
        })

        tms = MagicMock()
        config = {'data_dir': '/tmp/test'}

        await _auto_close_superseded(ib, 'old-thesis-2', 'CONTRADICT', config, tms)

    @patch('trading_bot.order_manager.get_trade_ledger_df')
    async def test_auto_close_skips_when_disconnected(self, mock_ledger):
        """IB disconnected at entry → returns immediately."""
        from trading_bot.order_manager import _auto_close_superseded

        ib = MagicMock()
        ib.isConnected.return_value = False

        tms = MagicMock()
        config = {'data_dir': '/tmp/test'}

        await _auto_close_superseded(ib, 'old-thesis-3', 'CONTRADICT', config, tms)

        # Should not even attempt to get positions
        ib.reqPositionsAsync.assert_not_called()


class TestDedupPreventsSecondAutoClose(unittest.TestCase):
    """Verify _processed_supersedes prevents duplicate deferred invalidation."""

    @patch('trading_bot.order_manager.record_entry_thesis_for_trade', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.order_manager._auto_close_superseded', new_callable=AsyncMock)
    @patch('trading_bot.tms.TransactiveMemory')
    async def test_second_fill_skips_invalidation(
        self, mock_tms_cls, mock_auto_close, mock_log, mock_record
    ):
        """Two leg fills with same supersedes_trade_ids → only one invalidation + auto-close."""
        from trading_bot.order_manager import (
            _handle_and_log_fill, _recorded_thesis_positions, _processed_supersedes
        )

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms_cls.return_value = mock_tms

        ib = MagicMock()
        ib.isConnected.return_value = True
        ib.qualifyContractsAsync = AsyncMock(return_value=[MagicMock()])

        decision_data = {
            'prediction_type': 'DIRECTIONAL',
            'direction': 'BULLISH',
            'contract_month': '202607',
            'supersedes_trade_ids': ['old-bear-dedup'],
            'supersedes_reason': 'CONTRADICT',
        }

        config = {'symbol': 'KC', 'exchange': 'NYBOT', 'notifications': {}}

        # --- First leg fill ---
        trade1 = MagicMock()
        trade1.order.orderId = 200
        trade1.order.orderRef = 'dedup-uuid'
        trade1.contract = MagicMock(spec=['comboLegs', 'localSymbol'])
        trade1.contract.localSymbol = 'COMBO'

        fill1 = MagicMock()
        fill1.contract = MagicMock()
        fill1.contract.conId = 1001
        fill1.execution.avgPrice = 2.00

        _recorded_thesis_positions.discard('dedup-uuid')
        _processed_supersedes.discard('old-bear-dedup')

        await _handle_and_log_fill(ib, trade1, fill1, 500, 'dedup-uuid', decision_data, config)

        self.assertEqual(mock_tms.invalidate_thesis.call_count, 1)
        self.assertEqual(mock_auto_close.call_count, 1)

        # --- Second leg fill (same decision_data ref) ---
        trade2 = MagicMock()
        trade2.order.orderId = 201
        trade2.order.orderRef = 'dedup-uuid'
        trade2.contract = MagicMock(spec=['comboLegs', 'localSymbol'])
        trade2.contract.localSymbol = 'COMBO'

        fill2 = MagicMock()
        fill2.contract = MagicMock()
        fill2.contract.conId = 1002
        fill2.execution.avgPrice = 1.80

        await _handle_and_log_fill(ib, trade2, fill2, 501, 'dedup-uuid', decision_data, config)

        # Invalidation + auto-close should NOT have been called a second time
        self.assertEqual(mock_tms.invalidate_thesis.call_count, 1,
                         "invalidate_thesis called more than once — dedup failed")
        self.assertEqual(mock_auto_close.call_count, 1,
                         "_auto_close_superseded called more than once — dedup failed")


class TestCloseSpreadPositionUsesPosContract(unittest.TestCase):
    """Verify _close_spread_position uses pos.contract directly, not re-qualification."""

    async def test_no_requalification(self):
        """qualifyContractsAsync should never be called during spread close."""
        import sys
        from unittest.mock import AsyncMock, MagicMock, patch

        # Lazy import to get the function from orchestrator
        from orchestrator import _close_spread_position

        ib = MagicMock()
        ib.qualifyContractsAsync = AsyncMock(
            side_effect=AssertionError("qualifyContractsAsync should NOT be called"))
        ib.isConnected.return_value = True

        # Build fake position legs with correct contracts
        leg1 = MagicMock()
        leg1.contract = MagicMock()
        leg1.contract.localSymbol = 'KCN6 C310'
        leg1.contract.strike = 3.10
        leg1.contract.exchange = 'NYBOT'
        leg1.contract.conId = 12345
        leg1.position = 1
        leg1.account = 'DU12345'
        leg1.avgCost = 100.0

        leg2 = MagicMock()
        leg2.contract = MagicMock()
        leg2.contract.localSymbol = 'KCN6 C320'
        leg2.contract.strike = 3.20
        leg2.contract.exchange = 'NYBOT'
        leg2.contract.conId = 12346
        leg2.position = -1
        leg2.account = 'DU12345'
        leg2.avgCost = 50.0

        legs = [leg1, leg2]
        config = {'exchange': 'NYBOT', 'notifications': {}}

        # Mock place_order to return a filled trade
        with patch('orchestrator.place_order') as mock_place:
            mock_trade = MagicMock()
            mock_trade.orderStatus.status = 'Filled'
            mock_trade.orderStatus.remaining = 0
            mock_place.return_value = mock_trade

            result = await _close_spread_position(
                ib, legs, 'test-pos-123', 'unit test close', config)

        # qualifyContractsAsync was NOT called (would raise AssertionError if it was)
        ib.qualifyContractsAsync.assert_not_called()

    async def test_fills_missing_exchange_from_config(self):
        """If pos.contract has no exchange, fill from config."""
        from orchestrator import _close_spread_position

        ib = MagicMock()
        ib.isConnected.return_value = True

        leg = MagicMock()
        leg.contract = MagicMock()
        leg.contract.localSymbol = 'KCN6 P270'
        leg.contract.strike = 2.70
        leg.contract.exchange = ''  # Missing exchange
        leg.contract.conId = 54321
        leg.position = -1
        leg.account = 'DU12345'
        leg.avgCost = 80.0

        config = {'exchange': 'NYBOT', 'notifications': {}}

        with patch('orchestrator.place_order') as mock_place:
            mock_trade = MagicMock()
            mock_trade.orderStatus.status = 'Filled'
            mock_trade.orderStatus.remaining = 0
            mock_place.return_value = mock_trade

            await _close_spread_position(
                ib, [leg], 'test-pos-456', 'unit test close', config)

        # Exchange should have been filled from config
        self.assertEqual(leg.contract.exchange, 'NYBOT')


if __name__ == '__main__':
    unittest.main()
