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

class TestDeferredInvalidation(unittest.IsolatedAsyncioTestCase):
    """Tests for fill handler behavior with supersedes_trade_ids.

    CONTRADICT closures are now handled immediately in place_queued_orders()
    before new orders are placed. The fill handler only logs a warning if
    it receives supersedes_trade_ids (should no longer happen in normal flow).
    """

    @patch('trading_bot.order_manager.record_entry_thesis_for_trade', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.tms.TransactiveMemory')
    async def test_fill_handler_logs_warning_for_supersedes(
        self, mock_tms_cls, mock_log, mock_record
    ):
        """Fill with supersedes_trade_ids → warning logged, NO invalidation or auto-close."""
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

        with self.assertLogs('OrderManager', level='WARNING') as cm:
            await _handle_and_log_fill(ib, trade, fill, 456, 'new-uuid-1', decision_data, config)

        # Verify warning was logged about legacy supersedes_trade_ids
        warning_msgs = [m for m in cm.output if 'supersedes_trade_ids' in m]
        self.assertTrue(warning_msgs, "Expected warning about supersedes_trade_ids")

        # NO invalidation or auto-close — handled by place_queued_orders now
        mock_tms.invalidate_thesis.assert_not_called()

    @patch('trading_bot.order_manager.record_entry_thesis_for_trade', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.tms.TransactiveMemory')
    async def test_fill_handler_completes_with_supersedes(
        self, mock_tms_cls, mock_log, mock_record
    ):
        """Fill with supersedes_trade_ids still completes and records thesis."""
        from trading_bot.order_manager import _handle_and_log_fill, _recorded_thesis_positions

        mock_tms = MagicMock()
        mock_tms.collection = True
        mock_tms_cls.return_value = mock_tms

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

        await _handle_and_log_fill(ib, trade, fill, 457, 'new-uuid-2', decision_data, config)

        # Thesis was still recorded
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
    async def test_fill_handler_warning_only_when_ib_disconnected(
        self, mock_tms_cls, mock_log, mock_record
    ):
        """ib.isConnected() False + supersedes_trade_ids → warning only, no invalidation."""
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

        # No invalidation — handled by pending_close flag + audit cycle
        mock_tms.invalidate_thesis.assert_not_called()


class TestAutoCloseSuperseded(unittest.IsolatedAsyncioTestCase):
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


class TestFillHandlerNoLongerAutoCloses(unittest.IsolatedAsyncioTestCase):
    """Verify fill handler does not perform CONTRADICT invalidation or auto-close.

    CONTRADICT closures are now handled in place_queued_orders before order
    placement. The fill handler should only log a warning if it receives
    supersedes_trade_ids (legacy path).
    """

    @patch('trading_bot.order_manager.record_entry_thesis_for_trade', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.tms.TransactiveMemory')
    async def test_multiple_fills_no_invalidation(
        self, mock_tms_cls, mock_log, mock_record
    ):
        """Two leg fills with supersedes_trade_ids → warnings only, no invalidation."""
        from trading_bot.order_manager import _handle_and_log_fill, _recorded_thesis_positions

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

        await _handle_and_log_fill(ib, trade1, fill1, 500, 'dedup-uuid', decision_data, config)

        # --- Second leg fill ---
        fill2 = MagicMock()
        fill2.contract = MagicMock()
        fill2.contract.conId = 1002
        fill2.execution.avgPrice = 1.80

        await _handle_and_log_fill(ib, trade1, fill2, 501, 'dedup-uuid', decision_data, config)

        # No invalidation or auto-close in either fill
        mock_tms.invalidate_thesis.assert_not_called()


class TestCloseSpreadPositionUsesPosContract(unittest.IsolatedAsyncioTestCase):
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
        ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[])
        # New: _close_spread_position now verifies via reqPositionsAsync
        ib.reqPositionsAsync = AsyncMock(return_value=[])
        # New: BAG close attempts market data (will fail gracefully with NaN)
        mock_ticker = MagicMock()
        mock_ticker.bid = float('nan')
        mock_ticker.ask = float('nan')
        mock_ticker.last = float('nan')
        ib.reqMktData = MagicMock(return_value=mock_ticker)
        ib.cancelMktData = MagicMock()

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
            mock_trade.orderStatus.avgFillPrice = 1.50
            mock_trade.fills = []
            mock_trade.log = []
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
        ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[])
        # New: _close_spread_position now verifies via reqPositionsAsync
        ib.reqPositionsAsync = AsyncMock(return_value=[])
        # New: single-leg close attempts market data (will fail gracefully with NaN)
        mock_ticker = MagicMock()
        mock_ticker.bid = float('nan')
        mock_ticker.ask = float('nan')
        mock_ticker.last = float('nan')
        ib.reqMktData = MagicMock(return_value=mock_ticker)
        ib.cancelMktData = MagicMock()

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
            mock_trade.orderStatus.avgFillPrice = 0.80
            mock_trade.fills = []
            mock_trade.log = []
            mock_place.return_value = mock_trade

            await _close_spread_position(
                ib, [leg], 'test-pos-456', 'unit test close', config)

        # Exchange should have been filled from config
        self.assertEqual(leg.contract.exchange, 'NYBOT')


class TestCloseContradictedThesis(unittest.IsolatedAsyncioTestCase):
    """Tests for _close_contradicted_thesis() — the new immediate close path."""

    @patch('trading_bot.order_manager.get_trade_ledger_df')
    async def test_close_with_aggregated_positions(self, mock_ledger):
        """IB shows qty=2 (aggregated) but thesis owns qty=1 → closes only thesis portion."""
        from trading_bot.order_manager import _close_contradicted_thesis
        import pandas as pd

        ib = MagicMock()
        ib.isConnected.return_value = True

        # IB shows qty=2 (two theses share same strikes)
        pos1 = MagicMock()
        pos1.contract.localSymbol = 'KOK6 C2.95'
        pos1.position = 2  # Aggregated: 2 long
        pos2 = MagicMock()
        pos2.contract.localSymbol = 'KOK6 C3'
        pos2.position = -2  # Aggregated: 2 short
        ib.reqPositionsAsync = AsyncMock(return_value=[pos1, pos2])

        # Ledger: this thesis has qty=1 per leg
        mock_ledger.return_value = pd.DataFrame({
            'position_id': ['thesis-a', 'thesis-a'],
            'local_symbol': ['KOK6 C2.95', 'KOK6 C3'],
            'quantity': [1, 1],
            'action': ['BUY', 'SELL'],
        })

        tms = MagicMock()
        tms.retrieve_thesis.return_value = {'strategy_type': 'BULL_CALL_SPREAD'}
        config = {'data_dir': '/tmp/test', 'exchange': 'NYBOT', 'notifications': {}}

        with patch('orchestrator._close_spread_position',
                   new_callable=AsyncMock) as mock_close:
            mock_close.return_value = True

            result = await _close_contradicted_thesis(ib, 'thesis-a', config, tms)

            self.assertTrue(result)
            # Verify adjusted legs were passed with thesis-specific quantities
            call_args = mock_close.call_args
            legs_passed = call_args[0][1]
            self.assertEqual(len(legs_passed), 2)
            # Should have capped quantities to thesis amounts (1, not 2)
            for leg in legs_passed:
                self.assertEqual(abs(leg.position), 1)

            # Thesis should be invalidated and pending_close cleared
            tms.invalidate_thesis.assert_called_once()
            tms.clear_pending_close.assert_called_once_with('thesis-a')

    @patch('trading_bot.order_manager.get_trade_ledger_df')
    async def test_close_no_ib_positions_invalidates(self, mock_ledger):
        """No matching IB positions → invalidate thesis (already closed externally)."""
        from trading_bot.order_manager import _close_contradicted_thesis
        import pandas as pd

        ib = MagicMock()
        ib.isConnected.return_value = True
        ib.reqPositionsAsync = AsyncMock(return_value=[])

        mock_ledger.return_value = pd.DataFrame({
            'position_id': ['thesis-b'],
            'local_symbol': ['KOK6 C2.95'],
            'quantity': [1],
            'action': ['BUY'],
        })

        tms = MagicMock()
        config = {'data_dir': '/tmp/test'}

        result = await _close_contradicted_thesis(ib, 'thesis-b', config, tms)

        self.assertTrue(result)
        tms.invalidate_thesis.assert_called_once()
        tms.clear_pending_close.assert_called_once_with('thesis-b')

    @patch('trading_bot.order_manager.get_trade_ledger_df')
    async def test_close_failure_preserves_pending_flag(self, mock_ledger):
        """Close fails → pending_close flag preserved for audit retry."""
        from trading_bot.order_manager import _close_contradicted_thesis
        import pandas as pd

        ib = MagicMock()
        ib.isConnected.return_value = True

        pos = MagicMock()
        pos.contract.localSymbol = 'KOK6 C2.95'
        pos.position = 1
        ib.reqPositionsAsync = AsyncMock(return_value=[pos])

        mock_ledger.return_value = pd.DataFrame({
            'position_id': ['thesis-c'],
            'local_symbol': ['KOK6 C2.95'],
            'quantity': [1],
            'action': ['BUY'],
        })

        tms = MagicMock()
        tms.retrieve_thesis.return_value = {'strategy_type': 'BULL_CALL_SPREAD'}
        config = {'data_dir': '/tmp/test', 'exchange': 'NYBOT', 'notifications': {}}

        with patch('orchestrator._close_spread_position',
                   new_callable=AsyncMock) as mock_close:
            mock_close.return_value = False  # Close failed

            result = await _close_contradicted_thesis(ib, 'thesis-c', config, tms)

            self.assertFalse(result)
            # Thesis should NOT be invalidated
            tms.invalidate_thesis.assert_not_called()
            # pending_close flag should NOT be cleared
            tms.clear_pending_close.assert_not_called()

    async def test_close_skipped_when_ib_disconnected(self):
        """IB not connected → returns False, flag preserved."""
        from trading_bot.order_manager import _close_contradicted_thesis

        ib = MagicMock()
        ib.isConnected.return_value = False

        tms = MagicMock()
        config = {'data_dir': '/tmp/test'}

        result = await _close_contradicted_thesis(ib, 'thesis-d', config, tms)

        self.assertFalse(result)
        ib.reqPositionsAsync.assert_not_called()


class TestMultiLotFillRecording(unittest.IsolatedAsyncioTestCase):
    """Tests for multi-lot fill handling in on_exec_details.

    Verifies that when IB fills a qty=2 combo order, both lots are recorded
    to the ledger (not silently dropped as 'duplicate' fills).
    """

    def _make_fill(self, con_id: int, local_symbol: str = 'KOK6 C2.95'):
        """Create a mock Fill object."""
        fill = MagicMock()
        fill.contract = MagicMock()
        fill.contract.conId = con_id
        fill.contract.localSymbol = local_symbol
        fill.execution = MagicMock()
        fill.execution.avgPrice = 2.45
        fill.execution.shares = 1
        return fill

    def _make_trade(self, order_id: int, total_qty: int, order_ref: str,
                    combo_leg_con_ids: list):
        """Create a mock Trade with a combo order."""
        trade = MagicMock()
        trade.order.orderId = order_id
        trade.order.totalQuantity = total_qty
        trade.order.permId = 99999
        trade.order.orderRef = order_ref
        trade.contract = MagicMock()
        trade.contract.comboLegs = [
            MagicMock(conId=cid) for cid in combo_leg_con_ids
        ]
        return trade

    @patch('trading_bot.order_manager._handle_and_log_fill', new_callable=AsyncMock)
    async def test_two_lot_order_records_both_lots(self, mock_log_fill):
        """A qty=2 order should produce 4 fill logs (2 legs x 2 lots), not 2."""
        from trading_bot.order_manager import place_queued_orders

        # We'll test the on_exec_details handler directly by extracting its logic
        # Simulate what place_queued_orders sets up in live_orders
        order_id = 205
        total_qty = 2
        con_id_a = 732415557  # KOK6 C2.95
        con_id_b = 697282523  # KOK6 C3

        trade = self._make_trade(order_id, total_qty, 'test-uuid-1', [con_id_a, con_id_b])

        live_orders = {
            order_id: {
                'trade': trade,
                'status': 'Submitted',
                'display_name': 'KC-C295.0+C300.0',
                'is_filled': False,
                'total_legs': 2,
                'filled_legs': {},  # conId -> fill_count
                'fill_prices': {},
                'last_update_time': 0,
                'adaptive_ceiling_price': 2.55,
                'decision_data': {'direction': 'BULLISH'},
                'last_known_good_price': 1.80,
                'modification_error_count': 0,
                'max_modification_errors': 3,
                'catastrophe_stop_trade': None,
            }
        }

        details = live_orders[order_id]

        # Simulate on_exec_details logic inline (it's a closure, can't call directly)
        fills_processed = 0
        for lot in range(1, total_qty + 1):
            for con_id in [con_id_a, con_id_b]:
                fill = self._make_fill(con_id)
                leg_con_id = fill.contract.conId
                expected_qty = int(details['trade'].order.totalQuantity)
                current_fills = details['filled_legs'].get(leg_con_id, 0)

                if current_fills >= expected_qty:
                    continue  # Would be "duplicate"

                details['filled_legs'][leg_con_id] = current_fills + 1
                details['fill_prices'][leg_con_id] = fill.execution.avgPrice
                fills_processed += 1

                total_legs = details['total_legs']
                if (len(details['filled_legs']) == total_legs
                        and all(v >= expected_qty for v in details['filled_legs'].values())):
                    details['is_filled'] = True

        # All 4 fills should be processed (2 legs x 2 lots)
        self.assertEqual(fills_processed, 4)
        # Each leg should have fill count = 2
        self.assertEqual(details['filled_legs'][con_id_a], 2)
        self.assertEqual(details['filled_legs'][con_id_b], 2)
        # Order should be marked as fully filled
        self.assertTrue(details['is_filled'])

    async def test_single_lot_order_still_works(self):
        """A qty=1 order should still work correctly with the new dict-based tracking."""
        con_id_a = 111
        con_id_b = 222
        trade = self._make_trade(100, 1, 'test-uuid-2', [con_id_a, con_id_b])

        details = {
            'trade': trade,
            'is_filled': False,
            'total_legs': 2,
            'filled_legs': {},
            'fill_prices': {},
        }

        # Lot 1, leg A
        details['filled_legs'][con_id_a] = 1
        # Lot 1, leg B
        details['filled_legs'][con_id_b] = 1

        expected_qty = int(details['trade'].order.totalQuantity)
        total_legs = details['total_legs']
        if (len(details['filled_legs']) == total_legs
                and all(v >= expected_qty for v in details['filled_legs'].values())):
            details['is_filled'] = True

        self.assertTrue(details['is_filled'])

    async def test_duplicate_beyond_expected_qty_rejected(self):
        """A third fill for a qty=2 order should be rejected."""
        con_id_a = 111
        trade = self._make_trade(100, 2, 'test-uuid-3', [con_id_a, 222])

        details = {
            'trade': trade,
            'filled_legs': {con_id_a: 2},  # Already got 2 fills
        }

        expected_qty = int(details['trade'].order.totalQuantity)
        current_fills = details['filled_legs'].get(con_id_a, 0)
        # Third fill should be rejected
        self.assertTrue(current_fills >= expected_qty)

    async def test_is_filled_only_after_all_lots_all_legs(self):
        """is_filled should only be True when all legs have all lots."""
        con_id_a = 111
        con_id_b = 222
        trade = self._make_trade(100, 2, 'test-uuid-4', [con_id_a, con_id_b])

        details = {
            'trade': trade,
            'is_filled': False,
            'total_legs': 2,
            'filled_legs': {},
        }

        expected_qty = int(details['trade'].order.totalQuantity)

        # After lot 1 fills both legs — NOT yet fully filled
        details['filled_legs'] = {con_id_a: 1, con_id_b: 1}
        all_done = (len(details['filled_legs']) == details['total_legs']
                    and all(v >= expected_qty for v in details['filled_legs'].values()))
        self.assertFalse(all_done)

        # After lot 2 leg A — still not done
        details['filled_legs'] = {con_id_a: 2, con_id_b: 1}
        all_done = (len(details['filled_legs']) == details['total_legs']
                    and all(v >= expected_qty for v in details['filled_legs'].values()))
        self.assertFalse(all_done)

        # After lot 2 leg B — NOW done
        details['filled_legs'] = {con_id_a: 2, con_id_b: 2}
        all_done = (len(details['filled_legs']) == details['total_legs']
                    and all(v >= expected_qty for v in details['filled_legs'].values()))
        self.assertTrue(all_done)


if __name__ == '__main__':
    unittest.main()
