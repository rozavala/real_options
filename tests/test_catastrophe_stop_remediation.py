"""Tests for catastrophe stop fill detection and auto-remediation.

Covers the scenario where GTC catastrophe stop orders fill after a bot restart
but are invisible to reqAllOpenOrdersAsync() due to Gateway cache staleness.
The system must detect untracked naked futures positions and auto-remediate.
"""
import asyncio
import csv
import os
import tempfile
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock IB objects
# ---------------------------------------------------------------------------

def _make_contract(local_symbol='NGK26', symbol='NG', sec_type='FUT', multiplier='10000'):
    c = SimpleNamespace(
        localSymbol=local_symbol,
        symbol=symbol,
        secType=sec_type,
        multiplier=multiplier,
        conId=12345,
    )
    return c


def _make_position(local_symbol='NGK26', symbol='NG', sec_type='FUT',
                   position=1.0, avg_cost=29000.0, multiplier='10000'):
    return SimpleNamespace(
        contract=_make_contract(local_symbol, symbol, sec_type, multiplier),
        position=position,
        avgCost=avg_cost,
        account='U12345',
    )


def _make_phantom_trade(local_symbol='NGK26', symbol='NG', action='BUY',
                        order_ref='CATASTROPHE_NGK26', aux_price=2.90,
                        order_id=100, status='Submitted'):
    order = SimpleNamespace(
        orderRef=order_ref,
        action=action,
        totalQuantity=1,
        auxPrice=aux_price,
        orderId=order_id,
        clientId=42,
    )
    order_status = SimpleNamespace(status=status)
    contract = _make_contract(local_symbol, symbol)
    return SimpleNamespace(
        order=order,
        orderStatus=order_status,
        contract=contract,
    )


def _make_config(ticker='NG'):
    return {
        'commodity': {'ticker': ticker},
        'notifications': {'pushover_user_key': 'test', 'pushover_api_token': 'test'},
        'symbol': ticker,
    }


def _make_ib(positions=None, open_orders=None, last_price=2.95):
    """Create a mock IB client with configurable positions and market data."""
    ib = MagicMock()

    async def _req_positions():
        return positions or []
    ib.reqPositionsAsync = _req_positions

    async def _req_all_open():
        return open_orders or []
    ib.reqAllOpenOrdersAsync = _req_all_open

    async def _qualify(*args):
        return args
    ib.qualifyContractsAsync = _qualify

    ticker_data = SimpleNamespace(last=last_price, close=last_price)
    ib.reqMktData.return_value = ticker_data
    ib.cancelMktData = MagicMock()

    placed_trade = SimpleNamespace(order=SimpleNamespace(orderId=999))
    ib.placeOrder.return_value = placed_trade
    ib.cancelOrder = MagicMock()

    return ib


# ---------------------------------------------------------------------------
# Tests for _check_positions_for_catastrophe_fills
# ---------------------------------------------------------------------------

class TestCheckPositionsForCatastropheFills:
    """Detection logic: match phantom orders to untracked IBKR positions."""

    @pytest.fixture(autouse=True)
    def _tmp_ledger(self, tmp_path):
        """Provide an empty ledger directory for each test."""
        self.ledger_dir = str(tmp_path)
        self.ledger_path = os.path.join(self.ledger_dir, 'trade_ledger.csv')

    async def _check(self, ib, phantoms, config):
        from trading_bot.order_manager import _check_positions_for_catastrophe_fills
        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            return await _check_positions_for_catastrophe_fills(ib, phantoms, config)

    async def test_phantom_order_with_no_fill_stays_harmless(self):
        """No matching positions → empty list (regression: existing behavior)."""
        ib = _make_ib(positions=[])
        phantom = _make_phantom_trade()
        result = await self._check(ib, [phantom], _make_config())
        assert result == []

    async def test_phantom_order_with_matching_position_detected(self):
        """BUY stop + long FUT position on same localSymbol → detected as fill."""
        pos = _make_position(local_symbol='NGK26', position=1.0, avg_cost=29000.0)
        ib = _make_ib(positions=[pos])
        phantom = _make_phantom_trade(local_symbol='NGK26', action='BUY')
        result = await self._check(ib, [phantom], _make_config())

        assert len(result) == 1
        assert result[0]['localSymbol'] == 'NGK26'
        assert result[0]['action'] == 'BUY'
        # fill_price = avgCost / multiplier = 29000 / 10000 = 2.90
        assert result[0]['fill_price'] == pytest.approx(2.90)

    async def test_sell_stop_short_position_detected(self):
        """SELL stop + short FUT position → detected."""
        pos = _make_position(local_symbol='KCK26', symbol='KC', position=-1.0,
                             avg_cost=250000.0, multiplier='37500')
        ib = _make_ib(positions=[pos])
        phantom = _make_phantom_trade(local_symbol='KCK26', symbol='KC', action='SELL')
        result = await self._check(ib, [phantom], _make_config('KC'))

        assert len(result) == 1
        assert result[0]['action'] == 'SELL'
        assert result[0]['fill_price'] == pytest.approx(250000.0 / 37500)

    async def test_false_positive_prevented_option_position(self):
        """Position with matching symbol but secType='FOP' → NOT flagged."""
        pos = _make_position(local_symbol='NGK26', sec_type='FOP', position=1.0)
        ib = _make_ib(positions=[pos])
        phantom = _make_phantom_trade(local_symbol='NGK26', action='BUY')
        result = await self._check(ib, [phantom], _make_config())
        assert result == []

    async def test_false_positive_prevented_different_local_symbol(self):
        """NG position with different localSymbol (NGM26 vs NGK26) → NOT flagged."""
        pos = _make_position(local_symbol='NGM26', position=1.0)
        ib = _make_ib(positions=[pos])
        phantom = _make_phantom_trade(local_symbol='NGK26', action='BUY')
        result = await self._check(ib, [phantom], _make_config())
        assert result == []

    async def test_false_positive_prevented_position_in_ledger(self):
        """Position already tracked in ledger → NOT flagged."""
        # Write a ledger entry for NGK26
        with open(self.ledger_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['local_symbol', 'action', 'quantity'])
            w.writeheader()
            w.writerow({'local_symbol': 'NGK26', 'action': 'BUY', 'quantity': 1})

        pos = _make_position(local_symbol='NGK26', position=1.0)
        ib = _make_ib(positions=[pos])
        phantom = _make_phantom_trade(local_symbol='NGK26', action='BUY')
        result = await self._check(ib, [phantom], _make_config())
        assert result == []

    async def test_multi_commodity_isolation(self):
        """NG phantom doesn't match KC positions."""
        pos = _make_position(local_symbol='KCK26', symbol='KC', position=1.0)
        ib = _make_ib(positions=[pos])
        phantom = _make_phantom_trade(local_symbol='NGK26', symbol='NG', action='BUY')
        result = await self._check(ib, [phantom], _make_config('NG'))
        assert result == []

    async def test_detection_failure_graceful(self):
        """reqPositionsAsync timeout → returns empty list."""
        ib = MagicMock()

        async def _timeout():
            raise asyncio.TimeoutError()
        ib.reqPositionsAsync = _timeout

        from trading_bot.order_manager import _check_positions_for_catastrophe_fills
        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            result = await _check_positions_for_catastrophe_fills(
                ib, [_make_phantom_trade()], _make_config()
            )
        assert result == []

    async def test_direction_mismatch_not_flagged(self):
        """BUY stop + SHORT position → NOT flagged (direction mismatch)."""
        pos = _make_position(local_symbol='NGK26', position=-1.0)
        ib = _make_ib(positions=[pos])
        phantom = _make_phantom_trade(local_symbol='NGK26', action='BUY')
        result = await self._check(ib, [phantom], _make_config())
        assert result == []


# ---------------------------------------------------------------------------
# Tests for _remediate_filled_catastrophe_stop
# ---------------------------------------------------------------------------

class TestRemediateFilledCatastropheStop:
    """Remediation logic: ledger, TMS, notification, close order."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Reset dedup guard and provide temp ledger dir."""
        import trading_bot.order_manager as om
        om._remediated_catastrophe_fills.clear()
        self.ledger_dir = str(tmp_path)
        self.ledger_path = os.path.join(self.ledger_dir, 'trade_ledger.csv')

    def _make_fill(self, local_symbol='NGK26', symbol='NG', action='BUY',
                   qty=1, avg_cost=29000.0, fill_price=2.90):
        return {
            'phantom_trade': _make_phantom_trade(local_symbol, symbol, action),
            'position': _make_position(local_symbol, symbol, position=qty, avg_cost=avg_cost),
            'localSymbol': local_symbol,
            'symbol': symbol,
            'action': action,
            'quantity': qty,
            'avg_cost': avg_cost,
            'fill_price': fill_price,
            'contract': _make_contract(local_symbol, symbol),
        }

    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.TransactiveMemory')
    async def test_remediation_logs_to_ledger(self, mock_tms_cls, mock_push):
        """Verify CATASTROPHE_FILL reason appears in ledger CSV."""
        from trading_bot.order_manager import _remediate_filled_catastrophe_stop
        mock_tms = MagicMock()
        mock_tms_cls.return_value = mock_tms

        ib = _make_ib(last_price=2.95)
        fill = self._make_fill()

        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            await _remediate_filled_catastrophe_stop(ib, fill, _make_config())

        assert os.path.exists(self.ledger_path)
        with open(self.ledger_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]['reason'] == 'CATASTROPHE_FILL (auto-detected)'
        assert rows[0]['local_symbol'] == 'NGK26'
        assert rows[0]['action'] == 'BUY'

    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.TransactiveMemory')
    async def test_remediation_sends_p0_notification(self, mock_tms_cls, mock_push):
        """Verify Pushover called with priority=2."""
        from trading_bot.order_manager import _remediate_filled_catastrophe_stop
        mock_tms_cls.return_value = MagicMock()

        ib = _make_ib(last_price=2.95)
        fill = self._make_fill()

        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            await _remediate_filled_catastrophe_stop(ib, fill, _make_config())

        # Should be called at least once with priority=2
        p2_calls = [c for c in mock_push.call_args_list if c.kwargs.get('priority') == 2
                     or (len(c.args) > 4 and c.args[4] == 2)]
        assert len(p2_calls) >= 1

    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.TransactiveMemory')
    async def test_remediation_places_bounded_limit_close(self, mock_tms_cls, mock_push):
        """Verify limit order (not market) with correct action + 1% bound."""
        from trading_bot.order_manager import _remediate_filled_catastrophe_stop
        mock_tms_cls.return_value = MagicMock()

        # Use price high enough that 1% bound survives tick rounding
        # 300.00 * 0.99 = 297.00 — clearly below 300.00 after any tick rounding
        ib = _make_ib(last_price=300.00)
        fill = self._make_fill(action='BUY', fill_price=290.0)  # BUY fill → SELL to close

        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            await _remediate_filled_catastrophe_stop(ib, fill, _make_config())

        ib.placeOrder.assert_called_once()
        args = ib.placeOrder.call_args
        placed_order = args[0][1]
        assert placed_order.action == 'SELL'
        assert placed_order.orderType == 'LMT'
        assert placed_order.tif == 'GTC'
        assert placed_order.outsideRth is True
        assert placed_order.orderRef == 'CATASTROPHE_CLOSE_NGK26'
        # Limit should be ~1% below 300.00 for SELL (tick-rounded)
        assert placed_order.lmtPrice <= 300.00
        assert placed_order.lmtPrice >= 295.00  # not unreasonably far

    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.TransactiveMemory')
    async def test_remediation_records_tms_thesis(self, mock_tms_cls, mock_push):
        """Verify TMS entry created with CATASTROPHE_HEDGE_FILL and immediately invalidated."""
        from trading_bot.order_manager import _remediate_filled_catastrophe_stop
        mock_tms = MagicMock()
        mock_tms_cls.return_value = mock_tms

        ib = _make_ib(last_price=2.95)
        fill = self._make_fill()

        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            await _remediate_filled_catastrophe_stop(ib, fill, _make_config())

        mock_tms.record_trade_thesis.assert_called_once()
        thesis_data = mock_tms.record_trade_thesis.call_args[0][1]
        assert thesis_data['strategy_type'] == 'CATASTROPHE_HEDGE_FILL'

        # Must be immediately invalidated to prevent position audit double-close
        mock_tms.invalidate_thesis.assert_called_once()
        inv_reason = mock_tms.invalidate_thesis.call_args[0][1]
        assert 'catastrophe fill remediation' in inv_reason.lower()

    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.TransactiveMemory')
    async def test_duplicate_remediation_prevented(self, mock_tms_cls, mock_push):
        """Same localSymbol on consecutive audit cycles → second call skips."""
        from trading_bot.order_manager import _remediate_filled_catastrophe_stop
        mock_tms_cls.return_value = MagicMock()

        ib = _make_ib(last_price=2.95)
        fill = self._make_fill(local_symbol='NGK26')

        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            await _remediate_filled_catastrophe_stop(ib, fill, _make_config())
            # Second call — should be deduplicated
            await _remediate_filled_catastrophe_stop(ib, fill, _make_config())

        # placeOrder should be called only once
        assert ib.placeOrder.call_count == 1

    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.TransactiveMemory')
    async def test_no_market_data_falls_back_to_market_order(self, mock_tms_cls, mock_push):
        """If ticker.last is NaN, uses MarketOrder as fallback."""
        from trading_bot.order_manager import _remediate_filled_catastrophe_stop
        mock_tms_cls.return_value = MagicMock()

        ib = _make_ib(last_price=float('nan'))
        # Also make close NaN
        ticker_data = SimpleNamespace(last=float('nan'), close=float('nan'))
        ib.reqMktData.return_value = ticker_data

        fill = self._make_fill()

        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            await _remediate_filled_catastrophe_stop(ib, fill, _make_config())

        ib.placeOrder.assert_called_once()
        placed_order = ib.placeOrder.call_args[0][1]
        assert placed_order.orderType == 'MKT'

    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.TransactiveMemory')
    async def test_remediation_failure_still_notifies(self, mock_tms_cls, mock_push):
        """If close order fails, escalation notification still sent."""
        from trading_bot.order_manager import _remediate_filled_catastrophe_stop
        mock_tms_cls.return_value = MagicMock()

        ib = _make_ib(last_price=2.95)
        ib.placeOrder.side_effect = Exception("Connection lost")

        fill = self._make_fill()

        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            await _remediate_filled_catastrophe_stop(ib, fill, _make_config())

        # Should have escalation notification mentioning failure
        failure_calls = [c for c in mock_push.call_args_list
                         if 'FAILED' in str(c.args[1]) or 'FAILED' in str(c)]
        assert len(failure_calls) >= 1


# ---------------------------------------------------------------------------
# Tests for cleanup function integration
# ---------------------------------------------------------------------------

class TestCleanupIntegration:
    """Integration: _cancel_orphaned_catastrophe_stops flow with fill detection."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        import trading_bot.order_manager as om
        om._remediated_catastrophe_fills.clear()
        self.ledger_dir = str(tmp_path)

    async def test_close_order_not_cancelled_by_cleanup(self):
        """An order with ref CATASTROPHE_CLOSE_NGK26 is skipped by cleanup."""
        close_order_trade = _make_phantom_trade(
            order_ref='CATASTROPHE_CLOSE_NGK26',
            status='Submitted',
        )
        # A regular catastrophe stop that IS eligible for cancel
        stop_trade = _make_phantom_trade(
            order_ref='CATASTROPHE_NGK26',
            status='Submitted',
        )

        ib = _make_ib(open_orders=[close_order_trade, stop_trade], positions=[])
        config = _make_config('NG')

        from trading_bot.order_manager import _cancel_orphaned_catastrophe_stops
        with patch('trading_bot.order_manager._check_positions_for_catastrophe_fills',
                   new_callable=AsyncMock, return_value=[]):
            await _cancel_orphaned_catastrophe_stops(ib, config)

        # cancelOrder should be called for the stop, NOT for the close order
        cancel_calls = ib.cancelOrder.call_args_list
        cancelled_refs = [c.args[0].orderRef for c in cancel_calls]
        assert 'CATASTROPHE_NGK26' in cancelled_refs
        assert 'CATASTROPHE_CLOSE_NGK26' not in cancelled_refs

    async def test_position_audit_does_not_double_close(self):
        """After remediation creates invalidated TMS thesis, position audit won't double-close."""
        from trading_bot.order_manager import _remediate_filled_catastrophe_stop
        import trading_bot.order_manager as om

        mock_tms = MagicMock()
        fill = {
            'phantom_trade': _make_phantom_trade(),
            'position': _make_position(),
            'localSymbol': 'NGK26',
            'symbol': 'NG',
            'action': 'BUY',
            'quantity': 1,
            'avg_cost': 29000.0,
            'fill_price': 2.90,
            'contract': _make_contract(),
        }

        with patch('trading_bot.order_manager.TransactiveMemory', return_value=mock_tms), \
             patch('trading_bot.order_manager.send_pushover_notification'), \
             patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            ib = _make_ib(last_price=2.95)
            await _remediate_filled_catastrophe_stop(ib, fill, _make_config())

        # Verify thesis was immediately invalidated (active=false)
        mock_tms.record_trade_thesis.assert_called_once()
        mock_tms.invalidate_thesis.assert_called_once()
        thesis_id = mock_tms.invalidate_thesis.call_args[0][0]
        assert thesis_id == 'catastrophe_NGK26'


# ---------------------------------------------------------------------------
# Tests for _log_catastrophe_fill_to_ledger
# ---------------------------------------------------------------------------

class TestLogCatastropheFillToLedger:
    """Schema-safe CSV append."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.ledger_dir = str(tmp_path)
        self.ledger_path = os.path.join(self.ledger_dir, 'trade_ledger.csv')

    def _make_fill(self):
        return {
            'localSymbol': 'NGK26',
            'action': 'BUY',
            'quantity': 1,
            'fill_price': 2.90,
        }

    def test_ledger_creates_new_file(self):
        """Creates ledger with header if none exists."""
        from trading_bot.order_manager import _log_catastrophe_fill_to_ledger
        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            _log_catastrophe_fill_to_ledger(self._make_fill(), 'pos_001', _make_config())

        assert os.path.exists(self.ledger_path)
        with open(self.ledger_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]['reason'] == 'CATASTROPHE_FILL (auto-detected)'

    def test_ledger_schema_compatibility(self):
        """Appended row matches existing CSV header with all columns."""
        # Write existing ledger with extra columns
        header = ['timestamp', 'position_id', 'combo_id', 'local_symbol', 'action',
                  'quantity', 'avg_fill_price', 'strike', 'right', 'total_value_usd',
                  'reason', 'exchange', 'currency', 'multiplier', 'sec_type']
        with open(self.ledger_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerow({h: 'existing' for h in header})

        from trading_bot.order_manager import _log_catastrophe_fill_to_ledger
        with patch('trading_bot.utils._get_data_dir', return_value=self.ledger_dir):
            _log_catastrophe_fill_to_ledger(self._make_fill(), 'pos_002', _make_config())

        with open(self.ledger_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        new_row = rows[1]
        # All header columns must be present (no KeyError, no extra columns)
        assert set(new_row.keys()) == set(header)
        # Known fields populated, unknown fields default to ''
        assert new_row['local_symbol'] == 'NGK26'
        assert new_row['exchange'] == ''
        assert new_row['currency'] == ''
