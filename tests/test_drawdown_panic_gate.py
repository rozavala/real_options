"""Tests for _panic_is_live flag gating in DrawdownGuard and PortfolioRiskGuard.

Covers:
- Loaded PANIC from disk does NOT trigger emergency close
- Loaded PANIC still blocks new entries
- Fresh PANIC evaluation triggers emergency close
- _panic_is_live is False for non-PANIC statuses
- Daily reset clears _panic_is_live
"""

from unittest.mock import patch

import pytest

from conftest import mock_ib, make_drawdown_guard, make_portfolio_risk_guard


# ---------------------------------------------------------------------------
# DrawdownGuard Panic Gate Tests
# ---------------------------------------------------------------------------

class TestDrawdownGuardPanicGate:
    """Tests for C2: _panic_is_live gating on should_panic_close."""

    def test_loaded_panic_does_not_trigger_close(self, tmp_path):
        """C2: Loaded PANIC should NOT trigger emergency close (panic_is_live=False)."""
        guard = make_drawdown_guard(tmp_path, status="PANIC")

        assert guard._panic_is_live is False
        assert guard.state['status'] == "PANIC"
        assert guard.should_panic_close() is False

    def test_loaded_panic_still_blocks_entries(self, tmp_path):
        """C2: Loaded PANIC should still block new entries (conservative)."""
        guard = make_drawdown_guard(tmp_path, status="PANIC")

        assert guard.is_entry_allowed() is False

    @pytest.mark.asyncio
    async def test_fresh_panic_triggers_close(self, tmp_path):
        """C2: Fresh PANIC evaluation should trigger emergency close."""
        guard = make_drawdown_guard(tmp_path, status="NORMAL", starting_equity=100000)

        # NLV = 95500 → drawdown = -4.5% → exceeds panic_pct=4.0%
        ib = mock_ib(95500)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['status'] == "PANIC"
        assert guard._panic_is_live is True
        assert guard.should_panic_close() is True

    @pytest.mark.asyncio
    async def test_panic_is_live_false_when_not_panic(self, tmp_path):
        """C2: _panic_is_live should be False when status is not PANIC."""
        guard = make_drawdown_guard(tmp_path, status="NORMAL", starting_equity=100000)

        # NLV = 98000 → drawdown = -2.0% → WARNING
        ib = mock_ib(98000)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['status'] == "WARNING"
        assert guard._panic_is_live is False

    def test_daily_reset_clears_panic_is_live(self, tmp_path):
        """C2: _reset_daily should clear _panic_is_live."""
        guard = make_drawdown_guard(tmp_path, status="PANIC")
        guard._panic_is_live = True

        guard.state['date'] = '1999-01-01'
        guard._reset_daily()

        assert guard._panic_is_live is False


# ---------------------------------------------------------------------------
# PortfolioRiskGuard Panic Gate Tests
# ---------------------------------------------------------------------------

class TestPortfolioRiskGuardPanicGate:
    """Tests for C1: _panic_is_live flag in PortfolioRiskGuard."""

    def test_loaded_panic_does_not_trigger_close(self, tmp_path):
        """C1: Loaded PANIC from disk should NOT trigger emergency close."""
        guard = make_portfolio_risk_guard(tmp_path, status="PANIC")

        assert guard._panic_is_live is False
        assert guard._status == "PANIC"
        assert guard.should_panic_close() is False

    def test_loaded_panic_blocks_entries(self, tmp_path):
        """C1: Loaded PANIC should block entries (conservative)."""
        guard = make_portfolio_risk_guard(tmp_path, status="PANIC")

        assert guard.is_entry_allowed() is False

    @pytest.mark.asyncio
    async def test_fresh_panic_triggers_close(self, tmp_path):
        """C1: After update_equity freshly evaluates PANIC, should_panic_close is True."""
        guard = make_portfolio_risk_guard(tmp_path, status="NORMAL", starting_equity=100000, current_equity=100000)

        # equity=95500 → drawdown=4.5% → exceeds panic_pct=4.0%
        await guard.update_equity(95500, -4500)

        assert guard._status == "PANIC"
        assert guard._panic_is_live is True
        assert guard.should_panic_close() is True

    @pytest.mark.asyncio
    async def test_panic_is_live_false_for_non_panic(self, tmp_path):
        """C1: _panic_is_live should be False when status is not PANIC."""
        guard = make_portfolio_risk_guard(tmp_path, status="NORMAL", starting_equity=100000, current_equity=100000)

        # equity=98500 → drawdown=1.5% → WARNING
        await guard.update_equity(98500, -1500)

        assert guard._status == "WARNING"
        assert guard._panic_is_live is False

    def test_daily_reset_clears_panic_is_live(self, tmp_path):
        """C1: _reset_daily should clear _panic_is_live."""
        guard = make_portfolio_risk_guard(tmp_path, status="PANIC")
        guard._panic_is_live = True

        guard._state_date = "1999-01-01"
        guard._reset_daily()

        assert guard._panic_is_live is False
