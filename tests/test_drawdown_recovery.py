"""
Tests for PANIC/HALT recovery de-escalation in DrawdownGuard and PortfolioRiskGuard.

Covers:
1. Status sticks during active drawdown (no recovery)
2. Recovery timer starts when drawdown improves below threshold
3. Recovery timer resets when drawdown worsens
4. Recovery completes after hold period → de-escalate to WARNING
5. Pushover notification sent on recovery
6. Daily reset clears recovery timer
7. Backward-compatible state loading (no recovery_start key)
8. PortfolioRiskGuard mirrors DrawdownGuard recovery behavior
9. Starting equity from previous close (daily_equity.csv)
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# DrawdownGuard Tests
# ---------------------------------------------------------------------------

class TestDrawdownGuardRecovery:
    """Tests for recovery logic in DrawdownGuard."""

    def _make_guard(self, tmpdir, status="PANIC", drawdown=-4.5, starting_equity=100000):
        """Create a DrawdownGuard with pre-set state."""
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard

        data_dir = str(tmpdir)
        config = {
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 1.5,
                'halt_pct': 2.5,
                'panic_pct': 4.0,
                'recovery_pct': 3.0,
                'recovery_hold_minutes': 30,
            },
            'notifications': {'enabled': False},
            'data_dir': data_dir,
        }

        # Pre-write state so constructor loads it
        state_file = os.path.join(data_dir, 'drawdown_state.json')
        os.makedirs(data_dir, exist_ok=True)
        state = {
            "status": status,
            "current_drawdown_pct": drawdown,
            "starting_equity": starting_equity,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        # Write daily_equity.csv so load_prev_close() returns starting_equity
        # (starting_equity is reset to 0.0 on load, then re-derived from this file)
        equity_file = os.path.join(data_dir, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            f.write(f"2026-01-01 17:00:00,{starting_equity}\n")

        guard = DrawdownGuard(config)
        return guard

    def _mock_ib(self, net_liq):
        """Create a mock IB that returns the given NetLiquidation."""
        ib = MagicMock()
        summary_item = MagicMock()
        summary_item.tag = 'NetLiquidation'
        summary_item.currency = 'USD'
        summary_item.value = str(net_liq)
        ib.accountSummaryAsync = AsyncMock(return_value=[summary_item])
        return ib

    @pytest.mark.asyncio
    async def test_sticks_during_active_drawdown(self, tmp_path):
        """PANIC status should stick when drawdown is still severe."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        ib = self._mock_ib(95000)  # -5% drawdown, still bad

        result = await guard.update_pnl(ib)

        assert result == "PANIC"
        assert guard._recovery_start is None

    @pytest.mark.asyncio
    async def test_halt_sticks_during_active_drawdown(self, tmp_path):
        """HALT status should stick when drawdown is still above recovery threshold."""
        guard = self._make_guard(tmp_path, status="HALT", starting_equity=100000)
        ib = self._mock_ib(96500)  # -3.5% drawdown, above recovery_pct=3% but below panic_pct=4%

        result = await guard.update_pnl(ib)

        assert result == "HALT"
        assert guard._recovery_start is None

    @pytest.mark.asyncio
    async def test_recovery_timer_starts(self, tmp_path):
        """Recovery timer should start when drawdown improves below recovery_pct."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        ib = self._mock_ib(98000)  # -2% drawdown, below recovery_pct=3%

        result = await guard.update_pnl(ib)

        assert result == "PANIC"  # Still holds during observation
        assert guard._recovery_start is not None

    @pytest.mark.asyncio
    async def test_recovery_timer_resets_on_worsening(self, tmp_path):
        """Recovery timer should reset if drawdown worsens again."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        guard._recovery_start = datetime.now(timezone.utc).isoformat()

        # Drawdown worsens back to -4% (abs > recovery_pct=3%)
        ib = self._mock_ib(96000)

        result = await guard.update_pnl(ib)

        assert result == "PANIC"
        assert guard._recovery_start is None

    @pytest.mark.asyncio
    async def test_recovery_completes_after_hold_period(self, tmp_path):
        """After sustained improvement for hold period, should de-escalate to WARNING."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        # Set recovery_start to 31 minutes ago
        guard._recovery_start = (
            datetime.now(timezone.utc) - timedelta(minutes=31)
        ).isoformat()

        ib = self._mock_ib(98000)  # -2% drawdown, below recovery_pct=3%

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification') as mock_notify:
            result = await guard.update_pnl(ib)

        assert result == "WARNING"
        assert guard._recovery_start is None
        # Recovery notification is the first call (status-change notification may also fire)
        assert mock_notify.call_count >= 1
        recovery_call = mock_notify.call_args_list[0]
        assert "Recovery" in recovery_call[0][1]

    @pytest.mark.asyncio
    async def test_recovery_notification_sent(self, tmp_path):
        """Pushover notification should be sent when recovery completes."""
        guard = self._make_guard(tmp_path, status="HALT", starting_equity=100000)
        guard._recovery_start = (
            datetime.now(timezone.utc) - timedelta(minutes=45)
        ).isoformat()

        ib = self._mock_ib(98500)  # -1.5% drawdown

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification') as mock_notify:
            result = await guard.update_pnl(ib)

        assert result == "WARNING"
        # Recovery notification is the first call
        assert mock_notify.call_count >= 1
        recovery_call = mock_notify.call_args_list[0]
        title = recovery_call[0][1]
        body = recovery_call[0][2]
        assert "HALT" in title
        assert "WARNING" in title
        assert "resumed with caution" in body

    def test_daily_reset_clears_recovery_timer(self, tmp_path):
        """Daily reset should clear recovery timer."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        guard._recovery_start = datetime.now(timezone.utc).isoformat()

        # Simulate new day by changing the state date
        guard.state['date'] = '1999-01-01'
        guard._reset_daily()

        assert guard._recovery_start is None
        assert guard.state['status'] == "NORMAL"

    def test_backward_compatible_state_loading(self, tmp_path):
        """Old state files without recovery_start should load fine."""
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard

        data_dir = str(tmp_path)
        state_file = os.path.join(data_dir, 'drawdown_state.json')
        os.makedirs(data_dir, exist_ok=True)
        # Old-format state (no recovery_start key)
        state = {
            "status": "HALT",
            "current_drawdown_pct": -3.0,
            "starting_equity": 100000,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        config = {
            'drawdown_circuit_breaker': {'enabled': True},
            'notifications': {},
            'data_dir': data_dir,
        }
        guard = DrawdownGuard(config)

        assert guard.state['status'] == "HALT"
        assert guard._recovery_start is None

    @pytest.mark.asyncio
    async def test_halt_to_panic_escalation_still_works(self, tmp_path):
        """HALT should still escalate to PANIC when drawdown worsens past panic threshold."""
        guard = self._make_guard(tmp_path, status="HALT", starting_equity=100000)
        ib = self._mock_ib(95500)  # -4.5% drawdown, past panic_pct=4%

        result = await guard.update_pnl(ib)

        assert result == "PANIC"
        assert guard._recovery_start is None

    @pytest.mark.asyncio
    async def test_recovery_state_persisted(self, tmp_path):
        """Recovery timer should be persisted to disk."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        ib = self._mock_ib(98000)  # Triggers recovery timer start

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard._recovery_start is not None

        # Read state file and verify recovery_start is saved
        with open(guard.state_file, 'r') as f:
            saved = json.load(f)
        assert 'recovery_start' in saved
        assert saved['recovery_start'] == guard._recovery_start


# ---------------------------------------------------------------------------
# PortfolioRiskGuard Tests
# ---------------------------------------------------------------------------

class TestPortfolioRiskGuardRecovery:
    """Tests for recovery logic in PortfolioRiskGuard."""

    def _make_guard(self, tmpdir, status="PANIC", starting_equity=100000, current_equity=95000):
        """Create a PortfolioRiskGuard with pre-set state."""
        from trading_bot.shared_context import PortfolioRiskGuard

        data_dir = str(tmpdir)
        config = {
            'data_dir_root': data_dir,
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 1.5,
                'halt_pct': 2.5,
                'panic_pct': 4.0,
                'recovery_pct': 3.0,
                'recovery_hold_minutes': 30,
            },
            'notifications': {'enabled': False},
        }

        # Pre-write state
        state_file = os.path.join(data_dir, 'portfolio_risk_state.json')
        os.makedirs(data_dir, exist_ok=True)
        state = {
            "status": status,
            "peak_equity": starting_equity,
            "current_equity": current_equity,
            "starting_equity": starting_equity,
            "daily_pnl": current_equity - starting_equity,
            "positions": {},
            "margin": {},
            "date": datetime.now(timezone.utc).date().isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        # Write daily_equity.csv in a subdirectory so _load_prev_close() finds it
        # (starting_equity is reset to 0.0 on load, then re-derived from this file)
        subdir = os.path.join(data_dir, 'KC')
        os.makedirs(subdir, exist_ok=True)
        equity_file = os.path.join(subdir, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            f.write(f"2026-01-01 17:00:00,{starting_equity}\n")

        guard = PortfolioRiskGuard(config=config)
        return guard

    @pytest.mark.asyncio
    async def test_sticks_during_active_drawdown(self, tmp_path):
        """PANIC should stick when drawdown is still severe."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        # equity=95000 → drawdown_pct=5.0%, above recovery_pct=3.0%
        await guard.update_equity(95000, -5000)

        assert guard._status == "PANIC"
        assert guard._recovery_start is None

    @pytest.mark.asyncio
    async def test_recovery_timer_starts(self, tmp_path):
        """Recovery timer should start when drawdown improves below recovery_pct."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        # equity=98000 → drawdown_pct=2.0%, below recovery_pct=3.0%
        await guard.update_equity(98000, -2000)

        assert guard._status == "PANIC"  # Holds during observation
        assert guard._recovery_start is not None

    @pytest.mark.asyncio
    async def test_recovery_timer_resets_on_worsening(self, tmp_path):
        """Recovery timer should reset if drawdown worsens."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        guard._recovery_start = datetime.now(timezone.utc).isoformat()

        # equity=96000 → drawdown_pct=4.0%, above recovery_pct=3.0%
        await guard.update_equity(96000, -4000)

        assert guard._status == "PANIC"
        assert guard._recovery_start is None

    @pytest.mark.asyncio
    async def test_recovery_completes_after_hold_period(self, tmp_path):
        """After sustained improvement, should de-escalate to WARNING."""
        guard = self._make_guard(tmp_path, status="HALT", starting_equity=100000)
        guard._recovery_start = (
            datetime.now(timezone.utc) - timedelta(minutes=35)
        ).isoformat()

        with patch('notifications.send_pushover_notification') as mock_notify:
            # equity=98000 → drawdown_pct=2.0%, below recovery_pct=3.0%
            await guard.update_equity(98000, -2000)

        assert guard._status == "WARNING"
        assert guard._recovery_start is None
        mock_notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_halt_to_panic_escalation(self, tmp_path):
        """HALT should escalate to PANIC when drawdown exceeds panic threshold."""
        guard = self._make_guard(tmp_path, status="HALT", starting_equity=100000)
        # equity=95500 → drawdown_pct=4.5%, above panic_pct=4.0%
        await guard.update_equity(95500, -4500)

        assert guard._status == "PANIC"

    @pytest.mark.asyncio
    async def test_backward_compatible_state_loading(self, tmp_path):
        """Old state files without recovery_start should load fine."""
        from trading_bot.shared_context import PortfolioRiskGuard

        data_dir = str(tmp_path)
        state_file = os.path.join(data_dir, 'portfolio_risk_state.json')
        os.makedirs(data_dir, exist_ok=True)
        state = {
            "status": "HALT",
            "peak_equity": 100000,
            "current_equity": 97000,
            "starting_equity": 100000,
            "daily_pnl": -3000,
            "positions": {},
            "margin": {},
            "date": datetime.now(timezone.utc).date().isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        config = {
            'data_dir_root': data_dir,
            'drawdown_circuit_breaker': {'enabled': True},
            'notifications': {},
        }
        guard = PortfolioRiskGuard(config=config)

        assert guard._status == "HALT"
        assert guard._recovery_start is None

    @pytest.mark.asyncio
    async def test_recovery_state_persisted(self, tmp_path):
        """Recovery start time should be saved to disk."""
        guard = self._make_guard(tmp_path, status="PANIC", starting_equity=100000)
        # equity=98000 → drawdown_pct=2.0%, triggers recovery timer
        await guard.update_equity(98000, -2000)

        assert guard._recovery_start is not None

        # Read persisted state
        state_file = os.path.join(str(tmp_path), 'portfolio_risk_state.json')
        with open(state_file, 'r') as f:
            saved = json.load(f)
        assert 'recovery_start' in saved
        assert saved['recovery_start'] == guard._recovery_start

    @pytest.mark.asyncio
    async def test_recovery_in_progress_holds_status(self, tmp_path):
        """During recovery observation (timer running, not yet elapsed), status should hold."""
        guard = self._make_guard(tmp_path, status="HALT", starting_equity=100000)
        guard._recovery_start = (
            datetime.now(timezone.utc) - timedelta(minutes=10)
        ).isoformat()

        # equity=98500 → drawdown_pct=1.5%, below recovery_pct=3.0%
        await guard.update_equity(98500, -1500)

        assert guard._status == "HALT"  # Still holding
        assert guard._recovery_start is not None  # Timer still running

    @pytest.mark.asyncio
    async def test_daily_reset_via_update_equity(self, tmp_path):
        """PortfolioRiskGuard should reset PANIC→NORMAL on date rollover."""
        from trading_bot.shared_context import PortfolioRiskGuard

        data_dir = str(tmp_path)
        config = {
            'data_dir_root': data_dir,
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 1.5,
                'halt_pct': 2.5,
                'panic_pct': 4.0,
                'recovery_pct': 3.0,
                'recovery_hold_minutes': 30,
            },
            'notifications': {'enabled': False},
        }

        # Pre-write state from "yesterday"
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        state_file = os.path.join(data_dir, 'portfolio_risk_state.json')
        os.makedirs(data_dir, exist_ok=True)
        state = {
            "status": "PANIC",
            "peak_equity": 100000,
            "current_equity": 95000,
            "starting_equity": 100000,
            "daily_pnl": -5000,
            "positions": {"KC": 2},
            "margin": {"KC": 3000},
            "date": yesterday,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "recovery_start": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        # Guard loads but sees stale date → bootstraps, _state_date empty
        guard = PortfolioRiskGuard(config=config)

        # Simulate first equity update of the new day
        # _reset_daily() should fire inside update_equity(), resetting to NORMAL
        await guard.update_equity(95000, 0)

        assert guard._status != "PANIC", "PANIC should not persist across day boundary"
        assert guard._recovery_start is None, "Recovery timer should clear on daily reset"
        assert guard._state_date == datetime.now(timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Previous Close (daily_equity.csv) Tests
# ---------------------------------------------------------------------------

class TestDrawdownGuardPrevClose:
    """Tests for starting_equity initialization from daily_equity.csv."""

    def _mock_ib(self, net_liq):
        ib = MagicMock()
        summary_item = MagicMock()
        summary_item.tag = 'NetLiquidation'
        summary_item.currency = 'USD'
        summary_item.value = str(net_liq)
        ib.accountSummaryAsync = AsyncMock(return_value=[summary_item])
        return ib

    def _write_equity_csv(self, data_dir, rows):
        """Write daily_equity.csv with given rows [(timestamp, value), ...]."""
        equity_file = os.path.join(data_dir, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            for ts, val in rows:
                f.write(f"{ts},{val}\n")

    def _make_guard(self, data_dir):
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard
        config = {
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 1.5,
                'halt_pct': 2.5,
                'panic_pct': 4.0,
            },
            'notifications': {'enabled': False},
            'data_dir': str(data_dir),
        }
        return DrawdownGuard(config)

    @pytest.mark.asyncio
    async def test_starting_equity_from_prev_close(self, tmp_path):
        """starting_equity should come from daily_equity.csv, not live NLV."""
        self._write_equity_csv(tmp_path, [
            ("2026-03-02 17:00:00", "35000.00"),
            ("2026-03-03 17:00:00", "35145.06"),
        ])
        guard = self._make_guard(tmp_path)
        ib = self._mock_ib(34000)  # Live NLV is lower

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['starting_equity'] == 35145.06
        # Should detect drawdown: (34000-35145)/35145 = -3.26% → HALT
        assert guard.state['status'] == "HALT"

    @pytest.mark.asyncio
    async def test_falls_back_to_live_nlv_when_csv_missing(self, tmp_path):
        """If daily_equity.csv doesn't exist, fall back to live NLV."""
        guard = self._make_guard(tmp_path)
        ib = self._mock_ib(35000)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['starting_equity'] == 35000

    @pytest.mark.asyncio
    async def test_falls_back_to_live_nlv_when_csv_empty(self, tmp_path):
        """If daily_equity.csv is empty, fall back to live NLV."""
        equity_file = os.path.join(tmp_path, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            f.write("")
        guard = self._make_guard(tmp_path)
        ib = self._mock_ib(35000)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['starting_equity'] == 35000

    @pytest.mark.asyncio
    async def test_handles_corrupt_csv(self, tmp_path):
        """Corrupt CSV should fall back to live NLV gracefully."""
        equity_file = os.path.join(tmp_path, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            f.write("not,a,valid\ncsv,file,here\n")
        guard = self._make_guard(tmp_path)
        ib = self._mock_ib(35000)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        # Should fall back gracefully (corrupt value raises ValueError)
        assert guard.state['starting_equity'] == 35000

    @pytest.mark.asyncio
    async def test_all_engines_get_same_starting_equity(self, tmp_path):
        """Multiple guards with same daily_equity.csv get identical starting_equity."""
        self._write_equity_csv(tmp_path, [
            ("2026-03-03 17:00:00", "35145.06"),
        ])

        guard1 = self._make_guard(tmp_path)
        guard2 = self._make_guard(tmp_path)
        guard3 = self._make_guard(tmp_path)

        # Different live NLVs (simulating staggered startup)
        ib1 = self._mock_ib(35100)
        ib2 = self._mock_ib(34800)
        ib3 = self._mock_ib(34500)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard1.update_pnl(ib1)
            await guard2.update_pnl(ib2)
            await guard3.update_pnl(ib3)

        # All should use the same prev close, not their respective live NLVs
        assert guard1.state['starting_equity'] == 35145.06
        assert guard2.state['starting_equity'] == 35145.06
        assert guard3.state['starting_equity'] == 35145.06

    @pytest.mark.asyncio
    async def test_first_call_evaluates_thresholds(self, tmp_path):
        """First update_pnl should evaluate thresholds (not just return NORMAL)."""
        self._write_equity_csv(tmp_path, [
            ("2026-03-03 17:00:00", "35000.00"),
        ])
        guard = self._make_guard(tmp_path)
        ib = self._mock_ib(33000)  # -5.7% drawdown

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            result = await guard.update_pnl(ib)

        # Should detect PANIC immediately, not return NORMAL
        assert result == "PANIC"


class TestPortfolioRiskGuardPrevClose:
    """Tests for PortfolioRiskGuard starting_equity from daily_equity.csv."""

    def _make_guard(self, data_dir):
        from trading_bot.shared_context import PortfolioRiskGuard
        config = {
            'data_dir_root': str(data_dir),
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 1.5,
                'halt_pct': 2.5,
                'panic_pct': 4.0,
                'recovery_pct': 3.0,
                'recovery_hold_minutes': 30,
            },
            'notifications': {'enabled': False},
        }
        return PortfolioRiskGuard(config=config)

    @pytest.mark.asyncio
    async def test_starting_equity_from_prev_close(self, tmp_path):
        """PortfolioRiskGuard should use prev close from daily_equity.csv."""
        # Create a commodity subdir with daily_equity.csv
        kc_dir = tmp_path / "KC"
        kc_dir.mkdir()
        equity_file = kc_dir / "daily_equity.csv"
        equity_file.write_text("2026-03-03 17:00:00,35145.06\n")

        guard = self._make_guard(tmp_path)
        await guard.update_equity(34000, -1145)

        assert guard._starting_equity == 35145.06

    @pytest.mark.asyncio
    async def test_falls_back_to_live_equity(self, tmp_path):
        """Without daily_equity.csv, should fall back to live equity."""
        guard = self._make_guard(tmp_path)
        await guard.update_equity(35000, 0)

        assert guard._starting_equity == 35000

    @pytest.mark.asyncio
    async def test_daily_reset_uses_prev_close(self, tmp_path):
        """On date rollover, _reset_daily should prefer prev close."""
        kc_dir = tmp_path / "KC"
        kc_dir.mkdir()
        equity_file = kc_dir / "daily_equity.csv"
        equity_file.write_text("2026-03-03 17:00:00,35145.06\n")

        guard = self._make_guard(tmp_path)
        guard._starting_equity = 34000
        guard._current_equity = 34500
        guard._peak_equity = 35000
        guard._state_date = "2026-03-03"  # Yesterday
        guard._status = "HALT"

        await guard.update_equity(34200, -945)

        # Should have reset and used prev close
        assert guard._starting_equity == 35145.06


# ---------------------------------------------------------------------------
# Workstream A/C/E: Starting Equity Reset, Panic Gate, Recovery Oscillation
# ---------------------------------------------------------------------------

class TestDrawdownGuardStartingEquityReset:
    """Tests for A2: starting_equity forced to 0.0 on load, re-derived from prev close."""

    def _mock_ib(self, net_liq):
        ib = MagicMock()
        summary_item = MagicMock()
        summary_item.tag = 'NetLiquidation'
        summary_item.currency = 'USD'
        summary_item.value = str(net_liq)
        ib.accountSummaryAsync = AsyncMock(return_value=[summary_item])
        return ib

    def test_load_state_resets_starting_equity_to_zero(self, tmp_path):
        """A2: _load_state must reset starting_equity to 0.0 regardless of persisted value."""
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard

        data_dir = str(tmp_path)
        state_file = os.path.join(data_dir, 'drawdown_state.json')
        os.makedirs(data_dir, exist_ok=True)
        state = {
            "status": "HALT",
            "current_drawdown_pct": -3.0,
            "starting_equity": 35182.91,  # Stale value from mid-day restart
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        config = {
            'drawdown_circuit_breaker': {'enabled': True},
            'notifications': {},
            'data_dir': data_dir,
        }
        guard = DrawdownGuard(config)

        # starting_equity should be 0.0 after load, not the persisted value
        assert guard.state['starting_equity'] == 0.0
        # But status should still be loaded
        assert guard.state['status'] == "HALT"

    @pytest.mark.asyncio
    async def test_starting_equity_rederived_from_prev_close_after_reset(self, tmp_path):
        """A2 + a897d2d: load→reset to 0→update_pnl→re-derives from daily_equity.csv."""
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard

        data_dir = str(tmp_path)
        os.makedirs(data_dir, exist_ok=True)

        # Write stale state with wrong starting_equity (from mid-day NLV)
        state_file = os.path.join(data_dir, 'drawdown_state.json')
        state = {
            "status": "HALT",
            "current_drawdown_pct": -3.0,
            "starting_equity": 35182.91,  # Wrong: was set from live NLV
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        # Write daily_equity.csv with the correct prev close
        equity_file = os.path.join(data_dir, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            f.write("2026-03-03 17:00:00,35145.06\n")

        config = {
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 2.0,
                'halt_pct': 4.0,
                'panic_pct': 6.0,
            },
            'notifications': {'enabled': False},
            'data_dir': data_dir,
        }
        guard = DrawdownGuard(config)

        # Verify starting_equity was reset to 0 on load
        assert guard.state['starting_equity'] == 0.0

        # Now call update_pnl — should re-derive from daily_equity.csv
        ib = self._mock_ib(33000)  # NLV = $33,000

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        # Should use prev close, NOT the stale persisted value
        assert guard.state['starting_equity'] == 35145.06
        # Drawdown: (33000 - 35145.06) / 35145.06 = -6.10% → PANIC
        assert guard.state['status'] == "PANIC"

    @pytest.mark.asyncio
    async def test_zero_guard_prevents_division_by_zero(self, tmp_path):
        """A3: update_pnl returns current status when starting_equity is 0."""
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard

        data_dir = str(tmp_path)
        os.makedirs(data_dir, exist_ok=True)

        config = {
            'drawdown_circuit_breaker': {'enabled': True},
            'notifications': {},
            'data_dir': data_dir,
        }
        guard = DrawdownGuard(config)
        guard.state['starting_equity'] = 0.0
        guard.state['status'] = "HALT"

        # Mock IB that returns 0 NLV (load_prev_close also returns None → stays 0)
        ib = MagicMock()
        summary_item = MagicMock()
        summary_item.tag = 'NetLiquidation'
        summary_item.currency = 'USD'
        summary_item.value = '0'
        ib.accountSummaryAsync = AsyncMock(return_value=[summary_item])

        result = await guard.update_pnl(ib)
        assert result == "HALT"  # Returns current status without crash


class TestDrawdownGuardPanicGate:
    """Tests for C2: _panic_is_live gating on should_panic_close."""

    def _make_guard(self, tmpdir, status="PANIC", starting_equity=100000):
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard

        data_dir = str(tmpdir)
        config = {
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 1.5,
                'halt_pct': 2.5,
                'panic_pct': 4.0,
            },
            'notifications': {'enabled': False},
            'data_dir': data_dir,
        }

        state_file = os.path.join(data_dir, 'drawdown_state.json')
        os.makedirs(data_dir, exist_ok=True)
        state = {
            "status": status,
            "current_drawdown_pct": -5.0,
            "starting_equity": starting_equity,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        equity_file = os.path.join(data_dir, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            f.write(f"2026-01-01 17:00:00,{starting_equity}\n")

        return DrawdownGuard(config)

    def _mock_ib(self, net_liq):
        ib = MagicMock()
        summary_item = MagicMock()
        summary_item.tag = 'NetLiquidation'
        summary_item.currency = 'USD'
        summary_item.value = str(net_liq)
        ib.accountSummaryAsync = AsyncMock(return_value=[summary_item])
        return ib

    def test_loaded_panic_does_not_trigger_close(self, tmp_path):
        """C2: Loaded PANIC should NOT trigger emergency close (panic_is_live=False)."""
        guard = self._make_guard(tmp_path, status="PANIC")

        # panic_is_live should be False after loading from disk
        assert guard._panic_is_live is False
        assert guard.state['status'] == "PANIC"
        assert guard.should_panic_close() is False

    def test_loaded_panic_still_blocks_entries(self, tmp_path):
        """C2: Loaded PANIC should still block new entries (conservative)."""
        guard = self._make_guard(tmp_path, status="PANIC")

        assert guard.is_entry_allowed() is False

    @pytest.mark.asyncio
    async def test_fresh_panic_triggers_close(self, tmp_path):
        """C2: Fresh PANIC evaluation should trigger emergency close."""
        guard = self._make_guard(tmp_path, status="NORMAL", starting_equity=100000)

        # NLV = 95500 → drawdown = -4.5% → exceeds panic_pct=4.0%
        ib = self._mock_ib(95500)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['status'] == "PANIC"
        assert guard._panic_is_live is True
        assert guard.should_panic_close() is True

    @pytest.mark.asyncio
    async def test_panic_is_live_false_when_not_panic(self, tmp_path):
        """C2: _panic_is_live should be False when status is not PANIC."""
        guard = self._make_guard(tmp_path, status="NORMAL", starting_equity=100000)

        # NLV = 98000 → drawdown = -2.0% → WARNING
        ib = self._mock_ib(98000)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['status'] == "WARNING"
        assert guard._panic_is_live is False

    def test_daily_reset_clears_panic_is_live(self, tmp_path):
        """C2: _reset_daily should clear _panic_is_live."""
        guard = self._make_guard(tmp_path, status="PANIC")
        guard._panic_is_live = True

        guard.state['date'] = '1999-01-01'
        guard._reset_daily()

        assert guard._panic_is_live is False


class TestPortfolioRiskGuardStartingEquityReset:
    """Tests for A1: PortfolioRiskGuard starting_equity reset on load."""

    def test_load_state_resets_starting_equity(self, tmp_path):
        """A1: _load_state must reset starting_equity to 0.0."""
        from trading_bot.shared_context import PortfolioRiskGuard

        data_dir = str(tmp_path)
        state_file = os.path.join(data_dir, 'portfolio_risk_state.json')
        os.makedirs(data_dir, exist_ok=True)
        state = {
            "status": "HALT",
            "peak_equity": 35200,
            "current_equity": 33500,
            "starting_equity": 35182.91,  # Stale
            "daily_pnl": -1682.91,
            "positions": {},
            "margin": {},
            "date": datetime.now(timezone.utc).date().isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        config = {
            'data_dir_root': data_dir,
            'drawdown_circuit_breaker': {'enabled': True},
            'notifications': {},
        }
        guard = PortfolioRiskGuard(config=config)

        # starting_equity reset, but status and peak preserved
        assert guard._starting_equity == 0.0
        assert guard._status == "HALT"
        assert guard._peak_equity == 35200

    @pytest.mark.asyncio
    async def test_starting_equity_rederived_from_prev_close(self, tmp_path):
        """A1 + a897d2d: load→reset→update_equity→re-derives from daily_equity.csv."""
        from trading_bot.shared_context import PortfolioRiskGuard

        data_dir = str(tmp_path)
        os.makedirs(data_dir, exist_ok=True)

        # Write stale state
        state_file = os.path.join(data_dir, 'portfolio_risk_state.json')
        state = {
            "status": "HALT",
            "peak_equity": 35200,
            "current_equity": 33500,
            "starting_equity": 35182.91,
            "daily_pnl": -1682.91,
            "positions": {},
            "margin": {},
            "date": datetime.now(timezone.utc).date().isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        # Write daily_equity.csv with correct prev close
        kc_dir = os.path.join(data_dir, 'KC')
        os.makedirs(kc_dir, exist_ok=True)
        with open(os.path.join(kc_dir, 'daily_equity.csv'), 'w') as f:
            f.write("2026-03-03 17:00:00,35145.06\n")

        config = {
            'data_dir_root': data_dir,
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 2.0,
                'halt_pct': 4.0,
                'panic_pct': 6.0,
            },
            'notifications': {'enabled': False},
        }
        guard = PortfolioRiskGuard(config=config)
        assert guard._starting_equity == 0.0

        # update_equity should re-derive from prev close
        await guard.update_equity(33000, -2145)

        assert guard._starting_equity == 35145.06
        # Drawdown: (35145 - 33000) / 35145 = 6.10% → PANIC
        assert guard._status == "PANIC"


class TestPortfolioRiskGuardPanicGate:
    """Tests for C1: _panic_is_live flag in PortfolioRiskGuard."""

    def _make_guard(self, tmpdir, status="PANIC", starting_equity=100000, current_equity=95000):
        from trading_bot.shared_context import PortfolioRiskGuard

        data_dir = str(tmpdir)
        config = {
            'data_dir_root': data_dir,
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 1.5,
                'halt_pct': 2.5,
                'panic_pct': 4.0,
            },
            'notifications': {'enabled': False},
        }

        state_file = os.path.join(data_dir, 'portfolio_risk_state.json')
        os.makedirs(data_dir, exist_ok=True)
        state = {
            "status": status,
            "peak_equity": starting_equity,
            "current_equity": current_equity,
            "starting_equity": starting_equity,
            "daily_pnl": current_equity - starting_equity,
            "positions": {},
            "margin": {},
            "date": datetime.now(timezone.utc).date().isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        kc_dir = os.path.join(data_dir, 'KC')
        os.makedirs(kc_dir, exist_ok=True)
        with open(os.path.join(kc_dir, 'daily_equity.csv'), 'w') as f:
            f.write(f"2026-01-01 17:00:00,{starting_equity}\n")

        return PortfolioRiskGuard(config=config)

    def test_loaded_panic_does_not_trigger_close(self, tmp_path):
        """C1: Loaded PANIC from disk should NOT trigger emergency close."""
        guard = self._make_guard(tmp_path, status="PANIC")

        assert guard._panic_is_live is False
        assert guard._status == "PANIC"
        assert guard.should_panic_close() is False

    def test_loaded_panic_blocks_entries(self, tmp_path):
        """C1: Loaded PANIC should block entries (conservative)."""
        guard = self._make_guard(tmp_path, status="PANIC")

        assert guard.is_entry_allowed() is False

    @pytest.mark.asyncio
    async def test_fresh_panic_triggers_close(self, tmp_path):
        """C1: After update_equity freshly evaluates PANIC, should_panic_close is True."""
        guard = self._make_guard(tmp_path, status="NORMAL", starting_equity=100000, current_equity=100000)

        # equity=95500 → drawdown=4.5% → exceeds panic_pct=4.0%
        await guard.update_equity(95500, -4500)

        assert guard._status == "PANIC"
        assert guard._panic_is_live is True
        assert guard.should_panic_close() is True

    @pytest.mark.asyncio
    async def test_panic_is_live_false_for_non_panic(self, tmp_path):
        """C1: _panic_is_live should be False when status is not PANIC."""
        guard = self._make_guard(tmp_path, status="NORMAL", starting_equity=100000, current_equity=100000)

        # equity=98500 → drawdown=1.5% → WARNING
        await guard.update_equity(98500, -1500)

        assert guard._status == "WARNING"
        assert guard._panic_is_live is False

    def test_daily_reset_clears_panic_is_live(self, tmp_path):
        """C1: _reset_daily should clear _panic_is_live."""
        guard = self._make_guard(tmp_path, status="PANIC")
        guard._panic_is_live = True

        guard._state_date = "1999-01-01"
        guard._reset_daily()

        assert guard._panic_is_live is False


class TestRecoveryTimerOscillation:
    """Tests for Workstream E: Recovery timer not reset on noise between recovery_pct and halt_pct."""

    def _make_dg(self, tmpdir, status="HALT", starting_equity=100000):
        """Create a DrawdownGuard for oscillation testing."""
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard

        data_dir = str(tmpdir)
        config = {
            'drawdown_circuit_breaker': {
                'enabled': True,
                'warning_pct': 1.5,
                'halt_pct': 2.5,
                'panic_pct': 4.0,
                'recovery_pct': 2.0,
                'recovery_hold_minutes': 30,
            },
            'notifications': {'enabled': False},
            'data_dir': data_dir,
        }

        state_file = os.path.join(data_dir, 'drawdown_state.json')
        os.makedirs(data_dir, exist_ok=True)
        state = {
            "status": status,
            "current_drawdown_pct": -3.0,
            "starting_equity": starting_equity,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        equity_file = os.path.join(data_dir, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            f.write(f"2026-01-01 17:00:00,{starting_equity}\n")

        return DrawdownGuard(config)

    def _mock_ib(self, net_liq):
        ib = MagicMock()
        summary_item = MagicMock()
        summary_item.tag = 'NetLiquidation'
        summary_item.currency = 'USD'
        summary_item.value = str(net_liq)
        ib.accountSummaryAsync = AsyncMock(return_value=[summary_item])
        return ib

    @pytest.mark.asyncio
    async def test_timer_not_reset_between_recovery_and_halt(self, tmp_path):
        """E: Timer should NOT reset when drawdown bounces between recovery_pct and halt_pct."""
        guard = self._make_dg(tmp_path, status="HALT", starting_equity=100000)
        # recovery_pct=2.0, halt_pct=2.5

        # Step 1: drawdown -1.5% → below recovery_pct → timer starts
        ib = self._mock_ib(98500)
        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)
        assert guard._recovery_start is not None
        saved_timer = guard._recovery_start

        # Step 2: drawdown -2.3% → between recovery_pct and halt_pct → timer should be preserved
        ib = self._mock_ib(97700)
        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)
        assert guard._recovery_start == saved_timer  # Timer NOT reset
        assert guard.state['status'] == "HALT"  # Status held

    @pytest.mark.asyncio
    async def test_timer_reset_above_halt_pct(self, tmp_path):
        """E: Timer SHOULD reset when drawdown crosses back above halt_pct."""
        guard = self._make_dg(tmp_path, status="HALT", starting_equity=100000)
        # recovery_pct=2.0, halt_pct=2.5

        # Step 1: Start recovery timer
        guard._recovery_start = datetime.now(timezone.utc).isoformat()

        # Step 2: drawdown worsens to -3.0% → above halt_pct=2.5% → reset timer
        ib = self._mock_ib(97000)
        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard._recovery_start is None  # Timer reset
        assert guard.state['status'] == "HALT"  # Status held

    @pytest.mark.asyncio
    async def test_oscillation_scenario_timer_survives(self, tmp_path):
        """E: Full oscillation scenario: start→bounce→return→complete."""
        guard = self._make_dg(tmp_path, status="HALT", starting_equity=100000)

        # Step 1: Drawdown improves to -1.8% (below recovery_pct=2.0%) → timer starts
        ib = self._mock_ib(98200)
        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)
        assert guard._recovery_start is not None

        # Step 2: Bounces to -2.2% (between recovery=2.0% and halt=2.5%) → timer preserved
        ib = self._mock_ib(97800)
        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)
        assert guard._recovery_start is not None

        # Step 3: Back to -1.5% and timer expired (simulate 31 min ago)
        guard._recovery_start = (
            datetime.now(timezone.utc) - timedelta(minutes=31)
        ).isoformat()
        ib = self._mock_ib(98500)
        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            result = await guard.update_pnl(ib)

        assert result == "WARNING"
        assert guard._recovery_start is None


class TestStaleRecoveryStartValidation:
    """Tests for A4: Stale recovery_start discarded on load."""

    def test_drawdown_guard_discards_stale_recovery_start(self, tmp_path):
        """A4: DrawdownGuard discards recovery_start from a previous day."""
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard

        data_dir = str(tmp_path)
        os.makedirs(data_dir, exist_ok=True)

        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        state_file = os.path.join(data_dir, 'drawdown_state.json')
        state = {
            "status": "HALT",
            "current_drawdown_pct": -3.0,
            "starting_equity": 100000,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
            "recovery_start": yesterday,  # From previous day
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        config = {
            'drawdown_circuit_breaker': {'enabled': True},
            'notifications': {},
            'data_dir': data_dir,
        }
        guard = DrawdownGuard(config)

        assert guard._recovery_start is None

    def test_drawdown_guard_keeps_today_recovery_start(self, tmp_path):
        """A4: DrawdownGuard preserves recovery_start from today."""
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard

        data_dir = str(tmp_path)
        os.makedirs(data_dir, exist_ok=True)

        today_ts = datetime.now(timezone.utc).isoformat()
        state_file = os.path.join(data_dir, 'drawdown_state.json')
        state = {
            "status": "HALT",
            "current_drawdown_pct": -2.0,
            "starting_equity": 100000,
            "last_updated": today_ts,
            "date": datetime.now(timezone.utc).date().isoformat(),
            "recovery_start": today_ts,
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        config = {
            'drawdown_circuit_breaker': {'enabled': True},
            'notifications': {},
            'data_dir': data_dir,
        }
        guard = DrawdownGuard(config)

        assert guard._recovery_start == today_ts

    def test_portfolio_guard_discards_stale_recovery_start(self, tmp_path):
        """A4: PortfolioRiskGuard discards recovery_start from a previous day."""
        from trading_bot.shared_context import PortfolioRiskGuard

        data_dir = str(tmp_path)
        os.makedirs(data_dir, exist_ok=True)

        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        state_file = os.path.join(data_dir, 'portfolio_risk_state.json')
        state = {
            "status": "HALT",
            "peak_equity": 100000,
            "current_equity": 97000,
            "starting_equity": 100000,
            "daily_pnl": -3000,
            "positions": {},
            "margin": {},
            "date": datetime.now(timezone.utc).date().isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "recovery_start": yesterday,
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        config = {
            'data_dir_root': data_dir,
            'drawdown_circuit_breaker': {'enabled': True},
            'notifications': {},
        }
        guard = PortfolioRiskGuard(config=config)

        assert guard._recovery_start is None
