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
