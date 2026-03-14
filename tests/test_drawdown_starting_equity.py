"""Tests for starting equity initialization, previous close, and stale state handling.

Covers:
- DrawdownGuard starting_equity from daily_equity.csv (previous close)
- PortfolioRiskGuard starting_equity from daily_equity.csv
- Starting equity forced to 0.0 on load, re-derived from prev close
- Stale recovery_start discarded on load
"""

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import mock_ib


# ---------------------------------------------------------------------------
# Module-level helpers for tests that need fresh guards (no pre-set state)
# ---------------------------------------------------------------------------

def _make_fresh_drawdown_guard(data_dir):
    """Create a DrawdownGuard without pre-written state."""
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


def _make_fresh_portfolio_guard(data_dir):
    """Create a PortfolioRiskGuard without pre-written state."""
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


def _write_equity_csv_rows(data_dir, rows):
    """Write daily_equity.csv with given rows [(timestamp, value), ...]."""
    equity_file = os.path.join(str(data_dir), 'daily_equity.csv')
    with open(equity_file, 'w') as f:
        for ts, val in rows:
            f.write(f"{ts},{val}\n")


# ---------------------------------------------------------------------------
# DrawdownGuard Previous Close Tests
# ---------------------------------------------------------------------------

class TestDrawdownGuardPrevClose:
    """Tests for starting_equity initialization from daily_equity.csv."""

    @pytest.mark.asyncio
    async def test_starting_equity_from_prev_close(self, tmp_path):
        """starting_equity should come from daily_equity.csv, not live NLV."""
        _write_equity_csv_rows(tmp_path, [
            ("2026-03-02 17:00:00", "35000.00"),
            ("2026-03-03 17:00:00", "35145.06"),
        ])
        guard = _make_fresh_drawdown_guard(tmp_path)
        ib = mock_ib(34000)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['starting_equity'] == 35145.06
        # Should detect drawdown: (34000-35145)/35145 = -3.26% → HALT
        assert guard.state['status'] == "HALT"

    @pytest.mark.asyncio
    async def test_falls_back_to_live_nlv_when_csv_missing(self, tmp_path):
        """If daily_equity.csv doesn't exist, fall back to live NLV."""
        guard = _make_fresh_drawdown_guard(tmp_path)
        ib = mock_ib(35000)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['starting_equity'] == 35000

    @pytest.mark.asyncio
    async def test_falls_back_to_live_nlv_when_csv_empty(self, tmp_path):
        """If daily_equity.csv is empty, fall back to live NLV."""
        equity_file = os.path.join(tmp_path, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            f.write("")
        guard = _make_fresh_drawdown_guard(tmp_path)
        ib = mock_ib(35000)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        assert guard.state['starting_equity'] == 35000

    @pytest.mark.asyncio
    async def test_handles_corrupt_csv(self, tmp_path):
        """Corrupt CSV should fall back to live NLV gracefully."""
        equity_file = os.path.join(tmp_path, 'daily_equity.csv')
        with open(equity_file, 'w') as f:
            f.write("not,a,valid\ncsv,file,here\n")
        guard = _make_fresh_drawdown_guard(tmp_path)
        ib = mock_ib(35000)

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            await guard.update_pnl(ib)

        # Should fall back gracefully (corrupt value raises ValueError)
        assert guard.state['starting_equity'] == 35000

    @pytest.mark.asyncio
    async def test_all_engines_get_same_starting_equity(self, tmp_path):
        """Multiple guards with same daily_equity.csv get identical starting_equity."""
        _write_equity_csv_rows(tmp_path, [
            ("2026-03-03 17:00:00", "35145.06"),
        ])

        guard1 = _make_fresh_drawdown_guard(tmp_path)
        guard2 = _make_fresh_drawdown_guard(tmp_path)
        guard3 = _make_fresh_drawdown_guard(tmp_path)

        # Different live NLVs (simulating staggered startup)
        ib1 = mock_ib(35100)
        ib2 = mock_ib(34800)
        ib3 = mock_ib(34500)

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
        _write_equity_csv_rows(tmp_path, [
            ("2026-03-03 17:00:00", "35000.00"),
        ])
        guard = _make_fresh_drawdown_guard(tmp_path)
        ib = mock_ib(33000)  # -5.7% drawdown

        with patch('trading_bot.drawdown_circuit_breaker.send_pushover_notification'):
            result = await guard.update_pnl(ib)

        # Should detect PANIC immediately, not return NORMAL
        assert result == "PANIC"


# ---------------------------------------------------------------------------
# PortfolioRiskGuard Previous Close Tests
# ---------------------------------------------------------------------------

class TestPortfolioRiskGuardPrevClose:
    """Tests for PortfolioRiskGuard starting_equity from daily_equity.csv."""

    @pytest.mark.asyncio
    async def test_starting_equity_from_prev_close(self, tmp_path):
        """PortfolioRiskGuard should use prev close from daily_equity.csv."""
        kc_dir = tmp_path / "KC"
        kc_dir.mkdir()
        equity_file = kc_dir / "daily_equity.csv"
        equity_file.write_text("2026-03-03 17:00:00,35145.06\n")

        guard = _make_fresh_portfolio_guard(tmp_path)
        await guard.update_equity(34000, -1145)

        assert guard._starting_equity == 35145.06

    @pytest.mark.asyncio
    async def test_falls_back_to_live_equity(self, tmp_path):
        """Without daily_equity.csv, should fall back to live equity."""
        guard = _make_fresh_portfolio_guard(tmp_path)
        await guard.update_equity(35000, 0)

        assert guard._starting_equity == 35000

    @pytest.mark.asyncio
    async def test_daily_reset_uses_prev_close(self, tmp_path):
        """On date rollover, _reset_daily should prefer prev close."""
        kc_dir = tmp_path / "KC"
        kc_dir.mkdir()
        equity_file = kc_dir / "daily_equity.csv"
        equity_file.write_text("2026-03-03 17:00:00,35145.06\n")

        guard = _make_fresh_portfolio_guard(tmp_path)
        guard._starting_equity = 34000
        guard._current_equity = 34500
        guard._peak_equity = 35000
        guard._state_date = "2026-03-03"  # Yesterday
        guard._status = "HALT"

        await guard.update_equity(34200, -945)

        # Should have reset and used prev close
        assert guard._starting_equity == 35145.06


# ---------------------------------------------------------------------------
# DrawdownGuard Starting Equity Reset Tests
# ---------------------------------------------------------------------------

class TestDrawdownGuardStartingEquityReset:
    """Tests for A2: starting_equity forced to 0.0 on load, re-derived from prev close."""

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
        ib = mock_ib(33000)  # NLV = $33,000

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


# ---------------------------------------------------------------------------
# PortfolioRiskGuard Starting Equity Reset Tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stale Recovery Start Validation Tests
# ---------------------------------------------------------------------------

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
