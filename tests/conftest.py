"""Shared test helpers for drawdown / portfolio-risk guard tests."""

import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock


def mock_ib(net_liq):
    """Create a mock IB returning the given NetLiquidation."""
    ib = MagicMock()
    summary_item = MagicMock()
    summary_item.tag = 'NetLiquidation'
    summary_item.currency = 'USD'
    summary_item.value = str(net_liq)
    ib.accountSummaryAsync = AsyncMock(return_value=[summary_item])
    return ib


def write_drawdown_state(data_dir, status, drawdown_pct, starting_equity):
    """Write drawdown_state.json to disk."""
    data_dir = str(data_dir)
    state_file = os.path.join(data_dir, 'drawdown_state.json')
    os.makedirs(data_dir, exist_ok=True)
    state = {
        "status": status,
        "current_drawdown_pct": drawdown_pct,
        "starting_equity": starting_equity,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "date": datetime.now(timezone.utc).date().isoformat(),
    }
    with open(state_file, 'w') as f:
        json.dump(state, f)


def write_equity_csv(data_dir, starting_equity):
    """Write daily_equity.csv for load_prev_close()."""
    data_dir = str(data_dir)
    equity_file = os.path.join(data_dir, 'daily_equity.csv')
    with open(equity_file, 'w') as f:
        f.write(f"2026-01-01 17:00:00,{starting_equity}\n")


def make_drawdown_guard(tmpdir, status="PANIC", starting_equity=100000, **config_overrides):
    """Create a DrawdownGuard with pre-set persisted state."""
    from trading_bot.drawdown_circuit_breaker import DrawdownGuard

    data_dir = str(tmpdir)
    cb_config = {
        'enabled': True,
        'warning_pct': 1.5,
        'halt_pct': 2.5,
        'panic_pct': 4.0,
        'recovery_pct': 3.0,
        'recovery_hold_minutes': 30,
    }
    cb_config.update(config_overrides)
    config = {
        'drawdown_circuit_breaker': cb_config,
        'notifications': {'enabled': False},
        'data_dir': data_dir,
    }

    write_drawdown_state(data_dir, status, -4.5, starting_equity)
    write_equity_csv(data_dir, starting_equity)

    return DrawdownGuard(config)


def make_portfolio_risk_guard(tmpdir, status="PANIC", starting_equity=100000, current_equity=95000, **config_overrides):
    """Create a PortfolioRiskGuard with pre-set persisted state."""
    from trading_bot.shared_context import PortfolioRiskGuard

    data_dir = str(tmpdir)
    cb_config = {
        'enabled': True,
        'warning_pct': 1.5,
        'halt_pct': 2.5,
        'panic_pct': 4.0,
        'recovery_pct': 3.0,
        'recovery_hold_minutes': 30,
    }
    cb_config.update(config_overrides)
    config = {
        'data_dir_root': data_dir,
        'drawdown_circuit_breaker': cb_config,
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
