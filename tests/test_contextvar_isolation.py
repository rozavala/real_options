"""
ContextVar isolation tests — MANDATORY Phase 1+2 gate.

Verifies that multiple concurrent asyncio.Tasks (one per CommodityEngine)
see their own data directory and engine runtime, and never cross-contaminate.
"""
import asyncio
import os
import pytest
from trading_bot.data_dir_context import (
    set_engine_data_dir, get_engine_data_dir, validate_data_dir_isolation,
    EngineRuntime, set_engine_runtime, get_engine_runtime,
)


@pytest.mark.asyncio
async def test_contextvar_isolation_two_engines():
    """CRITICAL: Verify KC and CC tasks see different data dirs."""
    failures = await validate_data_dir_isolation(["KC", "CC"])
    assert failures == [], f"ContextVar isolation failures: {failures}"


@pytest.mark.asyncio
async def test_contextvar_isolation_three_engines():
    """Verify isolation extends to 3+ engines (pre-NG readiness)."""
    failures = await validate_data_dir_isolation(["KC", "CC", "NG"])
    assert failures == [], f"ContextVar isolation failures: {failures}"


@pytest.mark.asyncio
async def test_contextvar_fallback():
    """In legacy mode (no ContextVar set), LookupError triggers module global fallback."""
    async def _check():
        # No default set → modules' _get_xxx() helpers should catch LookupError
        # and fall back to their module-level globals
        with pytest.raises(LookupError):
            get_engine_data_dir()
    await asyncio.create_task(_check())


@pytest.mark.asyncio
async def test_contextvar_post_trade_math():
    """Verify PortfolioRiskGuard uses post-trade counts (v2.2 fix)."""
    from trading_bot.shared_context import PortfolioRiskGuard
    guard = PortfolioRiskGuard(config={
        'data_dir_root': '/tmp/test_prg',
        'risk_management': {
            'max_total_positions': 10,
            'max_commodity_concentration_pct': 0.50,
            'max_correlated_exposure_pct': 0.70,
        }
    })
    # 5 KC positions, 0 CC. Proposing 1 CC.
    # Post-trade: 6 total, 1 CC → concentration = 1/6 = 16.7% (OK)
    guard._positions_by_commodity = {"KC": 5}
    guard._status = "NORMAL"
    allowed, reason = await guard.can_open_position("CC", 0)
    assert allowed, f"Should allow CC when concentration is low: {reason}"


@pytest.mark.asyncio
async def test_contextvar_set_and_get():
    """Basic set/get works within a single task."""
    async def _task():
        set_engine_data_dir('data/TEST')
        result = get_engine_data_dir()
        assert result == 'data/TEST', f"Expected 'data/TEST', got '{result}'"
    await asyncio.create_task(_task())


@pytest.mark.asyncio
async def test_portfolio_risk_guard_escalation_only():
    """PortfolioRiskGuard should never de-escalate within a day."""
    from trading_bot.shared_context import PortfolioRiskGuard
    guard = PortfolioRiskGuard(config={
        'data_dir_root': '/tmp/test_prg_esc',
        'drawdown_circuit_breaker': {
            'warning_pct': 1.5,
            'halt_pct': 2.5,
            'panic_pct': 4.0,
        }
    })
    guard._starting_equity = 100000.0

    # Push to WARNING (1.5% drawdown)
    await guard.update_equity(98500.0, -1500.0)
    assert guard._status == "WARNING"

    # Equity recovers — should NOT de-escalate
    await guard.update_equity(99500.0, -500.0)
    assert guard._status == "WARNING", "Should not de-escalate from WARNING to NORMAL"


@pytest.mark.asyncio
async def test_portfolio_risk_guard_halt():
    """PortfolioRiskGuard blocks trades at HALT status."""
    from trading_bot.shared_context import PortfolioRiskGuard
    guard = PortfolioRiskGuard(config={
        'data_dir_root': '/tmp/test_prg_halt',
    })
    guard._status = "HALT"
    allowed, reason = await guard.can_open_position("KC", 0)
    assert not allowed
    assert "HALT" in reason


@pytest.mark.asyncio
async def test_engine_runtime_isolation():
    """Verify EngineRuntime ContextVar isolates per-engine state across tasks."""
    results = {}

    async def _engine_task(ticker: str):
        rt = EngineRuntime(ticker=ticker)
        set_engine_runtime(rt)
        await asyncio.sleep(0.05)  # Yield to other tasks
        actual = get_engine_runtime()
        results[ticker] = actual.ticker

    t1 = asyncio.create_task(_engine_task("KC"))
    t2 = asyncio.create_task(_engine_task("CC"))
    await asyncio.gather(t1, t2)

    assert results["KC"] == "KC", f"KC saw {results['KC']}"
    assert results["CC"] == "CC", f"CC saw {results['CC']}"


@pytest.mark.asyncio
async def test_engine_runtime_none_in_legacy():
    """Without setting EngineRuntime, get_engine_runtime returns None."""
    async def _check():
        assert get_engine_runtime() is None
    await asyncio.create_task(_check())


@pytest.mark.asyncio
async def test_correlation_lookup():
    """Verify correlation matrix lookups are order-independent."""
    from trading_bot.shared_context import get_correlation
    assert get_correlation("KC", "CC") == get_correlation("CC", "KC")
    assert get_correlation("KC", "KC") == 1.0
    assert get_correlation("KC", "UNKNOWN") == 0.0
    assert get_correlation("NG", "CL") == 0.35
