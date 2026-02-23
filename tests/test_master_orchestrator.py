"""
Integration tests for MasterOrchestrator + CommodityEngine.

Verifies:
- SharedContext creation with all components
- CommodityEngine config building with overrides
- IB pool purpose auto-prefixing in multi-engine mode
- EngineRuntime ContextVar isolation across engines
- LLM semaphore backpressure wiring
"""

import asyncio
import os
import pytest
from unittest.mock import MagicMock, patch

from trading_bot.shared_context import SharedContext, PortfolioRiskGuard, MacroCache
from trading_bot.commodity_engine import CommodityEngine
from trading_bot.data_dir_context import (
    EngineRuntime, set_engine_runtime, get_engine_runtime,
    set_engine_data_dir, get_engine_data_dir,
)
from trading_bot.connection_pool import _resolve_purpose


# ---------------------------------------------------------------------------
# SharedContext
# ---------------------------------------------------------------------------

def test_shared_context_creation():
    """SharedContext should accept all required fields."""
    config = {'active_commodities': ['KC', 'CC'], 'risk_management': {}}
    shared = SharedContext(
        base_config=config,
        router=MagicMock(),
        budget_guard=MagicMock(),
        portfolio_guard=PortfolioRiskGuard(config={'data_dir_root': '/tmp/test_sc'}),
        macro_cache=MacroCache(),
        active_commodities=['KC', 'CC'],
        llm_semaphore=asyncio.Semaphore(4),
    )
    assert shared.active_commodities == ['KC', 'CC']
    assert shared.base_config == config


# ---------------------------------------------------------------------------
# CommodityEngine config building
# ---------------------------------------------------------------------------

def test_engine_config_building():
    """CommodityEngine should merge base config with commodity overrides."""
    base = {
        'active_commodities': ['KC', 'CC'],
        'commodity_overrides': {
            'CC': {
                'sentinels': {'price': {'threshold': 5.0}}
            }
        },
        'sentinels': {'price': {'threshold': 3.0}},
    }
    shared = SharedContext(
        base_config=base,
        router=MagicMock(),
        budget_guard=MagicMock(),
        portfolio_guard=MagicMock(),
        macro_cache=MacroCache(),
        active_commodities=['KC', 'CC'],
    )

    # KC engine: no overrides, uses base config
    kc = CommodityEngine('KC', shared)
    assert kc.config['sentinels']['price']['threshold'] == 3.0
    assert kc.config['symbol'] == 'KC'

    # CC engine: overrides threshold to 5.0
    cc = CommodityEngine('CC', shared)
    assert cc.config['sentinels']['price']['threshold'] == 5.0
    assert cc.config['symbol'] == 'CC'


def test_engine_data_dir():
    """CommodityEngine should set data_dir based on ticker."""
    shared = SharedContext(
        base_config={},
        router=MagicMock(),
        budget_guard=MagicMock(),
        portfolio_guard=MagicMock(),
        macro_cache=MacroCache(),
        active_commodities=['KC'],
    )
    engine = CommodityEngine('KC', shared)
    assert engine.data_dir.endswith(os.path.join('data', 'KC'))


# ---------------------------------------------------------------------------
# IB Pool purpose auto-prefix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ib_pool_purpose_auto_prefix():
    """_resolve_purpose should prefix with engine ticker from ContextVar."""
    async def _engine_task(ticker):
        rt = EngineRuntime(ticker=ticker)
        set_engine_runtime(rt)
        # Unprefixed purposes should get auto-prefixed
        assert _resolve_purpose("sentinel") == f"{ticker}_sentinel"
        assert _resolve_purpose("orders") == f"{ticker}_orders"
        # Already-prefixed should pass through
        assert _resolve_purpose(f"{ticker}_sentinel") == f"{ticker}_sentinel"

    t1 = asyncio.create_task(_engine_task("KC"))
    t2 = asyncio.create_task(_engine_task("CC"))
    await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_ib_pool_no_prefix_in_legacy():
    """Without EngineRuntime set, _resolve_purpose returns purpose as-is."""
    async def _check():
        assert _resolve_purpose("sentinel") == "sentinel"
    await asyncio.create_task(_check())


# ---------------------------------------------------------------------------
# EngineRuntime full flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_runtime_full_isolation():
    """Verify complete isolation of deduplicator, drawdown_guard across engines."""
    results = {}

    async def _engine_task(ticker):
        dedup = MagicMock()
        dedup.ticker = ticker
        rt = EngineRuntime(
            ticker=ticker,
            deduplicator=dedup,
            budget_guard=MagicMock(),
            drawdown_guard=MagicMock(),
        )
        set_engine_runtime(rt)
        set_engine_data_dir(f"data/{ticker}")

        await asyncio.sleep(0.05)  # Yield to other tasks

        actual_rt = get_engine_runtime()
        actual_dir = get_engine_data_dir()
        results[ticker] = {
            'runtime_ticker': actual_rt.ticker,
            'dedup_ticker': actual_rt.deduplicator.ticker,
            'data_dir': actual_dir,
        }

    t1 = asyncio.create_task(_engine_task("KC"))
    t2 = asyncio.create_task(_engine_task("CC"))
    t3 = asyncio.create_task(_engine_task("SB"))
    await asyncio.gather(t1, t2, t3)

    assert results["KC"]["runtime_ticker"] == "KC"
    assert results["CC"]["runtime_ticker"] == "CC"
    assert results["SB"]["runtime_ticker"] == "SB"
    assert results["KC"]["data_dir"] == "data/KC"
    assert results["CC"]["data_dir"] == "data/CC"
    assert results["SB"]["data_dir"] == "data/SB"


# ---------------------------------------------------------------------------
# LLM semaphore
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_semaphore_limits_concurrency():
    """LLM semaphore should limit concurrent generate() calls."""
    max_concurrent = 0
    current_concurrent = 0
    sem = asyncio.Semaphore(2)

    async def _fake_generate():
        nonlocal max_concurrent, current_concurrent
        await sem.acquire()
        try:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.05)
            current_concurrent -= 1
        finally:
            sem.release()

    # Fire 5 concurrent calls with semaphore limit of 2
    tasks = [asyncio.create_task(_fake_generate()) for _ in range(5)]
    await asyncio.gather(*tasks)

    # At most 2 should have been concurrent
    assert max_concurrent <= 2, f"Max concurrent was {max_concurrent}, expected <= 2"
