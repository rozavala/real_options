"""
MasterOrchestrator — the single-process coordinator.

Responsibilities:
1. Initialize SharedContext
2. Spawn CommodityEngine per active commodity (staggered startup)
3. Run shared services (equity, VaR, reconciliation, macro research)
4. Monitor engine health (restart on crash)

Full implementation in Phase 4. This skeleton ensures clean imports.
"""

import asyncio
import logging
import os

from trading_bot.shared_context import SharedContext, PortfolioRiskGuard, MacroCache

logger = logging.getLogger(__name__)

MAX_RESTARTS = 3
RESTART_DELAY_SECONDS = 30
ENGINE_STARTUP_JITTER_SECONDS = 2.5  # v2.1: stagger to prevent IB Gateway storm


async def run_engine_with_restart(engine, max_restarts=MAX_RESTARTS):
    """Run an engine with automatic restart on crash."""
    restarts = 0
    while restarts <= max_restarts:
        try:
            await engine.start()
            break
        except asyncio.CancelledError:
            break
        except Exception as e:
            restarts += 1
            logger.error(f"Engine [{engine.ticker}] crashed ({restarts}/{max_restarts}): {e}")
            if restarts <= max_restarts:
                logger.info(f"Restarting [{engine.ticker}] in {RESTART_DELAY_SECONDS}s...")
                await asyncio.sleep(RESTART_DELAY_SECONDS)
            else:
                logger.critical(f"Engine [{engine.ticker}] exceeded max restarts.")
                try:
                    from trading_bot.notifications import send_pushover_notification
                    send_pushover_notification(
                        engine.config.get('notifications', {}),
                        f"CRITICAL: {engine.ticker} Engine Dead",
                        f"Crashed {max_restarts} times. Last error: {e}"
                    )
                except Exception:
                    pass


async def main(active_commodities: list = None, single_commodity: str = None):
    """Main entry point for the MasterOrchestrator.

    Phase 4: Full implementation with equity service, VaR, reconciliation.
    """
    from config_loader import load_config
    from trading_bot.commodity_engine import CommodityEngine
    from config.commodity_profiles import get_commodity_profile

    config = load_config()
    if not config:
        logger.critical("Cannot start without valid configuration.")
        return

    # Determine commodities
    if single_commodity:
        tickers = [single_commodity.upper()]
    elif active_commodities:
        tickers = [t.upper() for t in active_commodities]
    else:
        from config_loader import get_active_commodities
        tickers = get_active_commodities(config)

    # Startup validation: every ticker must have a commodity profile
    for t in tickers:
        profile = get_commodity_profile(t)
        if profile is None:
            logger.critical(f"No commodity profile found for '{t}'. Cannot start.")
            return

    logger.info(f"MasterOrchestrator starting with commodities: {tickers}")

    # Build SharedContext
    from trading_bot.heterogeneous_router import HeterogeneousRouter
    from trading_bot.budget_guard import get_budget_guard

    shared = SharedContext(
        base_config=config,
        router=HeterogeneousRouter(config),
        budget_guard=get_budget_guard(config),
        portfolio_guard=PortfolioRiskGuard(
            config={**config, 'data_dir_root': 'data'}
        ),
        macro_cache=MacroCache(),
        active_commodities=tickers,
        llm_semaphore=asyncio.Semaphore(
            config.get('llm', {}).get('max_concurrent_calls', 4)
        ),
    )

    # Create engines
    engines = [CommodityEngine(ticker, shared) for ticker in tickers]

    try:
        # === STAGGERED ENGINE STARTUP (v2.1 HRO mandate) ===
        engine_tasks = []
        for i, engine in enumerate(engines):
            if i > 0:
                logger.info(
                    f"Staggering engine startup: waiting "
                    f"{ENGINE_STARTUP_JITTER_SECONDS}s before [{engine.ticker}]"
                )
                await asyncio.sleep(ENGINE_STARTUP_JITTER_SECONDS)
            task = asyncio.create_task(
                run_engine_with_restart(engine),
                name=f"engine-{engine.ticker}"
            )
            engine_tasks.append(task)

        # Phase 4: Add master-level shared services here
        # equity_task = asyncio.create_task(_equity_service(shared, config))

        await asyncio.gather(*engine_tasks)

    except asyncio.CancelledError:
        logger.info("MasterOrchestrator cancelled")
    finally:
        logger.info("MasterOrchestrator shutting down")
        from trading_bot.connection_pool import IBConnectionPool
        await IBConnectionPool.release_all()
