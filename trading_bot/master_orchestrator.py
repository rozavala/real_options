"""
MasterOrchestrator — the single-process coordinator.

Responsibilities:
1. Initialize SharedContext
2. Spawn CommodityEngine per active commodity (staggered startup)
3. Run shared services (equity, VaR, reconciliation, macro research)
4. Monitor engine health (restart on crash)

Phase 4: Full implementation with equity, macro, and post-close services.
"""

import asyncio
import logging
import os
import random
import time as time_module
from datetime import datetime, time, timezone

from trading_bot.shared_context import SharedContext, PortfolioRiskGuard, MacroCache

logger = logging.getLogger(__name__)

MAX_RESTARTS = 3
RESTART_DELAY_SECONDS = 30
ENGINE_STARTUP_JITTER_SECONDS = 2.5  # v2.1: stagger to prevent IB Gateway storm

# Master-level service intervals
EQUITY_POLL_INTERVAL_SECONDS = 300  # 5 minutes
MACRO_RESEARCH_HOUR_ET = 6  # 06:00 ET daily
POST_CLOSE_DELAY_MINUTES = 20  # Run 20 min after market close


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


# ==========================================================================
# Master-Level Shared Services
# ==========================================================================

async def _equity_service(shared: SharedContext, config: dict):
    """Poll account equity every 5 minutes during market hours.

    Updates PortfolioRiskGuard with current NetLiquidation and daily P&L.
    Runs at the master level so all engines share one equity feed — prevents
    duplicate IB accountSummary calls across engines.
    """
    import pytz
    from trading_bot.utils import is_market_open

    logger.info("Equity service started")

    while True:
        try:
            if not is_market_open(config):
                await asyncio.sleep(60)
                continue

            ib = None
            try:
                from ib_insync import IB
                from trading_bot.utils import configure_market_data_type

                ib = IB()
                host = config.get('connection', {}).get('host', '127.0.0.1')
                port = config.get('connection', {}).get('port', 4002)
                client_id = random.randint(3000, 3999)

                await ib.connectAsync(host, port, clientId=client_id)
                configure_market_data_type(ib)

                summary = await ib.accountSummaryAsync()

                net_liq = 0.0
                daily_pnl = 0.0
                for item in summary:
                    if item.tag == 'NetLiquidation':
                        try:
                            net_liq = float(item.value)
                        except (ValueError, TypeError):
                            pass
                    elif item.tag == 'RealizedPnL':
                        try:
                            daily_pnl = float(item.value)
                        except (ValueError, TypeError):
                            pass

                if net_liq > 0:
                    await shared.portfolio_guard.update_equity(net_liq, daily_pnl)
                    logger.debug(
                        f"Equity service: NLV=${net_liq:,.0f}, "
                        f"P&L=${daily_pnl:,.0f}, "
                        f"status={shared.portfolio_guard._status}"
                    )

            except Exception as e:
                logger.warning(f"Equity service poll failed (retry in {EQUITY_POLL_INTERVAL_SECONDS}s): {e}")
            finally:
                if ib is not None:
                    try:
                        ib.disconnect()
                    except Exception:
                        pass

            await asyncio.sleep(EQUITY_POLL_INTERVAL_SECONDS)

        except asyncio.CancelledError:
            logger.info("Equity service cancelled")
            break
        except Exception as e:
            logger.error(f"Equity service error: {e}")
            await asyncio.sleep(60)


async def _macro_research_service(shared: SharedContext, config: dict):
    """Run daily macro + geopolitical research at 06:00 ET.

    Uses the macro and geopolitical agents (existing in agents.py) to produce
    a shared macro thesis. Results are stored in MacroCache and read by each
    engine's council during signal cycles. Eliminates duplicate macro LLM calls.
    """
    import pytz

    logger.info("Macro research service started")

    while True:
        try:
            # Wait until 06:00 ET
            ny_tz = pytz.timezone('America/New_York')
            now_ny = datetime.now(timezone.utc).astimezone(ny_tz)
            target = now_ny.replace(
                hour=MACRO_RESEARCH_HOUR_ET, minute=0, second=0, microsecond=0
            )
            if now_ny >= target:
                # Already past 06:00 today — schedule for tomorrow
                from datetime import timedelta
                target += timedelta(days=1)

            wait_seconds = (target - now_ny).total_seconds()
            logger.info(
                f"Macro research service: next run at {target.strftime('%Y-%m-%d %H:%M ET')} "
                f"({wait_seconds/3600:.1f}h from now)"
            )
            await asyncio.sleep(wait_seconds)

            # Run macro research
            logger.info("Running daily macro research...")
            try:
                from trading_bot.agents import TradingCouncil
                council = TradingCouncil(config)

                # Use the macro and geopolitical agents to produce summaries
                macro_result = await council.research_topic(
                    'macro',
                    "Provide a comprehensive macro-economic outlook focusing on: "
                    "Fed policy, USD strength, inflation trends, and their impact "
                    "on commodity markets. Be specific about directional bias."
                )
                geo_result = await council.research_topic(
                    'geopolitical',
                    "Provide a geopolitical risk assessment focusing on: "
                    "trade policy, supply chain disruptions, sanctions, and weather "
                    "patterns affecting agricultural commodity production."
                )

                await shared.macro_cache.update(
                    macro={'summary': macro_result} if isinstance(macro_result, str) else macro_result,
                    geopolitical={'summary': geo_result} if isinstance(geo_result, str) else geo_result,
                )
                logger.info("Macro research complete — cache updated for all engines")

            except Exception as e:
                logger.error(f"Macro research failed (non-fatal): {e}")

        except asyncio.CancelledError:
            logger.info("Macro research service cancelled")
            break
        except Exception as e:
            logger.error(f"Macro research service error: {e}")
            await asyncio.sleep(3600)  # Retry in 1 hour


async def _post_close_service(shared: SharedContext, config: dict):
    """Run post-close reconciliation once daily, 20 minutes after market close.

    Handles equity sync, trade reconciliation, and Brier reconciliation at the
    master level — eliminates per-engine duplication of these account-wide tasks.
    """
    import pytz
    from trading_bot.utils import is_market_open

    logger.info("Post-close service started")

    while True:
        try:
            # Wait for market to close
            if is_market_open(config):
                await asyncio.sleep(60)
                continue

            # Check if we're in the post-close window
            ny_tz = pytz.timezone('America/New_York')
            now_ny = datetime.now(timezone.utc).astimezone(ny_tz)

            # Only run between 14:20-15:00 ET (post-close window for KC)
            # This covers the period after KC close (14:00 ET) + 20 min buffer
            post_close_start = time(14, 20)
            post_close_end = time(15, 0)

            if not (post_close_start <= now_ny.time() <= post_close_end):
                await asyncio.sleep(300)  # Check every 5 minutes
                continue

            # Check if already ran today
            state_file = os.path.join('data', 'post_close_state.json')
            today_str = now_ny.date().isoformat()
            already_ran = False
            if os.path.exists(state_file):
                try:
                    import json
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    if state.get('last_run_date') == today_str:
                        already_ran = True
                except Exception:
                    pass

            if already_ran:
                await asyncio.sleep(3600)  # Sleep 1 hour, check again tomorrow
                continue

            logger.info("=== Running post-close reconciliation ===")

            # 1. Equity sync (Flex query)
            try:
                from equity_logger import sync_equity_from_flex, log_equity_snapshot
                # Use the primary engine's config for equity (account-wide)
                primary_config = config.copy()
                primary_data_dir = os.path.join('data', shared.active_commodities[0])
                primary_config['data_dir'] = primary_data_dir
                await sync_equity_from_flex(primary_config)
                await log_equity_snapshot(primary_config)
            except Exception as e:
                logger.error(f"Post-close equity sync failed: {e}")

            # 2. Trade reconciliation
            try:
                from reconcile_trades import main as run_reconciliation
                for ticker in shared.active_commodities:
                    ticker_config = config.copy()
                    ticker_config['data_dir'] = os.path.join('data', ticker)
                    ticker_config['symbol'] = ticker
                    await run_reconciliation(ticker_config)
            except Exception as e:
                logger.error(f"Post-close reconciliation failed: {e}")

            # 3. Brier reconciliation
            try:
                from trading_bot.brier_reconciliation import resolve_with_cycle_aware_match
                for ticker in shared.active_commodities:
                    # Set data dir context for each commodity
                    from trading_bot.data_dir_context import set_engine_data_dir
                    set_engine_data_dir(os.path.join('data', ticker))
                    resolved = resolve_with_cycle_aware_match(dry_run=False)
                    logger.info(f"Brier reconciliation [{ticker}]: {resolved} resolved")
            except Exception as e:
                logger.error(f"Post-close Brier reconciliation failed: {e}")

            # Record completion
            try:
                import json
                os.makedirs(os.path.dirname(state_file) or '.', exist_ok=True)
                with open(state_file, 'w') as f:
                    json.dump({
                        'last_run_date': today_str,
                        'last_run_utc': datetime.now(timezone.utc).isoformat(),
                    }, f)
            except Exception:
                pass

            logger.info("=== Post-close reconciliation complete ===")
            await asyncio.sleep(3600)  # Don't re-run for at least 1 hour

        except asyncio.CancelledError:
            logger.info("Post-close service cancelled")
            break
        except Exception as e:
            logger.error(f"Post-close service error: {e}")
            await asyncio.sleep(300)


async def main(active_commodities: list = None, single_commodity: str = None):
    """Main entry point for the MasterOrchestrator.

    Spawns CommodityEngines with staggered startup, then runs shared services
    (equity polling, macro research, post-close reconciliation) alongside them.
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

    router = HeterogeneousRouter(config)
    llm_sem = asyncio.Semaphore(
        config.get('llm', {}).get('max_concurrent_calls', 4)
    )
    router.llm_semaphore = llm_sem  # Wire backpressure into router

    shared = SharedContext(
        base_config=config,
        router=router,
        budget_guard=get_budget_guard(config),
        portfolio_guard=PortfolioRiskGuard(
            config={**config, 'data_dir_root': 'data'}
        ),
        macro_cache=MacroCache(),
        active_commodities=tickers,
        llm_semaphore=llm_sem,
    )

    # Create engines
    engines = [CommodityEngine(ticker, shared) for ticker in tickers]

    # Service tasks to cancel on shutdown
    service_tasks = []

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

        # === Master-level shared services (Phase 4) ===
        equity_task = asyncio.create_task(
            _equity_service(shared, config),
            name="master-equity"
        )
        service_tasks.append(equity_task)

        macro_task = asyncio.create_task(
            _macro_research_service(shared, config),
            name="master-macro"
        )
        service_tasks.append(macro_task)

        post_close_task = asyncio.create_task(
            _post_close_service(shared, config),
            name="master-post-close"
        )
        service_tasks.append(post_close_task)

        logger.info(
            f"Master services started: equity (every {EQUITY_POLL_INTERVAL_SECONDS}s), "
            f"macro (daily {MACRO_RESEARCH_HOUR_ET}:00 ET), "
            f"post-close (daily after market)"
        )

        await asyncio.gather(*engine_tasks)

    except asyncio.CancelledError:
        logger.info("MasterOrchestrator cancelled")
    finally:
        logger.info("MasterOrchestrator shutting down")

        # Cancel shared services
        for t in service_tasks:
            t.cancel()
        if service_tasks:
            await asyncio.gather(*service_tasks, return_exceptions=True)

        from trading_bot.connection_pool import IBConnectionPool
        await IBConnectionPool.release_all()
