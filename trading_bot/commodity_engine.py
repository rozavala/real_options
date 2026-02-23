"""
CommodityEngine — the per-commodity async runtime.

Owns: sentinels, council, schedule, state, TMS, order execution.
Delegates: LLM calls, budget, portfolio risk to SharedContext.

Each engine runs as an asyncio.Task with its own ContextVar scope,
ensuring complete data isolation between commodities.

Phase 2: Full lifecycle — replaces delegation to orchestrator.main().
"""

import asyncio
import json
import logging
import os
import time as time_module
from datetime import datetime, time, timedelta, timezone

import pytz

from trading_bot.shared_context import SharedContext
from trading_bot.data_dir_context import set_engine_data_dir

logger = logging.getLogger(__name__)


class CommodityEngine:
    """Runs the full trading pipeline for a single commodity."""

    def __init__(self, ticker: str, shared: SharedContext):
        self.ticker = ticker.upper()
        self.shared = shared
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', self.ticker
        )
        self._running = False
        self._logger = logging.getLogger(f"Engine.{self.ticker}")
        self._runtime = None

        # Build per-commodity config (safe — uses base_config, no file path resolution)
        self.config = self._build_config()

        # WARNING: Do NOT import or instantiate any module that resolves file paths
        # here (e.g., StateManager, TMS, SentinelStats). All file-path-dependent
        # initialization MUST happen inside start() AFTER set_engine_data_dir().

    def _build_config(self) -> dict:
        """Merge base config with commodity-specific overrides."""
        import copy
        from config_loader import deep_merge
        config = copy.deepcopy(self.shared.base_config)
        overrides = config.get('commodity_overrides', {}).get(self.ticker, {})
        if overrides:
            config = deep_merge(config, overrides)
        config['data_dir'] = self.data_dir
        config['symbol'] = self.ticker
        config.setdefault('commodity', {})['ticker'] = self.ticker
        # Primary commodity owns account-wide equity tracking (NetLiquidation).
        # Non-primary commodities skip equity snapshots to avoid duplicate data.
        if self.shared.active_commodities:
            config['commodity']['is_primary'] = (self.ticker == self.shared.active_commodities[0])
        else:
            config['commodity']['is_primary'] = True
        return config

    async def start(self):
        """Initialize and run the engine's main loop.

        CRITICAL: set_engine_data_dir() MUST be the very first call.
        This sets the ContextVar for this task, ensuring all downstream
        modules resolve paths to data/{TICKER}/. No imports or class
        instantiations that resolve file paths may happen before this.
        """
        # === 1. TASK-LOCAL DATA DIRECTORY — FIRST CALL ===
        set_engine_data_dir(self.data_dir)

        # === 2. Per-commodity logging (only in single-engine mode) ===
        # In --multi mode, __main__ already configured the unified orchestrator_multi.log.
        # Calling setup_logging again would replace the root handler (force=True),
        # causing cross-engine log contamination since all engines share one process.
        if self.shared is None:
            from trading_bot.logging_config import setup_logging
            setup_logging(log_file=f"logs/orchestrator_{self.ticker.lower()}.log")

        self._logger.info("=============================================")
        self._logger.info(f"=== Starting CommodityEngine [{self.ticker}] ===")
        self._logger.info("=============================================")
        self._logger.info(f"Data directory: {self.data_dir}")

        # === 3. Set legacy module globals for backward compatibility ===
        from trading_bot.data_dir_context import configure_legacy_modules
        configure_legacy_modules(self.data_dir)

        # === 4. Compliance boot time (VaR stale grace period) ===
        from trading_bot.compliance import set_boot_time
        set_boot_time()

        # === 5. Create per-engine runtime and set ContextVar ===
        from trading_bot.data_dir_context import EngineRuntime, set_engine_runtime
        self._runtime = EngineRuntime(
            ticker=self.ticker,
            deduplicator=self._create_deduplicator(),
            budget_guard=self.shared.budget_guard,
            drawdown_guard=self._create_drawdown_guard(),
            shared=self.shared,
        )
        set_engine_runtime(self._runtime)

        # === 6. Trading mode ===
        from trading_bot.utils import set_trading_mode, is_trading_off
        from notifications import send_pushover_notification
        set_trading_mode(self.config)
        if is_trading_off():
            self._logger.warning("=" * 60)
            self._logger.warning("  TRADING MODE: OFF — Training/Observation Only")
            self._logger.warning("  No real orders will be placed via IB")
            self._logger.warning("=" * 60)
            send_pushover_notification(
                self.config.get('notifications', {}),
                f"Engine [{self.ticker}] Started (OFF Mode)",
                "Trading mode is OFF. Analysis pipeline runs normally. No orders will be placed."
            )

        # === 7. Remote Gateway indicator ===
        ib_host = self.config.get('connection', {}).get('host', '127.0.0.1')
        is_paper = self.config.get('connection', {}).get('paper', False)
        if ib_host not in ('127.0.0.1', 'localhost', '::1'):
            gw_label = "REMOTE/PAPER" if is_paper else "REMOTE"
            self._logger.warning("=" * 60)
            self._logger.warning(f"  IB GATEWAY: {gw_label} ({ib_host})")
            self._logger.warning(f"  Client IDs: DEV range (10-79)")
            self._logger.warning(f"  Trading Mode: {self.config.get('trading_mode', 'LIVE')}")
            self._logger.warning("=" * 60)
            if not is_trading_off() and not is_paper:
                self._logger.critical(
                    "SAFETY: Remote gateway with TRADING_MODE=LIVE! "
                    "Set TRADING_MODE=OFF or IB_PAPER=true in .env for dev environments."
                )
                send_pushover_notification(
                    self.config.get('notifications', {}),
                    "REMOTE GW + LIVE MODE",
                    f"Engine [{self.ticker}] on remote GW ({ib_host}) with TRADING_MODE=LIVE. "
                    "Likely a misconfiguration — set TRADING_MODE=OFF or IB_PAPER=true.",
                    priority=1
                )

        # === 8. Deduplicator config ===
        from orchestrator import _get_deduplicator
        _get_deduplicator().critical_severity_threshold = self.config.get(
            'sentinels', {}
        ).get('critical_severity_threshold', 9)

        self._logger.info(
            f"Budget Guard initialized. Daily limit: ${self.shared.budget_guard.daily_budget}"
        )
        self._logger.info("Drawdown Guard initialized.")

        # === 9. Validate expiry filter ===
        from config import get_active_profile
        profile = get_active_profile(self.config)
        if profile.min_dte >= profile.max_dte:
            raise ValueError(
                f"M6: Expiry filter overlap impossible: "
                f"min_dte ({profile.min_dte}) >= max_dte ({profile.max_dte})"
            )
        self._logger.info(f"Expiry filter window: {profile.min_dte}-{profile.max_dte} DTE")

        # === 10. Process deferred triggers ===
        from trading_bot.utils import is_market_open
        from orchestrator import process_deferred_triggers
        if is_market_open(self.config):
            await process_deferred_triggers(self.config)
        else:
            self._logger.info("Market Closed. Deferred triggers will remain queued.")

        # === 11. Build schedule ===
        from orchestrator import (
            build_schedule, apply_schedule_offset, get_next_task,
            recover_missed_tasks, RECOVERY_POLICY,
            run_sentinels, _set_startup_discovery_time,
        )
        from trading_bot.task_tracker import record_task_completion

        env_name = os.getenv("ENV_NAME", "DEV")
        is_prod = env_name.startswith("PROD")

        task_list = build_schedule(self.config)

        # Override function references with engine-scoped registry
        engine_registry = self._build_function_registry()
        for task in task_list:
            if task.func_name in engine_registry:
                task.function = engine_registry[task.func_name]

        # Runtime RECOVERY_POLICY validation
        cfg_func_names = {t.func_name for t in task_list}
        cfg_uncovered = cfg_func_names - set(RECOVERY_POLICY.keys())
        if cfg_uncovered:
            self._logger.warning(
                f"Config schedule has functions without RECOVERY_POLICY entries "
                f"(will use safe defaults): {cfg_uncovered}"
            )

        if not is_prod:
            schedule_offset_minutes = self.config.get('schedule', {}).get('dev_offset_minutes', -30)
            self._logger.info(
                f"Environment: {env_name}. Applying {schedule_offset_minutes} minute "
                f"'Civil War' avoidance offset."
            )
            task_list = apply_schedule_offset(task_list, offset_minutes=schedule_offset_minutes)
        else:
            schedule_offset_minutes = 0
            self._logger.info("Environment: PROD. Using standard master schedule.")

        # === 12. Write active schedule for dashboard ===
        self._write_active_schedule(task_list, env_name, is_prod, schedule_offset_minutes)

        # === 13. Missed task detection & recovery ===
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(timezone.utc).astimezone(ny_tz)

        missed_tasks = []
        for task in task_list:
            task_ny = now_ny.replace(
                hour=task.time_et.hour, minute=task.time_et.minute,
                second=0, microsecond=0
            )
            if task_ny < now_ny:
                missed_tasks.append(task)

        if missed_tasks:
            missed_names = [
                f"  - {t.time_et.strftime('%H:%M')} ET: {t.id}" for t in missed_tasks
            ]
            self._logger.warning(
                f"LATE START DETECTED: {len(missed_tasks)} scheduled tasks already passed:\n"
                + "\n".join(missed_names)
            )
            await recover_missed_tasks(missed_tasks, self.config)

        # === 14. Startup topic discovery ===
        await self._run_startup_topic_discovery()

        # === 15. Start sentinels ===
        sentinel_task = asyncio.create_task(run_sentinels(self.config))
        sentinel_task.add_done_callback(
            lambda t: self._logger.critical(f"SENTINEL TASK DIED: {t.exception()}")
            if not t.cancelled() and t.exception() else None
        )

        # === 16. Start self-healing monitor ===
        from trading_bot.self_healing import SelfHealingMonitor
        healer = SelfHealingMonitor(self.config)
        healing_task = asyncio.create_task(healer.run())

        # === 17. Run scheduler loop ===
        self._running = True
        try:
            await self._run_scheduler(task_list)
        except asyncio.CancelledError:
            self._logger.info(f"Engine [{self.ticker}] cancelled")
        except Exception as e:
            self._logger.critical(f"Engine [{self.ticker}] crashed: {e}", exc_info=True)
            raise
        finally:
            self._logger.info("Engine shutting down. Cleaning up...")
            healer.stop()
            healing_task.cancel()
            sentinel_task.cancel()

            # Cancel in-flight fire-and-forget tasks
            ift = self._runtime.inflight_tasks if self._runtime else set()
            if ift:
                self._logger.info(f"Cancelling {len(ift)} in-flight tasks...")
                for t in list(ift):
                    t.cancel()
                await asyncio.sleep(1)
                ift.clear()

            self._running = False
            await self._shutdown()

    async def _run_scheduler(self, task_list):
        """The main while-True scheduler loop.

        Mirrors orchestrator.main()'s scheduler, but uses engine-scoped state.
        """
        from orchestrator import get_next_task, _get_deduplicator
        from trading_bot.task_tracker import record_task_completion

        while True:
            try:
                now_utc = datetime.now(pytz.UTC)
                next_run_time, next_task = get_next_task(now_utc, task_list)
                wait_seconds = (next_run_time - now_utc).total_seconds()

                self._logger.info(
                    f"Next task '{next_task.id}' ({next_task.label}) scheduled for "
                    f"{next_run_time.strftime('%Y-%m-%d %H:%M:%S UTC')}. "
                    f"Waiting for {wait_seconds / 3600:.2f} hours."
                )

                await asyncio.sleep(wait_seconds)

                self._logger.info(
                    f"--- Running scheduled task: {next_task.id} [{next_task.func_name}] ---"
                )

                # Set global cooldown during scheduled cycle (10 mins)
                _get_deduplicator().set_cooldown("GLOBAL", 600)

                try:
                    if next_task.func_name == 'guarded_generate_orders':
                        await next_task.function(self.config, schedule_id=next_task.id)
                    else:
                        await next_task.function(self.config)
                    record_task_completion(next_task.id)
                finally:
                    _get_deduplicator().clear_cooldown("GLOBAL")

            except asyncio.CancelledError:
                self._logger.info("Scheduler loop cancelled.")
                break
            except Exception as e:
                self._logger.critical(
                    f"Critical error in scheduler loop: {e}", exc_info=True
                )
                await asyncio.sleep(60)

    def _write_active_schedule(self, task_list, env_name, is_prod, schedule_offset_minutes):
        """Write the effective schedule to JSON for dashboard consumption."""
        try:
            schedule_data = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "env": env_name,
                "offset_minutes": 0 if is_prod else schedule_offset_minutes,
                "tasks": [
                    {
                        "id": task.id,
                        "time_et": task.time_et.strftime('%H:%M'),
                        "name": task.func_name,
                        "label": task.label,
                    }
                    for task in task_list
                ]
            }
            schedule_file = os.path.join(self.data_dir, 'active_schedule.json')
            os.makedirs(os.path.dirname(schedule_file), exist_ok=True)
            with open(schedule_file, 'w') as f:
                json.dump(schedule_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            self._logger.debug(f"Active schedule written: {len(schedule_data['tasks'])} tasks")
        except Exception as e:
            self._logger.warning(f"Failed to write active schedule (non-fatal): {e}")

    async def _run_startup_topic_discovery(self):
        """Run TopicDiscoveryAgent on startup for PredictionMarketSentinel."""
        from orchestrator import _get_budget_guard, _set_startup_discovery_time
        discovery_config = self.config.get(
            'sentinels', {}
        ).get('prediction_markets', {}).get('discovery_agent', {})

        if not discovery_config.get('enabled', False):
            return

        try:
            from trading_bot.topic_discovery import TopicDiscoveryAgent
            self._logger.info("Running TopicDiscoveryAgent on startup...")
            agent = TopicDiscoveryAgent(self.config, budget_guard=_get_budget_guard())
            result = await agent.run_scan()
            self._logger.info(
                f"Startup TopicDiscovery: {result['metadata']['topics_discovered']} topics, "
                f"{result['changes']['summary']}"
            )
            _set_startup_discovery_time(time_module.time())
        except Exception as e:
            self._logger.warning(
                f"Startup TopicDiscovery failed (sentinel loop will retry): {e}"
            )

    def _build_function_registry(self) -> dict:
        """Build engine-scoped function registry for scheduled tasks.

        Returns a dict mapping func_name → callable that wraps the orchestrator
        function with self.config, ensuring engine-scoped execution via ContextVar.

        Functions in orchestrator.py already read config and engine state through
        ContextVar accessors (_get_deduplicator, etc.), so the main value here is
        ensuring the right config is passed and providing a single override point
        for future function extraction.
        """
        import orchestrator
        from equity_logger import log_equity_snapshot
        from trading_bot.order_manager import close_stale_positions

        return {
            'start_monitoring': orchestrator.start_monitoring,
            'process_deferred_triggers': orchestrator.process_deferred_triggers,
            'cleanup_orphaned_theses': orchestrator.cleanup_orphaned_theses,
            'guarded_generate_orders': orchestrator.guarded_generate_orders,
            'run_position_audit_cycle': orchestrator.run_position_audit_cycle,
            'close_stale_positions': close_stale_positions,
            'close_stale_positions_fallback': orchestrator.close_stale_positions_fallback,
            'emergency_hard_close': orchestrator.emergency_hard_close,
            'cancel_and_stop_monitoring': orchestrator.cancel_and_stop_monitoring,
            'log_equity_snapshot': log_equity_snapshot,
            'reconcile_and_analyze': orchestrator.reconcile_and_analyze,
            'run_brier_reconciliation': orchestrator.run_brier_reconciliation,
            'sentinel_effectiveness_check': orchestrator.sentinel_effectiveness_check,
        }

    def _create_deduplicator(self):
        """Create a per-engine TriggerDeduplicator."""
        from orchestrator import TriggerDeduplicator
        return TriggerDeduplicator(
            state_file=os.path.join(self.data_dir, 'deduplicator_state.json')
        )

    def _create_drawdown_guard(self):
        """Create a per-engine DrawdownGuard."""
        from trading_bot.drawdown_circuit_breaker import DrawdownGuard
        return DrawdownGuard(self.config)

    async def _get_ib(self, purpose: str):
        """Get an IB connection scoped to this engine.

        CRITICAL: All IB access MUST go through this method.
        The engine-scoped key ensures KC's sentinel and CC's sentinel
        get physically separate IB instances.
        """
        from trading_bot.connection_pool import IBConnectionPool
        scoped_purpose = f"{self.ticker}_{purpose}"
        return await IBConnectionPool.get_connection(scoped_purpose, self.config)

    async def _shutdown(self):
        """Graceful shutdown of engine-owned resources."""
        self._logger.info(f"Engine [{self.ticker}] shutting down...")
        from trading_bot.connection_pool import IBConnectionPool
        for purpose in ['sentinel', 'orders', 'microstructure', 'audit',
                        'emergency', 'drawdown_check', 'cleanup']:
            try:
                await IBConnectionPool.release_connection(f"{self.ticker}_{purpose}")
            except Exception:
                pass
