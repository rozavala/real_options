"""
CommodityEngine — the per-commodity async runtime.

Owns: sentinels, council, schedule, state, TMS, order execution.
Delegates: LLM calls, budget, portfolio risk to SharedContext.

Each engine runs as an asyncio.Task with its own ContextVar scope,
ensuring complete data isolation between commodities.

Full extraction from orchestrator.py happens in Phase 2.
This skeleton ensures clean imports during Phase 0-1.
"""

import asyncio
import logging
import os

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
        return config

    async def start(self):
        """Initialize and run the engine's main loop.

        CRITICAL: set_engine_data_dir() MUST be the very first call.
        This sets the ContextVar for this task, ensuring all downstream
        modules resolve paths to data/{TICKER}/. No imports or class
        instantiations that resolve file paths may happen before this.
        """
        # === TASK-LOCAL DATA DIRECTORY — FIRST CALL (v2.0 critical fix) ===
        set_engine_data_dir(self.data_dir)

        self._logger.info(f"=== Starting CommodityEngine [{self.ticker}] ===")
        self._logger.info(f"Data directory: {self.data_dir}")

        # Set legacy module globals for backward compatibility
        from trading_bot.data_dir_context import configure_legacy_modules
        configure_legacy_modules(self.data_dir)

        # === Create per-engine runtime and set ContextVar (Phase 2) ===
        from trading_bot.data_dir_context import EngineRuntime, set_engine_runtime
        self._runtime = EngineRuntime(
            ticker=self.ticker,
            deduplicator=self._create_deduplicator(),
            budget_guard=self.shared.budget_guard,
            drawdown_guard=self._create_drawdown_guard(),
        )
        set_engine_runtime(self._runtime)

        self._running = True
        try:
            # Delegate to orchestrator's main loop (reuses existing functions)
            import orchestrator
            await orchestrator.main(commodity_ticker=self.ticker)
        except asyncio.CancelledError:
            self._logger.info(f"Engine [{self.ticker}] cancelled")
        except Exception as e:
            self._logger.critical(f"Engine [{self.ticker}] crashed: {e}", exc_info=True)
            raise
        finally:
            self._running = False
            await self._shutdown()

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

        CRITICAL (v2.1): All IB access MUST go through this method.
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
