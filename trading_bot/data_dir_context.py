"""
Task-local data directory and engine runtime isolation using contextvars.

CRITICAL ARCHITECTURAL COMPONENT (v2.0):
In a single-process multi-engine setup, module-level globals for data paths
and engine state would cause cross-contamination between engines. ContextVar
provides per-Task isolation that asyncio manages automatically.

Usage:
    # At engine startup:
    from trading_bot.data_dir_context import set_engine_data_dir, get_engine_data_dir
    set_engine_data_dir('data/KC')

    # In any downstream module:
    data_dir = get_engine_data_dir()  # Returns 'data/KC' for KC engine's task

    # For engine-scoped state (Phase 2):
    from trading_bot.data_dir_context import get_engine_runtime
    runtime = get_engine_runtime()
    runtime.deduplicator.should_deduplicate(...)
"""

import asyncio
import contextvars
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The ContextVar — each asyncio.Task gets its own copy automatically.
# NO default: LookupError propagates to module helpers so they fall back to
# their own module-level globals. This is critical for test isolation — tests
# that monkeypatch module globals must NOT have the ContextVar silently win.
_engine_data_dir: contextvars.ContextVar[str] = contextvars.ContextVar(
    '_engine_data_dir'
)


def set_engine_data_dir(data_dir: str):
    """Set the data directory for the current engine/task.

    Called once at engine startup. All code running within this task's
    async context will see this value. Other engine tasks are unaffected.

    IMPORTANT (v2.1): This MUST be the very first call in CommodityEngine.start(),
    before any module imports or class instantiations that resolve file paths.
    StateManager and other class-level attributes must NOT be initialized before
    this call — they will resolve paths using the ContextVar at access time.
    """
    os.makedirs(data_dir, exist_ok=True)
    _engine_data_dir.set(data_dir)
    logger.info(f"Engine data_dir set to: {data_dir}")


def get_engine_data_dir() -> str:
    """Get the data directory for the current engine/task.

    This is THE function that all modules call instead of reading
    their own _data_dir global. Safe for concurrent multi-engine use.
    """
    return _engine_data_dir.get()


def get_engine_data_path(filename: str) -> str:
    """Convenience: join the engine's data dir with a filename."""
    return os.path.join(get_engine_data_dir(), filename)


# ==========================================================================
# Engine Runtime: per-engine state (Phase 2)
# ==========================================================================

@dataclass
class EngineRuntime:
    """Per-engine mutable state — isolated via ContextVar across concurrent engines.

    Each CommodityEngine creates one EngineRuntime and sets it as the ContextVar
    value for its asyncio.Task. Functions that previously read module-level globals
    (GLOBAL_DEDUPLICATOR, etc.) now call get_engine_runtime() instead.
    """
    ticker: str = "KC"
    deduplicator: Any = None       # TriggerDeduplicator instance
    budget_guard: Any = None       # BudgetGuard instance
    drawdown_guard: Any = None     # DrawdownGuard instance
    emergency_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inflight_tasks: set = field(default_factory=set)
    startup_discovery_time: float = 0.0
    shared: Any = None             # SharedContext reference (Phase 2)
    brier_zero_resolution_streak: int = 0


# No default — LookupError triggers fallback to module globals in legacy mode
_engine_runtime: contextvars.ContextVar[EngineRuntime] = contextvars.ContextVar(
    '_engine_runtime'
)


def set_engine_runtime(runtime: EngineRuntime):
    """Set the EngineRuntime for the current asyncio task."""
    _engine_runtime.set(runtime)
    logger.info(f"EngineRuntime set for [{runtime.ticker}]")


def get_engine_runtime() -> Optional[EngineRuntime]:
    """Get the EngineRuntime for the current task, or None in legacy mode."""
    try:
        return _engine_runtime.get()
    except LookupError:
        return None


# --- Future modules: decorator option (v2.1) ---
# Uncomment and use when adding new stateful modules that need data dir awareness.
# This provides a cleaner API than manually calling get_engine_data_dir() in every
# path function, but the explicit approach is preferred for existing modules to
# minimize migration risk.
#
# def engine_context(func):
#     """Decorator that injects 'data_dir' kwarg from the current engine's ContextVar.
#
#     Usage:
#         @engine_context
#         def save_report(report, data_dir=None):
#             path = os.path.join(data_dir, 'report.json')
#             ...
#     """
#     import functools
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         if 'data_dir' not in kwargs or kwargs['data_dir'] is None:
#             kwargs['data_dir'] = get_engine_data_dir()
#         return func(*args, **kwargs)
#     return wrapper


# --- Migration helpers for existing modules ---

def configure_legacy_modules(data_dir: str):
    """LEGACY MODE ONLY: Set module-level globals for single-engine backward compat.

    In multi-engine mode, this is NOT called. Instead, modules read from
    get_engine_data_dir() directly. This function exists only for the
    LEGACY_MODE=true code path.
    """
    _LEGACY_REGISTRY = [
        ("trading_bot.state_manager", "StateManager", "set_data_dir"),
        ("trading_bot.task_tracker", None, "set_data_dir"),
        ("trading_bot.decision_signals", None, "set_data_dir"),
        ("trading_bot.order_manager", None, "set_capital_state_dir"),
        ("trading_bot.sentinel_stats", None, "set_data_dir"),
        ("trading_bot.utils", None, "set_data_dir"),
        ("trading_bot.tms", None, "set_data_dir"),
        ("trading_bot.brier_bridge", None, "set_data_dir"),
        ("trading_bot.brier_scoring", None, "set_data_dir"),
        ("trading_bot.weighted_voting", None, "set_data_dir"),
        ("trading_bot.brier_reconciliation", None, "set_data_dir"),
        ("trading_bot.router_metrics", None, "set_data_dir"),
        ("trading_bot.agents", None, "set_data_dir"),
        ("trading_bot.prompt_trace", None, "set_data_dir"),
    ]

    os.makedirs(data_dir, exist_ok=True)
    for module_path, class_name, func_name in _LEGACY_REGISTRY:
        try:
            mod = __import__(module_path, fromlist=[func_name])
            if class_name:
                cls = getattr(mod, class_name)
                getattr(cls, func_name)(data_dir)
            else:
                getattr(mod, func_name)(data_dir)
        except Exception as e:
            logger.error(f"Failed to set data_dir for {module_path}: {e}")

    # VaR is portfolio-wide — set to parent directory
    try:
        from trading_bot.var_calculator import set_var_data_dir
        set_var_data_dir(os.path.dirname(data_dir))
    except Exception as e:
        logger.error(f"Failed to set VaR data dir: {e}")

    # Also set the ContextVar for consistency
    set_engine_data_dir(data_dir)
    logger.info(f"Legacy data directories configured for: {data_dir}")


# --- ContextVar Isolation Test (v2.1 — real implementation) ---

async def validate_data_dir_isolation(tickers: list = None) -> list:
    """Verify that ContextVar isolation works across concurrent tasks.

    Spawns one task per ticker, each sets its own data_dir, sleeps to
    allow interleaving, then asserts get_engine_data_dir() returns
    the correct value. Returns list of (ticker, expected, actual) failures.

    Python 3.9+ compatible (no asyncio.Barrier).
    Call during Phase 1 testing or at system startup in debug mode.
    """
    if tickers is None:
        tickers = ["KC", "CC"]

    failures = []
    # Use an Event to synchronize: all tasks set their dir, then all read
    ready_event = asyncio.Event()
    ready_count = {"n": 0}

    async def _engine_task(ticker: str):
        expected = os.path.join('data', ticker)
        set_engine_data_dir(expected)

        # Signal that this task has set its data dir
        ready_count["n"] += 1
        if ready_count["n"] >= len(tickers):
            ready_event.set()
        # Wait for all tasks to have set their dirs (maximizes race window)
        await ready_event.wait()

        # Additional sleep to further stress interleaving
        await asyncio.sleep(0.05)

        actual = get_engine_data_dir()
        if actual != expected:
            failures.append((ticker, expected, actual))
            logger.error(
                f"ISOLATION FAILURE: {ticker} expected '{expected}', got '{actual}'"
            )
        else:
            logger.info(f"ISOLATION OK: {ticker} → {actual}")

    tasks = [asyncio.create_task(_engine_task(t)) for t in tickers]
    await asyncio.gather(*tasks)
    return failures
