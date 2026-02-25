"""
SharedContext — Typed container for all cross-commodity shared resources.

Injected into every CommodityEngine. Eliminates the need for global singletons
(GLOBAL_BUDGET_GUARD, GLOBAL_DRAWDOWN_GUARD, GLOBAL_DEDUPLICATOR).

Full implementation in Phase 3. This skeleton ensures clean imports.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import asyncio
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


@dataclass
class MacroCache:
    """Stores daily macro/geopolitical research shared across all engines.

    Written by MasterOrchestrator's global scheduler at 06:00 ET.
    Read by each CommodityEngine's council during signal cycles.
    """
    macro_thesis: Optional[dict] = None
    geopolitical_brief: Optional[dict] = None
    last_updated: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def update(self, macro: dict = None, geopolitical: dict = None):
        async with self._lock:
            if macro is not None:
                self.macro_thesis = macro
            if geopolitical is not None:
                self.geopolitical_brief = geopolitical
            self.last_updated = time.time()

    async def get(self) -> dict:
        async with self._lock:
            return {
                "macro_thesis": self.macro_thesis,
                "geopolitical_brief": self.geopolitical_brief,
                "last_updated": self.last_updated,
            }

    def is_stale(self, max_age_seconds: int = 43200) -> bool:
        """Check if macro data is older than max_age (default 12h)."""
        return (time.time() - self.last_updated) > max_age_seconds


# --- Cross-Commodity Correlation Matrix (v2.1) ---
# Hard-coded initial values. Replace with rolling empirical estimates post-Phase 6.
# Values represent approximate daily return correlation between front-month futures.
CORRELATION_MATRIX = {
    # Keys MUST be alphabetically sorted tuples (get_correlation sorts before lookup)
    ("CC", "KC"): 0.30,   # Both softs, some shared demand/macro drivers
    ("KC", "SB"): 0.20,   # Brazil production overlap
    ("KC", "NG"): 0.05,   # Minimal fundamental linkage
    ("CL", "KC"): 0.10,   # Energy input cost
    ("CC", "SB"): 0.15,   # Tropics overlap
    ("CC", "NG"): 0.05,
    ("CC", "CL"): 0.08,
    ("NG", "SB"): 0.10,   # Ethanol linkage
    ("CL", "SB"): 0.15,   # Ethanol linkage
    ("CL", "NG"): 0.35,   # Both energy
}


def get_correlation(ticker_a: str, ticker_b: str) -> float:
    """Get pairwise correlation. Order-independent, defaults to 0.0."""
    if ticker_a == ticker_b:
        return 1.0
    pair = tuple(sorted([ticker_a.upper(), ticker_b.upper()]))
    return CORRELATION_MATRIX.get(pair, 0.0)


@dataclass
class PortfolioRiskGuard:
    """Account-wide risk aggregation across all commodity engines.

    Replaces per-process DrawdownGuard instances. Single source of truth for:
    total drawdown, total margin usage, position concentration, budget.

    PERSISTENCE: State is atomically saved to data/portfolio_risk_state.json
    on every update. On startup, bootstraps from existing per-commodity
    drawdown states to prevent risk blindness during migration.
    """
    config: dict = field(default_factory=dict)
    _state_file: str = ""
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # Portfolio-level state
    _peak_equity: float = 0.0
    _current_equity: float = 0.0
    _starting_equity: float = 0.0   # Daily open
    _daily_pnl: float = 0.0
    _positions_by_commodity: Dict[str, int] = field(default_factory=dict)
    _margin_by_commodity: Dict[str, float] = field(default_factory=dict)
    _status: str = "NORMAL"  # NORMAL, WARNING, HALT, PANIC

    def __post_init__(self):
        self._state_file = os.path.join(
            self.config.get('data_dir_root', 'data'),
            'portfolio_risk_state.json'
        )
        self._recovery_start = None
        self._recovery_active = False
        self._load_state()

    # --- Persistence ---

    def _load_state(self):
        """Load persisted state. If missing, bootstrap from per-commodity drawdown files."""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, 'r') as f:
                    saved = json.load(f)
                from datetime import datetime, timezone
                current_date = datetime.now(timezone.utc).date().isoformat()
                if saved.get('date') == current_date:
                    self._peak_equity = saved.get('peak_equity', 0.0)
                    self._current_equity = saved.get('current_equity', 0.0)
                    self._starting_equity = saved.get('starting_equity', 0.0)
                    self._daily_pnl = saved.get('daily_pnl', 0.0)
                    self._positions_by_commodity = saved.get('positions', {})
                    self._margin_by_commodity = saved.get('margin', {})
                    self._status = saved.get('status', 'NORMAL')
                    self._recovery_start = saved.get('recovery_start')
                    if self._recovery_start:
                        self._recovery_active = True
                    logger.info(
                        f"PortfolioRiskGuard loaded: {self._status}, "
                        f"equity=${self._current_equity:,.0f}, "
                        f"positions={self._positions_by_commodity}"
                    )
                    return
                else:
                    logger.info("PortfolioRiskGuard state is from previous day, resetting.")
            except Exception as e:
                logger.warning(f"Failed to load portfolio risk state: {e}")

        # Bootstrap from existing per-commodity drawdown states
        self._bootstrap_from_legacy()

    def _bootstrap_from_legacy(self):
        """Migration: Read existing data/{TICKER}/drawdown_state.json files.

        Prevents risk blindness during cutover from per-process DrawdownGuards.
        Takes the worst (most restrictive) status across all commodities.
        """
        data_root = self.config.get('data_dir_root', 'data')
        worst_status = "NORMAL"
        status_priority = {"NORMAL": 0, "WARNING": 1, "HALT": 2, "PANIC": 3}
        total_starting = 0.0

        if not os.path.isdir(data_root):
            return

        for ticker_dir in os.listdir(data_root):
            dd_file = os.path.join(data_root, ticker_dir, 'drawdown_state.json')
            if not os.path.exists(dd_file):
                continue
            try:
                with open(dd_file, 'r') as f:
                    dd = json.load(f)
                from datetime import datetime, timezone
                if dd.get('date') == datetime.now(timezone.utc).date().isoformat():
                    saved_status = dd.get('status', 'NORMAL')
                    if status_priority.get(saved_status, 0) > status_priority.get(worst_status, 0):
                        worst_status = saved_status
                    total_starting += dd.get('starting_equity', 0.0)
                    logger.info(
                        f"Bootstrapped from {ticker_dir}: "
                        f"status={saved_status}, drawdown={dd.get('current_drawdown_pct', 0):.2f}%"
                    )
            except Exception as e:
                logger.warning(f"Failed to read {dd_file}: {e}")

        if worst_status != "NORMAL":
            self._status = worst_status
            logger.warning(
                f"PortfolioRiskGuard bootstrapped with status={worst_status} "
                f"from legacy per-commodity drawdown states"
            )
        if total_starting > 0:
            self._starting_equity = total_starting

    def _persist(self):
        """Atomically save state to disk (tmp + fsync + rename)."""
        try:
            from datetime import datetime, timezone
            os.makedirs(os.path.dirname(self._state_file) or '.', exist_ok=True)
            state = {
                "peak_equity": self._peak_equity,
                "current_equity": self._current_equity,
                "starting_equity": self._starting_equity,
                "daily_pnl": self._daily_pnl,
                "positions": dict(self._positions_by_commodity),
                "margin": dict(self._margin_by_commodity),
                "status": self._status,
                "date": datetime.now(timezone.utc).date().isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            if self._recovery_start is not None:
                state["recovery_start"] = self._recovery_start
            temp = self._state_file + ".tmp"
            with open(temp, 'w') as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, self._state_file)
        except Exception as e:
            logger.error(f"Failed to persist PortfolioRiskGuard state: {e}")

    # --- Risk Checks ---

    async def can_open_position(self, commodity: str, max_risk_usd: float) -> tuple:
        """Check if a new position is allowed given account-wide risk limits.

        Returns: (allowed: bool, reason: str)
        """
        async with self._lock:
            risk_cfg = self.config.get('risk_management', {})

            # 1. Circuit breaker status
            if self._status in ("HALT", "PANIC"):
                return False, f"Portfolio circuit breaker: {self._status}"

            # 2. Global drawdown check
            if self._starting_equity > 0 and self._current_equity > 0:
                drawdown_pct = (
                    (self._starting_equity - self._current_equity) / self._starting_equity
                ) * 100
                halt_pct = self.config.get(
                    'drawdown_circuit_breaker', {}
                ).get('halt_pct', 2.5)
                if drawdown_pct >= halt_pct:
                    self._status = "HALT"
                    self._persist()
                    return False, f"Portfolio drawdown {drawdown_pct:.1f}% >= {halt_pct:.1f}% limit"

            # 3. Total position count (post-trade: +1 for the proposed position)
            total_positions = sum(self._positions_by_commodity.values()) + 1
            max_positions = risk_cfg.get('max_total_positions', 30)
            if total_positions > max_positions:
                return False, f"Total positions {total_positions} > {max_positions} limit"

            # 4. Per-commodity concentration (post-trade)
            max_concentration = risk_cfg.get('max_commodity_concentration_pct', 0.50)
            commodity_positions = self._positions_by_commodity.get(commodity, 0) + 1
            if total_positions > 1 and commodity_positions > 1:
                concentration = commodity_positions / total_positions
                if concentration >= max_concentration:
                    return False, (
                        f"{commodity} concentration {concentration:.0%} >= "
                        f"{max_concentration:.0%}"
                    )

            # 5. Cross-commodity correlation check (v2.1 + v2.2 post-trade fix)
            max_correlated_pct = risk_cfg.get('max_correlated_exposure_pct', 0.70)
            if total_positions > 1:
                correlated_positions = commodity_positions  # Already +1 (proposed)
                for other_ticker, other_count in self._positions_by_commodity.items():
                    if other_ticker == commodity or other_count == 0:
                        continue
                    corr = get_correlation(commodity, other_ticker)
                    correlated_positions += other_count * corr
                correlated_pct = correlated_positions / total_positions
                if correlated_pct >= max_correlated_pct:
                    return False, (
                        f"Correlated exposure {correlated_pct:.0%} >= "
                        f"{max_correlated_pct:.0%} "
                        f"(includes correlation-weighted positions)"
                    )

            return True, "Approved"

    async def update_equity(self, equity: float, daily_pnl: float):
        """Update account equity from IB. Called by Master-level equity service."""
        async with self._lock:
            self._current_equity = equity
            self._daily_pnl = daily_pnl
            if equity > self._peak_equity:
                self._peak_equity = equity
            if self._starting_equity == 0.0:
                self._starting_equity = equity
                logger.info(f"PortfolioRiskGuard starting equity: ${equity:,.2f}")

            # Evaluate thresholds
            dd_cfg = self.config.get('drawdown_circuit_breaker', {})
            if self._starting_equity > 0:
                drawdown_pct = (
                    (self._starting_equity - equity) / self._starting_equity
                ) * 100
                prev = self._status
                if drawdown_pct >= dd_cfg.get('panic_pct', 4.0):
                    self._status = "PANIC"
                elif drawdown_pct >= dd_cfg.get('halt_pct', 2.5):
                    self._status = "HALT"
                elif drawdown_pct >= dd_cfg.get('warning_pct', 1.5):
                    self._status = "WARNING"

                # Recovery-aware escalation guard
                recovery_pct = dd_cfg.get('recovery_pct', 3.0)
                recovery_hold = dd_cfg.get('recovery_hold_minutes', 30)
                status_priority = {"NORMAL": 0, "WARNING": 1, "HALT": 2, "PANIC": 3}

                if prev in ("PANIC", "HALT"):
                    if self._status == "PANIC" and prev != "PANIC":
                        # Allow HALT→PANIC escalation
                        self._recovery_start = None
                        self._recovery_active = False
                    elif drawdown_pct <= recovery_pct:
                        # Drawdown improved below recovery threshold
                        from datetime import datetime as _dt, timezone as _tz
                        if self._recovery_start is None:
                            self._recovery_start = _dt.now(_tz.utc).isoformat()
                            self._recovery_active = False
                            logger.info(f"PortfolioRiskGuard recovery timer started (drawdown {drawdown_pct:.2f}%)")
                        else:
                            recovery_start_dt = _dt.fromisoformat(self._recovery_start)
                            elapsed = (_dt.now(_tz.utc) - recovery_start_dt).total_seconds() / 60
                            if elapsed >= recovery_hold:
                                self._status = "WARNING"
                                self._recovery_start = None
                                self._recovery_active = True  # Allow de-escalation this cycle
                                logger.warning(f"PortfolioRiskGuard recovery: {prev} -> WARNING after {elapsed:.0f}min")
                                try:
                                    from notifications import send_pushover_notification
                                    send_pushover_notification(
                                        self.config.get('notifications', {}),
                                        f"📈 Portfolio Recovery: {prev} → WARNING",
                                        f"Portfolio drawdown improved to {drawdown_pct:.2f}%. Trading resumed with caution."
                                    )
                                except Exception:
                                    pass
                        if not self._recovery_active:
                            self._status = prev  # Hold during observation
                    else:
                        # Still in drawdown territory - reset recovery timer
                        if self._recovery_start is not None:
                            self._recovery_start = None
                            self._recovery_active = False
                            logger.info("PortfolioRiskGuard recovery timer reset (drawdown worsened)")
                        self._status = prev
                elif status_priority.get(self._status, 0) < status_priority.get(prev, 0):
                    if not self._recovery_active:
                        self._status = prev

                # Reset recovery_active flag after use
                if self._recovery_active and self._status == "WARNING":
                    self._recovery_active = False

            self._persist()

    async def update_positions(self, commodity: str, count: int, margin: float):
        async with self._lock:
            self._positions_by_commodity[commodity] = count
            self._margin_by_commodity[commodity] = margin
            self._persist()

    def is_entry_allowed(self) -> bool:
        return self._status in ("NORMAL", "WARNING")

    def should_panic_close(self) -> bool:
        return self._status == "PANIC"

    async def get_snapshot(self) -> dict:
        async with self._lock:
            return {
                "equity": self._current_equity,
                "peak_equity": self._peak_equity,
                "starting_equity": self._starting_equity,
                "daily_pnl": self._daily_pnl,
                "positions": dict(self._positions_by_commodity),
                "margin": dict(self._margin_by_commodity),
                "status": self._status,
            }


@dataclass
class SharedContext:
    """Injected into every CommodityEngine. Single source for shared resources."""
    base_config: dict                               # Config BEFORE commodity overrides
    router: Any = None                              # HeterogeneousRouter
    budget_guard: Any = None                        # BudgetGuard
    portfolio_guard: PortfolioRiskGuard = None       # Account-wide risk
    macro_cache: MacroCache = field(default_factory=MacroCache)
    active_commodities: list = field(default_factory=list)
    # Semaphore to limit concurrent LLM calls across engines (v2.0 backpressure fix)
    llm_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(4),
        repr=False
    )
