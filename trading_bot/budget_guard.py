"""
API Budget Guard — Prevents runaway costs on high-volatility days.

When daily_budget_usd is hit:
  1. Sends Pushover alert
  2. Disables LLM reasoning (council debate, reflexion loops)
  3. Keeps sentinels running (hard stops, price alerts only)
  4. Resets at midnight UTC

This is a CIRCUIT BREAKER, not a throttle. Once tripped, it stays tripped
until the next reset. This prevents oscillating between modes.
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from enum import IntEnum

logger = logging.getLogger(__name__)


class BudgetThrottledError(Exception):
    """Raised when an API call is blocked by budget throttling."""
    pass


class CallPriority(IntEnum):
    """API call priority levels for gradual throttling."""
    CRITICAL = 1    # Compliance checks, risk management
    HIGH = 2        # Order placement, position monitoring
    NORMAL = 3      # Scheduled analysis cycles
    LOW = 4         # Dashboard refresh, thesis cleanup
    BACKGROUND = 5  # Brier scoring, historical analysis


ROLE_PRIORITY = {
    'compliance': CallPriority.CRITICAL,
    'master': CallPriority.HIGH,
    'permabear': CallPriority.HIGH,
    'permabull': CallPriority.HIGH,
    'agronomist': CallPriority.NORMAL,
    'macro': CallPriority.NORMAL,
    'geopolitical': CallPriority.NORMAL,
    'sentiment': CallPriority.NORMAL,
    'technical': CallPriority.NORMAL,
    'volatility': CallPriority.NORMAL,
    'inventory': CallPriority.NORMAL,
    'supply_chain': CallPriority.NORMAL,
    'weather_sentinel': CallPriority.LOW,
    'logistics_sentinel': CallPriority.LOW,
    'news_sentinel': CallPriority.LOW,
    'price_sentinel': CallPriority.LOW,
    'microstructure_sentinel': CallPriority.LOW,
    'trade_analyst': CallPriority.LOW,  # Post-mortem utility, non-critical
}


class BudgetGuard:
    """Tracks cumulative daily API spend and enforces budget limits."""

    def __init__(self, config: dict):
        cost_config = config.get('cost_management', {})
        self.daily_budget = cost_config.get('daily_budget_usd', 15.0)
        self.warning_pct = cost_config.get('warning_threshold_pct', 0.75)
        self.sentinel_only_on_hit = cost_config.get('sentinel_only_mode_on_budget_hit', True)

        data_dir = config.get('data_dir', 'data')
        self.state_file = Path(os.path.join(data_dir, "budget_state.json"))
        self._costs_csv = Path(os.path.join(data_dir, "llm_daily_costs.csv"))
        self._daily_spend = 0.0
        self._cost_by_source: dict[str, float] = {}
        self._request_count = 0
        self._last_reset_date: Optional[str] = None
        self._budget_hit = False
        self._warning_sent = False
        self._x_api_calls = 0

        self._load_state()
        self._check_reset()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self._daily_spend = data.get('daily_spend', 0.0)
                    self._cost_by_source = data.get('cost_by_source', {})
                    self._request_count = data.get('request_count', 0)
                    self._last_reset_date = data.get('last_reset_date')
                    self._budget_hit = data.get('budget_hit', False)
                    self._warning_sent = data.get('warning_sent', False)
                    self._x_api_calls = data.get('x_api_calls', 0)
            except Exception as e:
                logger.warning(f"Failed to load budget state: {e}")

    def _save_state(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'daily_spend': self._daily_spend,
                'cost_by_source': self._cost_by_source,
                'request_count': self._request_count,
                'last_reset_date': self._last_reset_date,
                'budget_hit': self._budget_hit,
                'warning_sent': self._warning_sent,
                'x_api_calls': self._x_api_calls,
            }
            temp_path = str(self.state_file) + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(data, f)
            os.replace(temp_path, str(self.state_file))
        except Exception as e:
            logger.error(f"Failed to save budget state: {e}")

    def _archive_daily_costs(self):
        """Append yesterday's costs to llm_daily_costs.csv before resetting."""
        if self._daily_spend <= 0 and self._request_count == 0:
            return  # Nothing to archive

        try:
            write_header = not self._costs_csv.exists()
            with open(self._costs_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['date', 'total_usd', 'request_count', 'cost_by_source', 'x_api_calls'])
                writer.writerow([
                    self._last_reset_date,
                    round(self._daily_spend, 4),
                    self._request_count,
                    json.dumps(self._cost_by_source),
                    self._x_api_calls,
                ])
            logger.info(
                f"Archived daily LLM costs for {self._last_reset_date}: "
                f"${self._daily_spend:.2f}, {self._request_count} requests"
            )
        except Exception as e:
            logger.error(f"Failed to archive daily costs: {e}")

    def _check_reset(self):
        """Reset daily spend at midnight UTC."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if self._last_reset_date != today:
            self._archive_daily_costs()
            self._daily_spend = 0.0
            self._cost_by_source = {}
            self._request_count = 0
            self._budget_hit = False
            self._warning_sent = False
            self._x_api_calls = 0
            self._last_reset_date = today
            self._save_state()
            logger.info(f"Budget guard reset for {today}. Limit: ${self.daily_budget:.2f}")

    def check_budget(self, priority: CallPriority = CallPriority.NORMAL) -> bool:
        """
        H3 FIX: Priority-aware budget check with gradual throttling.

        Instead of binary gate, throttle based on remaining budget:
        - >50% remaining: All priorities allowed
        - 25-50% remaining: BACKGROUND blocked
        - 10-25% remaining: LOW + BACKGROUND blocked
        - <10% remaining: Only CRITICAL allowed
        """
        remaining_pct = self._get_remaining_budget_pct()

        if remaining_pct > 0.50:
            return True  # All clear
        elif remaining_pct > 0.25:
            allowed = priority <= CallPriority.LOW
            if not allowed:
                logger.info(f"Budget throttle: {priority.name} blocked ({remaining_pct:.0%} remaining)")
            return allowed
        elif remaining_pct > 0.10:
            allowed = priority <= CallPriority.NORMAL
            if not allowed:
                logger.warning(f"Budget throttle: {priority.name} blocked ({remaining_pct:.0%} remaining)")
            return allowed
        else:
            allowed = priority <= CallPriority.CRITICAL
            if not allowed:
                logger.warning(f"Budget CRITICAL: Only essential calls allowed ({remaining_pct:.0%} remaining)")
            return allowed

    def _get_remaining_budget_pct(self) -> float:
        """Calculate remaining daily budget as percentage."""
        if self.daily_budget <= 0:
            return 1.0
        return max(0.0, (self.daily_budget - self._daily_spend) / self.daily_budget)

    def record_cost(self, cost_usd: float, source: str = "unknown") -> bool:
        """
        Record a cost and check budget.

        Returns:
            True if within budget, False if budget hit (sentinel-only mode).
        """
        self._check_reset()
        self._daily_spend += cost_usd
        self._request_count += 1
        self._cost_by_source[source] = self._cost_by_source.get(source, 0.0) + cost_usd

        # Warning threshold
        if not self._warning_sent and self._daily_spend >= self.daily_budget * self.warning_pct:
            logger.warning(
                f"Budget warning: ${self._daily_spend:.2f} / ${self.daily_budget:.2f} "
                f"({self._daily_spend / self.daily_budget * 100:.0f}%) — source: {source}"
            )
            self._warning_sent = True
            self._save_state()

        # Hard limit
        if not self._budget_hit and self._daily_spend >= self.daily_budget:
            logger.error(
                f"BUDGET HIT: ${self._daily_spend:.2f} >= ${self.daily_budget:.2f}. "
                f"Switching to sentinel-only mode."
            )
            self._budget_hit = True

        self._save_state()
        return not self._budget_hit

    @property
    def is_budget_hit(self) -> bool:
        """Check if budget is hit (sentinel-only mode active)."""
        self._check_reset()
        return self._budget_hit

    @property
    def remaining_budget(self) -> float:
        """Get remaining budget for today."""
        self._check_reset()
        return max(0.0, self.daily_budget - self._daily_spend)

    def record_x_api_call(self):
        """Record an X/Twitter API call (separate from LLM spend)."""
        self._check_reset()
        self._x_api_calls += 1
        self._save_state()

    def get_status(self) -> dict:
        """Get current budget status for dashboard display."""
        self._check_reset()
        return {
            'daily_budget': self.daily_budget,
            'daily_spend': self._daily_spend,
            'remaining': self.remaining_budget,
            'pct_used': (self._daily_spend / self.daily_budget * 100) if self.daily_budget > 0 else 0,
            'sentinel_only_mode': self._budget_hit,
            'reset_date': self._last_reset_date,
            'cost_by_source': dict(self._cost_by_source),
            'request_count': self._request_count,
            'x_api_calls': self._x_api_calls,
        }


# --- Singleton Factory ---

_budget_guard_instance: Optional[BudgetGuard] = None


def get_budget_guard(config: dict = None) -> Optional[BudgetGuard]:
    """Get the BudgetGuard for the current engine context, or the singleton.

    In multi-engine mode, each CommodityEngine has its own BudgetGuard
    stored in EngineRuntime (via ContextVar). This ensures per-commodity
    cost tracking and state persistence to data/{TICKER}/budget_state.json.

    Falls back to the module-level singleton for single-engine mode or
    when called outside an engine context (e.g., during startup).
    """
    # Try per-engine instance first (multi-commodity mode)
    try:
        from trading_bot.data_dir_context import get_engine_runtime
        rt = get_engine_runtime()
        if rt and rt.budget_guard:
            return rt.budget_guard
    except (LookupError, ImportError):
        pass

    # Fallback to singleton (single-engine mode)
    global _budget_guard_instance
    if _budget_guard_instance is None and config is not None:
        _budget_guard_instance = BudgetGuard(config)
    return _budget_guard_instance


# --- Cost Calculation ---

_cost_config_cache: Optional[dict] = None


def _load_cost_config() -> dict:
    """Load API cost configuration with caching."""
    global _cost_config_cache
    if _cost_config_cache is not None:
        return _cost_config_cache
    cost_file = Path(__file__).parent.parent / "config" / "api_costs.json"
    try:
        with open(cost_file, 'r') as f:
            data = json.load(f)
            _cost_config_cache = data.get('costs_per_1k_tokens', {})
    except Exception as e:
        logger.warning(f"Failed to load api_costs.json: {e}")
        _cost_config_cache = {'default': {'input': 0.001, 'output': 0.002}}
    return _cost_config_cache


def calculate_api_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate API cost from actual token usage."""
    costs = _load_cost_config()
    model_lower = model_name.lower()

    # Find best matching model key
    model_cost = costs.get('default', {'input': 0.001, 'output': 0.002})
    for key in costs:
        if key != 'default' and key in model_lower:
            model_cost = costs[key]
            break

    if isinstance(model_cost, dict):
        return (input_tokens / 1000) * model_cost.get('input', 0.001) + \
               (output_tokens / 1000) * model_cost.get('output', 0.002)
    return ((input_tokens + output_tokens) / 1000) * model_cost
