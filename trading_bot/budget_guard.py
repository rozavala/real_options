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

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from enum import IntEnum

logger = logging.getLogger(__name__)


class CallPriority(IntEnum):
    """API call priority levels for gradual throttling."""
    CRITICAL = 1    # Compliance checks, risk management
    HIGH = 2        # Order placement, position monitoring
    NORMAL = 3      # Scheduled analysis cycles
    LOW = 4         # Dashboard refresh, thesis cleanup
    BACKGROUND = 5  # Brier scoring, historical analysis


class BudgetGuard:
    """Tracks cumulative daily API spend and enforces budget limits."""

    def __init__(self, config: dict):
        cost_config = config.get('cost_management', {})
        self.daily_budget = cost_config.get('daily_budget_usd', 15.0)
        self.warning_pct = cost_config.get('warning_threshold_pct', 0.75)
        self.sentinel_only_on_hit = cost_config.get('sentinel_only_mode_on_budget_hit', True)

        self.state_file = Path("data/budget_state.json")
        self._daily_spend = 0.0
        self._last_reset_date: Optional[str] = None
        self._budget_hit = False
        self._warning_sent = False

        self._load_state()
        self._check_reset()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self._daily_spend = data.get('daily_spend', 0.0)
                    self._last_reset_date = data.get('last_reset_date')
                    self._budget_hit = data.get('budget_hit', False)
                    self._warning_sent = data.get('warning_sent', False)
            except Exception as e:
                logger.warning(f"Failed to load budget state: {e}")

    def _save_state(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'daily_spend': self._daily_spend,
                'last_reset_date': self._last_reset_date,
                'budget_hit': self._budget_hit,
                'warning_sent': self._warning_sent
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save budget state: {e}")

    def _check_reset(self):
        """Reset daily spend at midnight UTC."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if self._last_reset_date != today:
            self._daily_spend = 0.0
            self._budget_hit = False
            self._warning_sent = False
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
        }
