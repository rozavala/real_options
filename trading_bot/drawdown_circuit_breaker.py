"""
Daily Drawdown Circuit Breaker.

Protects against aggregate portfolio losses that individual position stops miss.
Halts trading for the day if intraday P&L drops below configurable thresholds.

Thresholds (default):
- WARNING: -3.0% intraday -> Pushover alert
- HALT:    -6.0% intraday -> Block new trades
- PANIC:   -9.0% intraday -> Close ALL positions
"""

import csv
import logging
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional

from ib_insync import IB
from notifications import send_pushover_notification

logger = logging.getLogger(__name__)


def load_prev_close(data_dir: str) -> Optional[float]:
    """Load previous day's closing equity from daily_equity.csv.

    Returns the most recent entry's total_value_usd, or None if the file
    is missing, empty, or corrupt.  Used by DrawdownGuard and
    PortfolioRiskGuard to set a consistent starting_equity baseline
    independent of engine startup time.
    """
    equity_file = os.path.join(data_dir, 'daily_equity.csv')
    if not os.path.exists(equity_file):
        return None
    try:
        with open(equity_file, 'r') as f:
            reader = csv.reader(f)
            last_row = None
            for row in reader:
                if len(row) >= 2:
                    last_row = row
            if last_row is None:
                return None
            value = float(last_row[1])
            return value if value > 0 else None
    except Exception as e:
        logger.warning(f"Failed to load prev close from {equity_file}: {e}")
        return None


class DrawdownGuard:
    def __init__(self, config: dict):
        self.config = config.get('drawdown_circuit_breaker', {})
        self.notification_config = config.get('notifications', {})
        self.enabled = self.config.get('enabled', False)
        self.warning_pct = self.config.get('warning_pct', 3.0)
        self.halt_pct = self.config.get('halt_pct', 6.0)
        self.panic_pct = self.config.get('panic_pct', 9.0)
        self.recovery_pct = self.config.get('recovery_pct', 3.5)
        self.recovery_hold_minutes = self.config.get('recovery_hold_minutes', 30)
        self._recovery_start = None
        self._panic_is_live = False  # Only True after update_pnl() freshly evaluates PANIC
        self._data_dir = config.get('data_dir', 'data')
        # Always use per-commodity data_dir; ignore any legacy state_file in sub-config
        self.state_file = os.path.join(self._data_dir, 'drawdown_state.json')

        self.state = {
            "status": "NORMAL", # NORMAL, WARNING, HALT, PANIC
            "current_drawdown_pct": 0.0,
            "starting_equity": 0.0,
            "last_updated": None,
            "date": datetime.now(timezone.utc).date().isoformat()
        }

        self._load_state()
        self._reset_daily()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    saved = json.load(f)
                    # Check if saved state is from today
                    saved_date = saved.get('date')
                    current_date = datetime.now(timezone.utc).date().isoformat()
                    if saved_date == current_date:
                        self.state = saved
                        # Force starting_equity to 0.0 so update_pnl() re-derives
                        # from prev close (daily_equity.csv). Persisted value may
                        # have been set from live NLV during an earlier startup.
                        self.state['starting_equity'] = 0.0
                        self._recovery_start = saved.get('recovery_start')
                        # Discard stale recovery_start from previous day
                        if self._recovery_start:
                            try:
                                rs_date = datetime.fromisoformat(self._recovery_start).date()
                                if rs_date != datetime.now(timezone.utc).date():
                                    logger.info("Discarded stale recovery_start from previous day")
                                    self._recovery_start = None
                            except (ValueError, TypeError):
                                self._recovery_start = None
                        logger.info(
                            f"Loaded drawdown state: {self.state['status']} "
                            f"({self.state['current_drawdown_pct']:.2f}%), "
                            f"starting_equity=0 (will re-derive from prev close)"
                        )
                    else:
                        logger.info("Saved drawdown state is old. Starting fresh.")
            except Exception as e:
                logger.warning(f"Failed to load drawdown state: {e}")

    def _save_state(self):
        try:
            # Create dir if needed
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            if self._recovery_start is not None:
                self.state['recovery_start'] = self._recovery_start
            elif 'recovery_start' in self.state:
                del self.state['recovery_start']
            self.state['last_updated'] = datetime.now(timezone.utc).isoformat()
            temp_path = self.state_file + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(self.state, f, indent=2)
            os.replace(temp_path, self.state_file)
        except Exception as e:
            logger.warning(f"Failed to save drawdown state: {e}")

    def _reset_daily(self):
        """Reset state if new trading day."""
        current_date = datetime.now(timezone.utc).date().isoformat()
        if self.state['date'] != current_date:
            logger.info("Resetting DrawdownGuard for new day.")
            self._recovery_start = None
            self._panic_is_live = False
            self.state = {
                "status": "NORMAL",
                "current_drawdown_pct": 0.0,
                "starting_equity": 0.0, # Will be set on first check
                "last_updated": None,
                "date": current_date
            }
            self._save_state()

    async def update_pnl(self, ib: IB) -> str:
        """
        Calculate current intraday P&L and update status.
        Returns: Current Status (NORMAL, HALT, etc.)
        """
        if not self.enabled:
            return "NORMAL"

        self._reset_daily()

        try:
            # 1. Get Account Summary
            summary = await asyncio.wait_for(ib.accountSummaryAsync(), timeout=10)
            net_liq = 0.0

            for item in summary:
                if item.tag == 'NetLiquidation' and item.currency == 'USD':
                    net_liq = float(item.value)
                    break

            if net_liq == 0.0:
                logger.warning("Could not fetch NetLiquidation for drawdown check.")
                return self.state['status']

            # 2. Set Starting Equity if first run — prefer previous close
            if self.state['starting_equity'] == 0.0:
                prev_close = load_prev_close(self._data_dir)
                if prev_close is not None:
                    self.state['starting_equity'] = prev_close
                    logger.info(
                        f"DrawdownGuard initialized from prev close: "
                        f"${prev_close:,.2f} (live NLV: ${net_liq:,.2f})"
                    )
                else:
                    self.state['starting_equity'] = net_liq
                    logger.info(
                        f"DrawdownGuard initialized from live NLV: "
                        f"${net_liq:,.2f} (no daily_equity.csv available)"
                    )
                self._save_state()

            # 3. Calculate Drawdown
            start_eq = self.state['starting_equity']
            if start_eq <= 0:
                return self.state['status']  # Can't calculate drawdown yet
            pnl = net_liq - start_eq
            drawdown_pct = (pnl / start_eq) * 100

            self.state['current_drawdown_pct'] = drawdown_pct

            # 4. Check Thresholds
            prev_status = self.state['status']
            new_status = prev_status

            if drawdown_pct <= -self.panic_pct:
                new_status = "PANIC"
            elif drawdown_pct <= -self.halt_pct:
                new_status = "HALT"
            elif drawdown_pct <= -self.warning_pct:
                new_status = "WARNING"
            else:
                new_status = "NORMAL"

            # Recovery-aware escalation logic
            if prev_status in ("PANIC", "HALT"):
                if new_status == "PANIC" and prev_status != "PANIC":
                    # Allow HALT→PANIC escalation
                    self._recovery_start = None
                elif abs(drawdown_pct) <= self.recovery_pct:
                    # Drawdown improved below recovery threshold
                    if self._recovery_start is None:
                        self._recovery_start = datetime.now(timezone.utc).isoformat()
                        logger.info(f"Recovery timer started (drawdown {drawdown_pct:.2f}%, threshold {self.recovery_pct}%)")
                        new_status = prev_status  # Hold current status during observation
                    else:
                        recovery_start_dt = datetime.fromisoformat(self._recovery_start)
                        elapsed_minutes = (datetime.now(timezone.utc) - recovery_start_dt).total_seconds() / 60
                        if elapsed_minutes >= self.recovery_hold_minutes:
                            new_status = "WARNING"
                            self._recovery_start = None
                            logger.warning(
                                f"Recovery complete: {prev_status} -> WARNING after "
                                f"{elapsed_minutes:.0f}min (drawdown at completion: {drawdown_pct:.2f}%)"
                            )
                            send_pushover_notification(
                                self.notification_config,
                                f"📈 Recovery: {prev_status} → WARNING",
                                f"Drawdown improved to {drawdown_pct:.2f}% (held {elapsed_minutes:.0f}min). Trading resumed with caution.",
                                ticker="ALL"
                            )
                        else:
                            logger.info(f"Recovery in progress: {elapsed_minutes:.0f}/{self.recovery_hold_minutes}min")
                            new_status = prev_status  # Hold during observation
                elif abs(drawdown_pct) <= self.halt_pct:
                    # Between recovery_pct and halt_pct — hold status, keep timer intact.
                    # Timer continues running (wall-clock) — elapsed time includes
                    # danger-zone excursions. Recovery completion is only evaluated
                    # when drawdown is actually below recovery_pct at the moment of
                    # the check. Acceptable tradeoff vs. oscillation problem where
                    # any noise restarted the 30-min countdown indefinitely.
                    new_status = prev_status
                else:
                    # Back above halt_pct — genuine re-escalation, reset timer
                    if self._recovery_start is not None:
                        logger.info(f"Recovery timer reset (drawdown worsened past halt to {drawdown_pct:.2f}%)")
                        self._recovery_start = None
                    new_status = prev_status

            if new_status != prev_status:
                logger.warning(f"Drawdown Status Changed: {prev_status} -> {new_status} (PNL: {drawdown_pct:.2f}%)")
                self.state['status'] = new_status

                # Notifications
                if new_status == "WARNING":
                    send_pushover_notification(
                        self.notification_config,
                        "⚠️ Drawdown Warning",
                        f"Portfolio down {drawdown_pct:.2f}% today.",
                        ticker="ALL"
                    )
                elif new_status == "HALT":
                    send_pushover_notification(
                        self.notification_config,
                        "🛑 TRADING HALTED",
                        f"Daily loss limit hit ({drawdown_pct:.2f}%). New trades blocked.",
                        ticker="ALL"
                    )
                elif new_status == "PANIC":
                    send_pushover_notification(
                        self.notification_config,
                        "🚨 PANIC CLOSE TRIGGERED",
                        f"Critical loss ({drawdown_pct:.2f}%). Closing ALL positions.",
                        ticker="ALL"
                    )

            # Track whether PANIC was freshly evaluated (not loaded from disk)
            self._panic_is_live = (new_status == "PANIC")

            self._save_state()
            return new_status

        except Exception as e:
            logger.error(f"Drawdown check failed: {e}")
            return self.state['status']

    def is_entry_allowed(self) -> bool:
        """Check if new trades are allowed."""
        return self.state['status'] in ["NORMAL", "WARNING"]

    def should_panic_close(self) -> bool:
        """Check if we need to emergency close everything."""
        return self.state['status'] == "PANIC" and self._panic_is_live
