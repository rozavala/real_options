"""
Daily Drawdown Circuit Breaker.

Protects against aggregate portfolio losses that individual position stops miss.
Halts trading for the day if intraday P&L drops below configurable thresholds.

Thresholds (default):
- WARNING: -1.5% intraday -> Pushover alert
- HALT:    -2.5% intraday -> Block new trades
- PANIC:   -4.0% intraday -> Close ALL positions
"""

import logging
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional

from ib_insync import IB
from notifications import send_pushover_notification

logger = logging.getLogger(__name__)

class DrawdownGuard:
    def __init__(self, config: dict):
        self.config = config.get('drawdown_circuit_breaker', {})
        self.notification_config = config.get('notifications', {})
        self.enabled = self.config.get('enabled', False)
        self.warning_pct = self.config.get('warning_pct', 1.5)
        self.halt_pct = self.config.get('halt_pct', 2.5)
        self.panic_pct = self.config.get('panic_pct', 4.0)
        data_dir = config.get('data_dir', 'data')
        # Always use per-commodity data_dir; ignore any legacy state_file in sub-config
        self.state_file = os.path.join(data_dir, 'drawdown_state.json')

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
                        logger.info(f"Loaded drawdown state: {self.state['status']} ({self.state['current_drawdown_pct']:.2f}%)")
                    else:
                        logger.info("Saved drawdown state is old. Starting fresh.")
            except Exception as e:
                logger.warning(f"Failed to load drawdown state: {e}")

    def _save_state(self):
        try:
            # Create dir if needed
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
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

            # 2. Set Starting Equity if first run
            if self.state['starting_equity'] == 0.0:
                self.state['starting_equity'] = net_liq
                logger.info(f"DrawdownGuard initialized. Starting Equity: ${net_liq:,.2f}")
                self._save_state()
                return "NORMAL"

            # 3. Calculate Drawdown
            start_eq = self.state['starting_equity']
            pnl = net_liq - start_eq
            drawdown_pct = (pnl / start_eq) * 100

            # Only track negative drawdown (if we are up, drawdown is 0 for this purpose,
            # though technically we could track peak-to-trough if we updated starting_equity on highs.
            # For simplicity/safety, we stick to "Daily Loss Limit" logic (vs open).)
            # Wait, "Daily Drawdown" usually means from previous close.
            # If we restart mid-day, starting_equity might be mid-day equity.
            # To be robust, we should load yesterday's close from daily_equity.csv if starting_equity is 0.
            # But for this MVP, initializing on first run of day is acceptable,
            # assuming orchestrator runs before market open.

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

            # State Transition Logic (Escalation only, manual reset required for de-escalation usually,
            # but let's allow auto-recovery from WARNING. HALT/PANIC should stick?)
            # Prompt says "Reset daily at market open". So HALT persists for day.

            if prev_status == "PANIC":
                new_status = "PANIC" # Stick
            elif prev_status == "HALT" and new_status != "PANIC":
                new_status = "HALT" # Stick unless worsening

            if new_status != prev_status:
                logger.warning(f"Drawdown Status Changed: {prev_status} -> {new_status} (PNL: {drawdown_pct:.2f}%)")
                self.state['status'] = new_status

                # Notifications
                if new_status == "WARNING":
                    send_pushover_notification(
                        self.notification_config,
                        "⚠️ Drawdown Warning",
                        f"Portfolio down {drawdown_pct:.2f}% today."
                    )
                elif new_status == "HALT":
                    send_pushover_notification(
                        self.notification_config,
                        "🛑 TRADING HALTED",
                        f"Daily loss limit hit ({drawdown_pct:.2f}%). New trades blocked."
                    )
                elif new_status == "PANIC":
                    send_pushover_notification(
                        self.notification_config,
                        "🚨 PANIC CLOSE TRIGGERED",
                        f"Critical loss ({drawdown_pct:.2f}%). Closing ALL positions."
                    )

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
        return self.state['status'] == "PANIC"
