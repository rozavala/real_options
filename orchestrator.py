"""The main orchestrator for the automated trading bot.

This script serves as the central nervous system of the application. It runs
as a long-lived process, responsible for scheduling and executing all the
different components of the trading pipeline at the correct times.

It now supports an Event-Driven architecture via a Sentinel Loop.
"""

import asyncio
import logging
import sys
import traceback
import os
import random
import json
import hashlib
from collections import deque
import time as time_module
from datetime import datetime, time, timedelta
import pytz
from ib_insync import IB, util

from config_loader import load_config
from trading_bot.logging_config import setup_logging
from notifications import send_pushover_notification
from performance_analyzer import main as run_performance_analysis
from reconcile_trades import main as run_reconciliation, reconcile_active_positions
from trading_bot.reconciliation import reconcile_council_history
from trading_bot.order_manager import (
    generate_and_execute_orders,
    close_stale_positions,
    cancel_all_open_orders,
    place_queued_orders,
    ORDER_QUEUE
)
from trading_bot.utils import archive_trade_ledger, configure_market_data_type
from equity_logger import log_equity_snapshot, sync_equity_from_flex
from trading_bot.sentinels import PriceSentinel, WeatherSentinel, LogisticsSentinel, NewsSentinel, SentinelTrigger
from trading_bot.microstructure_sentinel import MicrostructureSentinel
from trading_bot.agents import CoffeeCouncil
from trading_bot.ib_interface import get_active_futures, build_option_chain, create_combo_order_object
from trading_bot.strategy import define_directional_strategy
from trading_bot.state_manager import StateManager
from trading_bot.connection_pool import IBConnectionPool

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("Orchestrator")

# --- Global Process Handle for the monitor ---
monitor_process = None

class TriggerDeduplicator:
    def __init__(self, window_seconds: int = 1800, state_file="data/deduplicator_state.json"):
        self.state_file = state_file
        self.window = window_seconds
        self.global_cooldown_until = 0
        self.recent_triggers = deque(maxlen=50)
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.global_cooldown_until = data.get('global_cooldown_until', 0)
                    for t in data.get('recent_triggers', []):
                         self.recent_triggers.append(tuple(t)) # (hash, timestamp)
            except Exception as e:
                logger.warning(f"Failed to load deduplicator state: {e}")

    def _save_state(self):
        try:
            # Create data dir if not exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {
                'global_cooldown_until': self.global_cooldown_until,
                'recent_triggers': list(self.recent_triggers)
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
             logger.warning(f"Failed to save deduplicator state: {e}")

    def should_process(self, trigger: SentinelTrigger) -> bool:
        now = time_module.time()

        # Global cooldown check
        if now < self.global_cooldown_until:
            logger.info(f"Global cooldown active. Skipping {trigger.source}")
            return False

        # Content-based deduplication
        trigger_hash = hashlib.md5(
            f"{trigger.reason[:50]}{trigger.payload}".encode()
        ).hexdigest()[:8]

        # Prune old triggers
        cutoff = now - self.window
        while self.recent_triggers and self.recent_triggers[0][1] < cutoff:
             self.recent_triggers.popleft()

        if any(t[0] == trigger_hash for t in self.recent_triggers):
            logger.info(f"Duplicate trigger detected: {trigger_hash}")
            return False

        self.recent_triggers.append((trigger_hash, now))
        self._save_state()
        return True

    def set_global_cooldown(self, seconds: int = 900):
        self.global_cooldown_until = time_module.time() + seconds
        self._save_state()

    def clear_global_cooldown(self):
        self.global_cooldown_until = 0
        self._save_state()

# Global Deduplicator Instance
GLOBAL_DEDUPLICATOR = TriggerDeduplicator()


async def log_stream(stream, logger_func):
    """Reads and logs lines from a subprocess stream."""
    while True:
        line = await stream.readline()
        if line:
            logger_func(line.decode('utf-8').strip())
        else:
            break


async def start_monitoring(config: dict):
    """Starts the `position_monitor.py` script as a background process."""
    global monitor_process
    if monitor_process and monitor_process.returncode is None:
        logger.warning("Monitoring process is already running.")
        return

    try:
        logger.info("--- Starting position monitoring process ---")
        monitor_process = await asyncio.create_subprocess_exec(
            sys.executable, 'position_monitor.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE  # Capture both stdout and stderr
        )
        logger.info(f"Successfully started position monitor with PID: {monitor_process.pid}")

        # Create tasks to log the output from the monitor process in the background
        asyncio.create_task(log_stream(monitor_process.stdout, logger.info))
        asyncio.create_task(log_stream(monitor_process.stderr, logger.error))

        send_pushover_notification(config.get('notifications', {}), "Orchestrator", "Started position monitoring service.")
    except Exception as e:
        logger.critical(f"Failed to start position monitor: {e}\n{traceback.format_exc()}")
        send_pushover_notification(config.get('notifications', {}), "Orchestrator CRITICAL", "Failed to start position monitor.")


async def stop_monitoring(config: dict):
    """Stops the background position monitoring process."""
    global monitor_process
    if not monitor_process or monitor_process.returncode is not None:
        logger.warning("Monitoring process is not running or has already terminated.")
        return

    try:
        logger.info(f"--- Stopping position monitoring process (PID: {monitor_process.pid}) ---")
        monitor_process.terminate()
        await monitor_process.wait()
        logger.info("Position monitoring process has been successfully terminated.")
        send_pushover_notification(config.get('notifications', {}), "Orchestrator", "Stopped position monitoring service.")
        monitor_process = None
    except ProcessLookupError:
        logger.warning("Process already terminated.")
    except Exception as e:
        logger.critical(f"An error occurred while stopping the monitor: {e}\n{traceback.format_exc()}")


async def cancel_and_stop_monitoring(config: dict):
    """Wrapper task to cancel open orders and then stop the monitor."""
    logger.info("--- Initiating end-of-day shutdown sequence ---")
    await cancel_all_open_orders(config)
    await stop_monitoring(config)
    logger.info("--- End-of-day shutdown sequence complete ---")


def get_next_task(now_gmt: datetime, schedule: dict) -> tuple[datetime, callable]:
    """Calculates the next scheduled task and its run time based on a schedule."""
    next_run_time, next_task = None, None
    sorted_times = sorted(schedule.keys(), key=lambda t: (t.hour, t.minute))

    for rt in sorted_times:
        run_datetime = now_gmt.replace(hour=rt.hour, minute=rt.minute, second=0, microsecond=0)
        if run_datetime > now_gmt:
            if next_run_time is None or run_datetime < next_run_time:
                next_run_time, next_task = run_datetime, schedule[rt]
                break

    if next_run_time is None:
        first_run_time_tomorrow = now_gmt.replace(
            hour=sorted_times[0].hour, minute=sorted_times[0].minute, second=0, microsecond=0
        ) + timedelta(days=1)
        next_run_time, next_task = first_run_time_tomorrow, schedule[sorted_times[0]]

    return next_run_time, next_task


async def analyze_and_archive(config: dict):
    """
    Triggers the performance analysis and then archives the trade ledger.
    """
    logger.info("--- Initiating end-of-day analysis and archiving ---")
    try:
        await run_performance_analysis()
        archive_trade_ledger()
        logger.info("--- End-of-day analysis and archiving complete ---")
    except Exception as e:
        logger.critical(f"An error occurred during the analysis and archiving process: {e}\n{traceback.format_exc()}")


async def reconcile_and_notify(config: dict):
    """Runs the trade reconciliation and sends a notification if discrepancies are found."""
    logger.info("--- Starting trade reconciliation ---")
    try:
        missing_df, superfluous_df = await run_reconciliation()

        if not missing_df.empty or not superfluous_df.empty:
            logger.warning("Trade reconciliation found discrepancies.")
            message = ""
            if not missing_df.empty:
                message += f"Found {len(missing_df)} missing trades in the local ledger.\n"
            if not superfluous_df.empty:
                message += f"Found {len(superfluous_df)} superfluous trades in the local ledger.\n"
            message += "Check the `archive_ledger` directory for details."

            send_pushover_notification(
                config.get('notifications', {}),
                "Trade Reconciliation Alert",
                message
            )
        else:
            logger.info("Trade reconciliation complete. No discrepancies found.")

        await reconcile_active_positions(config)
        await reconcile_council_history(config)

    except Exception as e:
        logger.critical(f"An error occurred during trade reconciliation: {e}\n{traceback.format_exc()}")


async def reconcile_and_analyze(config: dict):
    """Runs reconciliation, then analysis and archiving."""
    logger.info("--- Kicking off end-of-day reconciliation and analysis process ---")
    await sync_equity_from_flex(config)
    await reconcile_and_notify(config)
    await analyze_and_archive(config)
    logger.info("--- End-of-day reconciliation and analysis process complete ---")


# --- SENTINEL LOGIC ---

async def run_emergency_cycle(trigger: SentinelTrigger, config: dict, ib: IB):
    """
    Runs a specialized cycle triggered by a Sentinel.
    Executes trades if the Council approves.
    """
    logger.info(f"🚨 EMERGENCY CYCLE TRIGGERED by {trigger.source}: {trigger.reason}")
    send_pushover_notification(config.get('notifications', {}), f"Sentinel Trigger: {trigger.source}", trigger.reason)

    # --- DEFCON 1: Crash Protection ---
    # If price drops > 5% instantly, do NOT open new trades. Liquidation logic is complex, so we just Halt.
    if trigger.source == "PriceSentinel" and abs(trigger.payload.get('change', 0)) > 5.0:
        logger.critical("📉 FLASH CRASH DETECTED (>5%). Skipping Council. HALTING TRADING.")
        send_pushover_notification(config.get('notifications', {}), "FLASH CRASH ALERT", "Price moved >5%. Emergency Halt Triggered. No new orders.")
        # Future: await close_stale_positions(config, force=True)
        return

    try:
        # 1. Initialize Council
        council = CoffeeCouncil(config)

        # 2. Get Active Futures (We need a target contract)
        # For simplicity, target the Front Month or the one from the trigger payload
        contract_name_hint = trigger.payload.get('contract')

        active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=2)
        if not active_futures:
            logger.error("No active futures found for emergency cycle.")
            return

        # Select contract
        target_contract = active_futures[0]
        if contract_name_hint:
            # Try to match hint
            for f in active_futures:
                if f.localSymbol == contract_name_hint:
                    target_contract = f
                    break

        contract_name = f"{target_contract.localSymbol} ({target_contract.lastTradeDateOrContractMonth[:6]})"
        logger.info(f"Targeting contract: {contract_name}")

        # 3. Get Market Context (Snapshot)
        ticker = ib.reqMktData(target_contract, '', True, False)
        await asyncio.sleep(2)
        market_context_str = f"Live Price: {ticker.last if ticker.last else 'N/A'}"

        # 4. Load Cached ML Signal (Fix Dummy Signal Blindness)
        cached_state = StateManager.load_state()
        cached_ml_signals = cached_state.get('latest_ml_signals', {}).get('data', [])

        ml_signal = {
            "action": "NEUTRAL",
            "confidence": 0.5,
            "price": ticker.last,
            "reason": "Emergency Cycle: ML Signal unavailable/stale.",
            "regime": "UNKNOWN"
        }

        if cached_ml_signals:
            # Try to find signal for this contract month
            target_month = target_contract.lastTradeDateOrContractMonth[:6]
            found_signal = next((s for s in cached_ml_signals if s.get('contract_month') == target_month), None)
            if found_signal:
                ml_signal = found_signal
                logger.info(f"Loaded cached ML signal for {target_month}: {ml_signal.get('direction')}")

        # 5. Run Specialized Cycle
        decision = await council.run_specialized_cycle(trigger, contract_name, ml_signal, market_context_str)

        logger.info(f"Emergency Decision: {decision.get('direction')} ({decision.get('confidence')})")

        # 6. Execute if Actionable
        if decision.get('direction') in ['BULLISH', 'BEARISH'] and decision.get('confidence', 0) > config.get('strategy', {}).get('signal_threshold', 0.5):
            logger.info("Decision is actionable. Generating order...")

            # Build Strategy
            chain = await build_option_chain(ib, target_contract)
            if not chain:
                logger.warning("No option chain available.")
                return

            # Construct Signal Object for Strategy Definition
            signal_obj = {
                "contract_month": target_contract.lastTradeDateOrContractMonth[:6],
                "direction": decision['direction'],
                "confidence": decision['confidence'],
                "price": ticker.last,
                "prediction_type": "DIRECTIONAL"
            }

            strategy_def = define_directional_strategy(config, signal_obj, chain, ticker.last, target_contract)

            if strategy_def:
                order_objects = await create_combo_order_object(ib, config, strategy_def)
                if order_objects:
                    contract, order = order_objects

                    # Queue and Execute (Fix Global Queue Collision)
                    # Pass specific list to place_queued_orders so we don't wipe the global queue
                    emergency_order_list = [(contract, order, decision)]
                    await place_queued_orders(config, orders_list=emergency_order_list)
                    logger.info("Emergency Order Placed.")
        else:
            logger.info("Emergency Cycle concluded with no action.")

    except Exception as e:
        logger.error(f"Emergency Cycle Failed: {e}\n{traceback.format_exc()}")


async def run_sentinels(config: dict):
    """
    Main loop for Sentinels. Runs concurrently with the scheduler.
    """
    logger.info("--- Starting Sentinel Array ---")

    # Sentinel Config
    try:
        sentinel_ib = await IBConnectionPool.get_connection("sentinel", config)
        configure_market_data_type(sentinel_ib)
    except Exception as e:
        logger.error(f"Sentinel IB Connection Failed: {e}")
        sentinel_ib = IB() # Fallback

    price_sentinel = PriceSentinel(config, sentinel_ib)
    weather_sentinel = WeatherSentinel(config)
    logistics_sentinel = LogisticsSentinel(config)
    news_sentinel = NewsSentinel(config)
    microstructure_sentinel = MicrostructureSentinel(config, sentinel_ib)

    # Subscribe microstructure sentinel to front-month contract
    try:
        active_futures = await get_active_futures(sentinel_ib, config['symbol'], config['exchange'])
        if active_futures:
            await microstructure_sentinel.subscribe_contract(active_futures[0])
    except Exception as e:
        logger.error(f"Failed to subscribe microstructure sentinel: {e}")

    # Timing state
    last_weather = 0
    last_logistics = 0
    last_news = 0

    while True:
        try:
            now = time_module.time()

            # --- Auto-Reconnect for Zombie Sentinel Risk ---
            if not sentinel_ib.isConnected():
                try:
                    logger.warning("Sentinel IB connection lost. Attempting reconnect...")
                    sentinel_ib = await IBConnectionPool.get_connection("sentinel", config)
                    configure_market_data_type(sentinel_ib)
                    logger.info("Sentinel IB reconnected successfully.")
                except Exception as e:
                    logger.error(f"Sentinel IB Reconnect Failed: {e}")

            # Use asyncio.create_task for non-blocking execution (Fix Blocking Sentinel)

            # 1. Price Sentinel (Every 1 min) - Only if Connected
            if sentinel_ib.isConnected():
                trigger = await price_sentinel.check()
                if trigger and GLOBAL_DEDUPLICATOR.should_process(trigger):
                    asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                    GLOBAL_DEDUPLICATOR.set_global_cooldown(900)

            # 2. Weather Sentinel (Every 4 hours = 14400s)
            if now - last_weather > 14400:
                trigger = await weather_sentinel.check()
                if trigger and GLOBAL_DEDUPLICATOR.should_process(trigger):
                    asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                    GLOBAL_DEDUPLICATOR.set_global_cooldown(900)
                last_weather = now

            # 3. Logistics Sentinel (Every 6 hours = 21600s)
            if now - last_logistics > 21600:
                trigger = await logistics_sentinel.check()
                if trigger and GLOBAL_DEDUPLICATOR.should_process(trigger):
                    asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                    GLOBAL_DEDUPLICATOR.set_global_cooldown(900)
                last_logistics = now

            # 4. News Sentinel (Every 1 hour = 3600s)
            if now - last_news > 3600:
                trigger = await news_sentinel.check()
                if trigger and GLOBAL_DEDUPLICATOR.should_process(trigger):
                    asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                    GLOBAL_DEDUPLICATOR.set_global_cooldown(900)
                last_news = now

            # 5. Microstructure Sentinel (Every 1 min with Price Sentinel)
            if sentinel_ib.isConnected():
                micro_trigger = await microstructure_sentinel.check()
                if micro_trigger:
                    logger.warning(f"MICROSTRUCTURE ALERT: {micro_trigger.reason}")
                    if micro_trigger.severity >= 7:
                        asyncio.create_task(run_emergency_cycle(micro_trigger, config, sentinel_ib))
                    else:
                        # Log warning but don't trigger full cycle
                        send_pushover_notification(
                            config.get('notifications', {}),
                            "Microstructure Warning",
                            f"{micro_trigger.reason} (Severity: {micro_trigger.severity:.1f})"
                        )

            await asyncio.sleep(60) # Loop tick

        except asyncio.CancelledError:
            logger.info("Sentinel Loop Cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in Sentinel Loop: {e}")
            await asyncio.sleep(60)

    # IBConnectionPool manages disconnects, so explicit disconnect is not strictly needed,
    # but good practice if task is cancelled.
    # However, pool is shared (conceptually not shared here but...)
    # We leave it connected.


# --- Main Schedule ---

schedule = {
    time(8, 30): start_monitoring,
    time(14, 0): generate_and_execute_orders,
    time(17, 20): close_stale_positions,
    time(17, 22): cancel_and_stop_monitoring,
    time(17, 25): log_equity_snapshot,
    time(17, 35): reconcile_and_analyze
}

def apply_schedule_offset(original_schedule: dict, offset_minutes: int) -> dict:
    new_schedule = {}
    today = datetime.now().date()
    for run_time, task_func in original_schedule.items():
        dt_original = datetime.combine(today, run_time)
        dt_shifted = dt_original + timedelta(minutes=offset_minutes)
        new_schedule[dt_shifted.time()] = task_func
    return new_schedule

async def main():
    """The main long-running orchestrator process."""
    logger.info("=============================================")
    logger.info("=== Starting the Trading Bot Orchestrator ===")
    logger.info("=============================================")

    config = load_config()
    if not config:
        logger.critical("Orchestrator cannot start without a valid configuration."); return
    
    env_name = os.getenv("ENV_NAME", "DEV") 
    is_prod = env_name == "PROD 🚀"

    current_schedule = schedule
    if not is_prod:
        logger.info(f"Environment: {env_name}. Applying -5 minute 'Civil War' avoidance offset.")
        current_schedule = apply_schedule_offset(schedule, offset_minutes=-10)
    else:
        logger.info("Environment: PROD 🚀. Using standard master schedule.")

    # Start Sentinels in background
    sentinel_task = asyncio.create_task(run_sentinels(config))

    try:
        while True:
            try:
                gmt = pytz.timezone('GMT')
                now_gmt = datetime.now(gmt)
                next_run_time, next_task_func = get_next_task(now_gmt, current_schedule)
                wait_seconds = (next_run_time - now_gmt).total_seconds()

                task_name = next_task_func.__name__
                logger.info(f"Next task '{task_name}' scheduled for {next_run_time.strftime('%Y-%m-%d %H:%M:%S GMT')}. "
                            f"Waiting for {wait_seconds / 3600:.2f} hours.")

                await asyncio.sleep(wait_seconds)

                logger.info(f"--- Running scheduled task: {task_name} ---")

                # Set global cooldown during scheduled cycle (e.g. 10 mins)
                # This prevents Sentinels from firing Emergency Cycles while we are busy
                GLOBAL_DEDUPLICATOR.set_global_cooldown(600)

                try:
                    await next_task_func(config)
                finally:
                    # Clear cooldown immediately after task finishes
                    GLOBAL_DEDUPLICATOR.clear_global_cooldown()

            except asyncio.CancelledError:
                logger.info("Orchestrator main loop cancelled."); break
            except Exception as e:
                error_msg = f"A critical error occurred in the main orchestrator loop: {e}\n{traceback.format_exc()}"
                logger.critical(error_msg)
                await asyncio.sleep(60)
    finally:
        logger.info("Orchestrator shutting down. Ensuring monitor is stopped.")
        sentinel_task.cancel()
        if monitor_process and monitor_process.returncode is None:
            await stop_monitoring(config)
        await IBConnectionPool.release_all()


async def sequential_main():
    for task in schedule.values():
        await task()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        asyncio.run(sequential_main())
    else:
        loop = asyncio.get_event_loop()
        main_task = None
        try:
            main_task = loop.create_task(main())
            loop.run_until_complete(main_task)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Orchestrator stopped by user.")
            if main_task:
                main_task.cancel()
                loop.run_until_complete(main_task)
        finally:
            loop.close()
