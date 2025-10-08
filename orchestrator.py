import asyncio
import logging
import os
import signal
import sys
import traceback
from datetime import datetime, time, timedelta
import pytz

from ib_insync import IB

from coffee_factors_data_pull_new import main as run_data_pull
from config_loader import load_config
from logging_config import setup_logging
from notifications import send_pushover_notification
from performance_analyzer import analyze_performance
from send_data_to_api import send_data_and_get_prediction
from trading_bot.main import main_runner as run_trading_bot
from trading_bot.signal_generator import generate_signals

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("Orchestrator")

# --- Global Process Handle ---
monitor_process = None


async def start_monitoring(config: dict):
    """Starts the position_monitor.py script as a background process."""
    global monitor_process
    if monitor_process and monitor_process.returncode is None:
        logger.warning("Monitoring process is already running.")
        return

    try:
        logger.info("--- Starting position monitoring process ---")
        # Using sys.executable to ensure the same python interpreter is used
        monitor_process = await asyncio.create_subprocess_exec(
            sys.executable, 'position_monitor.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        logger.info(f"Successfully started position monitor with PID: {monitor_process.pid}")
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


async def run_trading_cycle(config: dict):
    """
    Runs one complete cycle of the trading bot pipeline.
    """
    logger.info("--- Starting new trading cycle ---")
    try:
        logger.info("--- Step 1: Kicking off data pull process ---")
        if not run_data_pull(config):
            logger.error("Data pull process failed. Aborting the trading cycle.")
            return
        logger.info("--- Data pull process completed successfully. ---")

        logger.info("\n--- Step 2: Fetching predictions from the API ---")
        predictions = send_data_and_get_prediction(config)
        if not predictions:
            send_pushover_notification(config.get('notifications', {}), "Orchestrator Failure", "Failed to get predictions from the API.")
            return
        logger.info(f"--- Successfully received predictions from API. ---")

        logger.info("\n--- Step 2.5: Generating structured signals ---")
        ib = IB()
        signals = []
        try:
            conn_settings = config.get('connection', {})
            await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=conn_settings.get('clientId', 10))
            signals = await generate_signals(ib, predictions, config)
        finally:
            if ib.isConnected():
                ib.disconnect()

        if not signals:
            logger.info("No actionable trading signals were generated. Concluding cycle.")
            return

        logger.info("\n--- Step 3: Starting the main trading bot logic with generated signals ---")
        await run_trading_bot(config, signals=signals)

    except Exception as e:
        error_msg = f"A critical error occurred during a trading cycle: {e}\n{traceback.format_exc()}"
        logger.critical(error_msg)
        send_pushover_notification(config.get('notifications', {}), "Orchestrator CRITICAL ERROR", error_msg)


def get_next_task(now_gmt: datetime, schedule: dict) -> (datetime, callable):
    """Calculates the next scheduled task and its run time."""
    next_run_time = None
    next_task = None

    sorted_times = sorted(schedule.keys(), key=lambda t: (t.hour, t.minute))

    for rt in sorted_times:
        run_datetime = now_gmt.replace(hour=rt.hour, minute=rt.minute, second=0, microsecond=0)
        if run_datetime > now_gmt:
            if next_run_time is None or run_datetime < next_run_time:
                next_run_time = run_datetime
                next_task = schedule[rt]

    if next_run_time is None:
        first_run_time_tomorrow = now_gmt.replace(
            hour=sorted_times[0].hour, minute=sorted_times[0].minute, second=0, microsecond=0
        ) + timedelta(days=1)
        next_run_time = first_run_time_tomorrow
        next_task = schedule[sorted_times[0]]

    return next_run_time, next_task


async def main():
    """Main long-running orchestrator process."""
    logger.info("=============================================")
    logger.info("=== Starting the Trading Bot Orchestrator ===")
    logger.info("=============================================")

    config = load_config()
    if not config:
        logger.critical("Orchestrator cannot start without a valid configuration.")
        return

    # Schedule mapping run times (GMT) to functions
    # Market hours are assumed to be roughly 08:00 to 22:00 GMT
    schedule = {
        time(7, 55): start_monitoring,      # Start monitoring just before trading
        time(8, 0): run_trading_cycle,       # Run the main trading logic
        time(21, 55): stop_monitoring,      # Stop monitoring after market hours
        time(22, 0): analyze_performance    # Run performance analysis
    }

    try:
        while True:
            try:
                gmt = pytz.timezone('GMT')
                now_gmt = datetime.now(gmt)

                next_run_time, next_task_func = get_next_task(now_gmt, schedule)
                wait_seconds = (next_run_time - now_gmt).total_seconds()

                task_name = next_task_func.__name__
                logger.info(f"Next task '{task_name}' scheduled for {next_run_time.strftime('%Y-%m-%d %H:%M:%S GMT')}. "
                            f"Waiting for {wait_seconds / 3600:.2f} hours.")

                await asyncio.sleep(wait_seconds)

                logger.info(f"--- Running scheduled task: {task_name} ---")
                if asyncio.iscoroutinefunction(next_task_func):
                    await next_task_func(config)
                else:
                    next_task_func(config)

            except Exception as e:
                error_msg = f"A critical error occurred in the main orchestrator loop: {e}\n{traceback.format_exc()}"
                logger.critical(error_msg)
                await asyncio.sleep(60)
    finally:
        logger.info("Orchestrator shutting down. Ensuring monitor is stopped.")
        await stop_monitoring(config)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Orchestrator stopped by user.")