"""The main orchestrator for the automated trading bot.

This script serves as the central nervous system of the application. It runs
as a long-lived process, responsible for scheduling and executing all the
different components of the trading pipeline at the correct times.

The orchestrator manages a daily schedule that includes:
- Generating and queuing orders pre-market.
- Starting a separate position monitoring process.
- Placing the queued orders after the market opens.
- Closing all positions before the market closes.
- Canceling any remaining open orders.
- Stopping the position monitoring process.
- Running a performance analysis at the end of the day.
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime, time, timedelta
import pytz

from config_loader import load_config
from logging_config import setup_logging
from notifications import send_pushover_notification
from performance_analyzer import main as run_performance_analysis
from trading_bot.order_manager import (
    generate_and_queue_orders,
    place_queued_orders,
    close_all_open_positions,
    cancel_all_open_orders,
)
from trading_bot.utils import archive_trade_ledger

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("Orchestrator")

# --- Global Process Handle for the monitor ---
monitor_process = None


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
    The performance analyzer is responsible for generating and sending the report.
    """
    logger.info("--- Initiating end-of-day analysis and archiving ---")
    try:
        # 1. Run the full analysis and reporting process
        await run_performance_analysis()

        # 2. Archive the ledger
        archive_trade_ledger()

        logger.info("--- End-of-day analysis and archiving complete ---")

    except Exception as e:
        logger.critical(f"An error occurred during the analysis and archiving process: {e}\n{traceback.format_exc()}")

# New schedule mapping run times (GMT) to functions
schedule = {
    time(8, 30): start_monitoring,
    time(8, 50): generate_and_queue_orders,
    time(8, 52): place_queued_orders,
    time(17, 20): close_all_open_positions,
    time(17, 22): cancel_and_stop_monitoring,
    time(17, 35): analyze_and_archive
}

async def main():
    """The main long-running orchestrator process."""
    logger.info("=============================================")
    logger.info("=== Starting the Trading Bot Orchestrator ===")
    logger.info("=============================================")

    config = load_config()
    if not config:
        logger.critical("Orchestrator cannot start without a valid configuration."); return


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
                # All scheduled tasks that take config are now async
                await next_task_func(config)

            except asyncio.CancelledError:
                logger.info("Orchestrator main loop cancelled."); break
            except Exception as e:
                error_msg = f"A critical error occurred in the main orchestrator loop: {e}\n{traceback.format_exc()}"
                logger.critical(error_msg)
                await asyncio.sleep(60)
    finally:
        logger.info("Orchestrator shutting down. Ensuring monitor is stopped.")
        if monitor_process and monitor_process.returncode is None:
            await stop_monitoring(config)


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
