"""The main orchestrator for the automated trading bot.

This script serves as the central nervous system of the application. It runs
as a long-lived process, responsible for scheduling and executing all the
different components of the trading pipeline at the correct times.

The orchestrator manages a daily schedule that includes:
- Starting a separate position monitoring process and waiting for it to be ready.
- Generating and queuing orders pre-market.
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
from performance_analyzer import analyze_performance
from trading_bot.order_manager import (
    generate_and_queue_orders,
    close_all_open_positions,
    cancel_all_open_orders,
)
from trading_bot.utils import archive_trade_ledger
from trading_bot.performance_graphs import generate_performance_chart

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("Orchestrator")

# --- Global Process Handle for the monitor ---
monitor_process = None


async def log_stream(stream, logger_func, readiness_event=None, readiness_signal=""):
    """
    Reads and logs lines from a subprocess stream. If a readiness_event is
    provided, it sets the event when the readiness_signal is detected.
    """
    while True:
        try:
            line_bytes = await stream.readline()
            if line_bytes:
                line = line_bytes.decode('utf-8').strip()
                logger_func(line)
                if readiness_event and readiness_signal and readiness_signal in line:
                    readiness_event.set()
            else:
                break
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in log_stream: {e}")
            break


async def start_monitoring(config: dict):
    """
    Starts the `position_monitor.py` script as a background process and waits
    for it to signal that it's ready before proceeding.
    """
    global monitor_process
    if monitor_process and monitor_process.returncode is None:
        logger.warning("Monitoring process is already running.")
        return

    try:
        logger.info("--- Starting position monitoring process ---")
        readiness_event = asyncio.Event()
        readiness_signal = "MONITOR_READY"

        monitor_process = await asyncio.create_subprocess_exec(
            sys.executable, 'position_monitor.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        logger.info(f"Successfully started position monitor with PID: {monitor_process.pid}")

        # Create tasks to log output and watch for the readiness signal
        asyncio.create_task(log_stream(monitor_process.stdout, logger.info, readiness_event, readiness_signal))
        asyncio.create_task(log_stream(monitor_process.stderr, logger.error))

        # Wait for the monitor to be ready, with a timeout
        logger.info("Waiting for position monitor to signal readiness...")
        await asyncio.wait_for(readiness_event.wait(), timeout=120.0)

        logger.info("--- Position monitor is ready. Orchestrator can now proceed. ---")
        send_pushover_notification(config.get('notifications', {}), "Orchestrator", "Position monitoring service is online and ready.")
    except asyncio.TimeoutError:
        logger.critical("Timeout: Position monitor did not become ready in 120 seconds. Terminating.")
        await stop_monitoring(config)
        raise
    except Exception as e:
        logger.critical(f"Failed to start position monitor: {e}\n{traceback.format_exc()}")
        send_pushover_notification(config.get('notifications', {}), "Orchestrator CRITICAL", "Failed to start position monitor.")
        raise


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
    Analyzes performance, generates a report with a chart, sends a notification,
    and then archives the trade ledger.
    """
    logger.info("--- Initiating end-of-day analysis, reporting, and archiving ---")
    try:
        analysis_result = analyze_performance(config)
        if not analysis_result:
            logger.error("Performance analysis failed. Skipping report and archiving.")
            return

        report_text, total_pnl, chart_path = analysis_result
        notification_title = f"Daily Report: P&L ${total_pnl:,.2f}"
        send_pushover_notification(
            config.get('notifications', {}),
            title=notification_title,
            message=report_text,
            attachment_path=chart_path
        )
        archive_trade_ledger()
        logger.info("--- End-of-day analysis, reporting, and archiving complete ---")
    except Exception as e:
        logger.critical(f"An error occurred during the analysis and archiving process: {e}\n{traceback.format_exc()}")

# The schedule is crucial. start_monitoring must run before generate_and_queue_orders.
schedule = {
    time(8, 30): start_monitoring,             # Start monitor well before trading
    time(8, 50): generate_and_queue_orders,    # Generate signals after monitor is ready
    time(17, 10): close_all_open_positions,   # Close positions before market close
    time(17, 8): cancel_and_stop_monitoring,  # Cancel leftovers and stop monitor
    time(18, 0): analyze_and_archive          # Analyze after everything is shut down
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

if __name__ == "__main__":
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