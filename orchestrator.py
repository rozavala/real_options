import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, time, timedelta
import pytz

from ib_insync import IB

from coffee_factors_data_pull_new import main as run_data_pull
from notifications import send_pushover_notification
from send_data_to_api import send_data_and_get_prediction
from trading_bot.main import main_runner as run_trading_bot
from trading_bot.signal_generator import generate_signals

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Orchestrator")


def load_config():
    """Loads the configuration from config.json."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        return None


async def run_trading_cycle():
    """
    Runs one complete cycle of the trading bot pipeline.
    """
    logger.info("--- Starting new trading cycle ---")
    config = load_config()
    if not config:
        logger.critical("Orchestrator cannot proceed without a valid configuration.")
        return

    try:
        # Step 1: Run the data pull process
        logger.info("--- Step 1: Kicking off data pull process ---")
        data_pull_success = run_data_pull()
        if not data_pull_success:
            logger.error("Data pull process failed. Aborting the trading cycle.")
            return
        logger.info("--- Data pull process completed successfully. ---")

        # Step 2: Send data to the API and get predictions
        logger.info("\n--- Step 2: Fetching predictions from the API ---")
        predictions = send_data_and_get_prediction()

        if not predictions:
            error_msg = "Failed to get predictions from the API. Aborting trading cycle."
            logger.error(error_msg)
            send_pushover_notification(config.get('notifications', {}), "Orchestrator Failure", error_msg)
            return
        logger.info(f"--- Successfully received predictions from API. ---")

        # Step 2.5: Generate structured signals from raw predictions
        logger.info("\n--- Step 2.5: Generating structured signals ---")
        ib = IB()
        signals = []
        try:
            conn_settings = config.get('connection', {})
            await ib.connectAsync(
                host=conn_settings.get('host', '127.0.0.1'),
                port=conn_settings.get('port', 7497),
                clientId=conn_settings.get('clientId', 10)  # Use a different client ID for this short-lived connection
            )
            signals = await generate_signals(ib, predictions, config)
        except Exception as e:
            logger.error(f"An error occurred during signal generation: {e}", exc_info=True)
            send_pushover_notification(config.get('notifications', {}), "Signal Generation Failed", str(e))
        finally:
            if ib.isConnected():
                ib.disconnect()

        if not signals:
            logger.info("No actionable trading signals were generated. Concluding cycle.")
            return

        # Step 3: Run the trading bot with the structured signals
        logger.info("\n--- Step 3: Starting the main trading bot logic with generated signals ---")
        await run_trading_bot(signals=signals)

    except Exception as e:
        error_msg = f"A critical error occurred during a trading cycle: {e}\n{traceback.format_exc()}"
        logger.critical(error_msg)
        send_pushover_notification(config.get('notifications', {}), "Orchestrator CRITICAL ERROR", error_msg)


def get_next_run_time(now_gmt: datetime, run_times_gmt: list) -> datetime:
    """Calculates the next scheduled run time."""
    next_run_time = None
    for rt in run_times_gmt:
        run_datetime = now_gmt.replace(hour=rt.hour, minute=rt.minute, second=0, microsecond=0)
        if run_datetime > now_gmt:
            if next_run_time is None or run_datetime < next_run_time:
                next_run_time = run_datetime

    # If all run times today are past, schedule for the first time tomorrow
    if next_run_time is None:
        first_run_time_tomorrow = now_gmt.replace(
            hour=run_times_gmt[0].hour, minute=run_times_gmt[0].minute, second=0, microsecond=0
        ) + timedelta(days=1)
        next_run_time = first_run_time_tomorrow

    return next_run_time


async def main():
    """
    Main long-running orchestrator process.
    """
    logger.info("=============================================")
    logger.info("=== Starting the Trading Bot Orchestrator ===")
    logger.info("=============================================")

    run_times_gmt = sorted([time(0, 30), time(8, 0)])

    while True:
        try:
            gmt = pytz.timezone('GMT')
            now_gmt = datetime.now(gmt)

            next_run = get_next_run_time(now_gmt, run_times_gmt)
            wait_seconds = (next_run - now_gmt).total_seconds()

            logger.info(f"Next trading cycle scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S GMT')}. "
                        f"Waiting for {wait_seconds / 3600:.2f} hours.")

            await asyncio.sleep(wait_seconds)

            await run_trading_cycle()

        except Exception as e:
            error_msg = f"A critical error occurred in the main orchestrator loop: {e}\n{traceback.format_exc()}"
            logger.critical(error_msg)
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())