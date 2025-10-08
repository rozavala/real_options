import asyncio
import logging
import os
import traceback

from coffee_factors_data_pull_new import main as run_data_pull
from send_data_to_api import send_data_and_get_prediction
from trading_bot.main import main_runner as run_trading_bot
from notifications import send_pushover_notification
import json

# --- Logging Setup ---
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'orchestrator.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
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


async def main():
    """
    Main orchestration function to run the trading bot pipeline.
    """
    logger.info("=============================================")
    logger.info("=== Starting the Trading Bot Orchestrator ===")
    logger.info("=============================================")

    config = load_config()
    if not config:
        logger.critical("Orchestrator cannot proceed without a valid configuration.")
        return

    try:
        # Step 1: Run the data pull process
        logger.info("--- Step 1: Kicking off data pull process ---")
        # The data pull script now returns True on success and False on failure
        data_pull_success = run_data_pull()
        if not data_pull_success:
            # The data pull script already sends a detailed notification on failure.
            # We just log it here and stop the process.
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
        logger.info(f"--- Successfully received {len(predictions)} prediction(s). ---")

        # Step 3: Run the trading bot with the predictions
        logger.info("\n--- Step 3: Starting the main trading bot logic ---")
        # The trading bot is a long-running process, so we await it.
        # It will handle its own notifications for trade execution.
        await run_trading_bot(signals=predictions)

    except Exception as e:
        error_msg = f"A critical error occurred in the orchestrator: {e}\n{traceback.format_exc()}"
        logger.critical(error_msg)
        send_pushover_notification(config.get('notifications', {}), "Orchestrator CRITICAL ERROR", error_msg)

    finally:
        logger.info("==========================================")
        logger.info("=== Orchestrator cycle has concluded. ===")
        logger.info("==========================================")


if __name__ == "__main__":
    asyncio.run(main())