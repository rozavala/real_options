"""
Script to backfill reconciliation data for the Council History.
This manually triggers the logic that normally runs in the Orchestrator.
"""

import asyncio
import logging
import random
from ib_insync import IB
from config_loader import load_config
from trading_bot.reconciliation import reconcile_council_history

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Manual Council History Backfill...")

    # 1. Load Config
    config = load_config()
    if not config:
        logger.error("Could not load config.")
        return

    # 2. Connect to IB
    ib = IB()
    try:
        host = config['connection']['host']
        port = config['connection']['port']
        client_id = random.randint(2000, 4999) # Use different ID range

        logger.info(f"Connecting to IB Gateway at {host}:{port} (ID: {client_id})...")
        await ib.connectAsync(host, port, clientId=client_id)
        logger.info("Connected.")

        # 3. Run Reconciliation
        await reconcile_council_history(ib, config)

    except Exception as e:
        logger.error(f"Error during backfill: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Backfill interrupted by user.")
