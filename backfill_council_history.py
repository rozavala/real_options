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

    # 2. Run Reconciliation (It handles connection)
    await reconcile_council_history(config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Backfill interrupted by user.")
