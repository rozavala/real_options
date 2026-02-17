"""
Script to backfill reconciliation data for the Council History.
This manually triggers the logic that normally runs in the Orchestrator.
"""

import asyncio
import logging
import os
import argparse
from config_loader import load_config
from trading_bot.reconciliation import reconcile_council_history

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main(commodity_ticker: str):
    logger.info(f"Starting Manual Council History Backfill for {commodity_ticker}...")

    # 1. Load Config
    config = load_config()
    if not config:
        logger.error("Could not load config.")
        return

    # 2. Inject data_dir for commodity isolation
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', commodity_ticker)
    config['data_dir'] = data_dir
    config.setdefault('commodity', {})['ticker'] = commodity_ticker

    # 3. Run Reconciliation (It handles connection)
    await reconcile_council_history(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Council History")
    parser.add_argument('--commodity', type=str,
                        default=os.environ.get("COMMODITY_TICKER", "KC"),
                        help="Commodity ticker (e.g. KC, CC)")
    args = parser.parse_args()
    ticker = args.commodity.upper()

    try:
        asyncio.run(main(ticker))
    except KeyboardInterrupt:
        logger.info("Backfill interrupted by user.")
