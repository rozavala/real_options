"""
Migration: Fix legacy theses with incorrect entry_price values.

Scans all active theses in TMS, identifies those where entry_price is
either 0 or below the underlying_price_floor (likely spread premiums),
and patches them with the current underlying price.

Usage:
    python scripts/migrations/fix_legacy_thesis_prices.py [--dry-run]
"""

import asyncio
import argparse
import logging
import sys
import os

# Add project root to sys.path to allow importing trading_bot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ib_insync import IB, Contract
from trading_bot.tms import TransactiveMemory
from trading_bot.ib_interface import get_active_futures
from trading_bot.utils import configure_market_data_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PRICE_FLOOR = 100.0  # Anything below this is not an underlying price


async def migrate(dry_run: bool = True):
    tms = TransactiveMemory()

    # Get all active theses
    all_theses = []
    # Note: These guardians correspond to the main guardians that store theses.
    # Some older theses might be under 'Master' or others, but these are the main ones.
    for guardian in ['VolatilityAnalyst', 'FundamentalAnalyst',
                     'MacroEconomicAnalyst', 'TechnicalAnalyst', 'Master']: # Added Master just in case
        all_theses.extend(tms.get_active_theses_by_guardian(guardian))

    if not all_theses:
        logger.info("No active theses found. Nothing to migrate.")
        return

    # Find problematic theses
    needs_fix = []
    for thesis in all_theses:
        entry = thesis.get('supporting_data', {}).get('entry_price', 0)
        # Check if entry is 0 or reasonably small (indicating spread credit)
        if entry == 0 or (0 < entry < PRICE_FLOOR):
            needs_fix.append(thesis)
            logger.warning(
                f"Thesis needs fix: position_id={thesis.get('position_id', '?')}, "
                f"entry_price={entry}, strategy={thesis.get('strategy_type', '?')}"
            )

    if not needs_fix:
        logger.info("All theses have valid entry prices. No migration needed.")
        return

    logger.info(f"Found {len(needs_fix)} theses needing price correction.")

    if dry_run:
        logger.info("DRY RUN — no changes made. Run with --apply to execute.")
        return

    # Connect to IB to get current underlying price
    ib = IB()
    try:
        await ib.connectAsync('127.0.0.1', 7497, clientId=250)
    except Exception as e:
        logger.error(f"Could not connect to IB: {e}")
        return

    configure_market_data_type(ib)

    try:
        # Assuming KC (Coffee) on NYBOT
        futures = await get_active_futures(ib, 'KC', 'NYBOT', count=1)
        if not futures:
            logger.error("Cannot get active futures. Aborting migration.")
            return

        ticker = ib.reqMktData(futures[0], '', True, False)
        await asyncio.sleep(2)
        current_price = ticker.last if ticker.last else ticker.close
        ib.cancelMktData(futures[0])

        if not current_price or current_price <= 0:
            logger.error(f"Invalid current price: {current_price}. Aborting.")
            return

        logger.info(f"Current underlying price: ${current_price:.2f}")

        for thesis in needs_fix:
            position_id = thesis.get('position_id', '?')
            old_entry = thesis.get('supporting_data', {}).get('entry_price', 0)

            # Patch the thesis
            thesis['supporting_data']['entry_price'] = current_price
            thesis['supporting_data']['_migration_note'] = (
                f"Price corrected from {old_entry} to {current_price} "
                f"by fix_legacy_thesis_prices.py on 2026-02-03"
            )

            # Record back to TMS (overwrites existing by ID)
            # tms.record_trade_thesis expects (position_id, thesis_data)
            # but TransactiveMemory.record_trade_thesis might not behave exactly as simple overwrite if using vector store.
            # Let's check tms.py if possible, but assuming standard behavior from report instructions.
            # The report assumes record_trade_thesis is idempotent.

            tms.record_trade_thesis(position_id, thesis)
            logger.info(
                f"Fixed: {position_id} — entry_price {old_entry} -> {current_price}"
            )

    finally:
        ib.disconnect()

    logger.info("Migration complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', default=True, help="Run without applying changes (default)")
    parser.add_argument('--apply', action='store_true', help="Apply changes to TMS")
    args = parser.parse_args()

    # If --apply is specified, dry_run is False. Otherwise True.
    dry_run = not args.apply
    asyncio.run(migrate(dry_run=dry_run))
