"""
One-time migration: Clean up ghost theses in ChromaDB.

Ghost theses have active=true in TMS but their positions are closed in IB.
This script connects to IB, checks for live positions, and invalidates
any active theses that don't have matching positions.

Usage:
    python scripts/migrations/cleanup_ghost_theses.py              # Dry run
    python scripts/migrations/cleanup_ghost_theses.py --apply      # Execute

Commodity-agnostic: Checks ALL active theses regardless of symbol.
"""

import asyncio
import argparse
import json
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ib_insync import IB
from config_loader import load_config
from trading_bot.tms import TransactiveMemory
from trading_bot.utils import configure_market_data_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GhostThesisCleanup")


async def cleanup(dry_run: bool = True):
    config = load_config()
    tms = TransactiveMemory()

    if not tms.collection:
        logger.error("TMS collection unavailable. Exiting.")
        return

    # 1. Get all active theses
    active_results = tms.collection.get(
        where={"active": "true"},
        include=['metadatas', 'documents']
    )

    active_ids = []
    for i, meta in enumerate(active_results.get('metadatas', [])):
        tid = meta.get('trade_id', 'UNKNOWN')
        doc = active_results['documents'][i] if active_results.get('documents') else '{}'
        try:
            thesis = json.loads(doc) if doc else {}
        except json.JSONDecodeError:
            thesis = {}

        strategy = meta.get('strategy_type', 'UNKNOWN')
        guardian = meta.get('guardian_agent', 'UNKNOWN')
        entry_ts = meta.get('entry_timestamp', 'UNKNOWN')

        active_ids.append(tid)
        logger.info(
            f"  Active thesis: {tid[:12]}... | "
            f"strategy={strategy} | guardian={guardian} | entry={entry_ts}"
        )

    if not active_ids:
        logger.info("No active theses found. Nothing to clean up.")
        return

    logger.info(f"\nTotal active theses: {len(active_ids)}")

    # 2. Connect to IB and check positions
    ib = IB()
    try:
        await ib.connectAsync(
            config['connection']['host'],
            config['connection']['port'],
            clientId=998  # Unique ID for migration script
        )
        configure_market_data_type(ib)

        positions = await ib.reqPositionsAsync()
        open_positions = [p for p in (positions or []) if p.position != 0]

        logger.info(f"IB reports {len(open_positions)} open position legs")

        if open_positions:
            logger.info("Open positions found:")
            for p in open_positions:
                logger.info(f"  {p.contract.localSymbol}: qty={p.position}")
            logger.warning(
                "\nThere are live positions. Only theses WITHOUT matching "
                "positions will be marked as ghosts."
            )
            # In this case, we'd need the full _find_position_id_for_contract
            # logic to determine which theses are ghosts. For safety, skip
            # auto-cleanup when positions exist — let the audit cycle handle it.
            logger.warning(
                "Skipping automatic cleanup. Let the audit cycle "
                "(with Part A + C fixes) handle reconciliation."
            )
            return

        # 3. No positions → ALL active theses are ghosts
        logger.info("\n🧹 IB confirms ZERO open positions.")
        logger.info(f"All {len(active_ids)} active theses are ghosts.\n")

        if dry_run:
            logger.info("DRY RUN — no changes made. Use --apply to clean up.")
            return

        # 4. Invalidate all
        success = 0
        for tid in active_ids:
            try:
                tms.invalidate_thesis(tid, "Migration: ghost thesis cleanup (no IB positions)")
                logger.info(f"  ✓ Invalidated: {tid[:12]}...")
                success += 1
            except Exception as e:
                logger.error(f"  ✗ Failed to invalidate {tid[:12]}...: {e}")

        logger.info(f"\n✅ Cleanup complete: {success}/{len(active_ids)} theses invalidated.")

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean up ghost theses in TMS")
    parser.add_argument('--apply', action='store_true', help="Apply changes (default is dry-run)")
    args = parser.parse_args()

    asyncio.run(cleanup(dry_run=not args.apply))
