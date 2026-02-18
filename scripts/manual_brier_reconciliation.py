#!/usr/bin/env python3
"""
One-shot manual Brier reconciliation.

Runs the two-step process:
  1. reconcile_council_history() — fills actual_trend_direction via IB historical data
  2. resolve_with_cycle_aware_match() + backfill_enhanced_from_csv() — grades Brier predictions

Usage (from project root):
    python scripts/manual_brier_reconciliation.py [--dry-run]

Requires an active IB Gateway connection (local or remote via IB_HOST).
"""
import asyncio
import argparse
import logging
import os
import sys

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ManualBrierReconciliation")


async def main(dry_run: bool = False):
    from config_loader import load_config
    config = load_config()
    if not config:
        logger.critical("Failed to load config.")
        return

    # Initialize module-level data paths (same as orchestrator startup)
    data_dir = config.get('data_dir')
    if data_dir:
        from trading_bot.brier_reconciliation import set_data_dir as set_brier_recon_dir
        from trading_bot.brier_bridge import set_data_dir as set_brier_bridge_dir
        from trading_bot.brier_scoring import set_data_dir as set_brier_scoring_dir
        set_brier_recon_dir(data_dir)
        set_brier_bridge_dir(data_dir)
        set_brier_scoring_dir(data_dir)
        logger.info(f"Data directory: {data_dir}")

    # --- Step 1: Council History Reconciliation (needs IB) ---
    logger.info("=" * 60)
    logger.info("STEP 1: Council History Reconciliation")
    logger.info("=" * 60)

    try:
        from trading_bot.reconciliation import reconcile_council_history
        await reconcile_council_history(config)
        logger.info("Council history reconciliation complete.")
    except Exception as e:
        logger.error(f"Council history reconciliation failed: {e}", exc_info=True)
        logger.warning("Continuing to Brier resolution (may still resolve some predictions)...")

    # --- Step 2: Brier Prediction Resolution (CSV-only, no IB needed) ---
    logger.info("=" * 60)
    logger.info("STEP 2: Brier Prediction Resolution")
    logger.info("=" * 60)

    try:
        from trading_bot.brier_reconciliation import resolve_with_cycle_aware_match
        resolved = resolve_with_cycle_aware_match(dry_run=dry_run)
        label = "(DRY RUN)" if dry_run else ""
        logger.info(f"Brier CSV resolution complete {label}: {resolved} predictions resolved")
    except Exception as e:
        logger.error(f"Brier CSV resolution failed: {e}", exc_info=True)

    # --- Step 3: Enhanced Brier JSON Backfill ---
    logger.info("=" * 60)
    logger.info("STEP 3: Enhanced Brier JSON Backfill")
    logger.info("=" * 60)

    if dry_run:
        logger.info("Skipping JSON backfill in dry-run mode.")
    else:
        try:
            from trading_bot.brier_bridge import backfill_enhanced_from_csv, reset_enhanced_tracker
            backfilled = backfill_enhanced_from_csv()
            if backfilled > 0:
                reset_enhanced_tracker()
            logger.info(f"Enhanced Brier backfill complete: {backfilled} predictions synced")
        except Exception as e:
            logger.error(f"Enhanced Brier backfill failed: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info("Manual Brier reconciliation finished.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Brier score reconciliation")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing changes")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
