#!/usr/bin/env python3
"""
One-shot manual Brier reconciliation for all active commodities.

Runs the two-step process per commodity:
  1. reconcile_council_history() — fills actual_trend_direction via IB historical data
  2. resolve_with_cycle_aware_match() + backfill_enhanced_from_csv() — grades Brier predictions

Usage (from project root):
    python scripts/manual_brier_reconciliation.py [--dry-run] [--commodity KC]

Requires an active IB Gateway connection (local or remote via IB_HOST).
"""
import asyncio
import argparse
import copy
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


def _build_commodity_config(base_config: dict, ticker: str) -> dict:
    """Build per-commodity config with deep_merge + exchange injection."""
    from config_loader import deep_merge
    from trading_bot.utils import get_ibkr_exchange

    config = copy.deepcopy(base_config)
    overrides = config.get('commodity_overrides', {}).get(ticker, {})
    if overrides:
        config = deep_merge(config, overrides)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['data_dir'] = os.path.join(project_root, 'data', ticker)
    config['symbol'] = ticker
    config.setdefault('commodity', {})['ticker'] = ticker
    config['exchange'] = get_ibkr_exchange(config)
    return config


async def _reconcile_commodity(config: dict, ticker: str, dry_run: bool):
    """Run full Brier reconciliation for a single commodity."""
    data_dir = config['data_dir']

    if not os.path.isdir(data_dir):
        logger.warning(f"[{ticker}] Data directory not found: {data_dir} — skipping")
        return

    # Initialize module-level data paths
    from trading_bot.brier_reconciliation import set_data_dir as set_brier_recon_dir
    from trading_bot.brier_bridge import set_data_dir as set_brier_bridge_dir
    from trading_bot.brier_scoring import set_data_dir as set_brier_scoring_dir
    set_brier_recon_dir(data_dir)
    set_brier_bridge_dir(data_dir)
    set_brier_scoring_dir(data_dir)
    logger.info(f"[{ticker}] Data directory: {data_dir}")

    # --- Step 1: Council History Reconciliation (needs IB) ---
    logger.info(f"[{ticker}] STEP 1: Council History Reconciliation")
    try:
        from trading_bot.reconciliation import reconcile_council_history
        await reconcile_council_history(config)
        logger.info(f"[{ticker}] Council history reconciliation complete.")
    except Exception as e:
        logger.error(f"[{ticker}] Council history reconciliation failed: {e}", exc_info=True)
        logger.warning(f"[{ticker}] Continuing to Brier resolution (may still resolve some predictions)...")

    # --- Step 2: Brier Prediction Resolution (CSV-only, no IB needed) ---
    logger.info(f"[{ticker}] STEP 2: Brier Prediction Resolution")
    try:
        from trading_bot.brier_reconciliation import resolve_with_cycle_aware_match
        resolved = resolve_with_cycle_aware_match(dry_run=dry_run)
        label = "(DRY RUN)" if dry_run else ""
        logger.info(f"[{ticker}] Brier CSV resolution complete {label}: {resolved} predictions resolved")
    except Exception as e:
        logger.error(f"[{ticker}] Brier CSV resolution failed: {e}", exc_info=True)

    # --- Step 3: Enhanced Brier JSON Backfill ---
    logger.info(f"[{ticker}] STEP 3: Enhanced Brier JSON Backfill")
    if dry_run:
        logger.info(f"[{ticker}] Skipping JSON backfill in dry-run mode.")
    else:
        try:
            from trading_bot.brier_bridge import backfill_enhanced_from_csv, reset_enhanced_tracker
            backfilled = backfill_enhanced_from_csv()
            if backfilled > 0:
                reset_enhanced_tracker()
            logger.info(f"[{ticker}] Enhanced Brier backfill complete: {backfilled} predictions synced")
        except Exception as e:
            logger.error(f"[{ticker}] Enhanced Brier backfill failed: {e}", exc_info=True)


async def main(dry_run: bool = False, commodity: str | None = None):
    from config_loader import load_config
    base_config = load_config()
    if not base_config:
        logger.critical("Failed to load config.")
        return

    # Determine which commodities to process
    if commodity:
        tickers = [commodity.upper()]
    else:
        tickers = base_config.get('active_commodities', ['KC'])

    logger.info("=" * 60)
    logger.info(f"Manual Brier Reconciliation — commodities: {tickers}")
    logger.info("=" * 60)

    for ticker in tickers:
        logger.info("=" * 60)
        logger.info(f"=== Processing {ticker} ===")
        logger.info("=" * 60)
        config = _build_commodity_config(base_config, ticker)
        await _reconcile_commodity(config, ticker, dry_run)

    logger.info("=" * 60)
    logger.info("Manual Brier reconciliation finished.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Brier score reconciliation")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing changes")
    parser.add_argument("--commodity", type=str, help="Process a single commodity (e.g., KC, CC, NG)")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, commodity=args.commodity))
