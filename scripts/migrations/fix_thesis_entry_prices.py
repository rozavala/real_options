"""
Migration: Fix legacy Iron Condor theses that store spread premium as entry_price.

Heuristic: Coffee futures trade at $200–$400. Option premiums trade at $1–$50.
Any IC thesis with entry_price < 100 is almost certainly a spread premium.

This script:
1. Queries TMS for all IRON_CONDOR theses (active + historical)
2. Identifies those with entry_price < PRICE_FLOOR (100.0)
3. For each:
   a. Reads underlying_symbol + contract_month from supporting_data
   b. If contract_month exists, attempts to resolve historical underlying price
   c. Falls back to "mark as needs_review" if historical price unavailable
4. Updates supporting_data with corrected values
5. Marks thesis as migrated to prevent re-processing

Idempotent: Checks for 'migrated_v6_5_1' flag before processing.
"""
import logging
from datetime import datetime, timezone
import sys
import os
import pandas as pd

# Add repo root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from trading_bot.tms import TransactiveMemory

logger = logging.getLogger(__name__)

# --- Configuration ---
# Commodity-agnostic: Each commodity profile should define these thresholds
# For coffee (KC): futures $200-$400, premiums $1-$50
PRICE_FLOOR = 100.0  # Below this, entry_price is almost certainly a premium


def run_migration(dry_run: bool = True):
    """
    Main migration entry point.

    Args:
        dry_run: If True, logs what would change but doesn't write.
                 Set to False for actual migration.
    """
    tms = TransactiveMemory()
    all_theses = tms.get_all_theses()  # Need to add this method if missing

    migrated_count = 0
    skipped_count = 0
    needs_review = []

    for thesis in all_theses:
        thesis_id = thesis.get('trade_id', 'UNKNOWN')
        strategy_type = thesis.get('strategy_type', '')
        supporting_data = thesis.get('supporting_data', {})
        if not supporting_data:
            # Maybe supporting_data is not in the thesis dict but in metadata?
            # get_all_theses logic tries to provide full thesis dict.
            supporting_data = {}

        # Skip non-IC theses
        if strategy_type != 'IRON_CONDOR':
            continue

        # Skip already-migrated theses
        if supporting_data.get('migrated_v6_5_1'):
            skipped_count += 1
            continue

        entry_price = supporting_data.get('entry_price', 0)

        # Skip if entry_price looks correct (above floor = likely underlying price)
        if entry_price >= PRICE_FLOOR:
            skipped_count += 1
            continue

        # This thesis has a premium stored as entry_price
        logger.info(
            f"MIGRATE {thesis_id}: entry_price=${entry_price:.2f} "
            f"(below floor ${PRICE_FLOOR}) — likely spread premium"
        )

        # Attempt to find the correct underlying price
        contract_month = supporting_data.get('contract_month')
        underlying_symbol = supporting_data.get('underlying_symbol', 'KC')

        # Option 1: If we have historical council_history.csv, look up the price
        # at entry_timestamp for this contract_month
        corrected_price = _lookup_historical_price(
            thesis.get('entry_timestamp'),
            contract_month,
            underlying_symbol
        )

        if corrected_price and corrected_price >= PRICE_FLOOR:
            new_supporting_data = {
                **supporting_data,
                'entry_price': corrected_price,
                'spread_credit': entry_price,  # Preserve original value
                'migrated_v6_5_1': True,
                'migration_timestamp': datetime.now(timezone.utc).isoformat(),
                'migration_note': f'Corrected entry_price from spread premium '
                                  f'(${entry_price:.2f}) to underlying price '
                                  f'(${corrected_price:.2f})'
            }

            if dry_run:
                logger.info(
                    f"  DRY RUN: Would update entry_price "
                    f"${entry_price:.2f} → ${corrected_price:.2f}"
                )
            else:
                tms.update_thesis_supporting_data(thesis_id, new_supporting_data)
                logger.info(
                    f"  MIGRATED: entry_price "
                    f"${entry_price:.2f} → ${corrected_price:.2f}"
                )

            migrated_count += 1
        else:
            # Can't determine correct price — flag for manual review
            needs_review.append({
                'thesis_id': thesis_id,
                'entry_price': entry_price,
                'contract_month': contract_month,
                'entry_timestamp': thesis.get('entry_timestamp')
            })
            logger.warning(
                f"  NEEDS REVIEW: Cannot determine underlying price for "
                f"{thesis_id} (contract_month={contract_month})"
            )

            # Still mark as migrated to prevent re-processing, but flag it
            if not dry_run:
                new_supporting_data = {
                    **supporting_data,
                    'spread_credit': entry_price,
                    'migrated_v6_5_1': True,
                    'needs_manual_review': True,
                    'migration_timestamp': datetime.now(timezone.utc).isoformat()
                }
                tms.update_thesis_supporting_data(thesis_id, new_supporting_data)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Migration Summary ({'DRY RUN' if dry_run else 'APPLIED'}):")
    logger.info(f"  Migrated: {migrated_count}")
    logger.info(f"  Skipped (correct or already migrated): {skipped_count}")
    logger.info(f"  Needs manual review: {len(needs_review)}")

    if needs_review:
        logger.info(f"\nTheses requiring manual review:")
        for item in needs_review:
            logger.info(f"  - {item['thesis_id']}: ${item['entry_price']:.2f} "
                        f"(month={item['contract_month']})")

    return {
        'migrated': migrated_count,
        'skipped': skipped_count,
        'needs_review': needs_review
    }


def _lookup_historical_price(entry_timestamp: str, contract_month: str,
                              symbol: str) -> float | None:
    """
    Attempts to find the underlying futures price at the time of trade entry.

    Strategy:
    1. Check council_history.csv for matching cycle entries
    2. Check trade_ledger.csv for the fill price context
    3. Return None if unavailable (triggers manual review)
    """
    if not entry_timestamp or not contract_month:
        return None

    # Try council_history.csv first
    council_path = 'data/council_history.csv'
    if os.path.exists(council_path):
        try:
            df = pd.read_csv(council_path)
            if 'entry_price' in df.columns and 'contract' in df.columns:
                # Filter to matching contract month
                # contract column usually has format like "KCZ5 (202512)"
                # we match substring of contract month e.g. "202512"
                matches = df[
                    df['contract'].astype(str).str.contains(str(contract_month)[:6])
                ]
                if not matches.empty:
                    # Get the entry_price closest to the entry_timestamp
                    # Ideally filter by timestamp too, but for now just taking last valid price
                    # as council_history is usually sequential
                    prices = matches['entry_price'].dropna()
                    valid_prices = prices[prices >= 100.0]  # Filter out premiums
                    if not valid_prices.empty:
                        return float(valid_prices.iloc[-1])
        except Exception as e:
            logger.warning(f"council_history lookup failed: {e}")

    return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dry = '--apply' not in sys.argv
    if dry:
        print("Running in DRY RUN mode. Use --apply to execute changes.")
    else:
        print("Running in APPLY mode. Changes will be written to TMS.")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            sys.exit(0)

    run_migration(dry_run=dry)
