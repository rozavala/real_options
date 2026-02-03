"""
One-time migration: Fix dollar-format strikes in trade_ledger.csv.

Scans for rows where strike < 100 and multiplier indicates KC options,
then converts to cents format (multiply by 100).

Usage:
    python scripts/migrations/fix_ledger_strike_format.py [--dry-run]
"""

import pandas as pd
import argparse
import shutil
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Correct path relative to where script is run (assuming run from root)
# If run from scripts/migrations, we need to adjust.
# Best to assume run from project root as per standard.
LEDGER_PATH = 'trade_ledger.csv'
BACKUP_SUFFIX = f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'


def migrate(dry_run: bool = True):
    if not os.path.exists(LEDGER_PATH):
        logger.error(f"Ledger file not found at {LEDGER_PATH}")
        return

    df = pd.read_csv(LEDGER_PATH)
    logger.info(f"Loaded {len(df)} ledger entries.")

    # Identify KC option rows with dollar-format strikes
    # KC options have localSymbol starting with "KO" (e.g., KOK6, KON6)
    # and strikes that should be in cents (100-500 range for coffee)
    # Also ensuring strike is numeric

    # Force strike to numeric, coercing errors to NaN
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')

    mask = (
        df['local_symbol'].str.startswith('KO', na=False) &
        (df['strike'] < 100.0) &
        (df['strike'] > 0)
    )

    affected = df[mask]
    logger.info(f"Found {len(affected)} rows with dollar-format strikes.")

    if affected.empty:
        logger.info("No rows need fixing.")
        return

    for idx, row in affected.iterrows():
        old_strike = row['strike']
        new_strike = round(old_strike * 100, 2)
        logger.info(
            f"  {row['local_symbol']}: {old_strike} -> {new_strike} "
            f"({row['timestamp']}, {row['reason']})"
        )

    if dry_run:
        logger.info("DRY RUN — no changes made. Use --apply to fix.")
        return

    # Backup
    backup_path = LEDGER_PATH + BACKUP_SUFFIX
    shutil.copy2(LEDGER_PATH, backup_path)
    logger.info(f"Backup created: {backup_path}")

    # Apply fix
    df.loc[mask, 'strike'] = df.loc[mask, 'strike'] * 100
    df.to_csv(LEDGER_PATH, index=False)
    logger.info(f"Fixed {len(affected)} rows. Ledger saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', default=False, help="Run without applying changes (default if no flags)")
    parser.add_argument('--apply', action='store_true', help="Apply changes to the file")
    args = parser.parse_args()

    # Default to dry-run if --apply is not specified
    is_dry_run = not args.apply
    if args.dry_run:
        is_dry_run = True

    migrate(dry_run=is_dry_run)
