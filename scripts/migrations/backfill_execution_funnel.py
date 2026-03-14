#!/usr/bin/env python3
"""
Migration: Backfill execution_funnel.csv from existing historical data.

Wraps scripts/backfill_execution_funnel.py for the migration runner framework.
Idempotent: skips tickers that already have execution_funnel.csv with data.

Accepts: [--dry-run] [data_dir1] [data_dir2] ...
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def needs_backfill(data_dir: str) -> bool:
    """Check if a commodity data dir needs funnel backfill."""
    funnel_path = os.path.join(data_dir, 'execution_funnel.csv')
    council_path = os.path.join(data_dir, 'council_history.csv')

    # No council history → nothing to backfill from
    if not os.path.exists(council_path):
        return False

    # No funnel file → needs backfill
    if not os.path.exists(funnel_path):
        return True

    # File exists but empty → needs backfill
    if os.path.getsize(funnel_path) == 0:
        return True

    # File exists with data — already backfilled
    return False


def main():
    dry_run = '--dry-run' in sys.argv
    data_dirs = [a for a in sys.argv[1:] if a != '--dry-run']

    if not data_dirs:
        print("No data directories provided")
        return

    # Import the actual backfill logic
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "backfill_execution_funnel_main",
        os.path.join(PROJECT_ROOT, "scripts", "backfill_execution_funnel.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    backfill_commodity = mod.backfill_commodity

    backfilled = 0
    skipped = 0

    for data_dir in data_dirs:
        ticker = os.path.basename(data_dir.rstrip('/'))
        parent_dir = os.path.dirname(data_dir.rstrip('/'))

        if not needs_backfill(data_dir):
            print(f"  {ticker}: execution_funnel.csv already exists, skipping")
            skipped += 1
            continue

        if dry_run:
            print(f"  {ticker}: Would backfill execution_funnel.csv")
            backfilled += 1
            continue

        print(f"  {ticker}: Running execution funnel backfill...")
        try:
            count = backfill_commodity(ticker, parent_dir)
            if count > 0:
                print(f"  {ticker}: Backfilled {count} events")
                backfilled += 1
            else:
                print(f"  {ticker}: No data to backfill")
                skipped += 1
        except Exception as e:
            print(f"  {ticker}: ERROR — {e}")
            skipped += 1

    print(f"\nExecution funnel backfill: {backfilled} backfilled, {skipped} skipped")


if __name__ == '__main__':
    main()
