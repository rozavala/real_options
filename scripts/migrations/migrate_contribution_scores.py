#!/usr/bin/env python3
"""
Migration: Backfill contribution_scores.json from council_history.csv.

Wraps scripts/migrate_contribution_scores.py for the migration runner framework.
Idempotent: skips tickers that already have contribution_scores.json with data.

Accepts: [--dry-run] [data_dir1] [data_dir2] ...
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def needs_migration(data_dir: str) -> bool:
    """Check if a commodity data dir needs contribution score migration."""
    scores_path = os.path.join(data_dir, 'contribution_scores.json')
    council_path = os.path.join(data_dir, 'council_history.csv')

    # No council history → nothing to migrate
    if not os.path.exists(council_path):
        return False

    # No scores file → needs migration
    if not os.path.exists(scores_path):
        return True

    # Scores file exists but empty or has no agent data → needs migration
    try:
        with open(scores_path) as f:
            data = json.load(f)
        agent_scores = data.get('agent_scores', {})
        if not agent_scores:
            return True
        # Has agent data — already migrated
        return False
    except (json.JSONDecodeError, Exception):
        return True


def main():
    dry_run = '--dry-run' in sys.argv
    data_dirs = [a for a in sys.argv[1:] if a != '--dry-run']

    if not data_dirs:
        print("No data directories provided")
        return

    # Import the actual migration logic from scripts/migrate_contribution_scores.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "migrate_contribution_scores_main",
        os.path.join(PROJECT_ROOT, "scripts", "migrate_contribution_scores.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    migrate_ticker = mod.migrate_ticker

    migrated = 0
    skipped = 0

    for data_dir in data_dirs:
        # data_dir is like /path/to/data/KC — ticker is the basename
        ticker = os.path.basename(data_dir.rstrip('/'))
        parent_dir = os.path.dirname(data_dir.rstrip('/'))

        if not needs_migration(data_dir):
            print(f"  {ticker}: contribution_scores.json already exists, skipping")
            skipped += 1
            continue

        print(f"  {ticker}: Running contribution scores migration...")
        try:
            result = migrate_ticker(ticker, parent_dir, dry_run=dry_run)
            status = result.get('status', 'unknown')
            if status == 'migrated':
                print(f"  {ticker}: Migrated {result['scored_cycles']} cycles, "
                      f"avg_diff={result['avg_diff']:.3f}")
                migrated += 1
            else:
                print(f"  {ticker}: {status} ({result.get('reason', '')})")
                skipped += 1
        except Exception as e:
            print(f"  {ticker}: ERROR — {e}")
            skipped += 1

    print(f"\nContribution scores migration: {migrated} migrated, {skipped} skipped")


if __name__ == '__main__':
    main()
