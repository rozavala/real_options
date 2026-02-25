#!/usr/bin/env python3
"""One-time migration: align council_history.csv header with canonical schema.

Reads the CSV with the csv module (tolerates ragged rows), compares the header
against COUNCIL_HISTORY_FIELDNAMES from schema.py, backfills missing columns
with semantically honest defaults, and rewrites atomically with a .bak backup.

Safe to run multiple times — no-ops if header already matches.

Usage:
    python scripts/migrations/migrate_council_schema.py [--dry-run] [data_dir ...]

Examples:
    # Migrate all commodity data dirs
    python scripts/migrations/migrate_council_schema.py data/KC data/CC data/NG

    # Dry run (preview changes only)
    python scripts/migrations/migrate_council_schema.py --dry-run data/KC
"""

import csv
import os
import shutil
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.schema import (
    COUNCIL_HISTORY_FIELDNAMES,
    backfill_missing_columns,
)


def migrate_council_csv(file_path: str, dry_run: bool = False) -> dict:
    """Migrate a single council_history.csv to canonical schema.

    Returns:
        dict with keys: status ('migrated'|'already_current'|'not_found'|'error'),
        rows_processed, columns_added, columns_removed
    """
    if not os.path.exists(file_path):
        return {"status": "not_found", "file": file_path}

    canonical = list(COUNCIL_HISTORY_FIELDNAMES)

    # Read existing header
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)

    if existing_header is None:
        return {"status": "error", "file": file_path, "detail": "empty file"}

    if existing_header == canonical:
        return {"status": "already_current", "file": file_path}

    missing = set(canonical) - set(existing_header)
    extra = set(existing_header) - set(canonical)

    print(f"  File: {file_path}")
    print(f"  Current columns: {len(existing_header)}")
    print(f"  Canonical columns: {len(canonical)}")
    if missing:
        print(f"  Missing (will add): {sorted(missing)}")
    if extra:
        print(f"  Extra (will drop): {sorted(extra)}")

    if dry_run:
        return {
            "status": "would_migrate",
            "file": file_path,
            "columns_added": sorted(missing),
            "columns_removed": sorted(extra),
        }

    # Read all rows using old header
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        old_reader = csv.DictReader(f)
        rows = list(old_reader)

    # Backfill missing columns
    for row in rows:
        backfill_missing_columns(row)

    # Create backup
    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)
    print(f"  Backup: {backup_path}")

    # Write with canonical header (temp + atomic replace)
    tmp_path = file_path + ".migrate.tmp"
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=canonical, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)

    print(f"  Migrated: {len(rows)} rows, +{len(missing)} columns")
    return {
        "status": "migrated",
        "file": file_path,
        "rows_processed": len(rows),
        "columns_added": sorted(missing),
        "columns_removed": sorted(extra),
    }


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")
        print("=== DRY RUN MODE ===\n")

    # Default to common data dirs if none specified
    if not args:
        args = ["data/KC", "data/CC", "data/NG"]
        print(f"No data dirs specified, using defaults: {args}\n")

    results = []
    for data_dir in args:
        csv_path = os.path.join(data_dir, "council_history.csv")
        print(f"\nProcessing: {csv_path}")
        result = migrate_council_csv(csv_path, dry_run=dry_run)
        results.append(result)
        print(f"  Status: {result['status']}")

    # Summary
    print("\n=== Summary ===")
    for r in results:
        print(f"  {r.get('file', 'unknown')}: {r['status']}")


if __name__ == "__main__":
    main()
