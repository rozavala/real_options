#!/usr/bin/env python3
"""One-time backfill: set trading_mode_active='True' for all pre-v6 rows.

The trading_mode_active column was added in v6. All historical data was produced
while the system was in LIVE trading mode, so empty values should be 'True'.

Safe to run multiple times — skips rows that already have a value.

Usage:
    python scripts/migrations/backfill_trading_mode.py [--dry-run] [data_dir ...]

Examples:
    python scripts/migrations/backfill_trading_mode.py data/KC data/NG
    python scripts/migrations/backfill_trading_mode.py --dry-run data/KC
"""

import csv
import os
import shutil
import sys
import tempfile

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def backfill_trading_mode(file_path: str, dry_run: bool = False) -> dict:
    """Backfill trading_mode_active='True' for rows with empty values.

    Returns:
        dict with keys: status, rows_total, rows_backfilled
    """
    if not os.path.exists(file_path):
        return {"status": "not_found", "file": file_path}

    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if 'trading_mode_active' not in fieldnames:
            return {
                "status": "column_missing",
                "file": file_path,
                "detail": "Run migrate_council_schema.py first to add the column",
            }
        rows = list(reader)

    rows_backfilled = 0
    for row in rows:
        if row.get('trading_mode_active', '') == '':
            row['trading_mode_active'] = 'True'
            rows_backfilled += 1

    if rows_backfilled == 0:
        return {
            "status": "already_backfilled",
            "file": file_path,
            "rows_total": len(rows),
            "rows_backfilled": 0,
        }

    if dry_run:
        return {
            "status": "dry_run",
            "file": file_path,
            "rows_total": len(rows),
            "rows_backfilled": rows_backfilled,
        }

    # Backup
    bak_path = file_path + '.bak.trading_mode'
    shutil.copy2(file_path, bak_path)

    # Atomic write
    dir_name = os.path.dirname(file_path)
    with tempfile.NamedTemporaryFile(
        mode='w', newline='', encoding='utf-8',
        dir=dir_name, suffix='.tmp', delete=False
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = tmp.name

    os.replace(tmp_path, file_path)

    return {
        "status": "backfilled",
        "file": file_path,
        "rows_total": len(rows),
        "rows_backfilled": rows_backfilled,
        "backup": bak_path,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    dry_run = '--dry-run' in sys.argv

    if not args:
        args = ['data/KC', 'data/NG']

    for data_dir in args:
        csv_path = os.path.join(data_dir, 'council_history.csv')
        print(f"\n--- {csv_path} ---")
        result = backfill_trading_mode(csv_path, dry_run=dry_run)
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
