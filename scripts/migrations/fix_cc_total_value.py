#!/usr/bin/env python3
"""
Migration: Fix CC (Cocoa) total_value_usd in trade_ledger.csv

Problem: log_trade_to_ledger() was dividing all total_value by 100.0 (cents divisor),
which is correct for KC (cents/lb) but wrong for CC ($/metric ton) and NG ($/mmBtu).
CC values were recorded 100x too small.

Fix: Multiply CC rows' total_value_usd by 100.0 to undo the erroneous division.

This migration is idempotent: it writes a marker column 'migrated_cc_divisor' and
skips rows that already have it set.

Usage:
    python scripts/migrations/fix_cc_total_value.py [--dry-run] [--data-dir data/CC]
"""

import argparse
import csv
import os
import sys
import shutil
from datetime import datetime


def fix_cc_total_value(data_dir: str, dry_run: bool = False) -> dict:
    """Fix CC total_value_usd by multiplying by 100.

    Returns:
        dict with keys: rows_total, rows_fixed, rows_skipped, ledger_path
    """
    ledger_path = os.path.join(data_dir, 'trade_ledger.csv')
    if not os.path.exists(ledger_path):
        print(f"No ledger found at {ledger_path} — nothing to migrate.")
        return {'rows_total': 0, 'rows_fixed': 0, 'rows_skipped': 0, 'ledger_path': ledger_path}

    # Read all rows
    with open(ledger_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)

    if not rows:
        print(f"Ledger at {ledger_path} is empty — nothing to migrate.")
        return {'rows_total': 0, 'rows_fixed': 0, 'rows_skipped': 0, 'ledger_path': ledger_path}

    # Ensure marker column exists in fieldnames
    marker_col = 'migrated_cc_divisor'
    fieldnames = list(original_fieldnames)
    if marker_col not in fieldnames:
        fieldnames.append(marker_col)

    rows_fixed = 0
    rows_skipped = 0

    for row in rows:
        # Skip already-migrated rows
        if row.get(marker_col):
            rows_skipped += 1
            continue

        # Only fix rows with CC symbols
        local_sym = row.get('local_symbol', '')
        if not local_sym.startswith('CC'):
            row[marker_col] = ''  # Not applicable, leave unmarked
            continue

        try:
            old_val = float(row.get('total_value_usd', 0))
            new_val = old_val * 100.0
            if not dry_run:
                row['total_value_usd'] = f"{new_val:.2f}"
                row[marker_col] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
            rows_fixed += 1
            print(f"  {'[DRY RUN] ' if dry_run else ''}Row {local_sym}: "
                  f"${old_val:.2f} -> ${new_val:.2f}")
        except (ValueError, TypeError) as e:
            print(f"  WARNING: Could not convert total_value_usd for {local_sym}: {e}")

    if not dry_run and rows_fixed > 0:
        # Backup original
        backup_path = ledger_path + f'.bak.{datetime.utcnow().strftime("%Y%m%dT%H%M%S")}'
        shutil.copy2(ledger_path, backup_path)
        print(f"Backup saved to {backup_path}")

        # Write corrected file atomically
        tmp_path = ledger_path + '.tmp'
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, ledger_path)

    result = {
        'rows_total': len(rows),
        'rows_fixed': rows_fixed,
        'rows_skipped': rows_skipped,
        'ledger_path': ledger_path,
    }
    print(f"\nSummary: {rows_fixed} rows fixed, {rows_skipped} already migrated, "
          f"{len(rows)} total rows in {ledger_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Fix CC total_value_usd in trade_ledger.csv')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without writing')
    parser.add_argument('--data-dir', default='data/CC', help='Path to CC data directory')
    args = parser.parse_args()

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Fixing CC total_value_usd divisor bug...")
    fix_cc_total_value(args.data_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
