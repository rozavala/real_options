#!/usr/bin/env python3
"""
Legacy Data Purge Script v6.0

Removes all feedback-loop data that predates the cycle_id system.
This eliminates orphaned predictions, permanently stuck PENDINGs,
and cross-cycle contaminated Brier scores.

SAFE TO RUN: Creates timestamped backups before any modification.
IDEMPOTENT: Running multiple times produces the same result.

Usage:
    python scripts/purge_legacy_data.py --dry-run   # Preview changes
    python scripts/purge_legacy_data.py              # Execute purge
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import shutil
import argparse
import re
from datetime import datetime, timezone

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# === CONFIGURATION ===
DATA_DIR = "data"
STRUCTURED_FILE = os.path.join(DATA_DIR, "agent_accuracy_structured.csv")
LEGACY_ACCURACY_FILE = os.path.join(DATA_DIR, "agent_accuracy.csv")
ENHANCED_BRIER_FILE = os.path.join(DATA_DIR, "enhanced_brier.json")
COUNCIL_HISTORY_FILE = os.path.join(DATA_DIR, "council_history.csv")

# Valid cycle_id pattern: 2-4 uppercase letters, dash, 6-12 hex chars
# e.g., KC-c0f3b12c, CC-ab12ef34
CYCLE_ID_PATTERN = re.compile(r'^[A-Z]{2,4}-[0-9a-f]{6,12}$')

BACKUP_SUFFIX = datetime.now().strftime('%Y%m%d_%H%M%S')


def is_valid_cycle_id(value) -> bool:
    """Check if a value is a valid cycle_id."""
    if pd.isna(value) or value is None:
        return False
    s = str(value).strip()
    if s in ('', 'nan', 'None', 'null'):
        return False
    return bool(CYCLE_ID_PATTERN.match(s))


def backup_file(filepath: str) -> str:
    """Create a timestamped backup of a file. Returns backup path."""
    if not os.path.exists(filepath):
        return None
    backup_path = filepath.replace('.csv', f'_pre_purge_{BACKUP_SUFFIX}.csv')
    backup_path = backup_path.replace('.json', f'_pre_purge_{BACKUP_SUFFIX}.json')
    shutil.copy2(filepath, backup_path)
    logger.info(f"  Backup: {filepath} → {backup_path}")
    return backup_path


def purge_structured_predictions(dry_run: bool) -> dict:
    """
    Purge agent_accuracy_structured.csv — keep only rows with valid cycle_id.

    Returns dict with before/after counts.
    """
    stats = {'file': STRUCTURED_FILE, 'exists': False}

    if not os.path.exists(STRUCTURED_FILE):
        logger.info(f"  {STRUCTURED_FILE} not found — skipping")
        return stats

    stats['exists'] = True
    df = pd.read_csv(STRUCTURED_FILE)
    stats['before_total'] = len(df)

    if 'cycle_id' not in df.columns:
        logger.warning(f"  {STRUCTURED_FILE} has no cycle_id column — nothing to filter")
        stats['after_total'] = len(df)
        return stats

    # Count by status before purge
    stats['before_pending'] = (df['actual'] == 'PENDING').sum()
    stats['before_orphaned'] = (df['actual'] == 'ORPHANED').sum()
    stats['before_resolved'] = stats['before_total'] - stats['before_pending'] - stats['before_orphaned']

    # Filter: keep only rows with valid cycle_id
    valid_mask = df['cycle_id'].apply(is_valid_cycle_id)
    df_clean = df[valid_mask].copy()

    stats['after_total'] = len(df_clean)
    stats['after_pending'] = (df_clean['actual'] == 'PENDING').sum()
    stats['after_orphaned'] = (df_clean['actual'] == 'ORPHANED').sum()
    stats['after_resolved'] = stats['after_total'] - stats['after_pending'] - stats['after_orphaned']
    stats['purged'] = stats['before_total'] - stats['after_total']

    if not dry_run and stats['purged'] > 0:
        backup_file(STRUCTURED_FILE)
        df_clean.to_csv(STRUCTURED_FILE, index=False)
        logger.info(f"  Saved: {stats['after_total']} rows (purged {stats['purged']})")

    return stats


def purge_legacy_accuracy(dry_run: bool, cutoff_timestamp: pd.Timestamp = None) -> dict:
    """
    Purge agent_accuracy.csv — keep only rows dated ≥ cutoff.

    The cutoff is derived from the earliest valid cycle_id entry in
    agent_accuracy_structured.csv if not provided explicitly.
    """
    stats = {'file': LEGACY_ACCURACY_FILE, 'exists': False}

    if not os.path.exists(LEGACY_ACCURACY_FILE):
        logger.info(f"  {LEGACY_ACCURACY_FILE} not found — skipping")
        return stats

    stats['exists'] = True
    df = pd.read_csv(LEGACY_ACCURACY_FILE)
    stats['before_total'] = len(df)

    if cutoff_timestamp is None:
        logger.warning("  No cutoff timestamp provided — skipping legacy accuracy purge")
        stats['after_total'] = len(df)
        return stats

    # Parse timestamps
    if 'timestamp' not in df.columns:
        logger.warning(f"  {LEGACY_ACCURACY_FILE} has no timestamp column — skipping")
        stats['after_total'] = len(df)
        return stats

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    valid_time_mask = df['timestamp'].notna()
    after_cutoff_mask = df['timestamp'] >= cutoff_timestamp

    df_clean = df[valid_time_mask & after_cutoff_mask].copy()

    stats['after_total'] = len(df_clean)
    stats['purged'] = stats['before_total'] - stats['after_total']

    if not dry_run and stats['purged'] > 0:
        backup_file(LEGACY_ACCURACY_FILE)
        df_clean.to_csv(LEGACY_ACCURACY_FILE, index=False)
        logger.info(f"  Saved: {stats['after_total']} rows (purged {stats['purged']})")

    return stats


def purge_enhanced_brier(dry_run: bool) -> dict:
    """
    Purge enhanced_brier.json — keep only predictions with valid cycle_id,
    reset computed agent_scores and calibration_buckets.
    """
    stats = {'file': ENHANCED_BRIER_FILE, 'exists': False}

    if not os.path.exists(ENHANCED_BRIER_FILE):
        logger.info(f"  {ENHANCED_BRIER_FILE} not found — skipping")
        return stats

    stats['exists'] = True

    with open(ENHANCED_BRIER_FILE, 'r') as f:
        data = json.load(f)

    predictions = data.get('predictions', [])
    stats['before_total'] = len(predictions)

    # Filter: keep only predictions with valid cycle_id
    clean_predictions = [
        p for p in predictions
        if is_valid_cycle_id(p.get('cycle_id', ''))
    ]

    stats['after_total'] = len(clean_predictions)
    stats['purged'] = stats['before_total'] - stats['after_total']

    if not dry_run and stats['purged'] > 0:
        backup_file(ENHANCED_BRIER_FILE)

        # Rebuild the JSON with clean predictions and reset computed fields
        clean_data = {
            'schema_version': data.get('schema_version', '1.0'),
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'predictions': clean_predictions,
            'agent_scores': {},            # Reset — will be recomputed from clean data
            'calibration_buckets': {},     # Reset — will be recomputed from clean data
        }

        with open(ENHANCED_BRIER_FILE, 'w') as f:
            json.dump(clean_data, f, indent=2)

        logger.info(f"  Saved: {stats['after_total']} predictions (purged {stats['purged']})")
        logger.info(f"  Reset: agent_scores and calibration_buckets (will recompute)")

    return stats


def purge_council_history(dry_run: bool) -> dict:
    """
    Purge council_history.csv — keep only rows with valid cycle_id.
    """
    stats = {'file': COUNCIL_HISTORY_FILE, 'exists': False}

    if not os.path.exists(COUNCIL_HISTORY_FILE):
        logger.info(f"  {COUNCIL_HISTORY_FILE} not found — skipping")
        return stats

    stats['exists'] = True
    df = pd.read_csv(COUNCIL_HISTORY_FILE)
    stats['before_total'] = len(df)

    if 'cycle_id' not in df.columns:
        logger.warning(f"  {COUNCIL_HISTORY_FILE} has no cycle_id column — skipping")
        stats['after_total'] = len(df)
        return stats

    # Count reconciliation status before purge
    if 'actual_trend_direction' in df.columns:
        reconciled_mask = (
            df['actual_trend_direction'].notna() &
            (df['actual_trend_direction'].astype(str).str.strip() != '')
        )
        stats['before_reconciled'] = reconciled_mask.sum()
        stats['before_unreconciled'] = stats['before_total'] - stats['before_reconciled']

    # Filter: keep only rows with valid cycle_id
    valid_mask = df['cycle_id'].apply(is_valid_cycle_id)
    df_clean = df[valid_mask].copy()

    stats['after_total'] = len(df_clean)
    stats['purged'] = stats['before_total'] - stats['after_total']

    if 'actual_trend_direction' in df_clean.columns:
        reconciled_mask_clean = (
            df_clean['actual_trend_direction'].notna() &
            (df_clean['actual_trend_direction'].astype(str).str.strip() != '')
        )
        stats['after_reconciled'] = reconciled_mask_clean.sum()
        stats['after_unreconciled'] = stats['after_total'] - stats['after_reconciled']

    if not dry_run and stats['purged'] > 0:
        backup_file(COUNCIL_HISTORY_FILE)
        df_clean.to_csv(COUNCIL_HISTORY_FILE, index=False)
        logger.info(f"  Saved: {stats['after_total']} rows (purged {stats['purged']})")

    return stats


def derive_cutoff_timestamp() -> pd.Timestamp:
    """
    Derive the cutoff timestamp from the earliest valid cycle_id entry
    in agent_accuracy_structured.csv.

    This ensures agent_accuracy.csv (which has no cycle_id column)
    is filtered to the same baseline date.
    """
    if not os.path.exists(STRUCTURED_FILE):
        return None

    df = pd.read_csv(STRUCTURED_FILE)
    if 'cycle_id' not in df.columns or 'timestamp' not in df.columns:
        return None

    valid_mask = df['cycle_id'].apply(is_valid_cycle_id)
    valid_rows = df[valid_mask]

    if valid_rows.empty:
        return None

    timestamps = pd.to_datetime(valid_rows['timestamp'], utc=True, errors='coerce')
    cutoff = timestamps.min()

    logger.info(f"  Derived cutoff timestamp: {cutoff}")
    return cutoff


def print_summary(all_stats: dict):
    """Print a comprehensive summary table."""
    logger.info("\n" + "=" * 70)
    logger.info("PURGE SUMMARY")
    logger.info("=" * 70)

    for name, stats in all_stats.items():
        if not stats.get('exists', False):
            logger.info(f"\n  {name}: FILE NOT FOUND — skipped")
            continue

        logger.info(f"\n  {name}:")
        logger.info(f"    Before: {stats.get('before_total', '?')} rows")
        logger.info(f"    After:  {stats.get('after_total', '?')} rows")
        logger.info(f"    Purged: {stats.get('purged', 0)} rows")

        # Extra detail for structured predictions
        if 'before_pending' in stats:
            logger.info(f"    --- Before breakdown ---")
            logger.info(f"      Resolved:  {stats['before_resolved']}")
            logger.info(f"      Pending:   {stats['before_pending']}")
            logger.info(f"      Orphaned:  {stats['before_orphaned']}")
            logger.info(f"    --- After breakdown ---")
            logger.info(f"      Resolved:  {stats['after_resolved']}")
            logger.info(f"      Pending:   {stats['after_pending']}")
            logger.info(f"      Orphaned:  {stats['after_orphaned']}")

        # Extra detail for council history
        if 'before_reconciled' in stats:
            logger.info(f"    --- Before breakdown ---")
            logger.info(f"      Reconciled:   {stats['before_reconciled']}")
            logger.info(f"      Unreconciled: {stats['before_unreconciled']}")
            logger.info(f"    --- After breakdown ---")
            logger.info(f"      Reconciled:   {stats.get('after_reconciled', '?')}")
            logger.info(f"      Unreconciled: {stats.get('after_unreconciled', '?')}")

    logger.info("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Purge legacy pre-cycle_id data from feedback loop files"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Preview changes without writing files"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("LEGACY DATA PURGE v6.0")
    logger.info("Removing pre-cycle_id data from feedback loop files")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("🔍 DRY RUN MODE — no files will be modified\n")
    else:
        logger.info("🔥 LIVE MODE — files will be modified (backups created)\n")

    all_stats = {}

    # Step 1: Derive cutoff from structured file BEFORE purging it
    logger.info("Step 1: Deriving cutoff timestamp from earliest valid cycle_id...")
    cutoff = derive_cutoff_timestamp()

    # Step 2: Purge structured predictions (primary file)
    logger.info("\nStep 2: Purging agent_accuracy_structured.csv...")
    all_stats['structured_predictions'] = purge_structured_predictions(args.dry_run)

    # Step 3: Purge legacy accuracy (date-based, using derived cutoff)
    logger.info("\nStep 3: Purging agent_accuracy.csv...")
    all_stats['legacy_accuracy'] = purge_legacy_accuracy(args.dry_run, cutoff)

    # Step 4: Purge enhanced Brier JSON
    logger.info("\nStep 4: Purging enhanced_brier.json...")
    all_stats['enhanced_brier'] = purge_enhanced_brier(args.dry_run)

    # Step 5: Purge council history
    logger.info("\nStep 5: Purging council_history.csv...")
    all_stats['council_history'] = purge_council_history(args.dry_run)

    # Summary
    print_summary(all_stats)

    if args.dry_run:
        logger.info("\n✅ Dry run complete. No files were modified.")
        logger.info("   Run without --dry-run to execute the purge.")
    else:
        logger.info("\n✅ Purge complete. Backups created for all modified files.")
        logger.info("   ⚠️  IMPORTANT: Restart the orchestrator to reset the in-memory")
        logger.info("      BrierScoreTracker singleton with clean data.")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
