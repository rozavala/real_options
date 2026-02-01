#!/usr/bin/env python3
"""
Data Migration Script v5: Cycle-Aware Brier Resolution

WHAT THIS DOES:
1. Backs up all data files
2. Migrates agent_accuracy_structured.csv to include cycle_id column
3. Migrates council_history.csv to include cycle_id column
4. Resolves PENDING predictions using cycle-aware matching algorithm
5. Fixes legacy agent_accuracy.csv schema corruption

SAFE TO RUN MULTIPLE TIMES (idempotent).

Usage:
    python scripts/fix_brier_data.py
    python scripts/fix_brier_data.py --dry-run  # Preview without changes
"""

import pandas as pd
import numpy as np
import os
import sys
import shutil
import argparse
from datetime import datetime, timezone

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add to imports section:
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from trading_bot.timestamps import parse_ts_column

# Paths
DATA_DIR = "data"
STRUCTURED_FILE = os.path.join(DATA_DIR, "agent_accuracy_structured.csv")
COUNCIL_FILE = os.path.join(DATA_DIR, "council_history.csv")
ACCURACY_FILE = os.path.join(DATA_DIR, "agent_accuracy.csv")


def backup_files():
    """Create timestamped backups before any modification."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backed_up = []

    for filepath in [STRUCTURED_FILE, COUNCIL_FILE, ACCURACY_FILE]:
        if os.path.exists(filepath):
            backup_path = filepath.replace('.csv', f'_pre_v5_migration_{timestamp}.csv')
            shutil.copy2(filepath, backup_path)
            backed_up.append(backup_path)
            logger.info(f"Backed up: {filepath} → {backup_path}")

    return backed_up


def migrate_structured_csv():
    """
    Add cycle_id column to agent_accuracy_structured.csv if missing.
    Existing rows get empty cycle_id (will use nearest-match fallback).
    """
    if not os.path.exists(STRUCTURED_FILE):
        logger.info(f"{STRUCTURED_FILE} doesn't exist — skipping migration")
        return

    df = pd.read_csv(STRUCTURED_FILE)

    if 'cycle_id' in df.columns:
        logger.info(f"{STRUCTURED_FILE} already has cycle_id column")
        return

    # Add cycle_id as first column with empty values
    df.insert(0, 'cycle_id', '')
    df.to_csv(STRUCTURED_FILE, index=False)
    logger.info(f"Added cycle_id column to {STRUCTURED_FILE} ({len(df)} rows)")


def migrate_council_history_csv():
    """
    Add cycle_id column to council_history.csv if missing.
    Existing rows get empty cycle_id.
    """
    if not os.path.exists(COUNCIL_FILE):
        logger.info(f"{COUNCIL_FILE} doesn't exist — skipping migration")
        return

    df = pd.read_csv(COUNCIL_FILE)

    if 'cycle_id' in df.columns:
        logger.info(f"{COUNCIL_FILE} already has cycle_id column")
        return

    df.insert(0, 'cycle_id', '')
    df.to_csv(COUNCIL_FILE, index=False)
    logger.info(f"Added cycle_id column to {COUNCIL_FILE} ({len(df)} rows)")


def fix_legacy_accuracy_file():
    """
    Fix schema corruption in agent_accuracy.csv.
    Handles both 5-column (correct) and 7-column (corrupted) formats.
    """
    if not os.path.exists(ACCURACY_FILE):
        logger.info(f"{ACCURACY_FILE} doesn't exist — skipping")
        return

    with open(ACCURACY_FILE, 'r') as f:
        lines = f.readlines()

    if not lines:
        return

    header = lines[0].strip()
    logger.info(f"Legacy accuracy header: {header}")

    correct_columns = ['timestamp', 'agent', 'predicted', 'actual', 'correct']

    valid_rows = []
    skipped = 0

    for i, line in enumerate(lines[1:], start=2):
        parts = line.strip().split(',')

        if len(parts) == 5:
            valid_rows.append({
                'timestamp': parts[0],
                'agent': parts[1].lower(),
                'predicted': parts[2].upper(),
                'actual': parts[3].upper(),
                'correct': int(parts[4]) if parts[4].strip().isdigit() else 0
            })
        elif len(parts) == 7:
            # Corrupted 7-column format
            valid_rows.append({
                'timestamp': parts[0],
                'agent': parts[1].lower(),
                'predicted': parts[2].upper(),
                'actual': parts[5].upper(),
                'correct': int(parts[6]) if parts[6].strip().isdigit() else 0
            })
        else:
            skipped += 1

    df = pd.DataFrame(valid_rows)
    if df.empty:
        logger.warning("No valid data in legacy file")
        return

    before_dedup = len(df)
    df = df.drop_duplicates()
    logger.info(f"Legacy file: {len(df)} valid rows ({skipped} skipped, {before_dedup - len(df)} deduped)")

    df.to_csv(ACCURACY_FILE, index=False, columns=correct_columns)
    logger.info(f"Wrote clean {ACCURACY_FILE}")


def resolve_with_cycle_aware_match(dry_run: bool = False):
    """
    Resolve PENDING predictions using cycle-aware matching.

    IMPROVEMENT OVER v4 nearest-match:
    - v4 matched predictions to nearest RECONCILED decision (cross-cycle risk)
    - v5 matches predictions to nearest decision (ANY), then checks if reconciled
    - This prevents cross-cycle contamination of accuracy scores

    Steps:
    1. Match each prediction to its own cycle (nearest council decision, any status)
    2. Check if that cycle's council decision has been reconciled
    3. Only resolve if reconciled; otherwise classify as "awaiting reconciliation"
    """
    if not os.path.exists(STRUCTURED_FILE) or not os.path.exists(COUNCIL_FILE):
        logger.info("Missing required files for resolution")
        return 0

    predictions_df = pd.read_csv(STRUCTURED_FILE)
    council_df = pd.read_csv(COUNCIL_FILE)

    if predictions_df.empty or council_df.empty:
        return 0

    # Parse timestamps
    predictions_df['timestamp'] = parse_ts_column(predictions_df['timestamp'])
    council_df['timestamp'] = parse_ts_column(council_df['timestamp'])

    # Filter to PENDING
    pending_mask = predictions_df['actual'] == 'PENDING'
    pending_count = pending_mask.sum()

    if pending_count == 0:
        logger.info("No pending predictions to resolve")
        return 0

    logger.info(f"Found {pending_count} pending predictions")

    # === KEY CHANGE: Use ALL council decisions for cycle matching ===
    all_decisions = council_df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Total council decisions: {len(all_decisions)}")

    # Identify reconciled decisions (have actual_trend_direction)
    reconciled_mask = (
        all_decisions['actual_trend_direction'].notna() &
        (all_decisions['actual_trend_direction'] != '') &
        (all_decisions['actual_trend_direction'].astype(str).str.strip() != '')
    )
    reconciled_count = reconciled_mask.sum()
    logger.info(f"Reconciled council decisions: {reconciled_count}/{len(all_decisions)}")

    # Direction normalization
    direction_map = {
        'UP': 'BULLISH', 'DOWN': 'BEARISH', 'BULLISH': 'BULLISH',
        'BEARISH': 'BEARISH', 'NEUTRAL': 'NEUTRAL', 'FLAT': 'NEUTRAL'
    }

    # === PHASE 1: Match each prediction to its own cycle ===
    resolved_count = 0
    awaiting_reconciliation = 0
    orphaned = 0
    gap_stats = []

    for idx in predictions_df[pending_mask].index:
        pred_time = predictions_df.loc[idx, 'timestamp']

        # Find nearest council decision (ANY status)
        time_diffs = abs(all_decisions['timestamp'] - pred_time)
        nearest_idx = time_diffs.idxmin()
        nearest_gap = time_diffs[nearest_idx]

        # Safety cap: 2 hours max (prevents matching to totally unrelated cycles)
        if nearest_gap > pd.Timedelta(hours=2):
            orphaned += 1
            continue

        # === PHASE 2: Check if this cycle's decision is reconciled ===
        if not reconciled_mask.iloc[nearest_idx]:
            # This prediction's own cycle hasn't been reconciled yet
            awaiting_reconciliation += 1
            continue

        # === PHASE 3: Resolve with correct cycle's outcome ===
        raw_actual = str(all_decisions.loc[nearest_idx, 'actual_trend_direction']).upper().strip()
        actual = direction_map.get(raw_actual)

        if actual:
            predictions_df.loc[idx, 'actual'] = actual
            resolved_count += 1
            gap_stats.append(nearest_gap.total_seconds() / 60)

    # === DIAGNOSTICS ===
    logger.info(f"\n{'='*50}")
    logger.info(f"RESOLUTION BREAKDOWN:")
    logger.info(f"  Total pending:              {pending_count}")
    logger.info(f"  ✅ Resolved:                 {resolved_count}")
    logger.info(f"  ⏳ Awaiting reconciliation:  {awaiting_reconciliation}")
    logger.info(f"  ❌ Orphaned (>2h gap):       {orphaned}")
    logger.info(f"{'='*50}")

    if gap_stats:
        logger.info(f"Match gap stats (minutes): "
                    f"min={min(gap_stats):.1f}, max={max(gap_stats):.1f}, "
                    f"mean={np.mean(gap_stats):.1f}, median={np.median(gap_stats):.1f}")

    # Estimate potential from running reconciliation
    if awaiting_reconciliation > 0:
        logger.info(f"\n💡 Running 'python backfill_council_history.py' could unlock "
                    f"up to {awaiting_reconciliation} more predictions")

    if resolved_count > 0 and not dry_run:
        predictions_df.to_csv(STRUCTURED_FILE, index=False)
        logger.info(f"Saved updated {STRUCTURED_FILE}")

        # Sync to legacy file
        newly_resolved = predictions_df[
            (predictions_df['actual'] != 'PENDING') &
            predictions_df.index.isin(predictions_df[pending_mask].index)
        ].copy()

        if not newly_resolved.empty:
            newly_resolved['correct'] = (
                newly_resolved['direction'].str.upper() == newly_resolved['actual'].str.upper()
            ).astype(int)

            with open(ACCURACY_FILE, 'a') as f:
                for _, row in newly_resolved.iterrows():
                    agent = str(row.get('agent', '')).lower()
                    f.write(f"{row['timestamp']},{agent},{row['direction']},{row['actual']},{row['correct']}\n")

            logger.info(f"Appended {len(newly_resolved)} rows to {ACCURACY_FILE}")

    return resolved_count


def print_summary():
    """Print post-migration summary statistics."""
    logger.info("\n=== POST-MIGRATION SUMMARY ===")

    if os.path.exists(STRUCTURED_FILE):
        df = pd.read_csv(STRUCTURED_FILE)
        pending = (df['actual'] == 'PENDING').sum() if 'actual' in df.columns else 'N/A'
        resolved = len(df) - pending if isinstance(pending, int) else 'N/A'
        has_cycle_id = ('cycle_id' in df.columns)
        logger.info(f"Structured predictions: {len(df)} total, {resolved} resolved, {pending} pending, cycle_id column: {has_cycle_id}")

    if os.path.exists(COUNCIL_FILE):
        df = pd.read_csv(COUNCIL_FILE)
        reconciled = df['actual_trend_direction'].notna().sum() if 'actual_trend_direction' in df.columns else 'N/A'
        has_cycle_id = ('cycle_id' in df.columns)
        logger.info(f"Council history: {len(df)} total, {reconciled} reconciled, cycle_id column: {has_cycle_id}")

    if os.path.exists(ACCURACY_FILE):
        df = pd.read_csv(ACCURACY_FILE)
        logger.info(f"Legacy accuracy: {len(df)} rows, agents: {df['agent'].nunique() if 'agent' in df.columns else 'N/A'}")


def main():
    parser = argparse.ArgumentParser(description="Brier Score Data Migration v5")
    parser.add_argument('--dry-run', action='store_true', help="Preview changes without writing")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BRIER SCORE DATA MIGRATION v5 — DEFINITIVE")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE — no files will be modified")

    # Step 1: Backup
    if not args.dry_run:
        backups = backup_files()
        logger.info(f"Created {len(backups)} backups")

    # Step 2: Schema migrations
    if not args.dry_run:
        migrate_structured_csv()
        migrate_council_history_csv()
        fix_legacy_accuracy_file()

    # Step 3: Resolve pending predictions
    resolved = resolve_with_cycle_aware_match(dry_run=args.dry_run)

    # Step 4: Summary
    if not args.dry_run:
        print_summary()

    logger.info("=" * 60)
    logger.info(f"MIGRATION COMPLETE — Resolved {resolved} predictions")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
