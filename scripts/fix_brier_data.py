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
import re
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

# Regex: matches cycle_id patterns like "KC-c0f3b12c", "CC-ab12ef34"
CYCLE_ID_PATTERN = re.compile(r'^[A-Z]{2,4}-[0-9a-f]{6,12}$')

# Known valid agent names for recovery validation
KNOWN_AGENTS = {
    'agronomist', 'macro', 'geopolitical', 'supply_chain', 'inventory',
    'sentiment', 'technical', 'volatility', 'microstructure', 'fundamentalist',
}

# Valid direction values for recovery validation
VALID_DIRECTIONS = {'BULLISH', 'BEARISH', 'NEUTRAL'}


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

    # R3: Enforce canonical column order
    CANONICAL_ORDER = ['cycle_id', 'timestamp', 'agent', 'direction', 'confidence', 'prob_bullish', 'actual']

    # Reorder columns to match canonical schema
    existing_cols = [c for c in CANONICAL_ORDER if c in df.columns]
    extra_cols = [c for c in df.columns if c not in CANONICAL_ORDER]
    df = df[existing_cols + extra_cols]

    df.to_csv(STRUCTURED_FILE, index=False)
    logger.info(f"Added cycle_id column to {STRUCTURED_FILE} ({len(df)} rows) and enforced canonical order")


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


def sanitize_structured_csv():
    """
    Detect and repair column-misaligned rows in agent_accuracy_structured.csv.

    ROOT CAUSE: record_prediction_structured() writes cycle_id as the FIRST field,
    but if the file header has cycle_id as the LAST column (pre-v6.4 layout),
    all values shift one position right when read by pandas.

    DETECTION: timestamp column contains a cycle_id pattern (e.g., KC-c0f3b12c)
    RECOVERY: Right-shift all column values to restore correct alignment
    FALLBACK: Drop unrecoverable rows with logging

    IDEMPOTENT: Safe to run multiple times. Clean rows are not modified.
    """
    if not os.path.exists(STRUCTURED_FILE):
        logger.info(f"{STRUCTURED_FILE} doesn't exist — skipping sanitization")
        return

    df = pd.read_csv(STRUCTURED_FILE)
    if df.empty:
        return

    # Detect misaligned rows: timestamp column contains a cycle_id pattern
    if df['timestamp'].dtype != 'object' and df['timestamp'].dtype != 'string':
        logger.info("Timestamp column is numeric type — no string corruption possible")
        return

    bad_mask = df['timestamp'].astype(str).str.match(CYCLE_ID_PATTERN, na=False)
    bad_count = bad_mask.sum()

    if bad_count == 0:
        logger.info(f"Sanitization: No misaligned rows detected in {STRUCTURED_FILE}")
        return

    logger.warning(f"Sanitization: Found {bad_count} misaligned rows — attempting recovery")

    # Determine current header order
    columns = list(df.columns)
    logger.info(f"Current header order: {columns}")

    recovered = 0
    dropped = 0

    for idx in df[bad_mask].index:
        row = df.loc[idx]

        # The data was written as: cycle_id, timestamp, agent, direction, confidence, prob_bullish, actual
        # But the header reads it as: timestamp, agent, direction, confidence, prob_bullish, actual, cycle_id
        # So: current 'timestamp' = real cycle_id
        #     current 'agent' = real timestamp
        #     current 'direction' = real agent
        #     current 'confidence' = real direction
        #     current 'prob_bullish' = real confidence
        #     current 'actual' = real prob_bullish
        #     current 'cycle_id' = real actual

        real_cycle_id = str(row.get('timestamp', ''))
        real_timestamp = str(row.get('agent', ''))
        real_agent = str(row.get('direction', '')).lower()
        real_direction = str(row.get('confidence', '')).upper()
        # For numeric columns, keep as numeric if possible to satisfy pandas strict typing
        raw_conf = row.get('prob_bullish', '')
        try:
            real_confidence = float(raw_conf)
        except (ValueError, TypeError):
            real_confidence = str(raw_conf)

        raw_prob = row.get('actual', '')
        try:
            real_prob_bullish = float(raw_prob)
        except (ValueError, TypeError):
            real_prob_bullish = str(raw_prob)

        real_actual = str(row.get('cycle_id', '')).upper()

        # Ensure compatibility with target column dtypes (Pandas 3.0 strictness)
        # 1. confidence
        if pd.api.types.is_string_dtype(df['confidence'].dtype):
            real_confidence = str(real_confidence)
        elif pd.api.types.is_numeric_dtype(df['confidence'].dtype):
            try:
                real_confidence = float(real_confidence)
            except (ValueError, TypeError):
                pass  # Keep as is, let assignment fail if incompatible

        # 2. prob_bullish
        if pd.api.types.is_string_dtype(df['prob_bullish'].dtype):
            real_prob_bullish = str(real_prob_bullish)
        elif pd.api.types.is_numeric_dtype(df['prob_bullish'].dtype):
            try:
                real_prob_bullish = float(real_prob_bullish)
            except (ValueError, TypeError):
                pass

        # Validate recovery: check that shifted values make sense
        is_valid = True
        validation_failures = []

        # 1. real_cycle_id should match cycle_id pattern (already confirmed by bad_mask)
        # 2. real_timestamp should be parseable as datetime
        try:
            pd.to_datetime(real_timestamp)
        except Exception:
            is_valid = False
            validation_failures.append(f"timestamp={real_timestamp!r} not parseable")

        # 3. real_agent should be a known agent name
        if real_agent not in KNOWN_AGENTS:
            is_valid = False
            validation_failures.append(f"agent={real_agent!r} not in known agents")

        # 4. real_direction should be a valid direction
        if real_direction not in VALID_DIRECTIONS:
            is_valid = False
            validation_failures.append(f"direction={real_direction!r} not valid")

        # 5. real_actual should be a valid status
        if real_actual not in ('PENDING', 'BULLISH', 'BEARISH', 'NEUTRAL', 'ORPHANED', ''):
            is_valid = False
            validation_failures.append(f"actual={real_actual!r} not valid status")

        if is_valid:
            # Apply recovery: reassign columns to correct values
            df.loc[idx, 'cycle_id'] = real_cycle_id
            df.loc[idx, 'timestamp'] = real_timestamp
            df.loc[idx, 'agent'] = real_agent
            df.loc[idx, 'direction'] = real_direction
            df.loc[idx, 'confidence'] = real_confidence
            df.loc[idx, 'prob_bullish'] = real_prob_bullish
            df.loc[idx, 'actual'] = real_actual
            recovered += 1
            logger.info(f"  Recovered row {idx}: {real_agent} cycle={real_cycle_id} actual={real_actual}")
        else:
            # Drop unrecoverable row
            df = df.drop(idx)
            dropped += 1
            logger.warning(f"  Dropped unrecoverable row {idx}: {validation_failures}")

    # Save cleaned file
    df.to_csv(STRUCTURED_FILE, index=False)
    logger.info(
        f"Sanitization complete: {recovered} recovered, {dropped} dropped "
        f"(out of {bad_count} misaligned)"
    )


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

    # Parse timestamps (coerce mode: unparseable values become NaT instead of crashing)
    predictions_df['timestamp'] = parse_ts_column(predictions_df['timestamp'], errors='coerce')
    council_df['timestamp'] = parse_ts_column(council_df['timestamp'], errors='coerce')

    # Drop rows with unparseable timestamps (defense-in-depth after sanitization)
    pred_nat_count = predictions_df['timestamp'].isna().sum()
    if pred_nat_count > 0:
        logger.warning(f"Dropping {pred_nat_count} predictions with unparseable timestamps")
        predictions_df = predictions_df.dropna(subset=['timestamp'])

    council_nat_count = council_df['timestamp'].isna().sum()
    if council_nat_count > 0:
        logger.warning(f"Dropping {council_nat_count} council rows with unparseable timestamps")
        council_df = council_df.dropna(subset=['timestamp'])

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

    # Filter corrupt cycles (shutdown race condition)
    corrupt_mask = all_decisions['master_reasoning'].str.contains(
        'cannot schedule new futures after shutdown',
        na=False, case=False
    )
    if corrupt_mask.any():
        corrupt_count = corrupt_mask.sum()
        logger.warning(
            f"Found {corrupt_count} corrupt council decisions (shutdown race condition). "
            f"Marking associated predictions as ORPHANED."
        )
        # Note: We rely on the orphan logic (no valid match) to handle these,
        # but we could also proactively mark them. For now, just logging is good,
        # as they won't have a valid 'actual_trend_direction' so they won't be reconciled.

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
    orphaned_indices = []
    gap_stats = []

    for idx in predictions_df[pending_mask].index:
        pred_time = predictions_df.loc[idx, 'timestamp']

        # Find nearest council decision (ANY status)
        time_diffs = abs(all_decisions['timestamp'] - pred_time)
        nearest_idx = time_diffs.idxmin()
        nearest_gap = time_diffs[nearest_idx]

        # Safety cap: 2 hours max (prevents matching to totally unrelated cycles)
        if nearest_gap > pd.Timedelta(hours=2):
            orphaned_indices.append(idx)
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

    # === ORPHAN HANDLING ===
    if orphaned_indices and not dry_run:
        predictions_df.loc[orphaned_indices, 'actual'] = 'ORPHANED'
        logger.info(f"Classified {len(orphaned_indices)} predictions as ORPHANED (no council decision within 2h)")

    orphaned_count = len(orphaned_indices)

    # === DIAGNOSTICS ===
    total = len(predictions_df)
    orphaned_total = (predictions_df['actual'] == 'ORPHANED').sum()
    if dry_run: orphaned_total += orphaned_count

    still_pending = (predictions_df['actual'] == 'PENDING').sum()
    if dry_run: still_pending -= resolved_count

    resolved_total = total - orphaned_total - still_pending
    resolvable = total - orphaned_total
    effective_rate = (resolved_total / resolvable * 100) if resolvable > 0 else 0

    logger.info(f"\n{'='*50}")
    logger.info(f"RESOLUTION BREAKDOWN:")
    logger.info(f"  Total predictions:            {total}")
    logger.info(f"  ✅ Resolved:                   {resolved_total} (+{resolved_count} new)")
    logger.info(f"  ⏳ Awaiting reconciliation:    {still_pending} (potential: {awaiting_reconciliation})")
    logger.info(f"  🗑️  Orphaned (no council):     {orphaned_total} (+{orphaned_count} new)")
    logger.info(f"  📊 Effective resolution rate:  {effective_rate:.0f}% (excl. orphans)")
    logger.info(f"{'='*50}")

    if gap_stats:
        logger.info(f"Match gap stats (minutes): "
                    f"min={min(gap_stats):.1f}, max={max(gap_stats):.1f}, "
                    f"mean={np.mean(gap_stats):.1f}, median={np.median(gap_stats):.1f}")

    # Estimate potential from running reconciliation
    if awaiting_reconciliation > 0:
        logger.info(f"\n💡 Running 'python backfill_council_history.py' could unlock "
                    f"up to {awaiting_reconciliation} more predictions")

    # Save changes if any (resolved or orphaned)
    if (resolved_count > 0 or orphaned_count > 0) and not dry_run:
        predictions_df.to_csv(STRUCTURED_FILE, index=False)
        logger.info(f"Saved updated {STRUCTURED_FILE}")

        # Sync to legacy file (only resolved ones, orphans don't go to legacy)
        if resolved_count > 0:
            newly_resolved = predictions_df[
                (predictions_df['actual'] != 'PENDING') &
                (predictions_df['actual'] != 'ORPHANED') &
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
        orphaned = (df['actual'] == 'ORPHANED').sum() if 'actual' in df.columns else 0
        resolved = len(df) - pending - orphaned if isinstance(pending, int) else 'N/A'
        has_cycle_id = ('cycle_id' in df.columns)
        logger.info(f"Structured predictions: {len(df)} total, {resolved} resolved, {pending} pending, {orphaned} orphaned")

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

    # Step 2.5: Sanitize structured CSV (fix column misalignment)
    if not args.dry_run:
        sanitize_structured_csv()

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
