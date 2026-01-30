#!/usr/bin/env python3
"""
One-time script to fix Brier score data files.
Run this ONCE after deploying the code fixes.

Usage: python scripts/fix_brier_data.py
"""

import pandas as pd
import os
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
ACCURACY_FILE = os.path.join(DATA_DIR, "agent_accuracy.csv")
STRUCTURED_FILE = os.path.join(DATA_DIR, "agent_accuracy_structured.csv")
COUNCIL_FILE = os.path.join(DATA_DIR, "council_history.csv")

def backup_files():
    """Create timestamped backups before modification."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for filepath in [ACCURACY_FILE, STRUCTURED_FILE]:
        if os.path.exists(filepath):
            backup_path = filepath.replace('.csv', f'_backup_{timestamp}.csv')
            shutil.copy(filepath, backup_path)
            logger.info(f"Backed up {filepath} to {backup_path}")

def fix_legacy_accuracy_file():
    """
    Fix schema corruption in agent_accuracy.csv.

    The file may have a 7-column header but 5-column data.
    We need to:
    1. Read with correct column count
    2. Remove duplicate/corrupt rows
    3. Write back with correct header
    """
    if not os.path.exists(ACCURACY_FILE):
        logger.info("No agent_accuracy.csv to fix")
        return

    logger.info("Fixing agent_accuracy.csv schema...")

    # Read raw to detect format
    with open(ACCURACY_FILE, 'r') as f:
        lines = f.readlines()

    if not lines:
        logger.info("File is empty")
        return

    header = lines[0].strip()
    logger.info(f"Current header: {header}")

    # Expected 5-column format
    correct_header = "timestamp,agent,predicted,actual,correct"
    correct_columns = ['timestamp', 'agent', 'predicted', 'actual', 'correct']

    # Parse data rows (skip header)
    valid_rows = []
    skipped = 0

    for i, line in enumerate(lines[1:], start=2):
        parts = line.strip().split(',')

        # Handle 5-column rows (correct format)
        if len(parts) == 5:
            valid_rows.append({
                'timestamp': parts[0],
                'agent': parts[1].lower(),  # Normalize to lowercase
                'predicted': parts[2].upper(),
                'actual': parts[3].upper(),
                'correct': int(parts[4]) if parts[4].isdigit() else 0
            })
        # Handle 7-column rows (wrong format) - try to extract useful data
        elif len(parts) == 7:
            # timestamp,agent,predicted,confidence,prob_bullish,actual,correct
            valid_rows.append({
                'timestamp': parts[0],
                'agent': parts[1].lower(),
                'predicted': parts[2].upper(),
                'actual': parts[5].upper(),
                'correct': int(parts[6]) if parts[6].isdigit() else 0
            })
        else:
            logger.warning(f"Line {i}: Unexpected format ({len(parts)} columns), skipping")
            skipped += 1

    logger.info(f"Parsed {len(valid_rows)} valid rows, skipped {skipped}")

    # Create DataFrame and deduplicate
    df = pd.DataFrame(valid_rows)

    if df.empty:
        logger.warning("No valid data found!")
        return

    # Remove exact duplicates
    before_dedup = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before_dedup - len(df)} duplicate rows")

    # Write back with correct header
    df.to_csv(ACCURACY_FILE, index=False, columns=correct_columns)
    logger.info(f"Wrote {len(df)} rows with correct schema")

def resolve_old_pending_predictions():
    """
    Attempt to resolve old PENDING predictions using council_history.
    Uses a wider time window (30 minutes) for historical data.
    """
    if not os.path.exists(STRUCTURED_FILE) or not os.path.exists(COUNCIL_FILE):
        logger.info("Missing required files for resolution")
        return

    logger.info("Resolving old PENDING predictions...")

    predictions_df = pd.read_csv(STRUCTURED_FILE)
    council_df = pd.read_csv(COUNCIL_FILE)

    if predictions_df.empty or council_df.empty:
        logger.info("Empty dataframes")
        return

    # Parse timestamps
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'], utc=True)
    council_df['timestamp'] = pd.to_datetime(council_df['timestamp'], utc=True)

    # Filter to PENDING only
    pending_mask = predictions_df['actual'] == 'PENDING'
    pending_count = pending_mask.sum()

    if pending_count == 0:
        logger.info("No pending predictions")
        return

    logger.info(f"Found {pending_count} pending predictions")

    # Get reconciled council decisions
    reconciled = council_df[
        (council_df['actual_trend_direction'].notna()) &
        (council_df['actual_trend_direction'] != '')
    ].copy()

    logger.info(f"Found {len(reconciled)} reconciled council decisions")

    resolved_count = 0

    for idx in predictions_df[pending_mask].index:
        pred_time = predictions_df.loc[idx, 'timestamp']

        # Use wider window (30 min) for historical backfill
        time_window = pd.Timedelta(minutes=30)

        matches = reconciled[
            (reconciled['timestamp'] >= pred_time - time_window) &
            (reconciled['timestamp'] <= pred_time + time_window)
        ]

        if not matches.empty:
            actual = str(matches.iloc[0]['actual_trend_direction']).upper().strip()

            # Normalize direction
            if actual == 'UP':
                actual = 'BULLISH'
            elif actual == 'DOWN':
                actual = 'BEARISH'

            if actual in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                predictions_df.loc[idx, 'actual'] = actual
                resolved_count += 1

    logger.info(f"Resolved {resolved_count} predictions")

    if resolved_count > 0:
        # Save updated structured file
        predictions_df.to_csv(STRUCTURED_FILE, index=False)

        # Append newly resolved to legacy file
        newly_resolved = predictions_df[
            (predictions_df['actual'] != 'PENDING') &
            predictions_df.index.isin(predictions_df[pending_mask].index)
        ].copy()

        if not newly_resolved.empty:
            # Calculate correctness
            newly_resolved['correct'] = (
                newly_resolved['direction'].str.upper() == newly_resolved['actual'].str.upper()
            ).astype(int)

            # Append to legacy file
            with open(ACCURACY_FILE, 'a') as f:
                for _, row in newly_resolved.iterrows():
                    agent = row['agent'].lower()  # Normalize
                    f.write(f"{row['timestamp']},{agent},{row['direction']},{row['actual']},{row['correct']}\n")

            logger.info(f"Appended {len(newly_resolved)} rows to legacy file")

def main():
    logger.info("=== Brier Score Data Fix Script ===")

    # Safety backup
    backup_files()

    # Fix schema corruption
    fix_legacy_accuracy_file()

    # Resolve old predictions
    resolve_old_pending_predictions()

    logger.info("=== Done ===")

if __name__ == "__main__":
    main()
