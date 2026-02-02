"""
Migration 002: Backfill decision_signals.csv from council_history.csv

Extracts lightweight decision records from the full council history,
populating the new decision_signals.csv with all historical decisions.

IDEMPOTENT: If decision_signals.csv already exists with data, this migration
only appends rows whose (timestamp, contract) pair doesn't already exist.

SOURCE: data/council_history.csv (authoritative — 30+ column forensic record)
TARGET: decision_signals.csv (lightweight — 11 column operational summary)
"""

import os
import sys
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths (relative to repo root — run_migrations.sh cd's there)
COUNCIL_HISTORY_PATH = 'data/council_history.csv'
DECISION_SIGNALS_PATH = 'decision_signals.csv'
OLD_MODEL_SIGNALS_PATH = 'model_signals.csv'
ARCHIVE_PATH = 'archive/ml_pipeline/model_signals_final.csv'


def run():
    """Main migration logic."""

    # --- Step 1: Load council_history ---
    if not os.path.exists(COUNCIL_HISTORY_PATH):
        logger.warning(f"No council_history.csv found at {COUNCIL_HISTORY_PATH}. Nothing to migrate.")
        return True

    try:
        council_df = pd.read_csv(COUNCIL_HISTORY_PATH)
    except Exception as e:
        logger.error(f"Failed to read council_history.csv: {e}")
        return False

    if council_df.empty:
        logger.info("council_history.csv is empty. Nothing to migrate.")
        return True

    logger.info(f"Loaded {len(council_df)} rows from council_history.csv")

    # --- Step 2: Transform to decision_signals schema ---
    signals_df = pd.DataFrame({
        'timestamp': council_df['timestamp'],
        'cycle_id': council_df.get('cycle_id', ''),
        'contract': council_df['contract'],
        'signal': council_df['master_decision'].fillna('NEUTRAL'),
        'prediction_type': council_df.get('prediction_type', 'DIRECTIONAL').fillna('DIRECTIONAL'),
        'strategy': council_df.get('strategy_type', 'NONE').fillna('NONE'),
        'price': council_df['entry_price'],
        'sma_200': None,  # Not tracked in council_history
        'confidence': council_df.get('master_confidence', None),
        'regime': 'UNKNOWN',  # Not tracked in council_history
        'trigger_type': council_df.get('trigger_type', 'SCHEDULED').fillna('SCHEDULED'),
    })

    # Clean: replace empty strings with proper defaults
    signals_df['cycle_id'] = signals_df['cycle_id'].fillna('')
    signals_df['trigger_type'] = signals_df['trigger_type'].replace('', 'SCHEDULED')
    signals_df['strategy'] = signals_df['strategy'].replace('', 'NONE')

    # Clamp confidence values
    if 'confidence' in signals_df.columns:
        signals_df['confidence'] = pd.to_numeric(signals_df['confidence'], errors='coerce')
        signals_df['confidence'] = signals_df['confidence'].clip(0.0, 1.0)

    # Round prices
    for col in ['price', 'sma_200']:
        if col in signals_df.columns:
            signals_df[col] = pd.to_numeric(signals_df[col], errors='coerce').round(2)

    logger.info(f"Transformed {len(signals_df)} decision signal rows")

    # --- Step 3: Deduplicate against existing decision_signals (idempotent) ---
    if os.path.exists(DECISION_SIGNALS_PATH):
        try:
            existing_df = pd.read_csv(DECISION_SIGNALS_PATH)
            if not existing_df.empty:
                # Create dedup key: timestamp + contract
                existing_keys = set(
                    existing_df['timestamp'].astype(str) + '|' + existing_df['contract'].astype(str)
                )
                new_keys = signals_df['timestamp'].astype(str) + '|' + signals_df['contract'].astype(str)

                before_count = len(signals_df)
                signals_df = signals_df[~new_keys.isin(existing_keys)]
                skipped = before_count - len(signals_df)

                if skipped > 0:
                    logger.info(f"Skipped {skipped} rows already present in decision_signals.csv")

                if signals_df.empty:
                    logger.info("All rows already exist. Migration is a no-op.")
                    return True

                # Append to existing
                combined = pd.concat([existing_df, signals_df], ignore_index=True)
                combined.to_csv(DECISION_SIGNALS_PATH, index=False)
                logger.info(f"Appended {len(signals_df)} new rows (total: {len(combined)})")
                return True
        except pd.errors.EmptyDataError:
            pass  # File exists but empty — treat as new

    # --- Step 4: Write new file ---
    signals_df.to_csv(DECISION_SIGNALS_PATH, index=False)
    logger.info(f"Created decision_signals.csv with {len(signals_df)} rows")

    # --- Step 5: Archive old model_signals.csv if it exists ---
    if os.path.exists(OLD_MODEL_SIGNALS_PATH):
        os.makedirs(os.path.dirname(ARCHIVE_PATH), exist_ok=True)
        os.rename(OLD_MODEL_SIGNALS_PATH, ARCHIVE_PATH)
        logger.info(f"Archived {OLD_MODEL_SIGNALS_PATH} → {ARCHIVE_PATH}")

    return True


if __name__ == '__main__':
    success = run()
    sys.exit(0 if success else 1)
