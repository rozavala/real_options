#!/usr/bin/env python3
"""
Migration 003: Normalize trigger_type values in decision_signals.csv

Fixes:
- "TriggerType.SCHEDULED" → "SCHEDULED"
- "TriggerType.WEATHER" → "WEATHER"
- "scheduled" → "SCHEDULED"
- Any "TriggerType.X" → "X"

Idempotent: safe to run multiple times.
"""

import os
import re
import logging
import pandas as pd
import sys

# Add project root to sys.path to ensure local imports work if needed (though not used here)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIGNALS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'decision_signals.csv'
)


def run_migration() -> bool:
    if not os.path.exists(SIGNALS_PATH):
        logger.info("No decision_signals.csv found. Nothing to migrate.")
        return True

    try:
        df = pd.read_csv(SIGNALS_PATH)
        if df.empty:
            logger.info("decision_signals.csv is empty.")
            return True

        if 'trigger_type' not in df.columns:
            logger.info("trigger_type column missing in decision_signals.csv.")
            return True

        original_values = df['trigger_type'].value_counts().to_dict()
        logger.info(f"Before normalization: {original_values}")

        def normalize_trigger(val):
            if pd.isna(val) or val == '':
                return 'SCHEDULED'
            val = str(val).strip()
            val = re.sub(r'^TriggerType\.', '', val)
            return val.upper()

        df['trigger_type'] = df['trigger_type'].apply(normalize_trigger)

        normalized_values = df['trigger_type'].value_counts().to_dict()
        logger.info(f"After normalization: {normalized_values}")

        df.to_csv(SIGNALS_PATH, index=False)
        logger.info(f"Normalized {len(df)} rows in decision_signals.csv")
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


if __name__ == '__main__':
    success = run_migration()
    exit(0 if success else 1)
