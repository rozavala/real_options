#!/usr/bin/env python3
"""
One-time backfill: Seed enhanced_brier.json from legacy structured CSV.

Reads agent_accuracy_structured.csv, creates EnhancedBrierTracker predictions
for all resolved entries, and saves to disk.

SAFE TO RUN MULTIPLE TIMES: Checks for existing data and skips duplicates.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.enhanced_brier import EnhancedBrierTracker, MarketRegime
from trading_bot.brier_bridge import _confidence_to_probs
from trading_bot.cycle_id import is_valid_cycle_id

STRUCTURED_FILE = "data/agent_accuracy_structured.csv"
ENHANCED_FILE = "data/enhanced_brier.json"


def backfill():
    if not os.path.exists(STRUCTURED_FILE):
        print(f"❌ {STRUCTURED_FILE} not found")
        return

    try:
        df = pd.read_csv(STRUCTURED_FILE)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    # Check for timestamp column
    if 'timestamp' not in df.columns:
        print("❌ 'timestamp' column missing in CSV")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    # Filter to resolved predictions only
    if 'actual' not in df.columns:
        print("❌ 'actual' column missing in CSV")
        return

    resolved = df[
        (df['actual'] != 'PENDING') &
        (df['actual'] != 'ORPHANED') &
        (df['timestamp'].notna())
    ].copy()

    print(f"📊 Found {len(resolved)} resolved predictions to backfill")

    tracker = EnhancedBrierTracker(data_path=ENHANCED_FILE)
    existing_count = len(tracker.predictions)
    print(f"📂 Existing enhanced predictions: {existing_count}")

    # Build set of existing prediction keys to avoid duplicates
    existing_keys = set()
    for p in tracker.predictions:
        key = f"{p.agent}_{p.timestamp.isoformat()}_{p.cycle_id}"
        existing_keys.add(key)

    recorded = 0
    resolved_count = 0

    for _, row in resolved.iterrows():
        agent = str(row.get('agent', '')).strip()
        direction = str(row.get('direction', 'NEUTRAL')).upper().strip()
        confidence = float(row.get('confidence', 0.5))
        actual = str(row.get('actual', '')).upper().strip()
        ts = row['timestamp'].to_pydatetime()
        cycle_id = str(row.get('cycle_id', '')).strip()

        if not agent or actual not in ('BULLISH', 'BEARISH', 'NEUTRAL'):
            continue

        # Check for duplicate
        key = f"{agent}_{ts.isoformat()}_{cycle_id}"
        if key in existing_keys:
            continue

        # Record the prediction
        prob_b, prob_n, prob_be = _confidence_to_probs(direction, confidence)
        tracker.record_prediction(
            agent=agent,
            prob_bullish=prob_b,
            prob_neutral=prob_n,
            prob_bearish=prob_be,
            regime=MarketRegime.NORMAL,
            contract='',
            timestamp=ts,
            cycle_id=cycle_id,
        )
        recorded += 1

        # Immediately resolve it
        brier = tracker.resolve_prediction(
            agent=agent,
            actual_outcome=actual,
            cycle_id=cycle_id if is_valid_cycle_id(cycle_id) else '',
            timestamp=ts,
        )
        if brier is not None:
            resolved_count += 1

    # Final save (resolve_prediction saves per-prediction, but ensure final state)
    tracker._save()

    print(f"✅ Backfill complete:")
    print(f"   Recorded: {recorded}")
    print(f"   Resolved with Brier scores: {resolved_count}")
    print(f"   Total predictions in tracker: {len(tracker.predictions)}")
    print(f"   Agents with scores: {list(tracker.agent_scores.keys())}")


if __name__ == '__main__':
    backfill()
