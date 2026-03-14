#!/usr/bin/env python3
"""Backfill trade journal entries with metadata from council_history.csv.

One-shot migration script that:
1. Enriches journal entries with thesis_strength, primary_catalyst, dissent_acknowledged
   from council_history.csv (matched by position_id = cycle_id)
2. Infers schedule_id from entry timestamp for scheduled triggers
3. Optionally re-runs LLM narratives for entries flagged as truncation-affected

Usage:
    python scripts/backfill_trade_journal.py --data-dir data/KC
    python scripts/backfill_trade_journal.py --data-dir data/KC --dry-run
    python scripts/backfill_trade_journal.py --data-dir data/KC --regen-narratives
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_council_history(data_dir: str) -> dict:
    """Load council_history.csv and index by cycle_id."""
    csv_path = os.path.join(data_dir, "council_history.csv")
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found")
        return {}

    indexed = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cycle_id = row.get('cycle_id', '').strip()
            if cycle_id:
                indexed[cycle_id] = row
    print(f"  Loaded {len(indexed)} council_history rows (indexed by cycle_id)")
    return indexed


def infer_schedule_id(timestamp_str: str) -> str:
    """Infer schedule_id from entry timestamp (ET-based schedule windows).

    Schedule windows (from orchestrator default schedule):
      08:30-10:30 ET → signal_early
      10:30-12:00 ET → signal_euro
      12:00-14:00 ET → signal_us_open
      14:00-16:00 ET → signal_peak
      16:00-18:00 ET → signal_settlement
    """
    try:
        # Parse ISO timestamp (UTC)
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        dt = datetime.fromisoformat(timestamp_str)

        # Convert to ET (UTC-5 / UTC-4 depending on DST)
        # Simple heuristic: March-November is EDT (UTC-4), else EST (UTC-5)
        month = dt.month
        offset_hours = 4 if 3 <= month <= 11 else 5
        et_hour = dt.hour - offset_hours
        if et_hour < 0:
            et_hour += 24

        if 8 <= et_hour < 10 or (et_hour == 10 and dt.minute < 30):
            return "signal_early"
        elif (et_hour == 10 and dt.minute >= 30) or et_hour == 11:
            return "signal_euro"
        elif 12 <= et_hour < 14:
            return "signal_us_open"
        elif 14 <= et_hour < 16:
            return "signal_peak"
        elif 16 <= et_hour < 18:
            return "signal_settlement"
        else:
            return ""
    except Exception:
        return ""


def backfill_metadata(entries: list, council_index: dict) -> dict:
    """Backfill thesis_strength, primary_catalyst, dissent_acknowledged, schedule_id."""
    stats = {'matched': 0, 'unmatched': 0, 'schedule_inferred': 0}

    for entry in entries:
        position_id = entry.get('position_id', '')
        row = council_index.get(position_id)

        if row:
            stats['matched'] += 1
            # Backfill from CSV if not already present
            if not entry.get('thesis_strength'):
                entry['thesis_strength'] = row.get('thesis_strength', '')
            if not entry.get('primary_catalyst'):
                entry['primary_catalyst'] = row.get('primary_catalyst', '')
            if not entry.get('dissent_acknowledged'):
                entry['dissent_acknowledged'] = row.get('dissent_acknowledged', '')
            if not entry.get('schedule_id') and row.get('schedule_id'):
                entry['schedule_id'] = row.get('schedule_id', '')
        else:
            stats['unmatched'] += 1

        # Infer schedule_id from timestamp for scheduled triggers
        trigger = entry.get('trigger_type', '').lower()
        if not entry.get('schedule_id') and 'scheduled' in trigger:
            inferred = infer_schedule_id(entry.get('timestamp', ''))
            if inferred:
                entry['schedule_id'] = inferred
                stats['schedule_inferred'] += 1

    return stats


def find_truncation_affected(entries: list) -> list:
    """Find entries whose narratives mention truncation/incomplete thesis."""
    affected = []
    for i, entry in enumerate(entries):
        narrative = entry.get('narrative', {})
        if isinstance(narrative, dict):
            wrong = narrative.get('what_went_wrong', [])
            text = json.dumps(wrong).lower() if wrong else ''
        elif isinstance(narrative, str):
            text = narrative.lower()
        else:
            continue

        if 'truncat' in text or 'incomplete' in text:
            affected.append(i)
    return affected


def main():
    parser = argparse.ArgumentParser(description='Backfill trade journal metadata')
    parser.add_argument('--data-dir', required=True, help='Path to data directory (e.g., data/KC)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without writing')
    parser.add_argument('--regen-narratives', action='store_true',
                        help='Re-run LLM narratives for truncation-affected entries (requires API keys)')
    args = parser.parse_args()

    journal_path = os.path.join(args.data_dir, "trade_journal.json")
    if not os.path.exists(journal_path):
        print(f"ERROR: {journal_path} not found")
        sys.exit(1)

    # Load journal
    with open(journal_path, 'r') as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} journal entries from {journal_path}")

    # Load council history
    council_index = load_council_history(args.data_dir)

    # Step 1: Backfill metadata
    print("\n--- Step 1: Backfill metadata ---")
    stats = backfill_metadata(entries, council_index)
    print(f"  Matched: {stats['matched']}, Unmatched: {stats['unmatched']}")
    print(f"  Schedule IDs inferred from timestamp: {stats['schedule_inferred']}")

    # Step 2: Identify truncation-affected entries
    affected_indices = find_truncation_affected(entries)
    print(f"\n--- Step 2: Truncation-affected entries ---")
    print(f"  Found {len(affected_indices)} entries with truncation/incomplete flags")

    if args.regen_narratives and affected_indices:
        print("  Re-generating LLM narratives (requires running environment)...")
        try:
            import asyncio
            from dotenv import load_dotenv
            load_dotenv()
            from config_loader import load_config
            from trading_bot.heterogeneous_router import HeterogeneousRouter

            config = load_config()
            config['data_dir'] = args.data_dir
            router = HeterogeneousRouter(config)

            from trading_bot.trade_journal import TradeJournal
            journal = TradeJournal(config, router=router)

            regen_count = 0
            for idx in affected_indices:
                entry = entries[idx]
                try:
                    narrative = asyncio.run(journal._generate_llm_narrative(entry, {}))
                    entry['narrative'] = narrative
                    entry['key_lesson'] = narrative.get('lesson', 'No lesson extracted')
                    regen_count += 1
                    print(f"    Regenerated narrative for {entry.get('position_id', idx)}")
                    import time
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    print(f"    FAILED for {entry.get('position_id', idx)}: {e}")

            print(f"  Regenerated {regen_count}/{len(affected_indices)} narratives")
        except ImportError as e:
            print(f"  SKIPPED narrative regen (missing dependency: {e})")
            print("  Run with full environment to regenerate narratives.")
    elif affected_indices:
        print("  Use --regen-narratives to re-run LLM narratives for these entries")

    # Step 3: Save
    if args.dry_run:
        print(f"\n--- DRY RUN: Would update {journal_path} ---")
        # Show a sample entry
        if entries:
            sample = entries[0]
            print(f"  Sample entry fields: {list(sample.keys())}")
            print(f"  thesis_strength: {sample.get('thesis_strength', '(empty)')}")
            print(f"  primary_catalyst: {sample.get('primary_catalyst', '(empty)')}")
            print(f"  schedule_id: {sample.get('schedule_id', '(empty)')}")
    else:
        print(f"\n--- Step 3: Saving to {journal_path} ---")
        # Atomic write
        dir_name = os.path.dirname(journal_path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.json')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(entries, f, indent=2, default=str)
            os.replace(tmp_path, journal_path)
            print(f"  Saved {len(entries)} entries")
        except Exception:
            os.unlink(tmp_path)
            raise

    print("\nDone.")


if __name__ == '__main__':
    main()
