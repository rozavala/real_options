#!/usr/bin/env python3
"""
One-time migration: Clear stale Polymarket state after v6.3 topic update.
Prevents the sentinel from carrying over slug/price data from old irrelevant markets.
"""

import json
import os
import shutil
from datetime import datetime

STATE_FILE = "data/state.json"
BACKUP_SUFFIX = f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def main():
    if not os.path.exists(STATE_FILE):
        print(f"No state file found at {STATE_FILE}. Nothing to clear.")
        return

    # Backup
    backup_path = STATE_FILE + BACKUP_SUFFIX
    shutil.copy2(STATE_FILE, backup_path)
    print(f"Backed up state to: {backup_path}")

    # Load state
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    # Clear prediction market namespace
    pm_key = "prediction_market_state"
    if pm_key in state:
        old_data = state[pm_key]
        print(f"Clearing {len(old_data)} prediction market entries:")
        for topic, data in old_data.items():
            if isinstance(data, dict) and 'data' in data:
                slug = data['data'].get('slug', 'unknown')
            elif isinstance(data, dict):
                slug = data.get('slug', 'unknown')
            else:
                slug = 'unknown'
            print(f"  - {topic}: {slug}")

        state[pm_key] = {}
        print(f"Cleared prediction_market_state namespace.")
    else:
        print(f"No '{pm_key}' namespace found in state. Nothing to clear.")

    # Save
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"State file updated. Sentinel will re-discover markets on next cycle.")

if __name__ == "__main__":
    main()
