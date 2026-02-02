#!/usr/bin/env python3
"""
One-time migration: Remove deprecated agent entries from Brier scoring CSVs.
Creates backup before modifying.
"""
import pandas as pd
import shutil
from datetime import datetime
import os

DEPRECATED = {'latest_ml_signals', 'ml_model', 'strategy_iron_condor', 'strategy_long_straddle'}

def clean_csv(filepath, agent_col='agent'):
    """Remove rows for deprecated agents, backup first."""
    if not os.path.exists(filepath):
        print(f"Skipped {filepath} (not found)")
        return

    try:
        df = pd.read_csv(filepath)
        original_len = len(df)
        backup = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup)

        cleaned = df[~df[agent_col].isin(DEPRECATED)]
        removed = original_len - len(cleaned)

        cleaned.to_csv(filepath, index=False)
        print(f"{filepath}: Removed {removed}/{original_len} rows. Backup: {backup}")
    except Exception as e:
        print(f"Error cleaning {filepath}: {e}")

if __name__ == "__main__":
    clean_csv("data/agent_accuracy.csv")
    clean_csv("data/agent_accuracy_structured.csv")
    print("Done. Restart orchestrator to pick up clean data.")
