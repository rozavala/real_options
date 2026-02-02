"""
One-time repair script for agent_accuracy_structured.csv column misalignment.
"""
import pandas as pd
import shutil
import os
from datetime import datetime

STRUCTURED_FILE = "data/agent_accuracy_structured.csv"
BACKUP_FILE = f"data/agent_accuracy_structured_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def repair():
    if not os.path.exists(STRUCTURED_FILE):
        print(f"File not found: {STRUCTURED_FILE}")
        return

    shutil.copy2(STRUCTURED_FILE, BACKUP_FILE)
    print(f"Backed up to {BACKUP_FILE}")

    df = pd.read_csv(STRUCTURED_FILE)
    print(f"Total rows: {len(df)}")

    if df['timestamp'].dtype == 'object':
        # Check for values that look like cycle_ids (e.g. KC-c0f3b12c) in the timestamp column
        # Standard timestamp format: YYYY-MM-DD HH:MM:SS or ISO8601
        # Bad format: [A-Z]{2,4}-[a-f0-9]+
        bad_mask = df['timestamp'].str.match(r'^[A-Z]{2,4}-[a-f0-9]+$', na=False)
        bad_count = bad_mask.sum()
        print(f"Misaligned rows detected: {bad_count}")

        if bad_count > 0:
            df_clean = df[~bad_mask].copy()
            df_clean.to_csv(STRUCTURED_FILE, index=False)
            print(f"Cleaned file written. {len(df_clean)} rows remaining.")
        else:
            print("No obvious misalignment detected based on regex.")
    else:
        print("Timestamp column is not object type, assuming valid.")

if __name__ == "__main__":
    repair()
