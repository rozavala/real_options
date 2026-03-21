"""
Migration: Fix pnl_realized=0.0 for NEUTRAL non-trade rows in council_history.csv.

ROOT CAUSE:
  NEUTRAL+DIRECTIONAL decisions ("Cash is a Position") have pnl_realized=0.0
  instead of NaN. The 0.0 passes DataFrame.notna() checks in the dashboard,
  inflating the resolved-trade denominator and suppressing the win rate.

  KC example:
    - Before fix: 166 wins / 431 resolved = 38.5%
    - After fix:  166 wins / 302 resolved = 55.0%

CONDITION:
  A row is a non-trade if:
    master_decision = NEUTRAL
    AND strategy_type in (NONE, EMERGENCY, '', N/A, NaN)
    AND pnl_realized = 0.0

  NEUTRAL+VOLATILITY rows (straddles/condors) are intentionally preserved.

SAFE TO RUN MULTIPLE TIMES (idempotent).
"""

import os
import sys
import shutil
import pandas as pd
from datetime import datetime

def fix_neutral_nontrade_pnl(data_dir: str, dry_run: bool = False) -> dict:
    """Set pnl_realized=NaN for NEUTRAL non-trade rows in council_history.csv.

    Returns: dict with keys 'fixed', 'skipped', 'error'.
    """
    ch_path = os.path.join(data_dir, 'council_history.csv')
    if not os.path.exists(ch_path):
        print(f"  SKIP: {ch_path} not found")
        return {'fixed': 0, 'skipped': 0, 'error': 'file not found'}

    df = pd.read_csv(ch_path, low_memory=False)
    original_len = len(df)

    pnl = pd.to_numeric(df['pnl_realized'], errors='coerce')

    NON_TRADE_STRATEGIES = {'NONE', 'EMERGENCY', '', 'N/A'}

    mask = (
        (df['master_decision'].fillna('').str.upper() == 'NEUTRAL') &
        (df['strategy_type'].fillna('').isin(NON_TRADE_STRATEGIES)) &
        (pnl == 0.0)
    )

    n_fixed = int(mask.sum())
    if n_fixed == 0:
        print(f"  OK: {data_dir} — no rows need fixing")
        return {'fixed': 0, 'skipped': original_len, 'error': None}

    print(f"  Found {n_fixed} NEUTRAL non-trade rows with pnl_realized=0.0 in {data_dir}")

    if dry_run:
        print(f"  DRY RUN — would set {n_fixed} rows to NaN")
        return {'fixed': n_fixed, 'skipped': original_len - n_fixed, 'error': None}

    # Backup before modifying
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = ch_path + f'.bak_{ts}'
    shutil.copy2(ch_path, backup_path)
    print(f"  Backup: {backup_path}")

    df.loc[mask, 'pnl_realized'] = float('nan')
    df.to_csv(ch_path, index=False)

    # Verify
    pnl_after = pd.to_numeric(df['pnl_realized'], errors='coerce')
    n_wins = int((pnl_after > 0).sum())
    n_resolved = int(pnl_after.notna().sum())
    win_rate = n_wins / max(n_resolved, 1) * 100
    print(f"  Fixed {n_fixed} rows. New win rate: {n_wins}/{n_resolved} = {win_rate:.1f}%")

    return {'fixed': n_fixed, 'skipped': original_len - n_fixed, 'error': None}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix pnl_realized=0.0 for NEUTRAL non-trade rows')
    parser.add_argument('data_dirs', nargs='+', help='Data directories (e.g. data/KC data/CC data/NG)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without modifying files')
    args = parser.parse_args()

    dry = args.dry_run
    if dry:
        print("DRY RUN MODE — no files will be modified\n")

    total_fixed = 0
    for d in args.data_dirs:
        print(f"Processing: {d}")
        result = fix_neutral_nontrade_pnl(d, dry_run=dry)
        total_fixed += result['fixed']

    print(f"\nTotal rows fixed: {total_fixed}")
