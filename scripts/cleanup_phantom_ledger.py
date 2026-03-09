#!/usr/bin/env python3
"""
One-time cleanup: Remove PHANTOM_RECONCILIATION entries from archived trade ledgers.

These synthetic entries were created by _reconcile_phantom_ledger_entries() and
cause double-counting when combined with RECONCILIATION_MISSING entries that
capture the same close trades from Flex Query.

Also removes specific RECON_MISSING entries that are provably duplicates or
orphaned closes for positions already at zero.

Usage:
    python scripts/cleanup_phantom_ledger.py --commodity KC --dry-run
    python scripts/cleanup_phantom_ledger.py --commodity KC

Safe to run multiple times — only removes entries matching specific patterns.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cleanup_commodity(ticker: str, data_dir: str, dry_run: bool) -> dict:
    """Remove phantom entries from archived ledgers for one commodity."""
    archive_dir = os.path.join(data_dir, ticker, "archive_ledger")
    stats = {"files_modified": 0, "phantom_removed": 0, "recon_dupes_removed": 0}

    if not os.path.isdir(archive_dir):
        logger.info(f"[{ticker}] No archive directory — skipping")
        return stats

    # --- Pass 1: Remove PHANTOM_RECONCILIATION entries ---
    archive_files = sorted(
        f for f in os.listdir(archive_dir)
        if f.startswith("trade_ledger_") and f.endswith(".csv")
        and f != "trade_ledger_missing_trades.csv"
    )

    for filename in archive_files:
        filepath = os.path.join(archive_dir, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.warning(f"[{ticker}] Failed to read {filename}: {e}")
            continue

        if "reason" not in df.columns:
            continue

        reason_str = df["reason"].fillna("")
        phantom_mask = reason_str.str.contains("PHANTOM_RECONCILIATION", case=False)
        n_phantom = phantom_mask.sum()

        if n_phantom == 0:
            continue

        logger.info(f"[{ticker}] {filename}: {n_phantom} PHANTOM entries")
        stats["phantom_removed"] += n_phantom

        if not dry_run:
            clean_df = df[~phantom_mask]
            if clean_df.empty:
                # File would be empty — remove it entirely
                os.remove(filepath)
                logger.info(f"  Removed empty file {filename}")
            else:
                clean_df.to_csv(filepath, index=False, float_format="%.2f")
                logger.info(f"  Wrote {len(clean_df)} remaining entries to {filename}")
            stats["files_modified"] += 1

    # --- Pass 2: Remove duplicate/orphaned RECON_MISSING entries ---
    # Load full consolidated ledger to identify duplicates
    all_dfs = []
    for filename in archive_files:
        filepath = os.path.join(archive_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df["_source"] = filename
                all_dfs.append(df)
            except Exception:
                pass

    missing_path = os.path.join(archive_dir, "trade_ledger_missing_trades.csv")
    if os.path.exists(missing_path):
        try:
            df = pd.read_csv(missing_path)
            df["_source"] = "trade_ledger_missing_trades.csv"
            all_dfs.append(df)
        except Exception:
            pass

    if not all_dfs:
        return stats

    full = pd.concat(all_dfs, ignore_index=True)
    full["quantity"] = pd.to_numeric(full["quantity"], errors="coerce").fillna(0)
    full["signed"] = np.where(full["action"] == "BUY", full["quantity"], -full["quantity"])
    reason_col = full["reason"].fillna("")

    # --- Pass 2b: Remove "Ledger reconciliation: phantom RECONCILIATION_MISSING"
    # entries — these are counter-entries created to cancel RECON_MISSING entries
    # that were later deemed invalid. With PHANTOM cleanup, they become orphaned.
    phantom_recon_mask = reason_col.str.contains(
        "phantom RECONCILIATION_MISSING", case=False
    )
    phantom_recon_to_remove = set(full[phantom_recon_mask].index)
    if phantom_recon_to_remove:
        for idx in phantom_recon_to_remove:
            row = full.loc[idx]
            logger.info(
                f"  Orphaned counter-entry: {row['timestamp']} {row['action']} "
                f"{row['local_symbol']} ({row['reason'][:55]})"
            )

    # --- Pass 2c: Remove duplicate RECON_MISSING entries (same pid+symbol+action
    # as a base entry — proves the trade was already recorded)
    base_mask = ~reason_col.str.contains(
        "RECONCILIATION_MISSING|PHANTOM_RECONCILIATION", case=False
    )
    base = full[base_mask]
    recon = full[reason_col == "RECONCILIATION_MISSING"]

    dupes_to_remove = set()
    if not base.empty and not recon.empty:
        for idx, rrow in recon.iterrows():
            sym = rrow.get("local_symbol", "")
            pid = str(rrow.get("position_id", ""))
            action = rrow.get("action", "")

            # Check if a base entry exists with same pid, symbol, action
            matches = base[
                (base["local_symbol"] == sym)
                & (base["position_id"].astype(str) == pid)
                & (base["action"] == action)
            ]
            if not matches.empty:
                dupes_to_remove.add(idx)
                logger.info(
                    f"  Duplicate RECON_MISSING: {rrow['timestamp']} {action} {sym} "
                    f"pid={pid[:35]} (base has same trade)"
                )

    # --- Pass 2d: Remove paired RECON_MISSING entries ---
    # When one leg of a RECON_MISSING pair (same pid, opposite action) is flagged
    # as a duplicate, the other leg becomes orphaned and must also be removed.
    # Example: BUY+SELL RECON_MISSING with same pid — if SELL matches base
    # EMERGENCY_HARD_CLOSE, the BUY is also a duplicate of the original trade.
    paired_to_remove = set()
    if dupes_to_remove and not recon.empty:
        flagged_pids = {
            str(full.loc[idx, "position_id"])
            for idx in dupes_to_remove
            if idx in full.index
        }
        for idx, rrow in recon.iterrows():
            if idx in dupes_to_remove:
                continue
            pid = str(rrow.get("position_id", ""))
            if pid in flagged_pids:
                paired_to_remove.add(idx)
                logger.info(
                    f"  Paired RECON_MISSING: {rrow['timestamp']} {rrow['action']} "
                    f"{rrow['local_symbol']} pid={pid[:35]} (counterpart is duplicate)"
                )

    all_recon_to_remove = phantom_recon_to_remove | dupes_to_remove | paired_to_remove
    stats["recon_dupes_removed"] = len(all_recon_to_remove)

    if all_recon_to_remove and not dry_run:
        # Group removals by source file
        entries_to_remove = full.loc[list(all_recon_to_remove)]
        for src_file, group in entries_to_remove.groupby("_source"):
            filepath = os.path.join(archive_dir, src_file)
            if not os.path.exists(filepath):
                continue

            src_df = pd.read_csv(filepath)
            orig_len = len(src_df)

            # Match by timestamp + local_symbol + action + position_id to remove
            for _, row in group.iterrows():
                match_mask = (
                    (src_df["timestamp"].astype(str) == str(row["timestamp"]))
                    & (src_df["local_symbol"] == row["local_symbol"])
                    & (src_df["action"] == row["action"])
                    & (src_df["position_id"].astype(str) == str(row["position_id"]))
                    & (src_df["reason"].fillna("") == "RECONCILIATION_MISSING")
                )
                src_df = src_df[~match_mask]

            if len(src_df) < orig_len:
                if src_df.empty:
                    os.remove(filepath)
                    logger.info(f"  Removed empty file {src_file}")
                else:
                    src_df.to_csv(filepath, index=False, float_format="%.2f")
                    logger.info(
                        f"  Updated {src_file}: {orig_len} → {len(src_df)} entries"
                    )
                stats["files_modified"] += 1

    return stats


def verify_result(ticker: str, data_dir: str):
    """Verify the ledger state after cleanup."""
    archive_dir = os.path.join(data_dir, ticker, "archive_ledger")
    dfs = []
    for f in sorted(os.listdir(archive_dir)):
        if f.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(archive_dir, f))
                dfs.append(df)
            except Exception:
                pass

    main = os.path.join(data_dir, ticker, "trade_ledger.csv")
    if os.path.exists(main):
        try:
            dfs.append(pd.read_csv(main))
        except Exception:
            pass

    if not dfs:
        print(f"  [{ticker}] No ledger data to verify")
        return

    full = pd.concat(dfs, ignore_index=True)
    full["quantity"] = pd.to_numeric(full["quantity"], errors="coerce").fillna(0)
    full["signed"] = np.where(full["action"] == "BUY", full["quantity"], -full["quantity"])

    net = full.groupby("local_symbol")["signed"].sum()
    non_zero = net[net.abs() > 0.001]

    reason_str = full["reason"].fillna("")
    n_phantom = reason_str.str.contains("PHANTOM_RECONCILIATION", case=False).sum()
    n_recon = (reason_str == "RECONCILIATION_MISSING").sum()

    print(f"\n  [{ticker}] POST-CLEANUP VERIFICATION:")
    print(f"  Total entries: {len(full)}")
    print(f"  PHANTOM entries remaining: {n_phantom}")
    print(f"  RECON_MISSING entries remaining: {n_recon}")
    print(f"  Non-zero symbols: {len(non_zero)}")
    for sym, qty in sorted(non_zero.items()):
        print(f"    {sym}: {qty:+.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove PHANTOM_RECONCILIATION entries from archived trade ledgers"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--commodity", type=str, help="Single commodity (e.g., KC)")
    parser.add_argument(
        "--data-dir", type=str, default=str(PROJECT_ROOT / "data"),
        help="Path to data directory"
    )
    args = parser.parse_args()

    tickers = [args.commodity.upper()] if args.commodity else ["KC", "CC", "NG"]

    print("=" * 60)
    print(f"Phantom Ledger Cleanup — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Commodities: {tickers}")
    print("=" * 60)

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        stats = cleanup_commodity(ticker, args.data_dir, args.dry_run)
        print(
            f"  [{ticker}] phantom_removed={stats['phantom_removed']}, "
            f"recon_dupes_removed={stats['recon_dupes_removed']}, "
            f"files_modified={stats['files_modified']}"
        )
        if not args.dry_run:
            verify_result(ticker, args.data_dir)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN complete. Re-run without --dry-run to apply changes.")
    else:
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
