#!/usr/bin/env python3
"""
One-time re-scoring: Fix historical VOLATILITY/STAYED_FLAT Brier resolutions.

Prior to the volatility-aware fix, STAYED_FLAT cycles were resolved using
directional price movement (BULLISH/BEARISH). This penalized agents who
correctly predicted NEUTRAL. This script retroactively corrects those
resolutions to NEUTRAL and recalculates Brier scores.

NOTE: This script rebuilds agent_scores and calibration_buckets in
enhanced_brier.json. It does NOT rebuild weight_evolution.csv — those
weights will gradually adapt over the next 1–3 reconciliation cycles
as new Brier values flow into the voting pipeline.

Usage (must run from project root, e.g. /home/rodrigo/real_options):
    python scripts/rescore_volatility_brier.py --commodity KC
    python scripts/rescore_volatility_brier.py --commodity KC --dry-run
    python scripts/rescore_volatility_brier.py --commodity KC --output-csv
    python scripts/rescore_volatility_brier.py  # all active commodities

Safe to run multiple times — only modifies predictions that need correction.
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trading_bot.enhanced_brier import (
    EnhancedBrierTracker,
    resolve_outcome_for_cycle,
)


def rescore_commodity(ticker: str, dry_run: bool = False, output_csv: bool = False) -> dict:
    """
    Re-score historical VOLATILITY/STAYED_FLAT predictions for one commodity.

    Returns dict with stats: {checked, corrected, skipped, errors}
    """
    data_dir = f"data/{ticker}"
    enhanced_path = os.path.join(data_dir, "enhanced_brier.json")
    council_path = os.path.join(data_dir, "council_history.csv")

    stats = {"checked": 0, "corrected": 0, "skipped": 0, "errors": 0}

    if not os.path.exists(enhanced_path):
        print(f"  [{ticker}] No enhanced_brier.json found — skipping")
        return stats

    if not os.path.exists(council_path):
        print(f"  [{ticker}] No council_history.csv found — skipping")
        return stats

    # Load council_history to build cycle_id → (prediction_type, volatility_outcome) lookup
    try:
        ch_df = pd.read_csv(council_path, on_bad_lines="warn")
    except Exception as e:
        print(f"  [{ticker}] Failed to read council_history.csv: {e}")
        stats["errors"] += 1
        return stats

    cycle_info = {}
    for _, row in ch_df.iterrows():
        cid = str(row.get("cycle_id", "")).strip()
        if not cid:
            continue
        cycle_info[cid] = {
            "prediction_type": str(row.get("prediction_type", "")).strip(),
            "volatility_outcome": str(row.get("volatility_outcome", "")).strip(),
            "actual_trend_direction": str(row.get("actual_trend_direction", "")).strip(),
        }

    vol_stayed_flat_cycles = {
        cid for cid, info in cycle_info.items()
        if info["prediction_type"].upper() == "VOLATILITY"
        and info["volatility_outcome"].upper() == "STAYED_FLAT"
    }

    print(f"  [{ticker}] Found {len(vol_stayed_flat_cycles)} VOLATILITY/STAYED_FLAT cycles in council_history")

    if not vol_stayed_flat_cycles:
        print(f"  [{ticker}] Nothing to re-score")
        return stats

    # Load tracker
    tracker = EnhancedBrierTracker(data_path=enhanced_path)
    print(f"  [{ticker}] Loaded {len(tracker.predictions)} predictions from enhanced_brier.json")

    corrections = []

    for pred in tracker.predictions:
        if pred.cycle_id not in vol_stayed_flat_cycles:
            continue

        stats["checked"] += 1

        # Only fix predictions that were resolved with a directional outcome
        if pred.actual_outcome is None:
            stats["skipped"] += 1
            continue

        if pred.actual_outcome == "NEUTRAL":
            # Already correct — skip
            stats["skipped"] += 1
            continue

        if pred.actual_outcome == "ORPHANED":
            stats["skipped"] += 1
            continue

        # This prediction was resolved as BULLISH or BEARISH during a STAYED_FLAT cycle — fix it
        old_outcome = pred.actual_outcome
        new_outcome = "NEUTRAL"

        # Mutate-restore pattern: set new outcome to compute Brier, restore if dry-run
        pred.actual_outcome = new_outcome
        new_brier = pred.calc_brier_score()
        if dry_run:
            pred.actual_outcome = old_outcome

        corrections.append({
            "agent": pred.agent,
            "cycle_id": pred.cycle_id,
            "old_outcome": old_outcome,
            "new_outcome": new_outcome,
            "new_brier": f"{new_brier:.4f}" if new_brier is not None else "N/A",
            "timestamp": pred.timestamp.isoformat(),
        })
        stats["corrected"] += 1

    # Report
    if corrections:
        print(f"\n  [{ticker}] Corrections ({'DRY RUN' if dry_run else 'LIVE'}):")
        for c in corrections[:20]:  # Show first 20
            print(
                f"    {c['agent']:15s} | cycle={c['cycle_id'][:16]} | "
                f"{c['old_outcome']:>8s} → {c['new_outcome']:>8s} | "
                f"Brier={c['new_brier']}"
            )
        if len(corrections) > 20:
            print(f"    ... and {len(corrections) - 20} more")

    # Write audit trail CSV if requested
    if output_csv and corrections:
        csv_path = os.path.join(
            data_dir,
            f"rescore_corrections_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["agent", "cycle_id", "old_outcome", "new_outcome", "new_brier", "timestamp"])
            writer.writeheader()
            writer.writerows(corrections)
        print(f"  [{ticker}] Audit trail written to {csv_path}")

    if not dry_run and corrections:
        # Rebuild agent_scores from scratch (most accurate approach)
        tracker.agent_scores = {}
        for bucket_list in tracker.calibration_buckets.values():
            for b in bucket_list:
                b.predictions = 0
                b.correct = 0

        resolved_count = 0
        for pred in tracker.predictions:
            if pred.actual_outcome is None or pred.actual_outcome == "ORPHANED":
                continue
            brier = pred.calc_brier_score()
            if brier is not None:
                tracker._update_agent_score(pred.agent, pred.regime.value, brier)
                tracker._update_calibration(pred)
                resolved_count += 1

        tracker._save()
        print(f"\n  [{ticker}] Saved. Rebuilt scores from {resolved_count} resolved predictions.")
        print(f"  [{ticker}] NOTE: weight_evolution.csv is NOT rebuilt by this script.")
        print(f"  [{ticker}]       Weights will adapt over next 1-3 reconciliation cycles.")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Re-score historical VOLATILITY/STAYED_FLAT Brier predictions"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing changes")
    parser.add_argument("--commodity", type=str, help="Process a single commodity (e.g., KC, CC, NG)")
    parser.add_argument("--output-csv", action="store_true", help="Write corrections audit trail to CSV")
    args = parser.parse_args()

    # Determine commodities
    if args.commodity:
        tickers = [args.commodity.upper()]
    else:
        try:
            from config_loader import load_config
            config = load_config()
            tickers = config.get("active_commodities", ["KC"])
        except Exception:
            tickers = ["KC", "CC", "NG"]

    print("=" * 60)
    print(f"Volatility Brier Re-Scoring — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Commodities: {tickers}")
    print("=" * 60)

    total_stats = {"checked": 0, "corrected": 0, "skipped": 0, "errors": 0}

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        stats = rescore_commodity(ticker, dry_run=args.dry_run, output_csv=args.output_csv)
        for k in total_stats:
            total_stats[k] += stats[k]

    print("\n" + "=" * 60)
    print(f"Total: checked={total_stats['checked']}, "
          f"corrected={total_stats['corrected']}, "
          f"skipped={total_stats['skipped']}, "
          f"errors={total_stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
