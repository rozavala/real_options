"""Brier prediction reconciliation engine.

Extracted from scripts/fix_brier_data.py for runtime use by orchestrator.py.
Contains only the cycle-aware resolution logic needed for automated reconciliation.
"""

import os
import logging

import pandas as pd
import numpy as np

from trading_bot.timestamps import parse_ts_column

logger = logging.getLogger(__name__)

# Paths — set via set_data_dir() from orchestrator init
_data_dir = None


def set_data_dir(data_dir: str):
    """Set commodity-specific data directory for reconciliation paths."""
    global _data_dir
    _data_dir = data_dir


def _get_paths():
    """Return (structured_file, council_file, accuracy_file) for active commodity."""
    base = _data_dir or os.path.join("data", os.environ.get("COMMODITY_TICKER", "KC"))
    return (
        os.path.join(base, "agent_accuracy_structured.csv"),
        os.path.join(base, "council_history.csv"),
        os.path.join(base, "agent_accuracy.csv"),
    )


def resolve_with_cycle_aware_match(dry_run: bool = False):
    """
    Resolve PENDING predictions using cycle-aware matching.

    IMPROVEMENT OVER v4 nearest-match:
    - v4 matched predictions to nearest RECONCILED decision (cross-cycle risk)
    - v5 matches predictions to nearest decision (ANY), then checks if reconciled
    - This prevents cross-cycle contamination of accuracy scores

    Steps:
    1. Match each prediction to its own cycle (nearest council decision, any status)
    2. Check if that cycle's council decision has been reconciled
    3. Only resolve if reconciled; otherwise classify as "awaiting reconciliation"
    """
    STRUCTURED_FILE, COUNCIL_FILE, ACCURACY_FILE = _get_paths()

    if not os.path.exists(STRUCTURED_FILE) or not os.path.exists(COUNCIL_FILE):
        logger.info("Missing required files for resolution")
        return 0

    predictions_df = pd.read_csv(STRUCTURED_FILE)
    council_df = pd.read_csv(COUNCIL_FILE)

    if predictions_df.empty or council_df.empty:
        return 0

    # Parse timestamps (coerce mode: unparseable values become NaT instead of crashing)
    predictions_df['timestamp'] = parse_ts_column(predictions_df['timestamp'], errors='coerce')
    council_df['timestamp'] = parse_ts_column(council_df['timestamp'], errors='coerce')

    # Drop rows with unparseable timestamps (defense-in-depth after sanitization)
    pred_nat_count = predictions_df['timestamp'].isna().sum()
    if pred_nat_count > 0:
        logger.warning(f"Dropping {pred_nat_count} predictions with unparseable timestamps")
        predictions_df = predictions_df.dropna(subset=['timestamp'])

    council_nat_count = council_df['timestamp'].isna().sum()
    if council_nat_count > 0:
        logger.warning(f"Dropping {council_nat_count} council rows with unparseable timestamps")
        council_df = council_df.dropna(subset=['timestamp'])

    # Filter to PENDING
    pending_mask = predictions_df['actual'] == 'PENDING'
    pending_count = pending_mask.sum()

    if pending_count == 0:
        logger.info("No pending predictions to resolve")
        return 0

    logger.info(f"Found {pending_count} pending predictions")

    # === KEY CHANGE: Use ALL council decisions for cycle matching ===
    all_decisions = council_df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Total council decisions: {len(all_decisions)}")

    # Filter corrupt cycles (shutdown race condition)
    corrupt_mask = all_decisions['master_reasoning'].str.contains(
        'cannot schedule new futures after shutdown',
        na=False, case=False
    )
    if corrupt_mask.any():
        corrupt_count = corrupt_mask.sum()
        logger.warning(
            f"Found {corrupt_count} corrupt council decisions (shutdown race condition). "
            f"Marking associated predictions as ORPHANED."
        )

    # Identify reconciled decisions (have actual_trend_direction)
    reconciled_mask = (
        all_decisions['actual_trend_direction'].notna() &
        (all_decisions['actual_trend_direction'] != '') &
        (all_decisions['actual_trend_direction'].astype(str).str.strip() != '')
    )
    reconciled_count = reconciled_mask.sum()
    logger.info(f"Reconciled council decisions: {reconciled_count}/{len(all_decisions)}")

    # Direction normalization
    direction_map = {
        'UP': 'BULLISH', 'DOWN': 'BEARISH', 'BULLISH': 'BULLISH',
        'BEARISH': 'BEARISH', 'NEUTRAL': 'NEUTRAL', 'FLAT': 'NEUTRAL'
    }

    # === PHASE 1: Match each prediction to its own cycle ===
    resolved_count = 0
    awaiting_reconciliation = 0
    orphaned_indices = []
    gap_stats = []

    for idx in predictions_df[pending_mask].index:
        pred_time = predictions_df.loc[idx, 'timestamp']

        # Find nearest council decision (ANY status)
        time_diffs = abs(all_decisions['timestamp'] - pred_time)
        nearest_idx = time_diffs.idxmin()
        nearest_gap = time_diffs[nearest_idx]

        # Safety cap: 2 hours max (prevents matching to totally unrelated cycles)
        if nearest_gap > pd.Timedelta(hours=2):
            orphaned_indices.append(idx)
            continue

        # === PHASE 2: Check if this cycle's decision is reconciled ===
        if not reconciled_mask.iloc[nearest_idx]:
            # This prediction's own cycle hasn't been reconciled yet
            awaiting_reconciliation += 1
            continue

        # === PHASE 3: Resolve with correct cycle's outcome ===
        raw_actual = str(all_decisions.loc[nearest_idx, 'actual_trend_direction']).upper().strip()
        actual = direction_map.get(raw_actual)

        if actual:
            predictions_df.loc[idx, 'actual'] = actual
            resolved_count += 1
            gap_stats.append(nearest_gap.total_seconds() / 60)

    # === ORPHAN HANDLING ===
    if orphaned_indices and not dry_run:
        predictions_df.loc[orphaned_indices, 'actual'] = 'ORPHANED'
        logger.info(f"Classified {len(orphaned_indices)} predictions as ORPHANED (no council decision within 2h)")

    orphaned_count = len(orphaned_indices)

    # === DIAGNOSTICS ===
    total = len(predictions_df)
    orphaned_total = (predictions_df['actual'] == 'ORPHANED').sum()
    if dry_run: orphaned_total += orphaned_count

    still_pending = (predictions_df['actual'] == 'PENDING').sum()
    if dry_run: still_pending -= resolved_count

    resolved_total = total - orphaned_total - still_pending
    resolvable = total - orphaned_total
    effective_rate = (resolved_total / resolvable * 100) if resolvable > 0 else 0

    logger.info(f"{'='*50}")
    logger.info(f"RESOLUTION BREAKDOWN:")
    logger.info(f"  Total predictions:            {total}")
    logger.info(f"  Resolved:                     {resolved_total} (+{resolved_count} new)")
    logger.info(f"  Awaiting reconciliation:      {still_pending} (potential: {awaiting_reconciliation})")
    logger.info(f"  Orphaned (no council):        {orphaned_total} (+{orphaned_count} new)")
    logger.info(f"  Effective resolution rate:     {effective_rate:.0f}% (excl. orphans)")
    logger.info(f"{'='*50}")

    if gap_stats:
        logger.info(f"Match gap stats (minutes): "
                    f"min={min(gap_stats):.1f}, max={max(gap_stats):.1f}, "
                    f"mean={np.mean(gap_stats):.1f}, median={np.median(gap_stats):.1f}")

    # Estimate potential from running reconciliation
    if awaiting_reconciliation > 0:
        logger.info(f"Running reconciliation could unlock "
                    f"up to {awaiting_reconciliation} more predictions")

    # Save changes if any (resolved or orphaned)
    if (resolved_count > 0 or orphaned_count > 0) and not dry_run:
        predictions_df.to_csv(STRUCTURED_FILE, index=False)
        logger.info(f"Saved updated {STRUCTURED_FILE}")

        # Sync to legacy file (only resolved ones, orphans don't go to legacy)
        if resolved_count > 0:
            newly_resolved = predictions_df[
                (predictions_df['actual'] != 'PENDING') &
                (predictions_df['actual'] != 'ORPHANED') &
                predictions_df.index.isin(predictions_df[pending_mask].index)
            ].copy()

            if not newly_resolved.empty:
                newly_resolved['correct'] = (
                    newly_resolved['direction'].str.upper() == newly_resolved['actual'].str.upper()
                ).astype(int)

                with open(ACCURACY_FILE, 'a') as f:
                    for _, row in newly_resolved.iterrows():
                        agent = str(row.get('agent', '')).lower()
                        f.write(f"{row['timestamp']},{agent},{row['direction']},{row['actual']},{row['correct']}\n")

                logger.info(f"Appended {len(newly_resolved)} rows to {ACCURACY_FILE}")

    return resolved_count
