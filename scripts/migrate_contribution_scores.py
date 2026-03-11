#!/usr/bin/env python3
"""
Migrate historical council_history.csv data into contribution_scores.json.

Usage:
    python scripts/migrate_contribution_scores.py                    # All tickers
    python scripts/migrate_contribution_scores.py --ticker KC        # Single ticker
    python scripts/migrate_contribution_scores.py --ticker KC --dry-run  # Preview only

Reads council_history.csv for each commodity, applies the new contribution
scoring formula to all resolved cycles, and writes contribution_scores.json.
Then compares new multipliers to current Brier multipliers as a validation gate.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from trading_bot.contribution_scorer import ContributionTracker, compress_confidence, _is_vol_strategy_correct
from trading_bot.enhanced_brier import normalize_regime
from trading_bot.agent_names import normalize_agent_name, DEPRECATED_AGENTS
from trading_bot.brier_bridge import get_agent_reliability as get_brier_reliability

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Regimes to compare multipliers across
COMPARE_REGIMES = ["NORMAL", "HIGH_VOL", "RANGE_BOUND"]
# Max acceptable average absolute difference for validation gate
MAX_AVG_DIFF = 0.20


def migrate_ticker(ticker: str, data_dir: str, dry_run: bool = False) -> dict:
    """
    Migrate one commodity's council_history into contribution scores.

    Returns dict with migration stats and multiplier comparison.
    """
    csv_path = os.path.join(data_dir, ticker, "council_history.csv")
    output_path = os.path.join(data_dir, ticker, "contribution_scores.json")

    if not os.path.exists(csv_path):
        logger.warning(f"No council_history.csv for {ticker} at {csv_path}")
        return {"ticker": ticker, "status": "skipped", "reason": "no CSV"}

    df = pd.read_csv(csv_path)
    logger.info(f"{ticker}: Loaded {len(df)} rows from council_history.csv")

    # Filter to resolved rows (have actual_trend_direction)
    resolved_mask = (
        df["actual_trend_direction"].notna() &
        (df["actual_trend_direction"].astype(str).str.strip() != "") &
        (df["actual_trend_direction"].astype(str).str.upper() != "NAN")
    )
    resolved = df[resolved_mask].copy()
    logger.info(f"{ticker}: {len(resolved)} resolved cycles to migrate")

    if resolved.empty:
        return {"ticker": ticker, "status": "skipped", "reason": "no resolved cycles"}

    # Load commodity profile for materiality threshold
    try:
        from config.commodity_profiles import get_commodity_profile
        profile = get_commodity_profile(ticker)
        threshold = profile.neutral_move_threshold_pct
    except Exception:
        threshold = 0.008
    logger.info(f"{ticker}: Using materiality threshold {threshold:.4%}")

    # Create tracker (in-memory, not loading existing file)
    tracker = ContributionTracker.create_empty(output_path)

    scored_cycles = 0
    skipped_cycles = 0

    for _, row in resolved.iterrows():
        # Parse vote_breakdown
        vb_raw = row.get("vote_breakdown", "")
        if not vb_raw or pd.isna(vb_raw) or str(vb_raw).strip() in ("", "[]"):
            skipped_cycles += 1
            continue

        try:
            vote_data = json.loads(vb_raw) if isinstance(vb_raw, str) else vb_raw
        except (json.JSONDecodeError, TypeError):
            skipped_cycles += 1
            continue

        if not vote_data or not isinstance(vote_data, list):
            skipped_cycles += 1
            continue

        # Extract cycle metadata
        master_dir = str(row.get("master_decision", "NEUTRAL")).upper().strip()
        master_conf = float(row.get("master_confidence", 0.5) or 0.5)
        pred_type = str(row.get("prediction_type", "DIRECTIONAL")).upper().strip()
        strat_type = str(row.get("strategy_type", "")).upper().strip()
        vol_outcome = str(row.get("volatility_outcome", "")).upper().strip()
        regime = str(row.get("entry_regime", "NORMAL")).upper().strip()
        cycle_id = str(row.get("cycle_id", ""))
        contract = str(row.get("contract", ""))

        # Re-apply materiality threshold to actual_trend_direction
        raw_trend = str(row.get("actual_trend_direction", "")).upper().strip()
        entry_price = float(row.get("entry_price", 0) or 0)
        exit_price = float(row.get("exit_price", 0) or 0)

        if entry_price > 0 and exit_price > 0:
            pct_change = abs((exit_price - entry_price) / entry_price)
            if pct_change < threshold:
                actual_outcome = "NEUTRAL"
            else:
                actual_outcome = raw_trend if raw_trend in ("BULLISH", "BEARISH") else "NEUTRAL"
        else:
            actual_outcome = raw_trend if raw_trend in ("BULLISH", "BEARISH", "NEUTRAL") else "NEUTRAL"

        # Clean NaN-like strings
        if vol_outcome in ("NAN", "NONE", ""):
            vol_outcome = ""
        if pred_type in ("NAN", "NONE", ""):
            pred_type = "DIRECTIONAL"
        if regime in ("NAN", "NONE", ""):
            regime = "NORMAL"

        canonical_regime = normalize_regime(regime).value

        # Score each agent
        for vote in vote_data:
            agent = vote.get("agent", "")
            if not agent or agent in DEPRECATED_AGENTS:
                continue

            agent = normalize_agent_name(agent)
            direction = vote.get("direction", "NEUTRAL")
            confidence = float(vote.get("confidence", 0.5))
            weight = float(vote.get("final_weight", 1.0))

            score = tracker._compute_score(
                agent_name=agent,
                agent_direction=direction,
                agent_confidence=confidence,
                master_direction=master_dir,
                actual_outcome=actual_outcome,
                prediction_type=pred_type,
                strategy_type=strat_type,
                volatility_outcome=vol_outcome,
                influence_weight=weight,
            )

            # Store
            if agent not in tracker.agent_scores:
                tracker.agent_scores[agent] = {}
            if canonical_regime not in tracker.agent_scores[agent]:
                tracker.agent_scores[agent][canonical_regime] = []
            tracker.agent_scores[agent][canonical_regime].append(score)

        # Score Master as self-assessment (direction vs outcome)
        if master_dir and master_dir != "NEUTRAL":
            master_score = tracker._compute_score(
                agent_name="master_decision",
                agent_direction=master_dir,
                agent_confidence=master_conf,
                master_direction=master_dir,
                actual_outcome=actual_outcome,
                prediction_type=pred_type,
                strategy_type=strat_type,
                volatility_outcome=vol_outcome,
                influence_weight=1.0,
            )
            if "master_decision" not in tracker.agent_scores:
                tracker.agent_scores["master_decision"] = {}
            if canonical_regime not in tracker.agent_scores["master_decision"]:
                tracker.agent_scores["master_decision"][canonical_regime] = []
            tracker.agent_scores["master_decision"][canonical_regime].append(master_score)

        scored_cycles += 1

    logger.info(f"{ticker}: Scored {scored_cycles} cycles, skipped {skipped_cycles}")

    # Trim scores to MAX_SCORES_PER_REGIME
    for agent in tracker.agent_scores:
        for regime in tracker.agent_scores[agent]:
            scores = tracker.agent_scores[agent][regime]
            if len(scores) > 200:
                tracker.agent_scores[agent][regime] = scores[-200:]

    # === VALIDATION: Compare new multipliers to current Brier multipliers ===
    comparison = []
    agents = sorted(tracker.agent_scores.keys())

    # Temporarily set brier_bridge data_dir for this ticker
    from trading_bot.brier_bridge import set_data_dir as set_brier_dir
    set_brier_dir(os.path.join(data_dir, ticker))

    for agent in agents:
        for regime in COMPARE_REGIMES:
            new_mult = tracker.get_agent_reliability(agent, regime)
            old_mult = get_brier_reliability(agent, regime)
            diff = abs(new_mult - old_mult)
            comparison.append({
                "agent": agent,
                "regime": regime,
                "old_brier": round(old_mult, 3),
                "new_contribution": round(new_mult, 3),
                "diff": round(diff, 3),
            })

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  {ticker} Multiplier Comparison (Brier vs Contribution)")
    print(f"{'='*70}")
    fmt = "{:<20} {:<15} {:>10} {:>15} {:>8}"
    print(fmt.format("Agent", "Regime", "Old Brier", "New Contrib", "Diff"))
    print("-" * 70)
    for row in comparison:
        flag = " !!" if row["diff"] > MAX_AVG_DIFF else ""
        print(fmt.format(
            row["agent"], row["regime"],
            f"{row['old_brier']:.3f}", f"{row['new_contribution']:.3f}",
            f"{row['diff']:.3f}{flag}"
        ))

    avg_diff = np.mean([r["diff"] for r in comparison]) if comparison else 0
    max_diff = max([r["diff"] for r in comparison]) if comparison else 0
    print(f"\nAverage diff: {avg_diff:.3f} | Max diff: {max_diff:.3f}")

    if max_diff > MAX_AVG_DIFF:
        print(f"!!  Some agents exceed {MAX_AVG_DIFF} threshold -- review before enabling")
    else:
        print(f"OK  All within {MAX_AVG_DIFF} threshold -- safe to enable")

    # Save
    if not dry_run:
        tracker._save()
        logger.info(f"{ticker}: Wrote {output_path}")
    else:
        logger.info(f"{ticker}: DRY RUN -- would write {output_path}")

    return {
        "ticker": ticker,
        "status": "migrated",
        "scored_cycles": scored_cycles,
        "skipped_cycles": skipped_cycles,
        "agents": len(agents),
        "avg_diff": round(avg_diff, 3),
        "max_diff": round(max_diff, 3),
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Migrate to contribution scoring")
    parser.add_argument("--ticker", type=str, help="Single ticker (default: all)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    data_dir = args.data_dir
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        # Auto-detect tickers from data directories
        tickers = sorted([
            d.upper() for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and os.path.exists(os.path.join(data_dir, d, "council_history.csv"))
        ])

    if not tickers:
        print("No tickers found with council_history.csv")
        sys.exit(1)

    print(f"Migrating: {', '.join(tickers)}")
    results = []
    for ticker in tickers:
        result = migrate_ticker(ticker, data_dir, dry_run=args.dry_run)
        results.append(result)

    # Summary
    print(f"\n{'='*50}")
    print("  Migration Summary")
    print(f"{'='*50}")
    for r in results:
        status = r.get("status", "unknown")
        if status == "migrated":
            print(f"  {r['ticker']}: {r['scored_cycles']} cycles scored, "
                  f"avg_diff={r['avg_diff']:.3f}, max_diff={r['max_diff']:.3f}")
        else:
            print(f"  {r['ticker']}: {status} ({r.get('reason', '')})")


if __name__ == "__main__":
    main()
