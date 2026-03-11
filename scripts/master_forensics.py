#!/usr/bin/env python3
"""
Master Strategist Forensic Analysis.

Dissects every resolved council decision to understand WHERE and WHY
the Master adds or destroys value. Produces a structured report across
7 diagnostic dimensions.

Usage:
    python scripts/master_forensics.py                    # All tickers
    python scripts/master_forensics.py --ticker KC        # Single ticker
    python scripts/master_forensics.py --ticker KC --csv  # Also write CSV
    python scripts/master_forensics.py --json             # Export JSON results

Requires: council_history.csv with resolved cycles (actual_trend_direction populated).
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING & PREP
# ============================================================================

def prepare_forensic_df(df: pd.DataFrame, ticker: str = "KC") -> pd.DataFrame:
    """Apply forensic data prep to a council history DataFrame.

    Normalizes fields, applies materiality threshold, derives vote direction,
    classifies master correctness & overrides. Can be used on an already-loaded
    DataFrame (e.g. from the dashboard with date filtering applied).
    """
    if "actual_trend_direction" not in df.columns:
        return pd.DataFrame()

    df = df.copy()

    # Filter to resolved cycles
    df["actual_trend_direction"] = df["actual_trend_direction"].astype(str).str.strip().str.upper()
    resolved_mask = df["actual_trend_direction"].isin(["BULLISH", "BEARISH", "NEUTRAL"])
    df = df[resolved_mask].copy()

    if df.empty:
        return df

    # Normalize key fields (defensive: columns may not exist in all DataFrames)
    df["master_decision"] = df["master_decision"].astype(str).str.strip().str.upper()

    _defaults = {
        "prediction_type": "DIRECTIONAL",
        "trigger_type": "SCHEDULED",
        "thesis_strength": "UNKNOWN",
        "entry_regime": "NORMAL",
        "strategy_type": "",
        "volatility_outcome": "",
    }
    for col, default in _defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(str).str.strip().str.upper()

    # Clean NaN-like string values in entry_regime
    df.loc[df["entry_regime"].isin(["NAN", "NONE", ""]), "entry_regime"] = "NORMAL"

    # Parse numeric fields
    df["master_confidence"] = pd.to_numeric(df.get("master_confidence"), errors="coerce").fillna(0.5)
    df["weighted_score"] = pd.to_numeric(df.get("weighted_score"), errors="coerce").fillna(0.0)
    df["conviction_multiplier"] = pd.to_numeric(df.get("conviction_multiplier"), errors="coerce").fillna(1.0)

    # Apply materiality threshold to actual_trend_direction
    try:
        from config.commodity_profiles import get_commodity_profile
        profile = get_commodity_profile(ticker)
        threshold = profile.neutral_move_threshold_pct
    except Exception:
        threshold = 0.008

    df["entry_price"] = pd.to_numeric(df.get("entry_price"), errors="coerce")
    df["exit_price"] = pd.to_numeric(df.get("exit_price"), errors="coerce")

    valid_prices = (df["entry_price"] > 0) & (df["exit_price"] > 0)
    if valid_prices.any():
        pct_move = ((df["exit_price"] - df["entry_price"]) / df["entry_price"]).abs()
        sub_threshold = valid_prices & (pct_move < threshold)
        df.loc[sub_threshold, "actual_trend_direction"] = "NEUTRAL"

    # Derive vote direction from weighted_score
    df["vote_direction"] = "NEUTRAL"
    df.loc[df["weighted_score"] > 0.10, "vote_direction"] = "BULLISH"
    df.loc[df["weighted_score"] < -0.10, "vote_direction"] = "BEARISH"

    # Classify Master correctness
    # For DIRECTIONAL: Master direction matches outcome
    # For VOLATILITY: strategy outcome determines correctness
    df["master_correct"] = False

    dir_mask = df["prediction_type"] == "DIRECTIONAL"
    non_neutral_master = df["master_decision"].isin(["BULLISH", "BEARISH"])
    df.loc[dir_mask & non_neutral_master, "master_correct"] = (
        df.loc[dir_mask & non_neutral_master, "master_decision"]
        == df.loc[dir_mask & non_neutral_master, "actual_trend_direction"]
    )

    vol_mask = df["prediction_type"] == "VOLATILITY"
    if vol_mask.any():
        vol_correct = (
            ((df["strategy_type"] == "IRON_CONDOR") & (df["volatility_outcome"] == "STAYED_FLAT")) |
            ((df["strategy_type"] == "LONG_STRADDLE") & (df["volatility_outcome"] == "BIG_MOVE"))
        )
        df.loc[vol_mask, "master_correct"] = vol_correct[vol_mask]

    # Classify override vs aligned
    # For DIRECTIONAL: Master differs from vote direction
    # For VOLATILITY: Master NEUTRAL (vol trade) while vote was directional counts as override
    df["is_override"] = False
    df.loc[dir_mask, "is_override"] = (
        df.loc[dir_mask, "master_decision"] != df.loc[dir_mask, "vote_direction"]
    )
    if vol_mask.any():
        df.loc[vol_mask, "is_override"] = df.loc[vol_mask, "vote_direction"].isin(["BULLISH", "BEARISH"])

    # Parse dominant_agent
    df["dominant_agent"] = df.get("dominant_agent", pd.Series(dtype=str)).fillna("").astype(str).str.strip()

    # Parse vote_breakdown for per-agent analysis
    df["vote_data"] = df.get("vote_breakdown", pd.Series(dtype=str)).apply(_safe_parse_json)

    return df


def load_resolved_cycles(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    """Load council_history.csv and filter to resolved, scoreable cycles."""
    csv_path = os.path.join(data_dir, ticker, "council_history.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"No council_history.csv for {ticker}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    result = prepare_forensic_df(df, ticker)

    if not result.empty:
        try:
            from config.commodity_profiles import get_commodity_profile
            profile = get_commodity_profile(ticker)
            threshold = profile.neutral_move_threshold_pct
        except Exception:
            threshold = 0.008
        logger.info(f"{ticker}: {len(result)} resolved cycles loaded (threshold: {threshold:.4%})")

    return result


def _safe_parse_json(raw):
    """Safely parse JSON vote_breakdown."""
    if not raw or pd.isna(raw) or str(raw).strip() in ("", "[]"):
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


# ============================================================================
# DIMENSION 1: Override Rate vs Outcome (The 2x2 Quadrant)
# ============================================================================

def analyze_override_quadrants(df: pd.DataFrame) -> dict:
    """Classify every decision into Aligned/Override x Correct/Wrong."""
    # Filter to scoreable (non-NEUTRAL master on directional, or vol trades)
    scoreable = df[
        (df["master_decision"].isin(["BULLISH", "BEARISH"])) |
        (df["prediction_type"] == "VOLATILITY")
    ].copy()

    if scoreable.empty:
        return {"total": 0}

    aligned_correct = scoreable[~scoreable["is_override"] & scoreable["master_correct"]]
    aligned_wrong = scoreable[~scoreable["is_override"] & ~scoreable["master_correct"]]
    override_correct = scoreable[scoreable["is_override"] & scoreable["master_correct"]]
    override_wrong = scoreable[scoreable["is_override"] & ~scoreable["master_correct"]]

    total = len(scoreable)
    total_overrides = len(override_correct) + len(override_wrong)
    total_aligned = len(aligned_correct) + len(aligned_wrong)

    return {
        "total": total,
        "aligned_correct": len(aligned_correct),
        "aligned_wrong": len(aligned_wrong),
        "override_correct": len(override_correct),
        "override_wrong": len(override_wrong),
        "aligned_win_rate": _safe_pct(len(aligned_correct), total_aligned),
        "override_win_rate": _safe_pct(len(override_correct), total_overrides),
        "override_rate": _safe_pct(total_overrides, total),
        "override_value": _safe_pct(len(override_correct), total_overrides),
        # Key diagnostic: is the Master better or worse when it disagrees?
        "override_vs_aligned_delta": (
            _safe_pct(len(override_correct), total_overrides) -
            _safe_pct(len(aligned_correct), total_aligned)
        ) if total_overrides > 0 and total_aligned > 0 else None,
    }


# ============================================================================
# DIMENSION 2: Conviction Multiplier vs Outcome
# ============================================================================

def analyze_conviction_vs_outcome(df: pd.DataFrame) -> dict:
    """Win rate by conviction multiplier level."""
    scoreable = df[
        (df["master_decision"].isin(["BULLISH", "BEARISH"])) |
        (df["prediction_type"] == "VOLATILITY")
    ].copy()

    if scoreable.empty:
        return {}

    bins = {
        "full_alignment (1.0)": scoreable["conviction_multiplier"] == 1.0,
        "partial (0.71-0.99)": (scoreable["conviction_multiplier"] > 0.70) & (scoreable["conviction_multiplier"] < 1.0),
        "divergent (<= 0.70)": scoreable["conviction_multiplier"] <= 0.70,
    }

    result = {}
    for label, mask in bins.items():
        subset = scoreable[mask]
        if len(subset) > 0:
            result[label] = {
                "count": len(subset),
                "wins": int(subset["master_correct"].sum()),
                "win_rate": _safe_pct(int(subset["master_correct"].sum()), len(subset)),
            }
    return result


# ============================================================================
# DIMENSION 3: Weighted Score Strength vs Outcome
# ============================================================================

def analyze_consensus_strength(df: pd.DataFrame) -> dict:
    """Win rate by consensus strength band, with override breakdown."""
    scoreable = df[
        (df["master_decision"].isin(["BULLISH", "BEARISH"])) |
        (df["prediction_type"] == "VOLATILITY")
    ].copy()

    if scoreable.empty:
        return {}

    abs_ws = scoreable["weighted_score"].abs()

    bands = {
        "strong (|ws| > 0.30)": abs_ws > 0.30,
        "moderate (0.10 < |ws| < 0.30)": (abs_ws > 0.10) & (abs_ws <= 0.30),
        "deadlock (|ws| < 0.10)": abs_ws <= 0.10,
    }

    result = {}
    for label, mask in bands.items():
        subset = scoreable[mask]
        if len(subset) == 0:
            continue

        overrides = subset[subset["is_override"]]
        aligned = subset[~subset["is_override"]]

        result[label] = {
            "total": len(subset),
            "overall_win_rate": _safe_pct(int(subset["master_correct"].sum()), len(subset)),
            "override_count": len(overrides),
            "override_win_rate": _safe_pct(
                int(overrides["master_correct"].sum()), len(overrides)
            ) if len(overrides) > 0 else None,
            "aligned_count": len(aligned),
            "aligned_win_rate": _safe_pct(
                int(aligned["master_correct"].sum()), len(aligned)
            ) if len(aligned) > 0 else None,
            "follow_rate": _safe_pct(len(aligned), len(subset)),
        }

    return result


# ============================================================================
# DIMENSION 4: Regime-Specific Breakdown
# ============================================================================

def analyze_by_regime(df: pd.DataFrame) -> dict:
    """Override rate and win rate per regime."""
    scoreable = df[
        (df["master_decision"].isin(["BULLISH", "BEARISH"])) |
        (df["prediction_type"] == "VOLATILITY")
    ].copy()

    if scoreable.empty:
        return {}

    result = {}
    for regime, group in scoreable.groupby("entry_regime"):
        if len(group) < 3:
            continue  # Skip regimes with too few samples

        overrides = group[group["is_override"]]

        result[regime] = {
            "total": len(group),
            "win_rate": _safe_pct(int(group["master_correct"].sum()), len(group)),
            "override_rate": _safe_pct(len(overrides), len(group)),
            "override_win_rate": _safe_pct(
                int(overrides["master_correct"].sum()), len(overrides)
            ) if len(overrides) > 0 else None,
            "aligned_win_rate": _safe_pct(
                int(group[~group["is_override"]]["master_correct"].sum()),
                len(group[~group["is_override"]])
            ) if len(group[~group["is_override"]]) > 0 else None,
            # Avg weighted_score magnitude (how much consensus exists)
            "avg_consensus_strength": round(group["weighted_score"].abs().mean(), 3),
        }

    return result


# ============================================================================
# DIMENSION 5: Thesis Strength Calibration
# ============================================================================

def analyze_thesis_calibration(df: pd.DataFrame) -> dict:
    """Win rate by thesis_strength -- is the Master's self-assessment calibrated?"""
    scoreable = df[
        (df["master_decision"].isin(["BULLISH", "BEARISH"])) |
        (df["prediction_type"] == "VOLATILITY")
    ].copy()

    if scoreable.empty:
        return {}

    result = {}
    for thesis, group in scoreable.groupby("thesis_strength"):
        if thesis in ("", "UNKNOWN", "NAN", "NONE"):
            continue
        result[thesis] = {
            "count": len(group),
            "wins": int(group["master_correct"].sum()),
            "win_rate": _safe_pct(int(group["master_correct"].sum()), len(group)),
            "avg_confidence": round(group["master_confidence"].mean(), 3),
            "avg_conviction": round(group["conviction_multiplier"].mean(), 3),
        }

    # Calibration check: is PROVEN > PLAUSIBLE > SPECULATIVE?
    ordered = ["SPECULATIVE", "PLAUSIBLE", "PROVEN"]
    rates = [result.get(t, {}).get("win_rate", 0) for t in ordered]
    is_monotonic = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    result["_calibration_monotonic"] = is_monotonic
    result["_calibration_note"] = (
        "CALIBRATED: PROVEN > PLAUSIBLE > SPECULATIVE" if is_monotonic
        else "MISCALIBRATED: Thesis strength does not predict outcome"
    )

    return result


# ============================================================================
# DIMENSION 6: Trigger Type Breakdown
# ============================================================================

def analyze_by_trigger(df: pd.DataFrame) -> dict:
    """Override rate and win rate by trigger type (scheduled vs emergency)."""
    scoreable = df[
        (df["master_decision"].isin(["BULLISH", "BEARISH"])) |
        (df["prediction_type"] == "VOLATILITY")
    ].copy()

    if scoreable.empty:
        return {}

    # Normalize trigger types into broader categories
    def classify_trigger(t):
        t = str(t).upper().strip()
        if t in ("SCHEDULED", "MANUAL"):
            return "SCHEDULED"
        elif t in ("", "NAN", "NONE", "UNKNOWN"):
            return "SCHEDULED"  # Legacy rows without trigger_type
        else:
            return "EMERGENCY"

    scoreable["trigger_category"] = scoreable["trigger_type"].apply(classify_trigger)

    result = {}
    for trigger, group in scoreable.groupby("trigger_category"):
        overrides = group[group["is_override"]]
        result[trigger] = {
            "total": len(group),
            "win_rate": _safe_pct(int(group["master_correct"].sum()), len(group)),
            "override_rate": _safe_pct(len(overrides), len(group)),
            "override_win_rate": _safe_pct(
                int(overrides["master_correct"].sum()), len(overrides)
            ) if len(overrides) > 0 else None,
            "avg_confidence": round(group["master_confidence"].mean(), 3),
        }

    return result


# ============================================================================
# DIMENSION 7: Dominant Agent Override Analysis
# ============================================================================

def analyze_dominant_agent_overrides(df: pd.DataFrame) -> dict:
    """When Master overrides, which dominant agent is it overriding? Win rate?"""
    overrides = df[df["is_override"] & df["dominant_agent"].str.len() > 0].copy()

    if overrides.empty:
        return {}

    result = {}
    for agent, group in overrides.groupby("dominant_agent"):
        override_wr = _safe_pct(int(group["master_correct"].sum()), len(group))

        # What would have happened if Master followed the vote?
        # vote_direction == actual_trend_direction means vote was correct
        vote_correct = int((group["vote_direction"] == group["actual_trend_direction"]).sum())
        vote_wr = _safe_pct(vote_correct, len(group))

        result[agent] = {
            "override_count": len(group),
            "override_win_rate": override_wr,
            "vote_would_have_won": vote_wr,
            "master_override_net_value": override_wr - vote_wr,
        }

    return result


# ============================================================================
# SUPPLEMENTARY: Prediction Type Breakdown
# ============================================================================

def analyze_by_prediction_type(df: pd.DataFrame) -> dict:
    """Win rate for DIRECTIONAL vs VOLATILITY trades."""
    result = {}
    for pred_type, group in df.groupby("prediction_type"):
        scoreable = group[
            (group["master_decision"].isin(["BULLISH", "BEARISH"])) |
            (group["prediction_type"] == "VOLATILITY")
        ]
        if len(scoreable) < 3:
            continue
        result[pred_type] = {
            "total": len(scoreable),
            "win_rate": _safe_pct(int(scoreable["master_correct"].sum()), len(scoreable)),
            "avg_confidence": round(scoreable["master_confidence"].mean(), 3),
        }
    return result


# ============================================================================
# SUPPLEMENTARY: Master Confidence Calibration
# ============================================================================

def analyze_confidence_calibration(df: pd.DataFrame) -> dict:
    """Is Master confidence well-calibrated? 80% confidence should win ~80%."""
    scoreable = df[
        (df["master_decision"].isin(["BULLISH", "BEARISH"])) |
        (df["prediction_type"] == "VOLATILITY")
    ].copy()

    if scoreable.empty:
        return {}

    bins = [
        ("low (< 0.50)", scoreable["master_confidence"] < 0.50),
        ("medium (0.50-0.70)", (scoreable["master_confidence"] >= 0.50) & (scoreable["master_confidence"] < 0.70)),
        ("high (0.70-0.85)", (scoreable["master_confidence"] >= 0.70) & (scoreable["master_confidence"] < 0.85)),
        ("very_high (>= 0.85)", scoreable["master_confidence"] >= 0.85),
    ]

    result = {}
    for label, mask in bins:
        subset = scoreable[mask]
        if len(subset) > 0:
            result[label] = {
                "count": len(subset),
                "win_rate": _safe_pct(int(subset["master_correct"].sum()), len(subset)),
                "avg_confidence": round(subset["master_confidence"].mean(), 3),
            }

    # Check monotonicity
    rates = [result.get(label, {}).get("win_rate", 0) for label, _ in bins]
    non_zero = [r for r in rates if r > 0]
    is_monotonic = all(non_zero[i] <= non_zero[i + 1] for i in range(len(non_zero) - 1)) if len(non_zero) > 1 else True
    result["_calibration_note"] = (
        "CALIBRATED: Higher confidence predicts higher win rate" if is_monotonic
        else "MISCALIBRATED: Confidence does not predict outcome quality"
    )

    return result


# ============================================================================
# SUPPLEMENTARY: NEUTRAL Decision Analysis
# ============================================================================

def analyze_neutral_decisions(df: pd.DataFrame) -> dict:
    """How often does the Master say NEUTRAL on directional cycles?"""
    dir_cycles = df[df["prediction_type"] == "DIRECTIONAL"].copy()
    if dir_cycles.empty:
        return {}

    neutral_master = dir_cycles[dir_cycles["master_decision"] == "NEUTRAL"]
    non_neutral_master = dir_cycles[dir_cycles["master_decision"].isin(["BULLISH", "BEARISH"])]

    # When Master says NEUTRAL, how often was the market actually NEUTRAL?
    neutral_on_noise = neutral_master[neutral_master["actual_trend_direction"] == "NEUTRAL"]
    neutral_on_move = neutral_master[neutral_master["actual_trend_direction"].isin(["BULLISH", "BEARISH"])]

    neutral_pct = _safe_pct(len(neutral_master), len(dir_cycles))
    neutral_acc = _safe_pct(len(neutral_on_noise), len(neutral_master))
    non_neutral_wr = _safe_pct(
        int(non_neutral_master["master_correct"].sum()),
        len(non_neutral_master)
    ) if len(non_neutral_master) > 0 else 0

    note = (
        f"Master says NEUTRAL {neutral_pct:.0f}% of the time. "
        f"Of those, {neutral_acc:.0f}% were correct (market was flat). "
        f"When directional, Master wins {non_neutral_wr:.0f}%."
        if len(neutral_master) > 0 else "Master rarely says NEUTRAL."
    )

    return {
        "total_directional": len(dir_cycles),
        "master_neutral_count": len(neutral_master),
        "master_neutral_pct": neutral_pct,
        "neutral_on_noise": len(neutral_on_noise),
        "neutral_on_real_move": len(neutral_on_move),
        "neutral_accuracy": neutral_acc,
        "non_neutral_win_rate": non_neutral_wr,
        "note": note,
    }


# ============================================================================
# HELPERS
# ============================================================================

def _safe_pct(numerator: int, denominator: int) -> float:
    """Return percentage (0-100) or 0 if denominator is zero."""
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


# ============================================================================
# REPORT FORMATTING
# ============================================================================

def print_report(ticker: str, results: dict):
    """Print formatted forensic report."""
    W = 72
    print(f"\n{'=' * W}")
    print(f"  MASTER STRATEGIST FORENSIC REPORT -- {ticker}")
    print(f"{'=' * W}")

    # --- D1: Override Quadrants ---
    d1 = results["override_quadrants"]
    if d1["total"] > 0:
        print(f"\n  1. OVERRIDE QUADRANT ANALYSIS ({d1['total']} scoreable decisions)")
        print(f"  {'-' * (W - 4)}")
        print(f"                     {'Correct':>10} {'Wrong':>10} {'Win Rate':>10}")
        print(f"  Aligned (followed)  {d1['aligned_correct']:>10} {d1['aligned_wrong']:>10} {d1['aligned_win_rate']:>9.1f}%")
        print(f"  Override (deviated) {d1['override_correct']:>10} {d1['override_wrong']:>10} {d1['override_win_rate']:>9.1f}%")
        print(f"  Override rate: {d1['override_rate']:.1f}%")
        delta = d1.get("override_vs_aligned_delta")
        if delta is not None:
            direction = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
            print(f"  Override vs Aligned: {delta:+.1f}pp -- Master overrides are {direction}")
            if delta < -10:
                print(f"  !! CRITICAL: Master overrides destroy significant value")
            elif delta > 10:
                print(f"  OK Master overrides add significant value")
    else:
        print("\n  1. OVERRIDE QUADRANT: Insufficient data")

    # --- D2: Conviction vs Outcome ---
    d2 = results["conviction_vs_outcome"]
    if d2:
        print(f"\n  2. CONVICTION MULTIPLIER vs OUTCOME")
        print(f"  {'-' * (W - 4)}")
        for level, data in d2.items():
            print(f"  {level:<25} n={data['count']:>4}  wins={data['wins']:>4}  win_rate={data['win_rate']:>5.1f}%")

    # --- D3: Consensus Strength ---
    d3 = results["consensus_strength"]
    if d3:
        print(f"\n  3. CONSENSUS STRENGTH vs OUTCOME")
        print(f"  {'-' * (W - 4)}")
        for band, data in d3.items():
            override_str = f"override_wr={data['override_win_rate']:.1f}%" if data.get("override_win_rate") is not None else "no overrides"
            aligned_str = f"aligned_wr={data['aligned_win_rate']:.1f}%" if data.get("aligned_win_rate") is not None else "no aligned"
            print(
                f"  {band:<25} n={data['total']:>4}  "
                f"overall_wr={data['overall_win_rate']:>5.1f}%  "
                f"follow={data['follow_rate']:.0f}%  "
                f"{override_str}  {aligned_str}"
            )

    # --- D4: Regime Breakdown ---
    d4 = results["regime_breakdown"]
    if d4:
        print(f"\n  4. REGIME-SPECIFIC BREAKDOWN")
        print(f"  {'-' * (W - 4)}")
        for regime in sorted(d4.keys()):
            data = d4[regime]
            override_str = f"override_wr={data['override_win_rate']:.1f}%" if data.get("override_win_rate") is not None else "no overrides"
            print(
                f"  {regime:<18} n={data['total']:>4}  "
                f"win_rate={data['win_rate']:>5.1f}%  "
                f"override_rate={data['override_rate']:.1f}%  "
                f"{override_str}  "
                f"avg_consensus={data['avg_consensus_strength']:.3f}"
            )

    # --- D5: Thesis Calibration ---
    d5 = results["thesis_calibration"]
    if d5:
        print(f"\n  5. THESIS STRENGTH CALIBRATION")
        print(f"  {'-' * (W - 4)}")
        for thesis in ["SPECULATIVE", "PLAUSIBLE", "PROVEN"]:
            if thesis in d5:
                data = d5[thesis]
                print(
                    f"  {thesis:<14} n={data['count']:>4}  "
                    f"win_rate={data['win_rate']:>5.1f}%  "
                    f"avg_conf={data['avg_confidence']:.2f}  "
                    f"avg_conviction={data['avg_conviction']:.2f}"
                )
        print(f"  -> {d5.get('_calibration_note', 'N/A')}")

    # --- D6: Trigger Type ---
    d6 = results["trigger_breakdown"]
    if d6:
        print(f"\n  6. TRIGGER TYPE BREAKDOWN")
        print(f"  {'-' * (W - 4)}")
        for trigger, data in d6.items():
            override_str = f"override_wr={data['override_win_rate']:.1f}%" if data.get("override_win_rate") is not None else "no overrides"
            print(
                f"  {trigger:<14} n={data['total']:>4}  "
                f"win_rate={data['win_rate']:>5.1f}%  "
                f"override_rate={data['override_rate']:.1f}%  "
                f"{override_str}  "
                f"avg_conf={data['avg_confidence']:.2f}"
            )

    # --- D7: Dominant Agent Overrides ---
    d7 = results["dominant_agent_overrides"]
    if d7:
        print(f"\n  7. DOMINANT AGENT OVERRIDE ANALYSIS")
        print(f"  {'-' * (W - 4)}")
        print(f"  {'Agent':<18} {'Overrides':>10} {'Master WR':>10} {'Vote WR':>10} {'Net Value':>10}")
        for agent in sorted(d7.keys(), key=lambda a: d7[a]["override_count"], reverse=True):
            data = d7[agent]
            net = data.get("master_override_net_value", 0)
            net_str = f"{net:+.1f}pp"
            flag = " !!" if net < -15 else " OK" if net > 15 else ""
            print(
                f"  {agent:<18} {data['override_count']:>10} "
                f"{data['override_win_rate']:>9.1f}% "
                f"{data['vote_would_have_won']:>9.1f}% "
                f"{net_str:>10}{flag}"
            )

    # --- Supplementary: Prediction Type ---
    pt = results.get("prediction_type")
    if pt:
        print(f"\n  S1. PREDICTION TYPE")
        print(f"  {'-' * (W - 4)}")
        for ptype, data in pt.items():
            print(f"  {ptype:<18} n={data['total']:>4}  win_rate={data['win_rate']:>5.1f}%  avg_conf={data['avg_confidence']:.2f}")

    # --- Supplementary: Confidence Calibration ---
    cc = results.get("confidence_calibration")
    if cc:
        print(f"\n  S2. CONFIDENCE CALIBRATION")
        print(f"  {'-' * (W - 4)}")
        for level in ["low (< 0.50)", "medium (0.50-0.70)", "high (0.70-0.85)", "very_high (>= 0.85)"]:
            if level in cc:
                data = cc[level]
                print(f"  {level:<22} n={data['count']:>4}  win_rate={data['win_rate']:>5.1f}%  avg_conf={data['avg_confidence']:.2f}")
        print(f"  -> {cc.get('_calibration_note', 'N/A')}")

    # --- Supplementary: NEUTRAL Analysis ---
    na = results.get("neutral_analysis")
    if na and na.get("total_directional", 0) > 0:
        print(f"\n  S3. NEUTRAL DECISION ANALYSIS")
        print(f"  {'-' * (W - 4)}")
        print(f"  {na.get('note', 'N/A')}")

    # --- Summary / Recommendations ---
    print(f"\n  {'=' * (W - 4)}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"  {'=' * (W - 4)}")

    if d1["total"] > 0:
        delta = d1.get("override_vs_aligned_delta")
        if delta is not None and delta < -10:
            print(f"  !! Master overrides are net DESTRUCTIVE ({delta:+.1f}pp). Consider:")
            print(f"     - Raising conviction threshold for override trades")
            print(f"     - Requiring stronger weighted_score for Master to deviate")
        elif delta is not None and delta > 10:
            print(f"  OK Master overrides ADD VALUE ({delta:+.1f}pp). Master judgment is trustworthy.")
        elif delta is not None:
            print(f"  -- Master overrides are NEUTRAL relative to following consensus ({delta:+.1f}pp).")

    if d4:
        weak_regimes = [r for r, data in d4.items() if data["win_rate"] < 40 and data["total"] >= 5]
        if weak_regimes:
            print(f"  !! Weak regimes (win rate < 40%): {', '.join(weak_regimes)}")
            print(f"     Consider regime-aware trade gating for these regimes")

    if d5 and not d5.get("_calibration_monotonic", True):
        print(f"  !! Thesis strength is MISCALIBRATED -- PROVEN does not predict better outcomes")
        print(f"     Master cannot reliably assess its own conviction quality")

    cc_note = cc.get("_calibration_note", "") if cc else ""
    if "MISCALIBRATED" in cc_note:
        print(f"  !! Confidence is MISCALIBRATED -- higher confidence does not predict better outcomes")

    print()


# ============================================================================
# MAIN
# ============================================================================

def run_analysis(ticker: str, data_dir: str = "data") -> dict:
    """Run all forensic dimensions for one commodity."""
    df = load_resolved_cycles(ticker, data_dir)
    if df.empty:
        return {"ticker": ticker, "status": "no_data"}

    return {
        "ticker": ticker,
        "status": "ok",
        "total_resolved": len(df),
        "override_quadrants": analyze_override_quadrants(df),
        "conviction_vs_outcome": analyze_conviction_vs_outcome(df),
        "consensus_strength": analyze_consensus_strength(df),
        "regime_breakdown": analyze_by_regime(df),
        "thesis_calibration": analyze_thesis_calibration(df),
        "trigger_breakdown": analyze_by_trigger(df),
        "dominant_agent_overrides": analyze_dominant_agent_overrides(df),
        "prediction_type": analyze_by_prediction_type(df),
        "confidence_calibration": analyze_confidence_calibration(df),
        "neutral_analysis": analyze_neutral_decisions(df),
    }


def main():
    parser = argparse.ArgumentParser(description="Master Strategist Forensic Analysis")
    parser.add_argument("--ticker", type=str, help="Single ticker (default: all)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--csv", action="store_true", help="Export per-decision CSV")
    parser.add_argument("--json", action="store_true", help="Export JSON results")
    args = parser.parse_args()

    data_dir = args.data_dir
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = sorted([
            d.upper() for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and os.path.exists(os.path.join(data_dir, d, "council_history.csv"))
        ])

    if not tickers:
        print("No tickers found with council_history.csv")
        sys.exit(1)

    all_results = {}
    for ticker in tickers:
        results = run_analysis(ticker, data_dir)
        all_results[ticker] = results
        if results["status"] == "ok":
            print_report(ticker, results)
        else:
            print(f"\n{ticker}: No resolved data available")

    # Export options
    if args.json:
        output_path = os.path.join(data_dir, "master_forensics.json")
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nJSON exported to {output_path}")

    if args.csv:
        for ticker in tickers:
            df = load_resolved_cycles(ticker, data_dir)
            if df.empty:
                continue
            csv_path = os.path.join(data_dir, ticker, "master_forensics.csv")
            export_cols = [
                "cycle_id", "timestamp", "contract", "prediction_type",
                "master_decision", "master_confidence", "thesis_strength",
                "vote_direction", "weighted_score", "conviction_multiplier",
                "is_override", "master_correct", "actual_trend_direction",
                "entry_regime", "trigger_type", "dominant_agent",
                "strategy_type", "volatility_outcome",
            ]
            available = [c for c in export_cols if c in df.columns]
            df[available].to_csv(csv_path, index=False)
            print(f"\nCSV exported to {csv_path}")


if __name__ == "__main__":
    main()
