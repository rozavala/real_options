"""
Decision Signals Logger — Lightweight record of Council trading decisions.

WHY THIS EXISTS:
council_history.csv has 30+ columns (full agent text, debates, summaries).
It's essential for forensics but unusable for quick operational checks.
This module logs ONE clean row per contract per cycle with just the
decision-relevant fields, giving operators a fast "what did the system
decide today?" view.

REPLACES: archive/ml_pipeline/model_signals.py (ML-specific, now dead)

SCHEMA:
    timestamp       — UTC ISO8601
    cycle_id        — Links to council_history for drill-down
    contract        — Contract month (e.g., 202605)
    signal          — BULLISH / BEARISH / NEUTRAL / VOLATILITY
    prediction_type — DIRECTIONAL / VOLATILITY
    strategy        — BULL_CALL_SPREAD / BEAR_PUT_SPREAD / LONG_STRADDLE / IRON_CONDOR / NONE
    price           — Current market price at decision time
    sma_200         — 200-day SMA (if available)
    confidence      — Master confidence (0.0 – 1.0)
    regime          — TRENDING / RANGE_BOUND / UNKNOWN
    trigger_type    — SCHEDULED / EMERGENCY / DEFERRED

USAGE:
    from trading_bot.decision_signals import log_decision_signal, get_decision_signals_df

    # After Council decides:
    log_decision_signal(
        cycle_id="20260201_143000_KCH6",
        contract="202603",
        signal="BULLISH",
        prediction_type="DIRECTIONAL",
        strategy="BULL_CALL_SPREAD",
        price=354.05,
        sma_200=346.54,
        confidence=0.80,
        regime="TRENDING",
        trigger_type="SCHEDULED"
    )

    # For dashboard / reports:
    df = get_decision_signals_df()
"""

import os
import logging
import pandas as pd
from datetime import datetime, timezone
from trading_bot.timestamps import format_ts, parse_ts_column

logger = logging.getLogger(__name__)

# File lives at repo root alongside trade_ledger.csv
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS_FILE_PATH = os.path.join(_BASE_DIR, 'decision_signals.csv')

# Canonical schema — order matters for CSV columns
SCHEMA_COLUMNS = [
    'timestamp',
    'cycle_id',
    'contract',
    'signal',
    'prediction_type',
    'strategy',
    'price',
    'sma_200',
    'confidence',
    'regime',
    'trigger_type',
]


def log_decision_signal(
    cycle_id: str,
    contract: str,
    signal: str,
    prediction_type: str = "DIRECTIONAL",
    strategy: str = "NONE",
    price: float = None,
    sma_200: float = None,
    confidence: float = None,
    regime: str = "UNKNOWN",
    trigger_type: str = "SCHEDULED",
) -> bool:
    """
    Append one decision signal row to decision_signals.csv.

    Returns True on success, False on failure (never raises).
    """
    try:
        # Clamp confidence
        if confidence is not None:
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (ValueError, TypeError):
                confidence = None

        new_row = pd.DataFrame({
            'timestamp': [format_ts()],
            'cycle_id': [cycle_id],
            'contract': [contract],
            'signal': [signal],
            'prediction_type': [prediction_type],
            'strategy': [strategy],
            'price': [round(float(price), 2) if price is not None else None],
            'sma_200': [round(float(sma_200), 2) if sma_200 is not None else None],
            'confidence': [confidence],
            'regime': [regime],
            'trigger_type': [trigger_type],
        })

        if not os.path.exists(SIGNALS_FILE_PATH):
            logger.info(f"Creating new decision_signals.csv at {SIGNALS_FILE_PATH}")
            new_row.to_csv(SIGNALS_FILE_PATH, index=False)
        else:
            # Schema migration: check if columns match
            try:
                existing_header = pd.read_csv(SIGNALS_FILE_PATH, nrows=0)
                if list(existing_header.columns) != SCHEMA_COLUMNS:
                    logger.info("Schema mismatch in decision_signals.csv — migrating in-place.")
                    full_df = pd.read_csv(SIGNALS_FILE_PATH)
                    for col in SCHEMA_COLUMNS:
                        if col not in full_df.columns:
                            full_df[col] = None
                    full_df = full_df[SCHEMA_COLUMNS]
                    combined = pd.concat([full_df, new_row], ignore_index=True)
                    combined.to_csv(SIGNALS_FILE_PATH, index=False)
                else:
                    new_row.to_csv(SIGNALS_FILE_PATH, mode='a', header=False, index=False)
            except pd.errors.EmptyDataError:
                new_row.to_csv(SIGNALS_FILE_PATH, index=False)

        logger.info(f"Decision signal logged: {contract} → {signal} ({prediction_type}/{strategy})")
        return True

    except Exception as e:
        logger.error(f"Failed to log decision signal: {e}", exc_info=True)
        return False


def get_decision_signals_df() -> pd.DataFrame:
    """
    Read decision_signals.csv into a DataFrame with parsed timestamps.
    Returns empty DataFrame if file doesn't exist.
    """
    if not os.path.exists(SIGNALS_FILE_PATH):
        logger.info(f"No decision_signals.csv found at {SIGNALS_FILE_PATH}")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    try:
        df = pd.read_csv(SIGNALS_FILE_PATH)
        if 'timestamp' in df.columns:
            df['timestamp'] = parse_ts_column(df['timestamp'])
        return df
    except Exception as e:
        logger.error(f"Failed to read decision_signals.csv: {e}")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
