"""
Execution Funnel Logger — Structured event log for the signal-to-P&L pipeline.

WHY THIS EXISTS:
council_history.csv captures the intelligence layer (what the council decided).
trade_ledger.csv captures the financial layer (what actually traded).
Nothing captures the middle: conviction gates, compliance, liquidity checks,
adaptive walking, fills vs cancels. This module bridges that gap with an
append-only event log enabling funnel analysis to pinpoint where alpha leaks.

SCHEMA (execution_funnel.csv):
    timestamp       — UTC ISO8601
    cycle_id        — Links to council_history + decision_signals for drill-down
    contract        — Contract identifier (e.g., "KCK6 (202605)")
    stage           — FunnelStage enum value
    outcome         — PASS / BLOCK / PARTIAL / FAIL / INFO
    detail          — Context string (e.g., "score=0.34, threshold=0.10")
    price_snapshot  — Underlying price at event time (nullable)
    bid             — Combo bid at event time (nullable, ORDER_PLACED+ only)
    ask             — Combo ask at event time (nullable, ORDER_PLACED+ only)
    initial_limit   — First limit price set (nullable, ORDER_PLACED only)
    fill_price      — Actual fill price (nullable, ORDER_FILLED only)
    walk_away_price — Last limit before cancel (nullable, ORDER_CANCELLED only)
    regime          — Market regime at decision time
    source          — REALTIME or BACKFILL

USAGE:
    from trading_bot.execution_funnel import log_funnel_event, FunnelStage

    log_funnel_event(
        cycle_id="KC-abc123",
        contract="KCK6 (202605)",
        stage=FunnelStage.CONVICTION_GATE,
        outcome="PASS",
        detail=f"weighted_score={score:.4f}, threshold={threshold}",
        price_snapshot=market_ctx.get('price'),
        regime="TRENDING",
    )
"""

import os
import logging
import pandas as pd
from enum import Enum
from typing import Optional
from trading_bot.timestamps import format_ts, parse_ts_column

logger = logging.getLogger(__name__)

# --- Module-level path (legacy single-engine fallback) ---
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNNEL_FILE_PATH = os.path.join(_BASE_DIR, 'data', 'execution_funnel.csv')


def set_data_dir(data_dir: str):
    """Configure funnel path for a commodity-specific data directory."""
    global FUNNEL_FILE_PATH
    FUNNEL_FILE_PATH = os.path.join(data_dir, 'execution_funnel.csv')
    logger.info(f"ExecutionFunnel data_dir set to: {data_dir}")


def _get_funnel_file_path() -> str:
    """Resolve funnel file path via ContextVar (multi-engine) or module global."""
    try:
        from trading_bot.data_dir_context import get_engine_data_dir
        return os.path.join(get_engine_data_dir(), 'execution_funnel.csv')
    except LookupError:
        return FUNNEL_FILE_PATH


class FunnelStage(str, Enum):
    """Every stage in the signal-to-P&L pipeline. Ordered by execution flow."""

    # --- Intelligence Layer ---
    COUNCIL_DECISION = "COUNCIL_DECISION"

    # --- Decision Gates ---
    CONVICTION_GATE = "CONVICTION_GATE"
    COMPLIANCE_AUDIT = "COMPLIANCE_AUDIT"
    DA_REVIEW = "DA_REVIEW"

    # --- Order Generation ---
    CONFIDENCE_THRESHOLD = "CONFIDENCE_THRESHOLD"
    THESIS_COHERENCE = "THESIS_COHERENCE"
    CAPITAL_CHECK = "CAPITAL_CHECK"
    STRATEGY_SELECTION = "STRATEGY_SELECTION"
    ORDER_QUEUED = "ORDER_QUEUED"

    # --- Execution ---
    DRAWDOWN_GATE = "DRAWDOWN_GATE"
    LIQUIDITY_GATE = "LIQUIDITY_GATE"
    ORDER_PLACED = "ORDER_PLACED"
    PRICE_WALK_STEP = "PRICE_WALK_STEP"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIAL_FILL = "ORDER_PARTIAL_FILL"
    ORDER_CANCELLED = "ORDER_CANCELLED"

    # --- Position Management ---
    POSITION_OPENED = "POSITION_OPENED"
    RISK_TRIGGER = "RISK_TRIGGER"
    POSITION_CLOSED = "POSITION_CLOSED"

    # --- Reconciliation ---
    PNL_RECONCILED = "PNL_RECONCILED"
    TMS_ORPHAN_DETECTED = "TMS_ORPHAN_DETECTED"


# Canonical column order — must match CSV header
SCHEMA_COLUMNS = [
    'timestamp', 'cycle_id', 'contract', 'stage', 'outcome', 'detail',
    'price_snapshot', 'bid', 'ask', 'initial_limit', 'fill_price',
    'walk_away_price', 'regime', 'source',
]


def log_funnel_event(
    cycle_id: str,
    contract: str,
    stage: FunnelStage,
    outcome: str = "INFO",
    detail: str = "",
    price_snapshot: Optional[float] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    initial_limit: Optional[float] = None,
    fill_price: Optional[float] = None,
    walk_away_price: Optional[float] = None,
    regime: str = "UNKNOWN",
    source: str = "REALTIME",
) -> bool:
    """
    Append one event row to execution_funnel.csv.

    Returns True on success, False on failure (never raises — non-critical path).
    """
    try:
        # Guard: only log events inside a live orchestrator session (skip in tests/dashboard)
        if source == "REALTIME":
            try:
                from trading_bot.data_dir_context import get_engine_runtime
                if get_engine_runtime() is None:
                    return True
            except Exception:
                pass

        file_path = _get_funnel_file_path()
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

        new_row = pd.DataFrame({
            'timestamp': [format_ts()],
            'cycle_id': [cycle_id],
            'contract': [contract],
            'stage': [stage.value if isinstance(stage, FunnelStage) else str(stage)],
            'outcome': [outcome],
            'detail': [str(detail)[:500]],
            'price_snapshot': [round(float(price_snapshot), 2) if price_snapshot is not None else None],
            'bid': [round(float(bid), 4) if bid is not None else None],
            'ask': [round(float(ask), 4) if ask is not None else None],
            'initial_limit': [round(float(initial_limit), 4) if initial_limit is not None else None],
            'fill_price': [round(float(fill_price), 4) if fill_price is not None else None],
            'walk_away_price': [round(float(walk_away_price), 4) if walk_away_price is not None else None],
            'regime': [regime],
            'source': [source],
        })

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            new_row.to_csv(file_path, index=False)
        else:
            try:
                existing_header = pd.read_csv(file_path, nrows=0)
                if list(existing_header.columns) != SCHEMA_COLUMNS:
                    logger.info("Schema mismatch in execution_funnel.csv — migrating in-place.")
                    full_df = pd.read_csv(file_path)
                    for col in SCHEMA_COLUMNS:
                        if col not in full_df.columns:
                            full_df[col] = None
                    full_df = full_df[SCHEMA_COLUMNS]
                    combined = pd.concat([full_df, new_row], ignore_index=True)
                    temp_path = file_path + ".tmp"
                    combined.to_csv(temp_path, index=False)
                    os.replace(temp_path, file_path)
                else:
                    new_row.to_csv(file_path, mode='a', header=False, index=False)
            except pd.errors.EmptyDataError:
                new_row.to_csv(file_path, index=False)

        return True

    except Exception as e:
        logger.warning(f"Failed to log funnel event (non-fatal): {e}")
        return False


def get_funnel_df(ticker: str = None) -> pd.DataFrame:
    """Load execution funnel data for dashboard consumption."""
    try:
        if ticker:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, 'data', ticker, 'execution_funnel.csv')
        else:
            path = _get_funnel_file_path()

        if not os.path.exists(path):
            return pd.DataFrame(columns=SCHEMA_COLUMNS)

        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = parse_ts_column(df['timestamp'])
        return df

    except Exception as e:
        logger.warning(f"Failed to load funnel data: {e}")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
