"""
Canonical exit/close reasons for position lifecycle events.

Every position close event MUST use one of these values. No free-text exit
reasons allowed in trade_ledger.csv, TMS metadata, or funnel events.

Usage:
    from trading_bot.exit_reasons import ExitReason
    reason = ExitReason.STOP_LOSS
    log_funnel_event(..., detail=reason.value)
"""

from enum import Enum


class ExitReason(str, Enum):
    """Why a position was closed. Exhaustive — add new values here, not inline."""

    # === Entry (not an exit, but tracked for ledger symmetry) ===
    STRATEGY_EXECUTION = "Strategy Execution"

    # === Active risk management ===
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    RISK_MANAGEMENT = "Risk Management Closure"
    POSITION_MISALIGNED = "Position Misaligned"

    # === Time-based ===
    STALE_CLOSE = "Stale Position Close"
    FRIDAY_WEEKLY_CLOSE = "Friday Weekly Close"
    HOLIDAY_WEEKLY_CLOSE = "Holiday Tomorrow (Weekly Close)"
    POST_CLOSE_SWEEP = "POST_CLOSE_SWEEP"
    DTE_FORCE_CLOSE = "DTE_FORCE_CLOSE"

    # === Thesis lifecycle ===
    CONTRADICT_CLOSED = "CONTRADICT: position already closed"
    CONTRADICT_REVERSAL = "CONTRADICT: closed on direction reversal"
    MANUAL = "MANUAL"

    # === Options-specific terminal states ===
    EXPIRED_WORTHLESS = "EXPIRED_WORTHLESS"
    EXPIRED_ITM = "EXPIRED_ITM"
    ASSIGNED = "ASSIGNED"
    EXERCISED = "EXERCISED"

    # === Emergency ===
    EMERGENCY_HARD_CLOSE = "EMERGENCY_HARD_CLOSE"
    EMERGENCY_HARD_CLOSE_RETRY = "EMERGENCY_HARD_CLOSE_RETRY"
    EMERGENCY_HARD_CLOSE_RETRY_PARTIAL = "EMERGENCY_HARD_CLOSE_RETRY_PARTIAL"
    CATASTROPHE_FILL = "CATASTROPHE_FILL (auto-detected)"

    # === Spread close (normal exit flow) ===
    SPREAD_CLOSE = "SPREAD_CLOSE"
    SPREAD_CLOSE_NO_MKT_DATA = "SPREAD_CLOSE_NO_MKT_DATA"
    SPREAD_CLOSE_MARKET_FALLBACK = "SPREAD_CLOSE_MARKET_FALLBACK"

    # === Reconciliation / maintenance ===
    RECONCILIATION_MISSING = "RECONCILIATION_MISSING"
    PHANTOM_RECONCILIATION = "PHANTOM_RECONCILIATION"
    ORPHAN_CLEANUP = "ORPHAN_CLEANUP"

    # === Thesis invalidation (TMS, not trade ledger) ===
    REGIME_BREACH = "REGIME BREACH"
    PRICE_BREACH = "PRICE BREACH"
    NARRATIVE_INVALIDATION = "NARRATIVE INVALIDATION"

    # === Future (not instrumented yet) ===
    ROLLED = "ROLLED"


def classify_exit_reason(reason_text: str) -> ExitReason:
    """
    Best-effort classification of a free-text reason string into an ExitReason.

    Used by the backfill script to map historical reason strings to enum values.
    Returns the closest match, or RISK_MANAGEMENT as fallback.
    """
    if not reason_text:
        return ExitReason.STRATEGY_EXECUTION

    text = str(reason_text)

    # Exact matches first
    for member in ExitReason:
        if text == member.value:
            return member

    # Prefix / substring matches
    lower = text.lower()
    if 'stop-loss' in lower or 'stop_loss' in lower:
        return ExitReason.STOP_LOSS
    if 'take-profit' in lower or 'take_profit' in lower:
        return ExitReason.TAKE_PROFIT
    if 'stale position close' in lower:
        return ExitReason.STALE_CLOSE
    if lower.startswith('stale position'):
        return ExitReason.STALE_CLOSE
    if 'friday weekly' in lower:
        return ExitReason.FRIDAY_WEEKLY_CLOSE
    if 'holiday tomorrow' in lower:
        return ExitReason.HOLIDAY_WEEKLY_CLOSE
    if 'post-close sweep' in lower:
        return ExitReason.POST_CLOSE_SWEEP
    if 'contradict' in lower and 'reversal' in lower:
        return ExitReason.CONTRADICT_REVERSAL
    if 'contradict' in lower and 'closed' in lower:
        return ExitReason.CONTRADICT_CLOSED
    if 'contradict' in lower:
        return ExitReason.CONTRADICT_REVERSAL
    if 'emergency_hard_close_retry_partial' in lower:
        return ExitReason.EMERGENCY_HARD_CLOSE_RETRY_PARTIAL
    if 'emergency_hard_close_retry' in lower:
        return ExitReason.EMERGENCY_HARD_CLOSE_RETRY
    if 'emergency_hard_close' in lower:
        return ExitReason.EMERGENCY_HARD_CLOSE
    if 'catastrophe' in lower:
        return ExitReason.CATASTROPHE_FILL
    if 'dte force close' in lower or 'dte_force_close' in lower:
        return ExitReason.DTE_FORCE_CLOSE
    if 'spread close' in lower and 'no mkt' in lower:
        return ExitReason.SPREAD_CLOSE_NO_MKT_DATA
    if 'spread close' in lower and 'market fallback' in lower:
        return ExitReason.SPREAD_CLOSE_MARKET_FALLBACK
    if 'spread close' in lower:
        return ExitReason.SPREAD_CLOSE
    if 'reconciliation_missing' in lower:
        return ExitReason.RECONCILIATION_MISSING
    if 'phantom' in lower:
        return ExitReason.PHANTOM_RECONCILIATION
    if 'orphan' in lower:
        return ExitReason.ORPHAN_CLEANUP
    if 'position misaligned' in lower:
        return ExitReason.POSITION_MISALIGNED
    if 'regime breach' in lower:
        return ExitReason.REGIME_BREACH
    if 'price breach' in lower:
        return ExitReason.PRICE_BREACH
    if 'narrative' in lower:
        return ExitReason.NARRATIVE_INVALIDATION
    if lower == 'strategy execution' or lower == 'daily strategy fill':
        return ExitReason.STRATEGY_EXECUTION
    if 'risk management' in lower:
        return ExitReason.RISK_MANAGEMENT

    return ExitReason.RISK_MANAGEMENT
