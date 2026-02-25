"""Schema definitions for CSV data files."""

import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# Council History CSV Schema — Single Source of Truth
# =============================================================================
# Both log_council_decision() and migration scripts import from here.
# When adding columns: update ONLY this list + add backfill defaults.
# =============================================================================

COUNCIL_HISTORY_FIELDNAMES = [
    "cycle_id", "timestamp", "contract", "entry_price",
    "meteorologist_sentiment", "meteorologist_summary",
    "macro_sentiment", "macro_summary",
    "geopolitical_sentiment", "geopolitical_summary",
    "fundamentalist_sentiment", "fundamentalist_summary",
    "sentiment_sentiment", "sentiment_summary",
    "technical_sentiment", "technical_summary",
    "volatility_sentiment", "volatility_summary",
    "master_decision", "master_confidence", "master_reasoning",
    "prediction_type", "volatility_level", "strategy_type",
    "compliance_approved",
    "exit_price", "exit_timestamp", "pnl_realized", "actual_trend_direction",
    "volatility_outcome",
    # v2 — weighted voting
    "vote_breakdown", "dominant_agent", "weighted_score",
    "trigger_type", "schedule_id",
    # v3 — Judge & Jury
    "thesis_strength", "primary_catalyst", "conviction_multiplier", "dissent_acknowledged",
    # v4 — VaR & Compliance observability
    "compliance_reason", "var_utilization", "var_dampener", "risk_briefing_injected",
]

# Semantically honest defaults for backfilling old rows on schema migration.
# Key principle: empty string = "not measured / didn't exist yet"
#                specific value = known correct pre-feature-era default
COUNCIL_HISTORY_BACKFILL_DEFAULTS = {
    # v2 columns
    "vote_breakdown": "",
    "dominant_agent": "",
    "weighted_score": "",
    "trigger_type": "",
    "schedule_id": "",
    # v3 columns
    "thesis_strength": "",
    "primary_catalyst": "",
    "conviction_multiplier": "",
    "dissent_acknowledged": "",
    # v4 columns
    "compliance_reason": "",
    "var_utilization": "",
    "var_dampener": "1.0",
    "risk_briefing_injected": "False",
}


def backfill_missing_columns(row: dict, fieldnames: list = None, defaults: dict = None) -> dict:
    """Ensure a row dict has all canonical columns, filling missing ones with defaults.

    Args:
        row: The existing row dict (modified in place and returned).
        fieldnames: Column list. Defaults to COUNCIL_HISTORY_FIELDNAMES.
        defaults: Default values dict. Defaults to COUNCIL_HISTORY_BACKFILL_DEFAULTS.

    Returns:
        The row dict with all missing fields populated.
    """
    if fieldnames is None:
        fieldnames = COUNCIL_HISTORY_FIELDNAMES
    if defaults is None:
        defaults = COUNCIL_HISTORY_BACKFILL_DEFAULTS

    for field in fieldnames:
        if field not in row or row[field] is None:
            row[field] = defaults.get(field, "")
    return row


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.debug(f"Could not convert {value!r} to float, using default {default}")
        return default


@dataclass
class CouncilHistoryRow:
    """J2 FIX: Enforce council_history.csv schema.

    v5.1 FIX: Aligned field names with actual CSV schema in log_council_decision().
    - Renamed 'cycle_type' → 'trigger_type' to match CSV column name
    - Renamed 'direction' → 'master_decision' to match CSV column name
    - Renamed 'vol_level' → 'volatility_level' to match CSV column name
    - Renamed 'reason' → 'master_reasoning' to match CSV column name
    """
    timestamp: str
    cycle_id: str
    trigger_type: str  # SCHEDULED or EMERGENCY (was 'cycle_type' — CSV uses 'trigger_type')
    contract: str
    master_decision: str  # BULLISH / BEARISH / NEUTRAL (was 'direction')
    prediction_type: str
    volatility_level: str  # (was 'vol_level')
    master_confidence: float  # (was 'confidence')
    weighted_score: float
    thesis_strength: str
    regime: str  # NOTE: not in CSV fieldnames — captured via vote_breakdown or separate field
    compliance_approved: bool
    master_reasoning: str  # (was 'reason')

    @classmethod
    def validate_row(cls, row: dict) -> 'CouncilHistoryRow':
        """Validate and coerce a dict to the expected schema.

        v5.1: Field names now match log_council_decision() CSV columns exactly.
        """
        required_fields = ['timestamp', 'cycle_id', 'contract']
        for field in required_fields:
            if field not in row or not row[field]:
                raise ValueError(f"Missing required field: {field}")

        return cls(
            timestamp=str(row['timestamp']),
            cycle_id=str(row['cycle_id']),
            trigger_type=str(row.get('trigger_type', 'UNKNOWN')),
            contract=str(row['contract']),
            master_decision=str(row.get('master_decision', 'NEUTRAL')),
            prediction_type=str(row.get('prediction_type', 'UNKNOWN')),
            volatility_level=str(row.get('volatility_level', 'UNKNOWN')),
            master_confidence=_safe_float(row.get('master_confidence', 0.0)),
            weighted_score=_safe_float(row.get('weighted_score', 0.0)),
            thesis_strength=str(row.get('thesis_strength', 'SPECULATIVE')),
            regime=str(row.get('regime', 'UNKNOWN')),
            compliance_approved=bool(row.get('compliance_approved', False)),
            master_reasoning=str(row.get('master_reasoning', ''))
        )
