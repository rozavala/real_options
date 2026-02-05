"""Schema definitions for CSV data files."""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


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
            master_confidence=float(row.get('master_confidence', 0.0)),
            weighted_score=float(row.get('weighted_score', 0.0)),
            thesis_strength=str(row.get('thesis_strength', 'SPECULATIVE')),
            regime=str(row.get('regime', 'UNKNOWN')),
            compliance_approved=bool(row.get('compliance_approved', False)),
            master_reasoning=str(row.get('master_reasoning', ''))
        )
