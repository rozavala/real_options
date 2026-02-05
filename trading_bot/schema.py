"""Schema definitions for CSV data files."""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class CouncilHistoryRow:
    """J2 FIX: Enforce council_history.csv schema."""
    timestamp: str
    cycle_id: str
    cycle_type: str  # SCHEDULED or EMERGENCY
    contract: str
    direction: str
    prediction_type: str
    vol_level: str
    confidence: float
    weighted_score: float
    thesis_strength: str
    regime: str
    compliance_approved: bool
    reason: str

    @classmethod
    def validate_row(cls, row: dict) -> 'CouncilHistoryRow':
        """Validate and coerce a dict to the expected schema."""
        required_fields = ['timestamp', 'cycle_id', 'cycle_type', 'contract']
        for field in required_fields:
            if field not in row or not row[field]:
                raise ValueError(f"Missing required field: {field}")

        return cls(
            timestamp=str(row['timestamp']),
            cycle_id=str(row['cycle_id']),
            cycle_type=str(row.get('cycle_type', 'UNKNOWN')),
            contract=str(row['contract']),
            direction=str(row.get('direction', 'NEUTRAL')),
            prediction_type=str(row.get('prediction_type', 'UNKNOWN')),
            vol_level=str(row.get('vol_level', 'UNKNOWN')),
            confidence=float(row.get('confidence', 0.0)),
            weighted_score=float(row.get('weighted_score', 0.0)),
            thesis_strength=str(row.get('thesis_strength', 'SPECULATIVE')),
            regime=str(row.get('regime', 'UNKNOWN')),
            compliance_approved=bool(row.get('compliance_approved', False)),
            reason=str(row.get('reason', ''))
        )
