import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class OperationalHealthMonitor:
    """Monitors agent outputs for hallucination and quality degradation."""

    HALLUCINATION_INDICATORS = [
        r"USDA report \d{4}",  # Check if cited reports exist
        r"according to.*(\d{4})",  # Date references
        r"price of \$[\d,]+",  # Price claims
    ]

    def __init__(self, config: dict):
        self.config = config
        self.alerts = []

    async def validate_output(self, agent: str, output: str) -> tuple[bool, list]:
        """
        Validate agent output for potential hallucinations.
        Returns: (is_valid, warnings)
        """
        warnings = []

        # Check 1: Output length (context saturation indicator)
        if len(output) < 50:
            warnings.append(f"{agent}: Suspiciously short output (possible truncation)")

        # Check 2: Repetition (hallucination indicator)
        sentences = output.split('.')
        # Filter empty
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            unique_ratio = len(set(sentences)) / max(len(sentences), 1)
            if unique_ratio < 0.7:
                warnings.append(f"{agent}: High repetition detected (possible loop)")

        # Check 3: Confidence calibration
        if "100% certain" in output.lower() or "guaranteed" in output.lower():
            warnings.append(f"{agent}: Overconfident language detected")

        # Check 4: Citation validation (would need external API, placeholder)

        is_valid = len(warnings) == 0

        if not is_valid:
            logger.warning(f"Health check failed for {agent}: {warnings}")

        return is_valid, warnings

    def quarantine_agent(self, agent: str, reason: str):
        """Mark an agent as unreliable for this cycle."""
        self.alerts.append({
            "agent": agent,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        logger.error(f"AGENT QUARANTINED: {agent} - {reason}")
