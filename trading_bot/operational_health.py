"""Operational Health Monitoring for Agent System."""

import logging
from typing import Dict, List
from datetime import datetime, timezone
import re

logger = logging.getLogger(__name__)


class OperationalHealthMonitor:
    """Monitors agent system health and detects anomalies."""

    def __init__(self, config: dict):
        self.config = config
        self.agent_response_history: Dict[str, List[dict]] = {}
        self.hallucination_patterns = [
            r'USDA report from \d{4}',  # Fake report citations
            r'according to.*internal data',  # No internal data exists
            r'price of \$\d+\.\d{4}',  # Overly precise prices
        ]

    def check_response_consistency(self, agent: str, response: str) -> dict:
        """Check for hallucination patterns and inconsistencies."""

        issues = []

        # Pattern matching for common hallucinations
        for pattern in self.hallucination_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(f"Possible hallucination: pattern '{pattern}' detected")

        # Check for response length anomalies
        history = self.agent_response_history.get(agent, [])
        if history:
            avg_len = sum(h['length'] for h in history[-10:]) / min(10, len(history))
            if avg_len > 0 and len(response) > avg_len * 3:
                issues.append(f"Response length anomaly: {len(response)} vs avg {avg_len:.0f}")

        # Store response metrics
        if agent not in self.agent_response_history:
            self.agent_response_history[agent] = []

        self.agent_response_history[agent].append({
            'timestamp': datetime.now(timezone.utc),
            'length': len(response),
            'issues': issues
        })

        # Keep history bounded
        if len(self.agent_response_history[agent]) > 50:
            self.agent_response_history[agent].pop(0)

        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'quarantine': len(issues) > 2
        }

    def get_system_health_report(self) -> dict:
        """Generate overall system health report."""
        total_issues = sum(
            len(h['issues'])
            for history in self.agent_response_history.values()
            for h in history[-10:]
        )

        return {
            'status': 'HEALTHY' if total_issues < 5 else 'DEGRADED' if total_issues < 10 else 'CRITICAL',
            'total_recent_issues': total_issues,
            'agents_monitored': len(self.agent_response_history)
        }
