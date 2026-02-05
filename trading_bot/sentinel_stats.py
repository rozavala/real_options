"""
Sentinel Statistics Collector.

v3.1: Tracks sentinel performance for dashboard display.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

STATS_FILE = Path("data/sentinel_stats.json")


class SentinelStats:
    """Collect and expose sentinel performance statistics."""

    def __init__(self):
        self.stats = self._load_stats()

    def _load_stats(self) -> dict:
        """Load stats from file."""
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'sentinels': {}, 'last_updated': None}

    def _save_stats(self):
        """Save stats to file."""
        self.stats['last_updated'] = datetime.now(timezone.utc).isoformat()
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATS_FILE, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def record_alert(self, sentinel_name: str, triggered_trade: bool):
        """
        Record a sentinel alert.

        Args:
            sentinel_name: Name of the sentinel
            triggered_trade: Whether alert led to trade execution
        """
        if sentinel_name not in self.stats['sentinels']:
            self.stats['sentinels'][sentinel_name] = {
                'total_alerts': 0,
                'trades_triggered': 0,
                'last_alert': None,
                'daily_counts': {}
            }

        s = self.stats['sentinels'][sentinel_name]
        s['total_alerts'] += 1
        if triggered_trade:
            s['trades_triggered'] += 1
        s['last_alert'] = datetime.now(timezone.utc).isoformat()

        # Daily tracking
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        s['daily_counts'][today] = s['daily_counts'].get(today, 0) + 1

        self._save_stats()

    def get_dashboard_stats(self) -> dict:
        """Get stats formatted for dashboard display."""
        result = {}

        for name, data in self.stats.get('sentinels', {}).items():
            total = data['total_alerts']
            trades = data['trades_triggered']

            result[name] = {
                'total_alerts': total,
                'trades_triggered': trades,
                'conversion_rate': trades / total if total > 0 else 0,
                'last_alert': data['last_alert'],
                'alerts_today': data['daily_counts'].get(
                    datetime.now(timezone.utc).strftime('%Y-%m-%d'), 0
                )
            }

        return result


# Global instance
SENTINEL_STATS = SentinelStats()
