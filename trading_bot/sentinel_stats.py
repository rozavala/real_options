"""
Sentinel Statistics Collector.

v3.1: Tracks sentinel performance for dashboard display.
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)

STATS_FILE = Path(os.path.join("data", os.environ.get("COMMODITY_TICKER", "KC"), "sentinel_stats.json"))


def set_data_dir(data_dir: str):
    """Configure sentinel stats path for a commodity-specific data directory.

    Also reloads in-memory stats from the new path.
    """
    global STATS_FILE
    STATS_FILE = Path(data_dir) / "sentinel_stats.json"
    # Reload stats from new path into the module-level singleton
    SENTINEL_STATS.stats = SENTINEL_STATS._load_stats()
    logger.info(f"SentinelStats data_dir set to: {data_dir}")


def _get_stats_file() -> Path:
    """Resolve stats file via ContextVar (multi-engine) or module global (legacy)."""
    try:
        from trading_bot.data_dir_context import get_engine_data_dir
        return Path(get_engine_data_dir()) / "sentinel_stats.json"
    except LookupError:
        return STATS_FILE


class SentinelStats:
    """Collect and expose sentinel performance statistics.

    In multi-engine mode, multiple engines share this singleton but write to
    different files (via ContextVar-resolved _get_stats_file()). To prevent
    cross-engine data contamination, every mutating operation reloads from
    disk first (read-modify-write pattern).
    """

    def __init__(self):
        self.stats = self._load_stats()

    def _load_stats(self) -> dict:
        """Load stats from file."""
        stats_file = _get_stats_file()
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sentinel stats: {e}")
        return {'sentinels': {}, 'last_updated': None}

    def _save_stats(self):
        """Save stats to file."""
        stats_file = _get_stats_file()
        self.stats['last_updated'] = datetime.now(timezone.utc).isoformat()
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def _reload(self):
        """Reload stats from disk for the current engine's data dir.

        Prevents cross-engine contamination when a singleton is shared
        across CommodityEngines in multi-engine mode.
        """
        self.stats = self._load_stats()

    def record_alert(self, sentinel_name: str, triggered_trade: bool):
        """
        Record a sentinel alert.

        Args:
            sentinel_name: Name of the sentinel
            triggered_trade: Whether alert led to trade execution
        """
        self._reload()
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

    def record_error(self, sentinel_name: str, error_type: str):
        """
        Record a sentinel error (timeout, crash, API failure, etc.).

        Called by:
        - All 8 sentinel timeout handlers in run_sentinels() (Fix 0)
        - _emergency_cycle_done_callback() crash handler (Fix 8)

        Args:
            sentinel_name: Name of the sentinel (e.g., 'WeatherSentinel')
            error_type: Type of error (e.g., 'TIMEOUT', 'CRASH: ValueError')
        """
        self._reload()
        if sentinel_name not in self.stats['sentinels']:
            self.stats['sentinels'][sentinel_name] = {
                'total_alerts': 0,
                'trades_triggered': 0,
                'last_alert': None,
                'daily_counts': {},
                'errors': {}
            }

        s = self.stats['sentinels'][sentinel_name]

        # Ensure 'errors' key exists (backward compat with pre-existing stats)
        if 'errors' not in s:
            s['errors'] = {}

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        s['errors'][today] = s['errors'].get(today, 0) + 1

        self._save_stats()

    def get_all(self) -> dict:
        """
        Get raw stats dict for all sentinels.

        Called by sentinel_effectiveness_check() (Fix 7) which reads:
        - s.get('total_alerts', 0)
        - s.get('trades_triggered', 0)

        Returns:
            Dict of {sentinel_name: stats_dict}
        """
        self._reload()
        return self.stats.get('sentinels', {})

    def get_dashboard_stats(self) -> dict:
        """Get stats formatted for dashboard display."""
        self._reload()
        result = {}
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        for name, data in self.stats.get('sentinels', {}).items():
            total = data['total_alerts']
            trades = data['trades_triggered']

            result[name] = {
                'total_alerts': total,
                'trades_triggered': trades,
                'conversion_rate': trades / total if total > 0 else 0,
                'last_alert': data['last_alert'],
                'alerts_today': data['daily_counts'].get(today, 0),
                'errors_today': data.get('errors', {}).get(today, 0)
            }

        return result


# Global instance
SENTINEL_STATS = SentinelStats()
