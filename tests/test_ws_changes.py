import asyncio
import unittest
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from trading_bot.order_queue import OrderQueueManager
from trading_bot.sentinel_stats import SentinelStats, STATS_FILE
import json

class TestWSChanges(unittest.IsolatedAsyncioTestCase):
    async def test_order_queue(self):
        oq = OrderQueueManager()
        self.assertTrue(oq.is_empty())
        await oq.add("item1")
        self.assertFalse(oq.is_empty())
        self.assertEqual(len(oq), 1)

        items = await oq.pop_all()
        self.assertEqual(items, ["item1"])
        self.assertTrue(oq.is_empty())

    def test_sentinel_stats(self):
        # Clean up stats file
        if STATS_FILE.exists():
            os.remove(STATS_FILE)

        stats = SentinelStats()
        stats.record_alert("TestSentinel", True)

        dashboard = stats.get_dashboard_stats()
        self.assertIn("TestSentinel", dashboard)
        self.assertEqual(dashboard["TestSentinel"]["total_alerts"], 1)
        self.assertEqual(dashboard["TestSentinel"]["trades_triggered"], 1)
        self.assertEqual(dashboard["TestSentinel"]["conversion_rate"], 1.0)

    def test_sentinel_stats_record_error(self):
        """Test record_error method exists and works."""
        if STATS_FILE.exists():
            os.remove(STATS_FILE)

        stats = SentinelStats()
        stats.record_error("TestSentinel", "TIMEOUT")
        stats.record_error("TestSentinel", "TIMEOUT")

        raw = stats.get_all()
        self.assertIn("TestSentinel", raw)
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.assertEqual(raw["TestSentinel"]["errors"][today], 2)
        self.assertEqual(raw["TestSentinel"]["total_alerts"], 0)  # Errors != alerts

        # Check dashboard stats
        dashboard = stats.get_dashboard_stats()
        self.assertEqual(dashboard["TestSentinel"]["errors_today"], 2)

    def test_sentinel_stats_get_all(self):
        """Test get_all returns raw sentinel dict."""
        if STATS_FILE.exists():
            os.remove(STATS_FILE)

        stats = SentinelStats()
        stats.record_alert("Alpha", True)
        stats.record_alert("Beta", False)

        raw = stats.get_all()
        self.assertIn("Alpha", raw)
        self.assertIn("Beta", raw)
        self.assertEqual(raw["Alpha"]["trades_triggered"], 1)
        self.assertEqual(raw["Beta"]["trades_triggered"], 0)

    def test_sentinel_stats_backward_compat(self):
        """Test that record_error works on entries created by record_alert (no 'errors' key)."""
        if STATS_FILE.exists():
            os.remove(STATS_FILE)

        stats = SentinelStats()
        stats.record_alert("Legacy", True)  # Creates entry without 'errors' key

        # Remove 'errors' key to simulate pre-existing data
        # Note: In the new implementation record_alert doesn't add 'errors' key,
        # but let's be explicit about removing it if it was added implicitly
        if 'errors' in stats.stats['sentinels']['Legacy']:
            del stats.stats['sentinels']['Legacy']['errors']
        stats._save_stats()

        # Reload and record error — should not crash
        stats2 = SentinelStats()
        stats2.record_error("Legacy", "TIMEOUT")

        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.assertEqual(stats2.stats['sentinels']['Legacy']['errors'][today], 1)

if __name__ == "__main__":
    unittest.main()
