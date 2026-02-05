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

if __name__ == "__main__":
    unittest.main()
