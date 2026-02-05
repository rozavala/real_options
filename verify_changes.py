import asyncio
import sys
import os

sys.path.append(os.getcwd())

async def verify():
    print("Verifying imports...")
    try:
        from trading_bot.order_queue import OrderQueueManager
        oq = OrderQueueManager()
        await oq.add("test")
        print(f"OrderQueueManager: OK (len={len(oq)})")

        from trading_bot.sentinel_stats import SENTINEL_STATS
        SENTINEL_STATS.record_alert("TestSentinel", False)
        print("SentinelStats: OK")

        from config import get_active_profile
        profile = get_active_profile({'symbol': 'KC'})
        print(f"CommodityProfile: OK ({profile.name})")

        from trading_bot.compliance import log_compliance_decision
        print("Compliance Logging: OK")

        from trading_bot.state_manager import StateManager
        # Just check method existence
        assert hasattr(StateManager, 'migrate_legacy_entries')
        print("StateManager: OK")

        print("ALL CHECKS PASSED")
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify())
