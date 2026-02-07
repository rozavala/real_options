"""
Safe IB Gateway test wrapper for Claude Code sessions.
Forces development client IDs to prevent production conflicts.

ACTUAL PRODUCTION CLIENT ID MAP (from connection_pool.py + codebase audit):
    90-99    Claude Code dev/testing (THIS SCRIPT)
    100-109  Connection pool: main
    110-119  Connection pool: sentinel
    120-129  Connection pool: emergency
    130-139  Connection pool: monitor
    140-149  Connection pool: orchestrator_orders
    160-169  Connection pool: dashboard_orders
    180-189  Connection pool: dashboard_close
    200-209  Connection pool: microstructure
    220-229  Connection pool: test_utilities
    230-239  Connection pool: audit
    250-259  Connection pool: equity_logger
    260-269  Connection pool: reconciliation
    270-279  Connection pool: cleanup
    300-399  position_monitor.py
    998      verify_system_readiness (connection check)
    999      verify_system_readiness (market data check)
    1000-9999 Dashboard random, equity_logger standalone, reconciliation standalone

    60       .env IB_CLIENT_ID (DEAD CONFIG — loaded but never used by any code)

Usage:
    from tools.safe_ib_test import get_safe_connection
    client_id = get_safe_connection()
"""
import os
import sys
import random

# Development client ID range — safe gap between .env legacy (60) and pool base (100)
DEV_CLIENT_ID_START = 90
DEV_CLIENT_ID_END = 99

# All ranges used by the production system (DO NOT USE)
PRODUCTION_RANGES = [
    (100, 279, "Connection pool (orchestrator, sentinels, dashboard, audit, etc.)"),
    (300, 399, "Position monitor"),
    (998, 999, "Verify system readiness"),
    (1000, 9999, "Dashboard / equity logger / reconciliation (random)"),
]


def get_dev_client_id():
    """Get development client ID. Override with DEV_IB_CLIENT_ID env var."""
    env_id = os.environ.get('DEV_IB_CLIENT_ID')
    if env_id:
        return int(env_id)
    # Randomize within dev range to avoid collisions if multiple test scripts run
    return random.randint(DEV_CLIENT_ID_START, DEV_CLIENT_ID_END)


def validate_client_id(client_id: int) -> bool:
    """Check if a client ID is safe for development use."""
    if DEV_CLIENT_ID_START <= client_id <= DEV_CLIENT_ID_END:
        return True

    for start, end, purpose in PRODUCTION_RANGES:
        if start <= client_id <= end:
            print(f"⛔ REJECTED: Client ID {client_id} collides with {purpose} ({start}-{end})")
            return False

    # Unknown range — warn but allow
    print(f"⚠️  WARNING: Client ID {client_id} is outside all known ranges")
    return False


def verify_safe_to_test():
    """Pre-test safety checks. Returns a validated dev client ID."""
    client_id = get_dev_client_id()

    if not validate_client_id(client_id):
        print(f"⛔ ABORT: Client ID {client_id} is NOT in the safe dev range ({DEV_CLIENT_ID_START}-{DEV_CLIENT_ID_END})")
        sys.exit(1)

    print(f"✅ Using dev client_id={client_id}")
    print(f"   Dev range: {DEV_CLIENT_ID_START}-{DEV_CLIENT_ID_END}")
    print(f"   Production ranges (100-279, 300-399, 998-999, 1000-9999) are protected")
    return client_id


def get_safe_connection():
    """Return a safe development client ID after validation."""
    return verify_safe_to_test()


if __name__ == "__main__":
    client_id = get_safe_connection()
    print(f"\nReady to connect with client_id={client_id}")
    print(f"Example: ib_insync.IB().connect('127.0.0.1', 7497, clientId={client_id})")
