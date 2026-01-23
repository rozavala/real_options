#!/bin/bash
set -e

cd /home/rodrigo/real_options

# Verify critical modules exist before starting
CRITICAL_MODULES=(
    "trading_bot/__init__.py"
    "trading_bot/state_manager.py"
    "trading_bot/agents.py"
    "trading_bot/sentinels.py"
    "trading_bot/connection_pool.py"
)

echo "Verifying critical modules..."
for module in "${CRITICAL_MODULES[@]}"; do
    if [ ! -f "$module" ]; then
        echo "ERROR: Missing critical module: $module"
        exit 1
    fi
done

# Ensure filesystem is synced (important after git pull/deploy)
sync

# Activate venv and start
source venv/bin/activate
echo "Starting orchestrator..."
exec python orchestrator.py
