#!/bin/bash
set -e

# Detect Repo Root
if [ -f "pyproject.toml" ]; then
    REPO_ROOT=$(pwd)
elif [ -f "../pyproject.toml" ]; then
    REPO_ROOT=$(dirname $(pwd))
else
    REPO_ROOT=~/real_options
fi

cd "$REPO_ROOT"

# ===========================================================================
# CRITICAL_MODULES — Preflight existence check
#
# This list MUST be updated whenever a new module becomes load-bearing.
# Modules are checked in dependency order: configs → prompts → core → extensions
# ===========================================================================
CRITICAL_MODULES=(
    # === Foundation (existing) ===
    "trading_bot/__init__.py"
    "trading_bot/state_manager.py"
    "trading_bot/agents.py"
    "trading_bot/sentinels.py"
    "trading_bot/connection_pool.py"
    "trading_bot/tms.py"
    "trading_bot/brier_scoring.py"
    "trading_bot/heterogeneous_router.py"
    "trading_bot/weighted_voting.py"
    "trading_bot/semantic_router.py"

    # === HRO Phase 1: Commodity Profiles ===
    "config/__init__.py"
    "config/commodity_profiles.py"

    # === HRO Phase 1: Templatized Prompts ===
    "trading_bot/prompts/__init__.py"
    "trading_bot/prompts/base_prompts.py"
)

# Optional modules — logged as warnings, not failures
OPTIONAL_MODULES=(
    # === HRO Phase 3: Backtesting (not needed for live trading) ===
    "backtesting/__init__.py"
    "backtesting/simple_backtest.py"
    "backtesting/surrogate_models.py"

    # === HRO Phase 4-5: Enhancements ===
    "trading_bot/enhanced_brier.py"
    "trading_bot/observability.py"
    "trading_bot/budget_guard.py"
)

echo "Verifying critical modules..."
MISSING=0
for module in "${CRITICAL_MODULES[@]}"; do
    if [ ! -f "$module" ]; then
        echo "  ❌ MISSING critical module: $module"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo "ERROR: $MISSING critical module(s) missing. ABORTING."
    exit 1
fi
echo "  ✅ All ${#CRITICAL_MODULES[@]} critical modules present"

echo "Checking optional modules..."
for module in "${OPTIONAL_MODULES[@]}"; do
    if [ ! -f "$module" ]; then
        echo "  ⚠️  Optional module not yet deployed: $module"
    fi
done

# Ensure filesystem is synced (important after git pull/deploy)
sync

# Activate venv and start
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Starting orchestrator..."
exec python -u orchestrator.py
