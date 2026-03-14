#!/bin/bash
set -e

# Detect Repo Root
# I3 FIX: Use environment variable with sensible default
REPO_ROOT="${TRADING_BOT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

if [ ! -d "$REPO_ROOT" ]; then
    echo "ERROR: REPO_ROOT not found: $REPO_ROOT"
    echo "Set TRADING_BOT_ROOT environment variable or run from repo directory."
    exit 1
fi

cd "$REPO_ROOT"

# Detect mode: LEGACY_MODE=true runs single-commodity, otherwise MasterOrchestrator
LEGACY_MODE="${LEGACY_MODE:-false}"

# Ensure log directory exists
mkdir -p logs

if [ "$LEGACY_MODE" = "true" ]; then
    # Legacy: single-commodity mode
    COMMODITY="${1:-${COMMODITY_TICKER:-KC}}"
    export COMMODITY_TICKER="$COMMODITY"
    mkdir -p "data/$COMMODITY"
    echo "Starting orchestrator from $REPO_ROOT for commodity $COMMODITY (legacy mode)..."
    exec python -u orchestrator.py --commodity "$COMMODITY"
else
    # Default: MasterOrchestrator (all active commodities in one process)
    echo "Starting MasterOrchestrator from $REPO_ROOT..."
    exec python -u orchestrator.py
fi
