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

# Commodity ticker (default: KC)
COMMODITY="${1:-${COMMODITY_TICKER:-KC}}"
export COMMODITY_TICKER="$COMMODITY"

# Ensure log directory exists
mkdir -p logs

# Ensure data directory exists
mkdir -p "data/$COMMODITY"

# Start
echo "Starting orchestrator from $REPO_ROOT for commodity $COMMODITY..."
exec python -u orchestrator.py --commodity "$COMMODITY"
