#!/bin/bash
# === Coffee Bot Real Options - Log Collection (Environment Configurable) ===
# This script collects logs and output files for the specified environment.
#
# Safety: Uses a cleanup trap to ALWAYS switch back to the original branch,
# even if the script fails mid-execution.

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration with defaults
ENV_NAME="${LOG_ENV_NAME:-dev}"
REPO_DIR="${COFFEE_BOT_PATH:-$(pwd)}"
DEST_DIR="$REPO_DIR/$ENV_NAME"
BRANCH="${LOG_BRANCH:-logs}"
ORIGINAL_BRANCH=""
SWITCHED_BRANCH=false

echo "Coffee Bot Log Collection"
echo "Environment: $ENV_NAME"
echo "Repository: $REPO_DIR"
echo "Branch: $BRANCH"
echo "=========================="

# === CLEANUP TRAP: Always switch back to original branch ===
cleanup() {
    local exit_code=$?
    if [ "$SWITCHED_BRANCH" = true ] && [ -n "$ORIGINAL_BRANCH" ]; then
        echo "Cleanup: switching back to $ORIGINAL_BRANCH..."
        cd "$REPO_DIR" 2>/dev/null || true
        # Discard any uncommitted changes from the copy phase
        git checkout -- . 2>/dev/null || true
        git clean -fd "$DEST_DIR" 2>/dev/null || true
        git checkout "$ORIGINAL_BRANCH" 2>/dev/null || git checkout main 2>/dev/null || true
    fi
    exit $exit_code
}
trap cleanup EXIT

# Skip if deploy is in progress
DEPLOY_LOCK="/tmp/trading-bot-deploy.lock"
if [ -f "$DEPLOY_LOCK" ]; then
    LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$DEPLOY_LOCK") ))
    LOCK_PID=$(cat "$DEPLOY_LOCK" 2>/dev/null)
    if [ "$LOCK_AGE" -gt 1800 ]; then
        echo "Deploy lock is ${LOCK_AGE}s old (>30min), treating as stale"
        rm -f "$DEPLOY_LOCK"
    elif kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "Deploy in progress (PID $LOCK_PID, age ${LOCK_AGE}s), skipping"
        exit 0
    else
        echo "Stale deploy lock found (process gone), removing"
        rm -f "$DEPLOY_LOCK"
    fi
fi

# Create our own lock to prevent deploy from running git checkout mid-collection
COLLECT_LOCK="/tmp/trading-bot-collect.lock"
echo "$$" > "$COLLECT_LOCK"
# Remove collect lock on exit (in addition to branch cleanup above)
original_cleanup=$(trap -p EXIT | sed "s/trap -- '\(.*\)' EXIT/\1/")
trap "$original_cleanup; rm -f '$COLLECT_LOCK'" EXIT

cd "$REPO_DIR" || exit 1
ORIGINAL_BRANCH=$(git branch --show-current)

if [ -z "$ORIGINAL_BRANCH" ]; then
    echo "WARNING: Could not detect current branch, defaulting to 'main'"
    ORIGINAL_BRANCH="main"
fi
echo "Starting from branch: $ORIGINAL_BRANCH"

# Self-heal: if we're already on the logs branch (from a previous failed run),
# discard any dirty state before proceeding
if [ "$ORIGINAL_BRANCH" = "$BRANCH" ]; then
    echo "WARNING: Already on '$BRANCH' branch (previous run may have failed). Self-healing..."
    git checkout -- . 2>/dev/null || true
    git clean -fd 2>/dev/null || true
    ORIGINAL_BRANCH="main"
fi

# Prevent Git from using too much RAM on the 4GB droplet
git config pack.windowMemory 512m
git config pack.threads 1

# Fetch and switch to the logs branch
git fetch origin "$BRANCH" 2>/dev/null || echo "Creating new logs branch..."
git checkout "$BRANCH" 2>/dev/null || git checkout --orphan "$BRANCH"
SWITCHED_BRANCH=true
git reset --hard "origin/$BRANCH" 2>/dev/null || echo "New branch setup..."

# CLEAR & GATHER
# Wipe the old logs in the env folder so we don't keep junk
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

echo "Gathering logs for $ENV_NAME..."

# === LOG FILES ===
if [ -d "$REPO_DIR/logs" ]; then
    echo "Copying specific log files..."
    mkdir -p "$DEST_DIR/logs"

    for logfile in "orchestrator.log" "dashboard.log" "manual_test.log" "performance_analyzer.log" "equity_logger.log"; do
        if [ -f "$REPO_DIR/logs/$logfile" ]; then
            cp "$REPO_DIR/logs/$logfile" "$DEST_DIR/logs/"
        fi
    done
fi

# === DATA FILES ===
if [ -d "$REPO_DIR/data" ]; then
    echo "Copying specific data files..."
    mkdir -p "$DEST_DIR/data"

    cp "$REPO_DIR/data/"*.csv "$DEST_DIR/data/" 2>/dev/null || true
    cp "$REPO_DIR/data/"*.json "$DEST_DIR/data/" 2>/dev/null || true

    for f in "$REPO_DIR/data/"*.log; do
        if [ -f "$f" ]; then
            filename=$(basename "$f")
            if [[ ! "$filename" =~ -202[0-9]- ]]; then
                cp "$f" "$DEST_DIR/data/"
            fi
        fi
    done

    echo "Skipping .sqlite3 and TMS binaries to prevent repo bloat."
fi

# === TRADE FILES ===
if [ -f "$REPO_DIR/trade_ledger.csv" ]; then
    echo "Copying trade_ledger.csv..."
    cp "$REPO_DIR/trade_ledger.csv" "$DEST_DIR/"
fi

if [ -f "$REPO_DIR/decision_signals.csv" ]; then
    echo "Copying decision_signals.csv..."
    cp "$REPO_DIR/decision_signals.csv" "$DEST_DIR/"
fi

if [ -d "$REPO_DIR/archive_ledger" ]; then
    echo "Copying archive_ledger directory..."
    cp -r "$REPO_DIR/archive_ledger" "$DEST_DIR/"
fi

# === CONFIGURATION FILES ===
if [ -f "$REPO_DIR/config.json" ]; then
    echo "Copying config.json..."
    cp "$REPO_DIR/config.json" "$DEST_DIR/"
fi

# === STATE FILES ===
if [ -f "$REPO_DIR/state.json" ]; then
    echo "Copying state.json..."
    cp "$REPO_DIR/state.json" "$DEST_DIR/"
fi

# === PERFORMANCE FILES ===
if [ -f "$REPO_DIR/daily_performance.png" ]; then
    echo "Copying daily_performance.png..."
    cp "$REPO_DIR/daily_performance.png" "$DEST_DIR/"
fi

# === ENVIRONMENT-SPECIFIC REPORTS ===
if [ "$ENV_NAME" = "prod" ]; then
    echo "Creating production health report..."
    {
        echo "=== PRODUCTION HEALTH REPORT ==="
        echo "Timestamp: $(date)"
        echo "Environment: $ENV_NAME"
        echo "Hostname: $(hostname)"
        echo "Uptime: $(uptime)"
        echo ""

        echo "=== DISK USAGE ==="
        df -h
        echo ""

        echo "=== MEMORY USAGE ==="
        free -h
        echo ""

        echo "=== LOAD AVERAGE ==="
        cat /proc/loadavg
        echo ""

        echo "=== PROCESS STATUS ==="
        echo "Trading processes:"
        ps aux | grep -E "(orchestrator|position_monitor)" | grep -v grep || true
        echo ""
        echo "Dashboard processes:"
        ps aux | grep streamlit | grep -v grep || true
        echo ""

        echo "=== NETWORK STATUS ==="
        echo "Open ports (Trading & Dashboard):"
        netstat -tlnp 2>/dev/null | grep -E ":(8501|7497|7496)" || echo "netstat not available"
        echo ""

        echo "=== RECENT ERRORS (if any) ==="
        if [ -f "$REPO_DIR/logs/orchestrator.log" ]; then
            echo "Recent orchestrator errors:"
            grep -i "error\|critical\|exception" "$REPO_DIR/logs/orchestrator.log" | tail -10 || true
        fi

    } > "$DEST_DIR/production_health_report.txt"

    echo "Creating trading performance snapshot..."
    {
        echo "=== TRADING PERFORMANCE SNAPSHOT ==="
        echo "Generated: $(date)"
        echo ""

        if [ -f "$REPO_DIR/data/council_history.csv" ]; then
            echo "=== RECENT COUNCIL DECISIONS (Last 10) ==="
            tail -10 "$REPO_DIR/data/council_history.csv"
            echo ""
        fi

        if [ -f "$REPO_DIR/data/daily_equity.csv" ]; then
            echo "=== RECENT EQUITY DATA (Last 10 days) ==="
            tail -10 "$REPO_DIR/data/daily_equity.csv"
            echo ""
        fi

        if [ -f "$REPO_DIR/trade_ledger.csv" ]; then
            echo "=== RECENT TRADES (Last 10) ==="
            tail -10 "$REPO_DIR/trade_ledger.csv"
            echo ""
        fi

    } > "$DEST_DIR/trading_performance_snapshot.txt"
else
    echo "Creating system snapshot..."
    {
        echo "=== SYSTEM SNAPSHOT FOR $ENV_NAME ==="
        echo "Timestamp: $(date)"
        echo "Hostname: $(hostname)"
        echo "Uptime: $(uptime)"
        echo ""
        echo "=== DISK USAGE ==="
        df -h
        echo ""
        echo "=== MEMORY USAGE ==="
        free -h
        echo ""
        echo "=== PROCESS STATUS ==="
        echo "Python processes:"
        ps aux | grep python | grep -v grep || true
        echo ""
        echo "Streamlit processes:"
        ps aux | grep streamlit | grep -v grep || true

    } > "$DEST_DIR/system_snapshot.txt"
fi

touch "$DEST_DIR/.keep"

# === COMMIT & PUSH ===
git add "$DEST_DIR"

# Only commit if there are staged changes (prevents "nothing to commit" failure)
if git diff --cached --quiet; then
    echo "No changes since last snapshot, skipping commit."
else
    git commit -m "Snapshot $ENV_NAME: $(date +'%Y-%m-%d %H:%M')"
    git push origin "$BRANCH"
    echo "Snapshot pushed to $BRANCH branch."
fi

# Switch back (also handled by trap, but do it explicitly for clean exit message)
SWITCHED_BRANCH=false
echo "Switching back to original branch: $ORIGINAL_BRANCH"
git checkout "$ORIGINAL_BRANCH"

echo "Successfully collected $ENV_NAME logs!"
