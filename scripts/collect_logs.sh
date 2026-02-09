#!/bin/bash
# === Coffee Bot Real Options - Log Collection (Environment Configurable) ===
# This script collects logs and pushes them to the logs branch.
#
# CRITICAL: Uses git worktree instead of branch switching so the main repo
# working directory is NEVER modified. This prevents the dashboard and
# orchestrator from losing their files during collection.

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration with defaults
ENV_NAME="${LOG_ENV_NAME:-dev}"
REPO_DIR="${COFFEE_BOT_PATH:-$(pwd)}"
BRANCH="${LOG_BRANCH:-logs}"
WORKTREE_DIR="/tmp/coffee-bot-logs-worktree"

echo "Coffee Bot Log Collection"
echo "Environment: $ENV_NAME"
echo "Repository: $REPO_DIR"
echo "Branch: $BRANCH"
echo "=========================="

# === CLEANUP TRAP: Always remove worktree on exit ===
cleanup() {
    local exit_code=$?
    if [ -d "$WORKTREE_DIR" ]; then
        echo "Cleanup: removing worktree..."
        cd "$REPO_DIR" 2>/dev/null || true
        git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
    fi
    rm -f /tmp/trading-bot-collect.lock
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

# Create our own lock to prevent deploy from running mid-collection
echo "$$" > /tmp/trading-bot-collect.lock

cd "$REPO_DIR" || exit 1

# Prevent Git from using too much RAM on the 4GB droplet
git config pack.windowMemory 512m
git config pack.threads 1

# === SET UP WORKTREE FOR LOGS BRANCH ===
# This keeps the main repo on its current branch (dashboard + orchestrator stay running)

# Clean up any stale worktree from a previous failed run
if [ -d "$WORKTREE_DIR" ]; then
    echo "Removing stale worktree from previous run..."
    git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
fi

# Fetch the logs branch
git fetch origin "$BRANCH" 2>/dev/null || true

# Create worktree — if the logs branch doesn't exist yet, create it as orphan
if git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
    # Branch exists on remote — check it out into the worktree
    # First ensure local branch tracks remote
    git branch -f "$BRANCH" "origin/$BRANCH" 2>/dev/null || true
    git worktree add "$WORKTREE_DIR" "$BRANCH"
else
    echo "Creating new orphan logs branch..."
    git worktree add --detach "$WORKTREE_DIR"
    cd "$WORKTREE_DIR"
    git checkout --orphan "$BRANCH"
    git rm -rf . 2>/dev/null || true
    touch .keep
    git add .keep
    git commit -m "Initial logs branch"
    git push -u origin "$BRANCH"
fi

cd "$WORKTREE_DIR" || exit 1

# Reset to match remote (clean slate)
git reset --hard "origin/$BRANCH" 2>/dev/null || true

DEST_DIR="$WORKTREE_DIR/$ENV_NAME"

# CLEAR & GATHER
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

# === COMMIT & PUSH (in the worktree, not the main repo) ===
cd "$WORKTREE_DIR" || exit 1
git add "$ENV_NAME"

# Only commit if there are staged changes
if git diff --cached --quiet; then
    echo "No changes since last snapshot, skipping commit."
else
    git commit -m "Snapshot $ENV_NAME: $(date +'%Y-%m-%d %H:%M')"
    git push origin "$BRANCH"
    echo "Snapshot pushed to $BRANCH branch."
fi

# Worktree cleanup handled by trap
echo "Successfully collected $ENV_NAME logs!"
