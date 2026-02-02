#!/bin/bash
# === Coffee Bot Mission Control - Log Collection (Environment Configurable) ===
# This script collects logs and output files for the specified environment

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration with defaults
ENV_NAME="${LOG_ENV_NAME:-dev}"
REPO_DIR="${COFFEE_BOT_PATH:-$(pwd)}"
DEST_DIR="$REPO_DIR/$ENV_NAME"
BRANCH="${LOG_BRANCH:-logs}"

echo "🚀 Coffee Bot Log Collection"
echo "Environment: $ENV_NAME"
echo "Repository: $REPO_DIR"
echo "Branch: $BRANCH"
echo "=========================="

# 1. CAPTURE ORIGINAL BRANCH (Fix for Bug #1)
# We must do this BEFORE we switch branches
cd "$REPO_DIR" || exit
ORIGINAL_BRANCH=$(git branch --show-current)

if [ -z "$ORIGINAL_BRANCH" ]; then
    echo "⚠️  Could not detect current branch, defaulting to 'main'"
    ORIGINAL_BRANCH="main"
fi
echo "📍 Starting from branch: $ORIGINAL_BRANCH"

# 2. MEMORY SAFETY (Fix for Bug #2)
# Prevent Git from using too much RAM during operations on the 4GB droplet
git config pack.windowMemory 512m
git config pack.threads 1

# 1. Go to repo and get the latest logs branch state
cd "$REPO_DIR" || exit
git fetch origin $BRANCH 2>/dev/null || echo "Creating new logs branch..."
git checkout $BRANCH 2>/dev/null || git checkout --orphan $BRANCH
git reset --hard origin/$BRANCH 2>/dev/null || echo "New branch setup..."

# 2. CLEAR & GATHER
# Wipe the old logs in the env folder so we don't keep junk
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

echo "Gathering logs for $ENV_NAME..."

# === LOG FILES ===
# Specific log files only (not timestamped versions)
if [ -d "$REPO_DIR/logs" ]; then
    echo "Copying specific log files..."
    mkdir -p "$DEST_DIR/logs"
    
    # Copy only the main log files, not timestamped versions
    if [ -f "$REPO_DIR/logs/orchestrator.log" ]; then
        cp "$REPO_DIR/logs/orchestrator.log" "$DEST_DIR/logs/"
    fi
    
    if [ -f "$REPO_DIR/logs/dashboard.log" ]; then
        cp "$REPO_DIR/logs/dashboard.log" "$DEST_DIR/logs/"
    fi
fi

# === DATA FILES ===
# Main data directory check - prevents errors if data folder is missing
if [ -d "$REPO_DIR/data" ]; then
    echo "Copying specific data files..."
    mkdir -p "$DEST_DIR/data"

    # 1. Copy SAFE text-based files (CSV, JSON, LOG)
    # We use simple wildcards to grab relevant files while avoiding heavy binaries
    # 2>/dev/null ensures it doesn't complain if no files of that specific type exist
    cp "$REPO_DIR/data/"*.csv "$DEST_DIR/data/" 2>/dev/null
    cp "$REPO_DIR/data/"*.json "$DEST_DIR/data/" 2>/dev/null
    cp "$REPO_DIR/data/"*.log "$DEST_DIR/data/" 2>/dev/null

    # 2. EXPLICITLY SKIP HEAVY BINARIES
    # We do NOT copy the 'tms' folder or .sqlite3 files here
    echo "Skipping .sqlite3 and TMS binaries to prevent repo bloat."
fi

# === TRADE FILES ===
# Main trade ledger
if [ -f "$REPO_DIR/trade_ledger.csv" ]; then
    echo "Copying trade_ledger.csv..."
    cp "$REPO_DIR/trade_ledger.csv" "$DEST_DIR/"
fi

# Decision signals (Council output summary — replaces old model_signals.csv)
if [ -f "$REPO_DIR/decision_signals.csv" ]; then
    echo "Copying decision_signals.csv..."
    cp "$REPO_DIR/decision_signals.csv" "$DEST_DIR/"
fi

# Archive ledger directory
if [ -d "$REPO_DIR/archive_ledger" ]; then
    echo "Copying archive_ledger directory..."
    cp -r "$REPO_DIR/archive_ledger" "$DEST_DIR/"
fi

# === CONFIGURATION FILES ===
# Config file (sanitized - excludes .env for security)
if [ -f "$REPO_DIR/config.json" ]; then
    echo "Copying config.json..."
    cp "$REPO_DIR/config.json" "$DEST_DIR/"
fi

# === STATE FILES ===
# Root state file
if [ -f "$REPO_DIR/state.json" ]; then
    echo "Copying state.json..."
    cp "$REPO_DIR/state.json" "$DEST_DIR/"
fi

# === PERFORMANCE FILES ===
# Daily performance chart
if [ -f "$REPO_DIR/daily_performance.png" ]; then
    echo "Copying daily_performance.png..."
    cp "$REPO_DIR/daily_performance.png" "$DEST_DIR/"
fi

# === ENVIRONMENT-SPECIFIC REPORTS ===
if [ "$ENV_NAME" = "prod" ]; then
    # === PRODUCTION HEALTH REPORT ===
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
        ps aux | grep -E "(orchestrator|position_monitor)" | grep -v grep
        echo ""
        echo "Dashboard processes:"
        ps aux | grep streamlit | grep -v grep
        echo ""
        
        echo "=== NETWORK STATUS ==="
        echo "Open ports (Trading & Dashboard):"
        netstat -tlnp 2>/dev/null | grep -E ":(8501|7497|7496)" || echo "netstat not available"
        echo ""
        
        echo "=== RECENT ERRORS (if any) ==="
        if [ -f "$REPO_DIR/logs/orchestrator.log" ]; then
            echo "Recent orchestrator errors:"
            grep -i "error\|critical\|exception" "$REPO_DIR/logs/orchestrator.log" | tail -10
        fi
        
    } > "$DEST_DIR/production_health_report.txt"
    
    # === TRADING PERFORMANCE SNAPSHOT ===
    echo "Creating trading performance snapshot..."
    {
        echo "=== TRADING PERFORMANCE SNAPSHOT ==="
        echo "Generated: $(date)"
        echo ""
        
        # Council decisions summary
        if [ -f "$REPO_DIR/data/council_history.csv" ]; then
            echo "=== RECENT COUNCIL DECISIONS (Last 10) ==="
            tail -10 "$REPO_DIR/data/council_history.csv"
            echo ""
        fi
        
        # Equity curve summary
        if [ -f "$REPO_DIR/data/daily_equity.csv" ]; then
            echo "=== RECENT EQUITY DATA (Last 10 days) ==="
            tail -10 "$REPO_DIR/data/daily_equity.csv"
            echo ""
        fi
        
        # Trade ledger summary
        if [ -f "$REPO_DIR/trade_ledger.csv" ]; then
            echo "=== RECENT TRADES (Last 10) ==="
            tail -10 "$REPO_DIR/trade_ledger.csv"
            echo ""
        fi
        
    } > "$DEST_DIR/trading_performance_snapshot.txt"
else
    # === DEVELOPMENT SYSTEM INFO ===
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
        ps aux | grep python | grep -v grep
        echo ""
        echo "Streamlit processes:"
        ps aux | grep streamlit | grep -v grep
        
    } > "$DEST_DIR/system_snapshot.txt"
fi

# Keep the placeholder so the folder exists even if empty
touch "$DEST_DIR/.keep"

# 3. PUSH
git add "$DEST_DIR"
git commit -m "Snapshot $ENV_NAME: $(date +'%Y-%m-%d %H:%M')"
git push origin $BRANCH

# 4. CRITICAL: SWITCH BACK TO ORIGINAL BRANCH
echo "Switching back to original branch: $ORIGINAL_BRANCH"
git checkout "$ORIGINAL_BRANCH"

echo "✅ Successfully collected and pushed logs for $ENV_NAME"
echo "📁 Files included in snapshot:"
echo "   - logs/orchestrator.log and logs/dashboard.log (current only)"
echo "   - data/ directory (council_history.csv, daily_equity.csv, state.json, etc.)"
echo "   - data/tms/ directory (chroma.sqlite3 and TMS files)"
echo "   - trade_ledger.csv (model_signals.csv removed)"
echo "   - archive_ledger/ directory (if exists)"
echo "   - config.json and state.json"
if [ "$ENV_NAME" = "prod" ]; then
    echo "   - production_health_report.txt and trading_performance_snapshot.txt"
else
    echo "   - system_snapshot.txt"
fi
echo "   - log_summary.txt (generated report)"
