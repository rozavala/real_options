#!/bin/bash
# === Coffee Bot Real Options - Log Collection System Installer ===
# This script automates the setup of the repository-based log collection system

set -e  # Exit on any error

echo "☕ Coffee Bot Real Options - Log Collection System Installer"
echo "================================================================"

# Check if we're in the right directory
if [ ! -f "orchestrator.py" ] || [ ! -f "dashboard.py" ]; then
    echo "❌ Error: Please run this script from your Coffee Bot repository root"
    echo "   Expected files: orchestrator.py, dashboard.py"
    exit 1
fi

REPO_DIR=$(pwd)
echo "📁 Repository: $REPO_DIR"

# 1. Create scripts directory
echo ""
echo "📁 Creating scripts directory..."
mkdir -p scripts

# 2. Create collect_logs.sh
echo "📝 Creating collect_logs.sh..."
cat > scripts/collect_logs.sh << 'EOF'
#!/bin/bash
set -e
# === Coffee Bot Real Options - Log Collection (Environment Configurable) ===
# This script collects logs and pushes them to the logs branch.
#
# CRITICAL: Uses git worktree instead of branch switching so the main repo
# working directory is NEVER modified. This prevents the dashboard and
# orchestrator from losing their files during collection.

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs) || true
fi

# Configuration with defaults
ENV_NAME="${LOG_ENV_NAME:-dev}"
REPO_DIR="${COFFEE_BOT_PATH:-$(pwd)}"
BRANCH="${LOG_BRANCH:-logs}"
TICKER="${COMMODITY_TICKER:-KC}"
TICKER_LOWER=$(echo "$TICKER" | tr '[:upper:]' '[:lower:]')
DATA_DIR="$REPO_DIR/data/$TICKER"
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

    for logfile in "orchestrator_${TICKER_LOWER}.log" "dashboard_${TICKER_LOWER}.log" "manual_test.log" "performance_analyzer.log" "equity_logger.log"; do
        if [ -f "$REPO_DIR/logs/$logfile" ]; then
            cp "$REPO_DIR/logs/$logfile" "$DEST_DIR/logs/"
        fi
    done
fi

# === DATA FILES ===
if [ -d "$REPO_DIR/data" ]; then
    echo "Copying specific data files..."
    mkdir -p "$DEST_DIR/data"

    # Copy from per-commodity data directory
    if [ -d "$DATA_DIR" ]; then
        mkdir -p "$DEST_DIR/data/$TICKER"
        cp "$DATA_DIR/"*.csv "$DEST_DIR/data/$TICKER/" 2>/dev/null || true
        cp "$DATA_DIR/"*.json "$DEST_DIR/data/$TICKER/" 2>/dev/null || true
    fi
    # Also copy any legacy flat data/ files
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
# Check per-commodity directory first, then legacy project root
if [ -f "$DATA_DIR/trade_ledger.csv" ]; then
    echo "Copying trade_ledger.csv from $DATA_DIR..."
    cp "$DATA_DIR/trade_ledger.csv" "$DEST_DIR/"
elif [ -f "$REPO_DIR/trade_ledger.csv" ]; then
    echo "Copying trade_ledger.csv from project root (legacy)..."
    cp "$REPO_DIR/trade_ledger.csv" "$DEST_DIR/"
fi

if [ -f "$DATA_DIR/decision_signals.csv" ]; then
    echo "Copying decision_signals.csv from $DATA_DIR..."
    cp "$DATA_DIR/decision_signals.csv" "$DEST_DIR/"
elif [ -f "$REPO_DIR/decision_signals.csv" ]; then
    echo "Copying decision_signals.csv from project root (legacy)..."
    cp "$REPO_DIR/decision_signals.csv" "$DEST_DIR/"
fi

if [ -d "$DATA_DIR/archive_ledger" ]; then
    echo "Copying archive_ledger directory from $DATA_DIR..."
    cp -r "$DATA_DIR/archive_ledger" "$DEST_DIR/"
elif [ -d "$REPO_DIR/archive_ledger" ]; then
    echo "Copying archive_ledger directory from project root (legacy)..."
    cp -r "$REPO_DIR/archive_ledger" "$DEST_DIR/"
fi

# === CONFIGURATION FILES ===
if [ -f "$REPO_DIR/config.json" ]; then
    echo "Copying config.json (redacted)..."
    # Redact sensitive keys before saving to logs
    python3 -c "import json,sys,re; r=lambda o: {k:(r(v) if not re.search(r'(key|token|secret|password|sig)',k,re.I) else '[REDACTED]') for k,v in o.items()} if isinstance(o,dict) else [r(x) for x in o] if isinstance(o,list) else o; json.dump(r(json.load(sys.stdin)), sys.stdout, indent=2)" < "$REPO_DIR/config.json" > "$DEST_DIR/config.json"
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
        if [ -f "$REPO_DIR/logs/orchestrator_${TICKER_LOWER}.log" ]; then
            echo "Recent orchestrator errors:"
            grep -i "error\|critical\|exception" "$REPO_DIR/logs/orchestrator_${TICKER_LOWER}.log" | tail -10 || true
        fi

    } > "$DEST_DIR/production_health_report.txt"

    echo "Creating trading performance snapshot..."
    {
        echo "=== TRADING PERFORMANCE SNAPSHOT ==="
        echo "Generated: $(date)"
        echo ""

        if [ -f "$DATA_DIR/council_history.csv" ]; then
            echo "=== RECENT COUNCIL DECISIONS (Last 10) ==="
            tail -10 "$DATA_DIR/council_history.csv"
            echo ""
        fi

        if [ -f "$DATA_DIR/daily_equity.csv" ]; then
            echo "=== RECENT EQUITY DATA (Last 10 days) ==="
            tail -10 "$DATA_DIR/daily_equity.csv"
            echo ""
        fi

        if [ -f "$DATA_DIR/trade_ledger.csv" ]; then
            echo "=== RECENT TRADES (Last 10) ==="
            tail -10 "$DATA_DIR/trade_ledger.csv"
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
    # Retry push with rebase if another env pushed since our fetch
    for attempt in 1 2 3; do
        if git push origin "$BRANCH" 2>&1; then
            echo "Snapshot pushed to $BRANCH branch."
            break
        else
            echo "Push failed (attempt $attempt/3), rebasing and retrying..."
            git fetch origin "$BRANCH"
            git rebase "origin/$BRANCH"
        fi
    done
fi

# Worktree cleanup handled by trap
echo "Successfully collected $ENV_NAME logs!"
EOF

# 3. Create log_analysis.sh
echo "📝 Creating log_analysis.sh..."
cat > scripts/log_analysis.sh << 'EOF'
#!/bin/bash
set -e
# === Coffee Bot Real Options - Log Analysis Utilities ===
# This script provides utilities for analyzing collected logs and managing
# the logs branch effectively.

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

REPO_DIR="${COFFEE_BOT_PATH:-$(pwd)}"
BRANCH="${LOG_BRANCH:-logs}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper: Capture original branch
capture_original_branch() {
    ORIGINAL_BRANCH=$(git branch --show-current)
    if [ -z "$ORIGINAL_BRANCH" ]; then
        echo "⚠️  Could not detect current branch, defaulting to 'main'"
        ORIGINAL_BRANCH="main"
    fi
    echo "📍 Starting from branch: $ORIGINAL_BRANCH"
}

# Helper: Switch back to original branch
return_to_original_branch() {
    echo "Switching back to original branch: $ORIGINAL_BRANCH"
    git checkout "$ORIGINAL_BRANCH" > /dev/null 2>&1
}

usage() {
    echo "☕ Coffee Bot Real Options - Log Analysis Utilities"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  status                 Show current logs branch status"
    echo "  analyze ENV            Analyze logs for environment (dev/prod)"
    echo "  compare                Compare dev vs prod performance"
    echo "  health ENV             Show health report for environment"
    echo "  errors ENV [HOURS]     Show recent errors (default: 24 hours)"
    echo "  performance ENV        Show trading performance summary"
    echo "  collect ENV            Collect logs for environment"
    echo "  cleanup DAYS           Remove snapshots older than X days"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 analyze prod"
    echo "  $0 errors dev 12"
    echo "  $0 collect prod"
    echo ""
}

init_logs_branch() {
    cd "$REPO_DIR" || { echo -e "${RED}❌ Repository not found: $REPO_DIR${NC}"; exit 1; }

    # CRITICAL: Capture original branch BEFORE switching
    capture_original_branch

    # Switch to logs branch
    if ! git checkout $BRANCH > /dev/null 2>&1; then
        echo -e "${RED}❌ Cannot switch to logs branch. Run setup_logs_infrastructure.sh first.${NC}"
        exit 1
    fi

    # Pull latest
    git pull origin $BRANCH > /dev/null 2>&1
}

show_status() {
    echo -e "${BLUE}📊 Coffee Bot Logs Branch Status${NC}"
    echo "=================================="
    echo "Repository: $REPO_DIR"
    echo "Branch: $BRANCH"

    echo ""
    echo -e "${YELLOW}📅 Recent Snapshots:${NC}"
    git log --oneline --graph -10

    echo ""
    echo -e "${YELLOW}📁 Environment Status:${NC}"
    for env in dev prod; do
        if [ -d "$env" ]; then
            file_count=$(find "$env" -type f | wc -l)
            size=$(du -sh "$env" 2>/dev/null | cut -f1)

            echo "  $env: $file_count files, $size total"

            # Check for recent snapshot
            if [ -f "$env/system_snapshot.txt" ]; then
                last_snapshot=$(grep "Timestamp:" "$env/system_snapshot.txt" | cut -d: -f2-)
                echo "    Last snapshot:$last_snapshot"
            elif [ -f "$env/production_health_report.txt" ]; then
                last_snapshot=$(grep "Timestamp:" "$env/production_health_report.txt" | cut -d: -f2-)
                echo "    Last snapshot:$last_snapshot"
            fi
        else
            echo "  $env: No data"
        fi
    done

    echo ""
    echo -e "${YELLOW}💾 Branch Size:${NC}"
    total_size=$(du -sh . 2>/dev/null | cut -f1)
    echo "  Total: $total_size"
}

analyze_environment() {
    local env=$1

    if [ ! -d "$env" ]; then
        echo -e "${RED}❌ Environment '$env' not found${NC}"
        return 1
    fi
    
    echo -e "${BLUE}🔍 Analyzing $env Environment${NC}"
    echo "=============================="

    # System Health
    local health_file=""
    if [ "$env" = "prod" ] && [ -f "$env/production_health_report.txt" ]; then
        health_file="$env/production_health_report.txt"
    elif [ -f "$env/system_snapshot.txt" ]; then
        health_file="$env/system_snapshot.txt"
    fi

    if [ -n "$health_file" ]; then
        echo ""
        echo -e "${YELLOW}🖥️  System Health:${NC}"
        grep -A 3 "=== MEMORY USAGE ===" "$health_file" | tail -3
        grep -A 3 "=== DISK USAGE ===" "$health_file" | tail -3

        echo ""
        echo -e "${YELLOW}⚙️  Process Status:${NC}"
        grep -A 10 "=== PROCESS STATUS ===" "$health_file" | grep -v "==="
    fi

    # Trading Performance
    if [ -f "$env/data/council_history.csv" ]; then
        echo ""
        echo -e "${YELLOW}📈 Trading Activity:${NC}"
        local decisions=$(tail -n +2 "$env/data/council_history.csv" | wc -l)
        echo "  Total decisions: $decisions"

        if [ $decisions -gt 0 ]; then
            local recent_decisions=$(tail -5 "$env/data/council_history.csv" | wc -l)
            echo "  Recent decisions: $recent_decisions"
        fi
    fi

    # Log Analysis
    if [ -d "$env/logs" ]; then
        echo ""
        echo -e "${YELLOW}📋 Log Summary:${NC}"
        for logfile in "$env/logs"/*.log; do
            if [ -f "$logfile" ]; then
                filename=$(basename "$logfile")
                lines=$(wc -l < "$logfile")
                errors=$(grep -c -i "error\|exception\|critical" "$logfile" 2>/dev/null || echo "0")
                echo "  $filename: $lines lines, $errors errors"
            fi
        done
    fi
}

show_errors() {
    local env=$1
    local hours=${2:-24}

    if [ ! -d "$env" ]; then
        echo -e "${RED}❌ Environment '$env' not found${NC}"
        return 1
    fi

    echo -e "${BLUE}🚨 Recent Errors ($env - Last $hours hours)${NC}"
    echo "========================================="

    if [ -d "$env/logs" ]; then
        for logfile in "$env/logs"/*.log; do
            if [ -f "$logfile" ]; then
                filename=$(basename "$logfile")
                echo ""
                echo -e "${YELLOW}📄 $filename:${NC}"

                # Look for errors
                grep -i "error\|exception\|critical" "$logfile" | tail -10 | while read -r line; do
                    echo -e "  ${RED}●${NC} $line"
                done
            fi
        done
    else
        echo "No logs directory found for $env"
    fi
}

show_performance() {
    local env=$1

    if [ ! -d "$env" ]; then
        echo -e "${RED}❌ Environment '$env' not found${NC}"
        return 1
    fi

    echo -e "${BLUE}📈 Trading Performance Summary ($env)${NC}"
    echo "===================================="
    
    # Council History Analysis
    if [ -f "$env/data/council_history.csv" ]; then
        echo ""
        echo -e "${YELLOW}🧠 Council Decisions:${NC}"

        # Basic stats
        local total_decisions=$(tail -n +2 "$env/data/council_history.csv" | wc -l)
        echo "  Total decisions: $total_decisions"

        if [ $total_decisions -gt 0 ]; then
            # Recent decisions
            echo ""
            echo -e "${YELLOW}📊 Recent Decisions (Last 5):${NC}"
            head -1 "$env/data/council_history.csv"
            tail -5 "$env/data/council_history.csv"
        fi
    fi

    # Equity Data
    if [ -f "$env/data/daily_equity.csv" ]; then
        echo ""
        echo -e "${YELLOW}💰 Equity Curve:${NC}"
        echo "  Recent equity data (Last 5 days):"
        head -1 "$env/data/daily_equity.csv"
        tail -5 "$env/data/daily_equity.csv"
    fi

    # Trade Ledger
    if [ -f "$env/trade_ledger.csv" ]; then
        echo ""
        echo -e "${YELLOW}📋 Trade Ledger:${NC}"
        local total_trades=$(tail -n +2 "$env/trade_ledger.csv" | wc -l)
        echo "  Total trades: $total_trades"

        if [ $total_trades -gt 0 ]; then
            echo "  Recent trades (Last 3):"
            head -1 "$env/trade_ledger.csv"
            tail -3 "$env/trade_ledger.csv"
        fi
    fi
}

show_health() {
    local env=$1

    if [ ! -d "$env" ]; then
        echo -e "${RED}❌ Environment '$env' not found${NC}"
        return 1
    fi

    echo -e "${BLUE}🏥 System Health Report ($env)${NC}"
    echo "================================"

    # Check for health report files
    if [ "$env" = "prod" ] && [ -f "$env/production_health_report.txt" ]; then
        cat "$env/production_health_report.txt"
    elif [ -f "$env/system_snapshot.txt" ]; then
        cat "$env/system_snapshot.txt"
    else
        echo -e "${YELLOW}⚠️  No health report found for $env${NC}"
        echo "Run log collection first: ./scripts/collect_logs.sh"
    fi
}

collect_logs() {
    local env=$1

    if [ -z "$env" ]; then
        echo -e "${RED}❌ Please specify environment (dev/prod)${NC}"
        return 1
    fi

    echo -e "${BLUE}📥 Collecting logs for $env environment${NC}"

    # Note: We do NOT switch branches here anymore because
    # scripts/collect_logs.sh handles its own branch capturing and switching.
    # We just execute it.

    # Run collection script
    if [ -f "scripts/collect_logs.sh" ]; then
        LOG_ENV_NAME="$env" ./scripts/collect_logs.sh
    else
        echo -e "${RED}❌ Collection script not found: scripts/collect_logs.sh${NC}"
        return 1
    fi
}

# Main script logic
case "$1" in
    "status")
        init_logs_branch
        show_status
        return_to_original_branch
        ;;
    "analyze")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify environment (dev/prod)${NC}"
            usage
            exit 1
        fi
        init_logs_branch
        analyze_environment "$2"
        return_to_original_branch
        ;;
    "errors")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify environment (dev/prod)${NC}"
            usage
            exit 1
        fi
        init_logs_branch
        show_errors "$2" "$3"
        return_to_original_branch
        ;;
    "performance")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify environment (dev/prod)${NC}"
            usage
            exit 1
        fi
        init_logs_branch
        show_performance "$2"
        return_to_original_branch
        ;;
    "health")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify environment (dev/prod)${NC}"
            usage
            exit 1
        fi
        init_logs_branch
        show_health "$2"
        return_to_original_branch
        ;;
    "collect")
        # Note: collect_logs already handles branch switching internally
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify environment (dev/prod)${NC}"
            usage
            exit 1
        fi
        collect_logs "$2"
        ;;
    "compare")
        init_logs_branch
        echo -e "${BLUE}🔄 Comparing Dev vs Prod${NC}"
        echo "========================"
        echo -e "${YELLOW}Dev Environment:${NC}"
        analyze_environment "dev"
        echo ""
        echo -e "${YELLOW}Prod Environment:${NC}"
        analyze_environment "prod"
        return_to_original_branch
        ;;
    *)
        usage
        ;;
esac
EOF

# 4. Create setup_logs_infrastructure.sh
echo "📝 Creating setup_logs_infrastructure.sh..."
cat > scripts/setup_logs_infrastructure.sh << 'EOF'
#!/bin/bash
set -e
# === Coffee Bot Real Options - Logs Branch Setup ===
# This script sets up the logs branch infrastructure for collecting
# dev and prod environment logs and output files.

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

REPO_DIR="${COFFEE_BOT_PATH:-$(pwd)}"
BRANCH="${LOG_BRANCH:-logs}"

echo "🚀 Setting up Coffee Bot Real Options logs branch infrastructure..."
echo "Repository: $REPO_DIR"
echo "Branch: $BRANCH"

# 1. Navigate to repository
cd "$REPO_DIR" || { echo "❌ Repository directory not found: $REPO_DIR"; exit 1; }

# 2. Create and setup logs branch
echo "📋 Creating logs branch..."

# Check if logs branch already exists locally
if git show-ref --verify --quiet refs/heads/$BRANCH; then
    echo "✅ Logs branch already exists locally"
    git checkout $BRANCH
else
    # Check if logs branch exists on remote
    if git ls-remote --heads origin $BRANCH | grep $BRANCH > /dev/null; then
        echo "✅ Logs branch exists on remote, checking out..."
        git fetch origin $BRANCH
        git checkout -b $BRANCH origin/$BRANCH
    else
        echo "✨ Creating new logs branch..."
        # Create orphan branch (clean history for logs)
        git checkout --orphan $BRANCH

        # Clear all tracked files from the new branch
        git rm -rf . 2>/dev/null || true
        
        # Create initial structure
        mkdir -p dev prod scripts
        
        # Create README for the logs branch
        cat > README.md << 'EOL'
# Coffee Bot Real Options - Logs Branch

This branch contains operational logs and output files from both development and production environments.

## Structure

```
├── dev/           # Development environment logs and files
├── prod/          # Production environment logs and files
├── scripts/       # Log collection and analysis scripts
└── README.md      # This file
```

## Contents

Each environment folder contains:
- `logs/` - Current application log files (orchestrator.log, dashboard.log)
- `data/` - CSV files, state files, and operational data
  - `data/tms/` - Transactive Memory System data (ChromaDB files)
- `archive_ledger/` - Archived trading ledgers
- `trade_ledger.csv` - Current trade ledger
# model_signals.csv removed — ML pipeline archived v4.0
- `config.json` - Configuration snapshot (sanitized)
- `state.json` - System state
- `system_snapshot.txt` - System health at time of collection
- `log_summary.txt` - Summarized logs for quick review
- `production_health_report.txt` - Production-specific health metrics
- `trading_performance_snapshot.txt` - Trading performance summary

## File Structure

Files maintain their original directory structure within each environment folder.
For example: `/home/rodrigo/real_options/data/council_history.csv` becomes `prod/data/council_history.csv`

## Usage

### Manual Collection
```bash
# Collect dev logs
LOG_ENV_NAME=dev ./scripts/collect_logs.sh

# Collect prod logs
LOG_ENV_NAME=prod ./scripts/collect_logs.sh
```

### Automated Collection
Set up cron jobs in your main branch:
```bash
# Every hour during market hours (dev)
0 9-16 * * 1-5 cd /home/rodrigo/real_options && LOG_ENV_NAME=dev ./scripts/collect_logs.sh

# Every 30 minutes during market hours (prod)
0,30 9-16 * * 1-5 cd /home/rodrigo/real_options && LOG_ENV_NAME=prod ./scripts/collect_logs.sh
```

## Analysis

```bash
# Quick status
./scripts/log_analysis.sh status

# Analyze environment
./scripts/log_analysis.sh analyze prod

# Check recent errors
./scripts/log_analysis.sh errors dev 12
```

**Note**: This branch contains operational data only.
Do not merge into main development branches.
EOL

        # Create placeholder files
        touch dev/.keep prod/.keep

        # Initial commit
        git add .
        git commit -m "Initial logs branch setup"

        # Push to remote
        git push -u origin $BRANCH

        echo "✅ Created new logs branch and pushed to remote"
    fi
fi

# 3. Switch back to main branch for script setup
echo ""
echo "📁 Setting up log collection scripts in main branch..."
git checkout main 2>/dev/null || git checkout master 2>/dev/null || echo "⚠️  Could not switch to main branch"

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Check if log collection script exists
if [ ! -f scripts/collect_logs.sh ]; then
    echo "⚠️  Log collection script not found in scripts/ directory"
    echo "📥 Please add the collect_logs.sh script to your scripts/ directory"
fi

# 4. Environment setup suggestions
echo ""
echo "🤖 Environment Setup:"
echo ""
echo "1. Create/update your .env file with:"
echo "   # Coffee Bot Log Collection Settings"
echo "   COFFEE_BOT_PATH=/home/rodrigo/real_options"
echo "   LOG_BRANCH=logs"
echo ""
echo "2. Make scripts executable:"
echo "   chmod +x scripts/*.sh"
echo ""
echo "3. Test manual collection:"
echo "   LOG_ENV_NAME=dev ./scripts/collect_logs.sh"
echo "   LOG_ENV_NAME=prod ./scripts/collect_logs.sh"
echo ""
echo "4. Set up cron jobs for automation:"
echo "   crontab -e"
echo ""
echo "   # Add these lines:"
echo "   0 9-16 * * 1-5 cd $REPO_DIR && LOG_ENV_NAME=dev ./scripts/collect_logs.sh"
echo "   0,30 9-16 * * 1-5 cd $REPO_DIR && LOG_ENV_NAME=prod ./scripts/collect_logs.sh"
echo ""
echo "✅ Logs branch infrastructure ready!"
echo "🎯 Next: Add collect_logs.sh to your scripts/ directory and test!"
EOF

# 5. Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.sh

# 6. Add environment configuration to .env
echo ""
echo "⚙️  Adding environment configuration..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
fi

# Check if log settings already exist
if ! grep -q "COFFEE_BOT_PATH" .env; then
    echo "" >> .env
    echo "# === Log Collection Settings ===" >> .env
    echo "COFFEE_BOT_PATH=$REPO_DIR" >> .env
    echo "LOG_BRANCH=logs" >> .env
    echo "LOG_ENV_NAME=dev" >> .env
    echo "✅ Added log collection settings to .env"
else
    echo "✅ Log collection settings already exist in .env"
fi

# 7. Set up logs branch
echo ""
echo "🏗️  Setting up logs branch infrastructure..."
./scripts/setup_logs_infrastructure.sh

# 8. Test collection
echo ""
echo "🧪 Testing log collection..."
read -p "Do you want to test log collection now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing dev collection..."
    LOG_ENV_NAME=dev ./scripts/collect_logs.sh
    
    echo ""
    echo "Checking status..."
    ./scripts/log_analysis.sh status
fi

# 9. Git integration
echo ""
echo "📝 Adding to git repository..."
git add scripts/ .env
git commit -m "Add log collection system infrastructure"
git push

echo ""
echo "🎉 Installation Complete!"
echo ""
echo "📋 Next steps:"
echo "1. Test collection: LOG_ENV_NAME=prod ./scripts/collect_logs.sh"
echo "2. Check status: ./scripts/log_analysis.sh status"
echo "3. Set up cron automation (see documentation)"
echo ""
echo "📖 Available commands:"
echo "  ./scripts/log_analysis.sh collect dev    # Collect dev logs"
echo "  ./scripts/log_analysis.sh collect prod   # Collect prod logs"  
echo "  ./scripts/log_analysis.sh status         # Show status"
echo ""
echo "✨ Your log collection system is ready!"
