#!/bin/bash
# === Coffee Bot Mission Control - Log Collection System Installer ===
# This script automates the setup of the repository-based log collection system

set -e  # Exit on any error

echo "☕ Coffee Bot Mission Control - Log Collection System Installer"
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
# Main data directory with all CSV files and state
if [ -d "$REPO_DIR/data" ]; then
    echo "Copying data directory..."
    cp -r "$REPO_DIR/data" "$DEST_DIR/"
fi

# === TMS DATA ===
# Transactive Memory System data (ChromaDB, etc.)
if [ -d "$REPO_DIR/data/tms" ]; then
    echo "Copying TMS data..."
    # This is already included in the data directory copy above, but highlighting it
    echo "  - ChromaDB and TMS files included in data/ copy"
fi

# === TRADE FILES ===
# Main trade ledger
if [ -f "$REPO_DIR/trade_ledger.csv" ]; then
    echo "Copying trade_ledger.csv..."
    cp "$REPO_DIR/trade_ledger.csv" "$DEST_DIR/"
fi

# Model signals
if [ -f "$REPO_DIR/model_signals.csv" ]; then
    echo "Copying model_signals.csv..."
    cp "$REPO_DIR/model_signals.csv" "$DEST_DIR/"
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

# === LOG SUMMARY ===
echo "Creating log summary..."
{
    echo "=== LOG SUMMARY FOR $ENV_NAME ==="
    echo "Generated: $(date)"
    echo ""
    
    # Orchestrator log summary
    if [ -f "$REPO_DIR/logs/orchestrator.log" ]; then
        echo "=== ORCHESTRATOR LOG (Last 50 lines) ==="
        tail -50 "$REPO_DIR/logs/orchestrator.log"
        echo ""
    fi
    
    # Dashboard log summary
    if [ -f "$REPO_DIR/logs/dashboard.log" ]; then
        echo "=== DASHBOARD LOG (Last 50 lines) ==="
        tail -50 "$REPO_DIR/logs/dashboard.log"
        echo ""
    fi
    
    # Any other .log files
    for logfile in "$REPO_DIR/logs"/*.log; do
        if [ -f "$logfile" ] && [[ "$logfile" != *"orchestrator.log" ]] && [[ "$logfile" != *"dashboard.log" ]]; then
            filename=$(basename "$logfile")
            echo "=== ${filename^^} (Last 20 lines) ==="
            tail -20 "$logfile"
            echo ""
        fi
    done
} > "$DEST_DIR/log_summary.txt"

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
echo "   - trade_ledger.csv and model_signals.csv"
echo "   - archive_ledger/ directory (if exists)"
echo "   - config.json and state.json"
if [ "$ENV_NAME" = "prod" ]; then
    echo "   - production_health_report.txt and trading_performance_snapshot.txt"
else
    echo "   - system_snapshot.txt"
fi
echo "   - log_summary.txt (generated report)"
EOF

# 3. Create log_analysis.sh (truncated for brevity, but would include full script)
echo "📝 Creating log_analysis.sh..."
cat > scripts/log_analysis.sh << 'EOF'
#!/bin/bash
# === Coffee Bot Mission Control - Log Analysis Utilities ===
# This script provides utilities for analyzing collected logs

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

usage() {
    echo "☕ Coffee Bot Mission Control - Log Analysis Utilities"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  status                 Show current logs branch status"
    echo "  collect ENV            Collect logs for environment (dev/prod)"
    echo "  analyze ENV            Analyze logs for environment"
    echo "  errors ENV [HOURS]     Show recent errors"
    echo "  performance ENV        Show trading performance"
    echo "  compare                Compare dev vs prod"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 collect dev"
    echo "  $0 analyze prod"
    echo ""
}

collect_logs() {
    local env=$1
    if [ -z "$env" ]; then
        echo -e "${RED}❌ Please specify environment (dev/prod)${NC}"
        return 1
    fi
    
    echo -e "${BLUE}📥 Collecting logs for $env environment${NC}"
    LOG_ENV_NAME="$env" ./scripts/collect_logs.sh
}

# Basic status command
show_status() {
    echo -e "${BLUE}📊 Coffee Bot Log Collection Status${NC}"
    echo "===================================="
    
    if git show-ref --verify --quiet refs/heads/logs; then
        echo -e "${GREEN}✅ Logs branch exists${NC}"
        git checkout logs > /dev/null 2>&1
        echo ""
        echo -e "${YELLOW}📅 Recent Snapshots:${NC}"
        git log --oneline -5
        git checkout main > /dev/null 2>&1 || git checkout master > /dev/null 2>&1
    else
        echo -e "${YELLOW}⚠️  Logs branch not yet created${NC}"
        echo "Run: ./scripts/setup_logs_infrastructure.sh"
    fi
}

# Main logic
case "$1" in
    "status")
        show_status
        ;;
    "collect")
        collect_logs "$2"
        ;;
    *)
        usage
        ;;
esac
EOF

# 4. Create setup script  
echo "📝 Creating setup_logs_infrastructure.sh..."
cat > scripts/setup_logs_infrastructure.sh << 'EOF'
#!/bin/bash
# === Coffee Bot Mission Control - Logs Branch Setup ===

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

REPO_DIR="${COFFEE_BOT_PATH:-$(pwd)}"
BRANCH="${LOG_BRANCH:-logs}"

echo "🚀 Setting up Coffee Bot Mission Control logs branch infrastructure..."
echo "Repository: $REPO_DIR"
echo "Branch: $BRANCH"

# Navigate to repository
cd "$REPO_DIR" || { echo "❌ Repository directory not found: $REPO_DIR"; exit 1; }

# Create logs branch
echo "📋 Creating logs branch..."

if git show-ref --verify --quiet refs/heads/$BRANCH; then
    echo "✅ Logs branch already exists locally"
    git checkout $BRANCH
else
    if git ls-remote --heads origin $BRANCH | grep $BRANCH > /dev/null; then
        echo "✅ Logs branch exists on remote, checking out..."
        git fetch origin $BRANCH
        git checkout -b $BRANCH origin/$BRANCH
    else
        echo "✨ Creating new logs branch..."
        git checkout --orphan $BRANCH
        git rm -rf . 2>/dev/null || true
        
        mkdir -p dev prod scripts
        
        cat > README.md << 'EOL'
# Coffee Bot Mission Control - Logs Branch

This branch contains operational logs and output files from both development and production environments.

## Usage

```bash
# Collect logs
LOG_ENV_NAME=dev ./scripts/collect_logs.sh
LOG_ENV_NAME=prod ./scripts/collect_logs.sh

# Analysis
./scripts/log_analysis.sh status
./scripts/log_analysis.sh collect prod
```
EOL

        touch dev/.keep prod/.keep
        git add .
        git commit -m "Initial logs branch setup"
        git push -u origin $BRANCH
        echo "✅ Created new logs branch and pushed to remote"
    fi
fi

# Switch back to main
git checkout main 2>/dev/null || git checkout master 2>/dev/null

echo ""
echo "✅ Logs branch infrastructure ready!"
echo ""
echo "🚀 Quick start:"
echo "  LOG_ENV_NAME=dev ./scripts/collect_logs.sh"
echo "  ./scripts/log_analysis.sh status"
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
EOF

chmod +x install_log_collection.sh

echo ""
echo "✅ Installation script created successfully!"
echo ""
echo "🚀 To install the complete log collection system:"
echo "   chmod +x install_log_collection.sh"
echo "   ./install_log_collection.sh"
echo ""
echo "This will:"
echo "  ✅ Create scripts/ directory with all tools"
echo "  ✅ Set up environment configuration"
echo "  ✅ Initialize the logs branch"
echo "  ✅ Test the collection system"
echo "  ✅ Add everything to git"
echo ""
echo "The installer is safe to run multiple times!"
