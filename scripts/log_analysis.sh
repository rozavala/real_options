#!/bin/bash
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
