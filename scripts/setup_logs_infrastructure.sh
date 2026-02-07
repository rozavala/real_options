#!/bin/bash
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
