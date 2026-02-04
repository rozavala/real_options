#!/bin/bash
# =============================================================================
# Coffee Bot Mission Control — Production Deployment Script
#
# HRO-Enhanced: Adds directory scaffolding, migration framework, post-deploy
# verification gate, and automatic rollback on failure.
#
# Called by: .github/workflows/deploy.yml (via SSH after git pull)
# Rollback: If verify_deploy.sh fails, reverts to previous commit and restarts.
# =============================================================================
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

# =========================================================================
# STEP 0: Capture rollback point BEFORE any changes
# =========================================================================
PREV_COMMIT=$(git rev-parse HEAD~1 2>/dev/null || echo "")
CURR_COMMIT=$(git rev-parse HEAD)
echo "--- Deploy: $CURR_COMMIT (rollback target: ${PREV_COMMIT:-none}) ---"

# Define rollback function
rollback_and_restart() {
    echo ""
    echo "!!! ===================================================== !!!"
    echo "!!! DEPLOYMENT FAILED — INITIATING AUTOMATIC ROLLBACK      !!!"
    echo "!!! ===================================================== !!!"

    if [ -n "$PREV_COMMIT" ]; then
        echo "--- Rolling back to $PREV_COMMIT ---"
        git reset --hard "$PREV_COMMIT"

        # Re-run the safe parts of deploy with the old code
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        pip install -r requirements.txt --quiet

        # Ensure log directory exists and is writable before restarting
        mkdir -p logs
        chmod 755 logs 2>/dev/null || true
        touch logs/orchestrator.log logs/dashboard.log logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log 2>/dev/null || true
        chmod 664 logs/*.log 2>/dev/null || true

        # Restart with old code
        if [ -f "scripts/start_orchestrator.sh" ]; then
            chmod +x scripts/start_orchestrator.sh
            nohup ./scripts/start_orchestrator.sh >> logs/orchestrator.log 2>&1 &
        else
            nohup python -u orchestrator.py >> logs/orchestrator.log 2>&1 &
        fi

        nohup streamlit run dashboard.py --server.address 0.0.0.0 > logs/dashboard.log 2>&1 &

        echo "--- Rollback complete. Old version restarted. ---"
        echo "--- MANUAL INVESTIGATION REQUIRED ---"

        # Notify
        python -c "
from trading_bot.notifications import send_pushover_notification
send_pushover_notification({}, 'DEPLOY ROLLBACK', 'Deploy of $CURR_COMMIT failed. Rolled back to $PREV_COMMIT.')
" 2>/dev/null || true
    else
        echo "--- No previous commit available for rollback ---"
        echo "--- MANUAL INTERVENTION REQUIRED ---"
    fi

    exit 1
}

# =========================================================================
# STEP 1: Stop old processes
# =========================================================================
echo "--- 1. Stopping old processes... ---"
pkill -f orchestrator.py || true
pkill -f position_monitor.py || true
pkill -f dashboard.py || true
sleep 2

# =========================================================================
# STEP 2: Rotate logs
# =========================================================================
echo "--- 2. Rotating logs... ---"
mkdir -p logs
[ -f logs/orchestrator.log ] && mv logs/orchestrator.log logs/orchestrator-$(date --iso=s).log || true
[ -f logs/dashboard.log ] && mv logs/dashboard.log logs/dashboard-$(date --iso=s).log || true

# Ensure logs directory exists with correct permissions
chmod 755 logs
touch logs/orchestrator.log logs/dashboard.log logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log
chmod 664 logs/*.log
chown rodrigo:rodrigo logs/*.log 2>/dev/null || true

# =========================================================================
# STEP 3: Activate venv + install deps
# =========================================================================
echo "--- 3. Activating virtual environment... ---"
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "--- 4. Installing/updating dependencies... ---"
pip install -r requirements.txt

# =========================================================================
# STEP 5: Directory scaffolding (idempotent — safe to re-run)
# =========================================================================
echo "--- 5. Ensuring directory structure... ---"
mkdir -p config/profiles          # Custom JSON commodity profiles
mkdir -p trading_bot/prompts      # Templatized agent prompts
mkdir -p backtesting              # Backtesting framework
mkdir -p data/surrogate_models    # Surrogate model cache (runtime)
mkdir -p scripts/migrations       # Migration scripts
echo "  ✅ Directories OK"

# =========================================================================
# STEP 6: Run pending migrations (idempotent)
# =========================================================================
echo "--- 6. Running database migrations... ---"
chmod +x scripts/run_migrations.sh 2>/dev/null || true
if [ -f "scripts/run_migrations.sh" ]; then
    bash scripts/run_migrations.sh || rollback_and_restart
else
    echo "  ⏭️  No migration runner found, skipping"
fi

# =========================================================================
# STEP 7: Sync equity data
# =========================================================================
echo "--- 7. Syncing Equity Data... ---"
python equity_logger.py --sync || true

# =========================================================================
# STEP 8: Post-deploy verification gate
# =========================================================================
echo "--- 8. Running post-deploy verification... ---"
chmod +x scripts/verify_deploy.sh 2>/dev/null || true
if [ -f "scripts/verify_deploy.sh" ]; then
    if ! bash scripts/verify_deploy.sh; then
        rollback_and_restart
    fi
else
    echo "  ⏭️  No verification script found, skipping"
fi

# =========================================================================
# STEP 9: Start services
# =========================================================================
echo "--- 9. Starting the new bot & dashboard... ---"
if [ -f "scripts/start_orchestrator.sh" ]; then
    chmod +x scripts/start_orchestrator.sh
    nohup ./scripts/start_orchestrator.sh >> logs/orchestrator.log 2>&1 &
else
    nohup python -u orchestrator.py >> logs/orchestrator.log 2>&1 &
fi

nohup streamlit run dashboard.py --server.address 0.0.0.0 > logs/dashboard.log 2>&1 &

echo ""
echo "--- ✅ Deployment finished successfully! ---"
echo "--- Commit: $CURR_COMMIT ---"
echo "--- $(date) ---"
