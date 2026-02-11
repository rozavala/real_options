#!/bin/bash
set -e
# =============================================================================
# Coffee Bot Real Options — Production Deployment Script
#
# HRO-Enhanced: Adds directory scaffolding, migration framework, post-deploy
# verification gate, and automatic rollback on failure.
#
# Called by: .github/workflows/deploy.yml (via SSH after git pull)
# Rollback: If verify_deploy.sh fails, reverts to previous commit and restarts.
# =============================================================================

# Detect Repo Root
if [ -f "pyproject.toml" ]; then
    REPO_ROOT=$(pwd)
elif [ -f "../pyproject.toml" ]; then
    REPO_ROOT=$(dirname $(pwd))
else
    REPO_ROOT=~/real_options
fi

cd "$REPO_ROOT"

# Prevent collect_logs.sh from switching branches during deploy
DEPLOY_LOCK="/tmp/trading-bot-deploy.lock"
echo "$$" > "$DEPLOY_LOCK"
trap "rm -f '$DEPLOY_LOCK'" EXIT

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
        touch logs/orchestrator.log logs/dashboard.log logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log logs/sentinels.log 2>/dev/null || true
        chmod 664 logs/orchestrator.log logs/dashboard.log logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log logs/sentinels.log 2>/dev/null || true

        # Kill any orphaned processes before restarting
        pkill -9 -f "orchestrator.py" 2>/dev/null || true
        pkill -f "streamlit run dashboard.py" 2>/dev/null || true
        sleep 2

        # Restart orchestrator via systemd (keeps it visible to future deploys)
        sudo systemctl daemon-reload
        sudo systemctl start "$SERVICE_NAME"
        sleep 3

        # Verify systemd started it
        if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
            ORCH_PID=$(pgrep -f "orchestrator.py")
            echo "  ✅ Rollback: $SERVICE_NAME restarted via systemd (PID: ${ORCH_PID:-unknown})"
        else
            echo "  ⚠️  Rollback: systemd start failed, falling back to nohup"
            if [ -f "scripts/start_orchestrator.sh" ]; then
                chmod +x scripts/start_orchestrator.sh
                nohup ./scripts/start_orchestrator.sh >> logs/orchestrator.log 2>&1 &
            else
                nohup python -u orchestrator.py >> logs/orchestrator.log 2>&1 &
            fi
        fi

        # Restart dashboard (not managed by systemd — manual start is correct here)
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
# STEP 1: Stop old processes via systemd
# =========================================================================
echo "--- 1. Stopping old processes... ---"

# Stop the systemd service (passwordless sudo configured in /etc/sudoers.d/coffee-bot)
SERVICE_NAME="${BOT_SERVICE_NAME:-trading-bot}"
echo "  🛑 Stopping $SERVICE_NAME service..."
sudo systemctl stop "$SERVICE_NAME"

# Wait for systemd to fully stop the service
echo "  ⏳ Waiting for clean shutdown..."
sleep 5

# Verify the service is actually stopped
if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "  ❌ ERROR: Service still active after stop command!"
    echo "  📋 Service status:"
    sudo systemctl status "$SERVICE_NAME" --no-pager
    rollback_and_restart
fi

# Double-check no orchestrator processes remain
if pgrep -f "orchestrator.py" > /dev/null; then
    echo "  ⚠️  WARNING: Orphaned orchestrator processes detected!"
    ps aux | grep orchestrator.py | grep -v grep
    echo "  🔨 Force cleaning..."
    pkill -9 -f orchestrator.py
    sleep 2
fi

# Also stop dashboard and monitor (these aren't managed by systemd)
pkill -f "streamlit run dashboard.py" || true
pkill -f "position_monitor.py" || true
sleep 2

echo "  ✅ All processes stopped and verified"

# =========================================================================
# STEP 2: Rotate logs
# =========================================================================
echo "--- 2. Rotating logs... ---"
mkdir -p logs
[ -f logs/orchestrator.log ] && mv logs/orchestrator.log logs/orchestrator-$(date --iso=s).log || true
[ -f logs/dashboard.log ] && mv logs/dashboard.log logs/dashboard-$(date --iso=s).log || true
[ -f logs/equity_logger.log ] && mv logs/equity_logger.log logs/equity_logger-$(date --iso=s).log || true
[ -f logs/sentinels.log ] && mv logs/sentinels.log logs/sentinels-$(date --iso=s).log || true

# Clean up rotated logs older than 7 days
find logs/ -name "*-20*.log" -mtime +7 -delete 2>/dev/null || true

# Ensure logs directory exists with correct permissions
chmod 755 logs
touch logs/orchestrator.log logs/dashboard.log logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log logs/sentinels.log
chmod 664 logs/orchestrator.log logs/dashboard.log logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log logs/sentinels.log
chown rodrigo:rodrigo logs/orchestrator.log logs/dashboard.log logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log logs/sentinels.log 2>/dev/null || true

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
echo "  ✅ Directories OK"

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
# STEP 9: Start services via systemd
# =========================================================================
echo "--- 9. Starting services... ---"

# Reload systemd configuration (in case service file changed)
echo "  🔄 Reloading systemd..."
sudo systemctl daemon-reload

# Start the orchestrator via systemd
echo "  🚀 Starting $SERVICE_NAME service..."
sudo systemctl start "$SERVICE_NAME"

# Wait for service to initialize
sleep 5

# Verify service started successfully
if ! sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "  ❌ ERROR: $SERVICE_NAME service failed to start!"
    echo "  📋 Service status:"
    sudo systemctl status "$SERVICE_NAME" --no-pager
    echo ""
    echo "  📋 Recent logs:"
    tail -50 logs/orchestrator.log
    rollback_and_restart
fi

# Verify exactly ONE orchestrator process
ORCH_COUNT=$(pgrep -f "orchestrator.py" | wc -l)
if [ "$ORCH_COUNT" -ne 1 ]; then
    echo "  ❌ ERROR: Expected 1 orchestrator, found $ORCH_COUNT!"
    ps aux | grep orchestrator.py | grep -v grep
    sudo systemctl stop "$SERVICE_NAME"
    rollback_and_restart
fi

ORCH_PID=$(pgrep -f "orchestrator.py")
echo "  ✅ $SERVICE_NAME service started (PID: $ORCH_PID)"

# Start dashboard (not managed by systemd)
echo "  🚀 Starting dashboard..."
nohup streamlit run dashboard.py --server.address 0.0.0.0 > logs/dashboard.log 2>&1 &
DASH_PID=$!
echo "  ✅ Dashboard started (PID: $DASH_PID)"

# Final verification after 3 more seconds
sleep 3
FINAL_COUNT=$(pgrep -f "orchestrator.py" | wc -l)
if [ "$FINAL_COUNT" -ne 1 ]; then
    echo "  ❌ ERROR: Process count changed! Expected 1, got $FINAL_COUNT"
    ps aux | grep orchestrator.py | grep -v grep
    sudo systemctl stop "$SERVICE_NAME"
    rollback_and_restart
fi

# =========================================================================
# STEP 10: Sync Claude Code worktree (non-destructive, skips if unsafe)
# =========================================================================
echo "--- 10. Syncing Claude Code worktree... ---"
chmod +x scripts/sync_worktree.sh 2>/dev/null || true
if [ -f "scripts/sync_worktree.sh" ]; then
    bash scripts/sync_worktree.sh || echo "  ⚠️  Worktree sync encountered an issue (non-blocking)"
else
    echo "  ⏭️  No worktree sync script found, skipping"
fi

echo ""
echo "--- ✅ Deployment finished successfully! ---"
echo "--- Commit: $CURR_COMMIT ---"
echo "--- Services: trading-bot (systemd) + dashboard (manual) ---"
echo "--- $(date) ---"
