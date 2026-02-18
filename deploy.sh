#!/bin/bash
set -e
# =============================================================================
# Coffee Bot Real Options — Production Deployment Script
#
# HRO-Enhanced: Adds directory scaffolding, migration framework, post-deploy
# verification gate, and automatic rollback on failure.
#
# Multi-commodity: Manages all commodity services (KC, CC, ...) in a single deploy.
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

# Commodity ticker for per-commodity log/data paths
TICKER="${COMMODITY_TICKER:-KC}"
TICKER_LOWER=$(echo "$TICKER" | tr '[:upper:]' '[:lower:]')

# All commodity tickers to manage (KC always present, CC if service installed)
ALL_TICKERS=("KC")
if [ -f "/etc/systemd/system/trading-bot-cc.service" ]; then
    ALL_TICKERS+=("CC")
fi

# Primary service name (KC)
SERVICE_NAME="${BOT_SERVICE_NAME:-trading-bot}"

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
echo "--- Commodities: ${ALL_TICKERS[*]} ---"

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
        for _t in "${ALL_TICKERS[@]}"; do
            _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
            touch "logs/orchestrator_${_tl}.log" "logs/dashboard_${_tl}.log" 2>/dev/null || true
            chmod 664 "logs/orchestrator_${_tl}.log" "logs/dashboard_${_tl}.log" 2>/dev/null || true
        done
        touch logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log logs/sentinels.log 2>/dev/null || true

        # Kill any orphaned processes before restarting
        pkill -9 -f "orchestrator.py" 2>/dev/null || true
        pkill -f "streamlit run dashboard.py" 2>/dev/null || true
        sleep 2

        # Restart ALL commodity services via systemd
        sudo systemctl daemon-reload
        _rb_port=8501
        for _t in "${ALL_TICKERS[@]}"; do
            _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
            if [ "$_t" = "KC" ]; then _svc="$SERVICE_NAME"; else _svc="trading-bot-${_tl}"; fi
            sudo systemctl start "$_svc" 2>/dev/null || true
            sleep 3
            if sudo systemctl is-active --quiet "$_svc"; then
                echo "  Rollback: $_svc restarted via systemd"
            else
                echo "  Rollback: $_svc start failed, falling back to nohup"
                nohup python -u orchestrator.py --commodity "$_t" >> "logs/orchestrator_${_tl}.log" 2>&1 &
            fi
            # Restart dashboard for this commodity
            COMMODITY_TICKER="$_t" nohup streamlit run dashboard.py --server.address 0.0.0.0 --server.port "$_rb_port" > "logs/dashboard_${_tl}.log" 2>&1 &
            _rb_port=$((_rb_port + 1))
        done

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

# Stop ALL commodity services (passwordless sudo configured in /etc/sudoers.d/trading-bot)
for _t in "${ALL_TICKERS[@]}"; do
    _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
    if [ "$_t" = "KC" ]; then _svc="$SERVICE_NAME"; else _svc="trading-bot-${_tl}"; fi
    if sudo systemctl is-active --quiet "$_svc" 2>/dev/null; then
        echo "  Stopping $_svc service..."
        sudo systemctl stop "$_svc"
    fi
done

# Wait for systemd to fully stop the services
echo "  Waiting for clean shutdown..."
sleep 5

# Verify all services are actually stopped
for _t in "${ALL_TICKERS[@]}"; do
    _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
    if [ "$_t" = "KC" ]; then _svc="$SERVICE_NAME"; else _svc="trading-bot-${_tl}"; fi
    if sudo systemctl is-active --quiet "$_svc" 2>/dev/null; then
        echo "  ERROR: $_svc still active after stop command!"
        echo "  Service status:"
        sudo systemctl status "$_svc" --no-pager
        rollback_and_restart
    fi
done

# Double-check no orchestrator processes remain
if pgrep -f "orchestrator.py" > /dev/null; then
    echo "  WARNING: Orphaned orchestrator processes detected!"
    ps aux | grep orchestrator.py | grep -v grep
    echo "  Force cleaning..."
    pkill -9 -f orchestrator.py
    sleep 2
fi

# Also stop ALL dashboards and monitors (these aren't managed by systemd)
pkill -f "streamlit run dashboard.py" || true
pkill -f "position_monitor.py" || true
sleep 2

echo "  All processes stopped and verified"

# =========================================================================
# STEP 2: Rotate logs
# =========================================================================
echo "--- 2. Rotating logs... ---"
mkdir -p logs
ROTATE_DATE=$(date --iso=s)
for _t in "${ALL_TICKERS[@]}"; do
    _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
    [ -f "logs/orchestrator_${_tl}.log" ] && mv "logs/orchestrator_${_tl}.log" "logs/orchestrator_${_tl}-${ROTATE_DATE}.log" || true
    [ -f "logs/dashboard_${_tl}.log" ] && mv "logs/dashboard_${_tl}.log" "logs/dashboard_${_tl}-${ROTATE_DATE}.log" || true
done
[ -f logs/equity_logger.log ] && mv logs/equity_logger.log "logs/equity_logger-${ROTATE_DATE}.log" || true
[ -f logs/sentinels.log ] && mv logs/sentinels.log "logs/sentinels-${ROTATE_DATE}.log" || true

# Clean up rotated logs older than 7 days
find logs/ -name "*-20*.log" -mtime +7 -delete 2>/dev/null || true

# Ensure logs directory exists with correct permissions
chmod 755 logs
LOG_FILES=(logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log logs/sentinels.log)
for _t in "${ALL_TICKERS[@]}"; do
    _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
    LOG_FILES+=("logs/orchestrator_${_tl}.log" "logs/dashboard_${_tl}.log")
done
touch "${LOG_FILES[@]}"
chmod 664 "${LOG_FILES[@]}"
chown rodrigo:rodrigo "${LOG_FILES[@]}" 2>/dev/null || true

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
for _t in "${ALL_TICKERS[@]}"; do
    mkdir -p "data/$_t"           # Per-commodity data directory
done
echo "  Directories OK"

# =========================================================================
# STEP 6: Data migration (idempotent — moves legacy flat paths to data/KC/)
# =========================================================================
echo "--- 6. Running data migration... ---"
if [ -f "scripts/migrate_data_dirs.py" ]; then
    python scripts/migrate_data_dirs.py --force || echo "  Migration encountered an issue (non-blocking)"
else
    echo "  No migration script found, skipping"
fi

# =========================================================================
# STEP 7: Sync equity data (only for primary commodity — is_primary handles this)
# =========================================================================
echo "--- 7. Syncing Equity Data... ---"
python equity_logger.py --sync --commodity KC || true

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
    echo "  No verification script found, skipping"
fi

# =========================================================================
# STEP 9: Start services via systemd
# =========================================================================
echo "--- 9. Starting services... ---"

# Reload systemd configuration (in case service file changed)
echo "  Reloading systemd..."
sudo systemctl daemon-reload

# Start all commodity services and their dashboards
DASHBOARD_PORT=8501
for _t in "${ALL_TICKERS[@]}"; do
    _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
    if [ "$_t" = "KC" ]; then _svc="$SERVICE_NAME"; else _svc="trading-bot-${_tl}"; fi

    echo "  Starting $_svc service..."
    sudo systemctl start "$_svc"
    sleep 5

    # Verify service started successfully
    if ! sudo systemctl is-active --quiet "$_svc"; then
        echo "  ERROR: $_svc service failed to start!"
        echo "  Service status:"
        sudo systemctl status "$_svc" --no-pager
        echo ""
        echo "  Recent logs:"
        tail -50 "logs/orchestrator_${_tl}.log" 2>/dev/null || true
        rollback_and_restart
    fi

    # Verify orchestrator process is running (use systemd MainPID for reliability)
    SVC_PID=$(sudo systemctl show -p MainPID --value "$_svc" 2>/dev/null || true)
    SVC_PID="${SVC_PID#MainPID=}"
    if [ -z "$SVC_PID" ] || [ "$SVC_PID" = "0" ]; then
        SVC_PID=$(pgrep -f "python.*orchestrator\.py.*--commodity $_t" | head -1)
    fi

    if [ -z "$SVC_PID" ] || ! kill -0 "$SVC_PID" 2>/dev/null; then
        echo "  ERROR: No orchestrator process found for $_t!"
        ps aux | grep orchestrator.py | grep -v grep
        sudo systemctl stop "$_svc"
        rollback_and_restart
    fi
    echo "  $_svc service started (PID: $SVC_PID)"

    # Start dashboard for this commodity
    echo "  Starting $_t dashboard on port $DASHBOARD_PORT..."
    COMMODITY_TICKER="$_t" nohup streamlit run dashboard.py --server.address 0.0.0.0 --server.port "$DASHBOARD_PORT" > "logs/dashboard_${_tl}.log" 2>&1 &
    echo "  $_t dashboard started (PID: $!, port: $DASHBOARD_PORT)"

    DASHBOARD_PORT=$((DASHBOARD_PORT + 1))
done

# Final verification after 3 more seconds — check all services are still alive
sleep 3
for _t in "${ALL_TICKERS[@]}"; do
    _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
    if [ "$_t" = "KC" ]; then _svc="$SERVICE_NAME"; else _svc="trading-bot-${_tl}"; fi
    if ! sudo systemctl is-active --quiet "$_svc"; then
        echo "  ERROR: $_svc no longer running after 3s!"
        sudo systemctl stop "$_svc" 2>/dev/null || true
        rollback_and_restart
    fi
done

# Warn if extra orchestrator processes exist
ORCH_COUNT=$(pgrep -fc "python.*orchestrator\.py" 2>/dev/null || echo 0)
EXPECTED_COUNT=${#ALL_TICKERS[@]}
if [ "$ORCH_COUNT" -gt "$EXPECTED_COUNT" ]; then
    echo "  WARNING: $ORCH_COUNT orchestrator processes found (expected $EXPECTED_COUNT)"
    ps aux | grep orchestrator.py | grep -v grep
fi

# =========================================================================
# STEP 10: Sync Claude Code worktree (non-destructive, skips if unsafe)
# =========================================================================
echo "--- 10. Syncing Claude Code worktree... ---"
chmod +x scripts/sync_worktree.sh 2>/dev/null || true
if [ -f "scripts/sync_worktree.sh" ]; then
    bash scripts/sync_worktree.sh || echo "  Worktree sync encountered an issue (non-blocking)"
else
    echo "  No worktree sync script found, skipping"
fi

# =========================================================================
# STEP 11: Error reporter cron (hourly) — user crontab (no root needed)
# Hourly gives time to investigate and fix issues before auto-triage triggers.
# =========================================================================
echo "--- 11. Setting up error reporter cron... ---"
CRON_LINE="0 * * * * cd $REPO_ROOT && $REPO_ROOT/venv/bin/python scripts/error_reporter.py >> logs/error_reporter.log 2>&1"
# Add to user crontab if not already present (idempotent)
( crontab -l 2>/dev/null | grep -v 'error_reporter\.py'; echo "$CRON_LINE" ) | crontab -
echo "  Error reporter cron installed"

echo ""
echo "--- Deployment finished successfully! ---"
echo "--- Commit: $CURR_COMMIT ---"
echo "--- Commodities: ${ALL_TICKERS[*]} ---"
echo "--- $(date) ---"
