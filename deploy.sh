#!/bin/bash
set -e
# =============================================================================
# Coffee Bot Real Options — Production Deployment Script
#
# HRO-Enhanced: Adds directory scaffolding, migration framework, post-deploy
# verification gate, and automatic rollback on failure.
#
# Multi-commodity: MasterOrchestrator runs all engines in a single process.
# Set LEGACY_MODE=true to fall back to per-commodity systemd services.
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

# Detect mode: --multi (default) vs LEGACY_MODE=true (per-commodity fallback)
LEGACY_MODE="${LEGACY_MODE:-false}"

# All commodity tickers — discovered from data/ directories
ALL_TICKERS=()
for _dir in data/*/; do
    [ -d "$_dir" ] || continue
    _tk=$(basename "$_dir")
    # Only uppercase directory names (KC, CC, SB) — skip surrogate_models etc.
    [[ "$_tk" =~ ^[A-Z]{2,4}$ ]] && ALL_TICKERS+=("$_tk")
done
# Fallback: at least KC
[ ${#ALL_TICKERS[@]} -eq 0 ] && ALL_TICKERS=("KC")

# Primary service name
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
if [ "$LEGACY_MODE" = "true" ]; then
    echo "--- Mode: LEGACY (per-commodity services) ---"
else
    echo "--- Mode: MULTI (MasterOrchestrator) ---"
fi

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
            touch "logs/orchestrator_${_tl}.log" 2>/dev/null || true
            chmod 664 "logs/orchestrator_${_tl}.log" 2>/dev/null || true
        done
        touch logs/orchestrator_multi.log logs/dashboard.log logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log logs/sentinels.log 2>/dev/null || true

        # Kill any orphaned processes before restarting
        pkill -9 -f "orchestrator.py" 2>/dev/null || true
        pkill -f "streamlit run dashboard.py" 2>/dev/null || true
        sleep 2

        # Restart via systemd
        sudo systemctl daemon-reload
        if [ "$LEGACY_MODE" = "true" ]; then
            # Legacy: restart per-commodity services
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
            done
        else
            # Multi: try MasterOrchestrator, fall back to per-commodity nohup
            sudo systemctl start "$SERVICE_NAME" 2>/dev/null || true
            sleep 5
            if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
                echo "  Rollback: $SERVICE_NAME (--multi) restarted via systemd"
            else
                echo "  Rollback: --multi start failed, falling back to per-commodity nohup"
                for _t in "${ALL_TICKERS[@]}"; do
                    _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
                    nohup python -u orchestrator.py --commodity "$_t" >> "logs/orchestrator_${_tl}.log" 2>&1 &
                    sleep 3
                done
            fi
        fi

        # Single dashboard (commodity selection is in-app)
        echo "  Rollback: Starting dashboard on port 8501..."
        nohup streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501 > "logs/dashboard.log" 2>&1 &
        echo "  Rollback: Dashboard started (PID: $!, port: 8501)"

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

if [ "$LEGACY_MODE" = "true" ]; then
    # Legacy: stop per-commodity services
    for _t in "${ALL_TICKERS[@]}"; do
        _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
        if [ "$_t" = "KC" ]; then _svc="$SERVICE_NAME"; else _svc="trading-bot-${_tl}"; fi
        if sudo systemctl is-active --quiet "$_svc" 2>/dev/null; then
            echo "  Stopping $_svc service..."
            sudo systemctl stop "$_svc"
        fi
    done
else
    # Multi: stop single MasterOrchestrator service + any legacy services
    if sudo systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo "  Stopping $SERVICE_NAME service..."
        sudo systemctl stop "$SERVICE_NAME"
    fi
    # Also stop any legacy per-commodity services that might be running
    for _svc_file in /etc/systemd/system/trading-bot-*.service; do
        [ -f "$_svc_file" ] || continue
        _svc=$(basename "$_svc_file" .service)
        if sudo systemctl is-active --quiet "$_svc" 2>/dev/null; then
            echo "  Stopping legacy $_svc service..."
            sudo systemctl stop "$_svc"
        fi
    done
fi

# Wait for systemd to fully stop the services
echo "  Waiting for clean shutdown..."
sleep 5

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
done
[ -f logs/orchestrator_multi.log ] && mv logs/orchestrator_multi.log "logs/orchestrator_multi-${ROTATE_DATE}.log" || true
[ -f logs/dashboard.log ] && mv logs/dashboard.log "logs/dashboard-${ROTATE_DATE}.log" || true
[ -f logs/equity_logger.log ] && mv logs/equity_logger.log "logs/equity_logger-${ROTATE_DATE}.log" || true
[ -f logs/sentinels.log ] && mv logs/sentinels.log "logs/sentinels-${ROTATE_DATE}.log" || true

# Clean up rotated logs older than 7 days
find logs/ -name "*-20*.log" -mtime +7 -delete 2>/dev/null || true

# Ensure logs directory exists with correct permissions
chmod 755 logs
LOG_FILES=(logs/manual_test.log logs/performance_analyzer.log logs/equity_logger.log logs/sentinels.log logs/dashboard.log logs/orchestrator_multi.log)
for _t in "${ALL_TICKERS[@]}"; do
    _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
    LOG_FILES+=("logs/orchestrator_${_tl}.log")
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
# STEP 7: Sync equity data
# =========================================================================
echo "--- 7. Syncing Equity Data... ---"
if [ "$LEGACY_MODE" = "true" ]; then
    # Legacy: run equity sync for KC (account-wide)
    python equity_logger.py --sync --commodity KC || true
else
    # Multi: MasterOrchestrator's _post_close_service handles equity.
    # Still run initial sync to seed the data file.
    python equity_logger.py --sync --commodity KC || true
fi

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

# Sync service file if repo version differs from installed version
REPO_SERVICE="scripts/trading-bot.service"
LIVE_SERVICE="/etc/systemd/system/$SERVICE_NAME.service"
if [ -f "$REPO_SERVICE" ]; then
    if ! diff -q "$REPO_SERVICE" "$LIVE_SERVICE" >/dev/null 2>&1; then
        echo "  Syncing service file (repo differs from installed)..."
        sudo cp "$REPO_SERVICE" "$LIVE_SERVICE" || echo "  WARNING: Could not sync service file (check sudoers)"
    fi
fi

# Reload systemd configuration (in case service file changed)
echo "  Reloading systemd..."
sudo systemctl daemon-reload

if [ "$LEGACY_MODE" = "true" ]; then
    # Legacy: start per-commodity services
    for _t in "${ALL_TICKERS[@]}"; do
        _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
        if [ "$_t" = "KC" ]; then _svc="$SERVICE_NAME"; else _svc="trading-bot-${_tl}"; fi

        echo "  Starting $_svc service..."
        sudo systemctl start "$_svc"
        sleep 5

        # Verify service started successfully
        if ! sudo systemctl is-active --quiet "$_svc"; then
            echo "  ERROR: $_svc service failed to start!"
            sudo systemctl status "$_svc" --no-pager
            rollback_and_restart
        fi
        echo "  $_svc service started"
    done
else
    # Multi: single MasterOrchestrator service
    echo "  Starting $SERVICE_NAME service (--multi mode)..."
    sudo systemctl start "$SERVICE_NAME"
    sleep 10  # Allow staggered engine startup

    if ! sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        echo "  ERROR: $SERVICE_NAME service failed to start!"
        sudo systemctl status "$SERVICE_NAME" --no-pager
        echo ""
        echo "  Recent logs:"
        tail -50 logs/orchestrator_multi.log 2>/dev/null || true
        rollback_and_restart
    fi

    # Verify orchestrator process is running
    SVC_PID=$(sudo systemctl show -p MainPID --value "$SERVICE_NAME" 2>/dev/null || true)
    SVC_PID="${SVC_PID#MainPID=}"
    if [ -z "$SVC_PID" ] || [ "$SVC_PID" = "0" ]; then
        SVC_PID=$(pgrep -f "python.*orchestrator\.py" | head -1)
    fi

    if [ -z "$SVC_PID" ] || ! kill -0 "$SVC_PID" 2>/dev/null; then
        echo "  ERROR: No orchestrator process found!"
        ps aux | grep orchestrator.py | grep -v grep
        sudo systemctl stop "$SERVICE_NAME"
        rollback_and_restart
    fi
    echo "  $SERVICE_NAME service started (PID: $SVC_PID, engines: ${ALL_TICKERS[*]})"
fi

# Single dashboard (commodity selection is in-app)
echo "  Starting dashboard on port 8501..."
nohup streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501 > "logs/dashboard.log" 2>&1 &
echo "  Dashboard started (PID: $!, port: 8501)"

# Final verification after 3 more seconds
sleep 3
if [ "$LEGACY_MODE" = "true" ]; then
    for _t in "${ALL_TICKERS[@]}"; do
        _tl=$(echo "$_t" | tr '[:upper:]' '[:lower:]')
        if [ "$_t" = "KC" ]; then _svc="$SERVICE_NAME"; else _svc="trading-bot-${_tl}"; fi
        if ! sudo systemctl is-active --quiet "$_svc"; then
            echo "  ERROR: $_svc no longer running after 3s!"
            sudo systemctl stop "$_svc" 2>/dev/null || true
            rollback_and_restart
        fi
    done
else
    if ! sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        echo "  ERROR: $SERVICE_NAME no longer running after 3s!"
        sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
        rollback_and_restart
    fi
fi

# Warn if extra orchestrator processes exist
ORCH_COUNT=$(pgrep -fc "python.*orchestrator\.py" 2>/dev/null || echo 0)
if [ "$LEGACY_MODE" = "true" ]; then
    EXPECTED_COUNT=${#ALL_TICKERS[@]}
else
    EXPECTED_COUNT=1
fi
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
if [ "$LEGACY_MODE" = "true" ]; then
    echo "--- Mode: LEGACY ---"
else
    echo "--- Mode: MULTI (MasterOrchestrator) ---"
fi
echo "--- $(date) ---"
