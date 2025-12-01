#!/bin/bash
set -e

# Navigate to the project directory.
cd ~/real_options

echo "--- 1. Stopping old processes... ---"
# Stop all related processes
pkill -f orchestrator.py || true
pkill -f position_monitor.py || true
pkill -f dashboard.py || true

# Wait for ports to free up (Crucial for Streamlit)
sleep 2

echo "--- 2. Rotating logs... ---"
# Backup Orchestrator Log
mv ~/real_options/logs/orchestrator.log ~/real_options/logs/orchestrator-$(date --iso=s).log || true
# Backup Dashboard Log (New)
mv ~/real_options/logs/dashboard.log ~/real_options/logs/dashboard-$(date --iso=s).log || true

echo "--- 3. Activating virtual environment... ---"
source ~/Desktop/ib_env/bin/activate

echo "--- 4. Installing/updating dependencies... ---"
pip install -r requirements.txt

echo "--- 5. Syncing Equity Data... ---"
# Ensure local data is in sync with IBKR before starting
python ~/real_options/equity_logger.py --sync || true

echo "--- 6. Starting the new bot & dashboard... ---"
# Start Orchestrator
nohup python -u ~/real_options/orchestrator.py >> ~/real_options/logs/orchestrator.log 2>&1 &

# Start Dashboard
nohup streamlit run ~/real_options/dashboard.py --server.address 0.0.0.0 > ~/real_options/logs/dashboard.log 2>&1 &

echo "--- Deployment finished successfully! ---"
