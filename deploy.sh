#!/bin/bash
set -e

# Navigate to the project directory.
# The script assumes it's being run from the repo's root directory on the server.
cd ~/real_options

echo "--- 1. Stopping the old trading bot process... ---"
pkill -f orchestrator.py || true
pkill -f position_monitor.py || true
pkill -f dashboard.py || true
mv ~/real_options/logs/orchestrator.log ~/real_options/logs/orchestrator-$(date --iso=s).log  || true

echo "--- 2. Activating virtual environment... ---"
source ~/Desktop/ib_env/bin/activate

echo "--- 3. Installing/updating dependencies... ---"
pip install -r requirements.txt

echo "--- 4. Starting the new bot in the background... ---"
nohup python -u ~/real_options/orchestrator.py >> ~/real_options/logs/orchestrator.log 2>&1 &
nohup streamlit run ~/real_options/dashboard.py --server.address 0.0.0.0 > ~/real_options/dashboard.log 2>&1 &
echo "--- Deployment finished successfully! ---"
