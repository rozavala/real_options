#!/bin/bash
set -e

# Navigate to the project directory.
# The script assumes it's being run from the repo's root directory on the server.
cd ~/real_options

echo "--- 1. Stopping the old trading bot process... ---"
pkill -f prob_coffee_options.py || true

echo "--- 2. Activating virtual environment... ---"
source ~/Desktop/ib_env/bin/activate

echo "--- 3. Installing/updating dependencies... ---"
pip install -r requirements.txt

echo "--- 4. Starting the new bot in the background... ---"
nohup python -u ~/real_options/prob_coffee_options.py >> ~/real_options/logs/trading_bot.log 2>&1 &

echo "--- Deployment finished successfully! ---"