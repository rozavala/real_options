#!/bin/bash

# Navigate to the project directory (IMPORTANT: Use the absolute path)
cd /home/rodrigo/real_options || exit

# Activate your Python virtual environment
# This line assumes your venv is inside the project folder. Adjust if needed.
source venv/bin/activate

echo "--- Morning Pipeline Started at $(date) ---"

# Step 1: Pull the latest market and factor data
python coffee_factors_data_pull_new.py

# Step 2: Send the new data to the prediction API
python send_data_to_api.py

echo "--- Morning Pipeline Finished at $(date) ---"
