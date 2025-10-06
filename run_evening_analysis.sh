#!/bin/bash

# Navigate to the project directory (IMPORTANT: Use the absolute path)
cd /home/rodrigo/real_options || exit

# Activate your Python virtual environment
source venv/bin/activate

echo "--- Evening Analysis Started at $(date) ---"

# Run the performance analyzer
python performance_analyzer.py

echo "--- Evening Analysis Finished at $(date) ---"
