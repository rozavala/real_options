"""Client script for interacting with the prediction API.

This script is responsible for sending data to the prediction service and
retrieving the results. It performs the following steps:
1. Finds the most recent coffee futures data file in the project directory.
2. Reads the data and sends it to the `/predictions` endpoint of the API.
3. Polls the status URL provided in the initial API response until the
   prediction job is completed or fails.
4. Logs the final result or any errors that occur during the process.
"""

import requests
import time
import os
import pandas as pd
import json
import logging
from datetime import datetime

from config_loader import load_config
from logging_config import setup_logging

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("APIAgent")

# The directory where this script is located, used to find the data file.
DATA_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def find_latest_data_file(directory: str) -> str | None:
    """Finds the most recently created coffee data CSV in a directory.

    It scans the given directory for files matching the pattern
    'coffee_futures_data_*.csv' and returns the path to the one with the
    most recent modification time.

    Args:
        directory (str): The absolute path to the directory to search.

    Returns:
        The full path to the latest data file, or None if no matching
        file is found or an error occurs.
    """
    try:
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.startswith("coffee_futures_data_") and f.endswith(".csv")
        ]
        if not files:
            return None
        # Return the file with the latest modification time.
        return max(files, key=os.path.getmtime)
    except OSError as e:
        print(f"Error accessing directory {directory}: {e}")
        return None


def send_data_and_get_prediction(config: dict) -> dict | None:
    """Sends data to the prediction API and polls for the result.

    This function orchestrates the entire client-side prediction workflow.
    It finds the latest data file, sends its content to the API to start a
    prediction job, and then polls the job status endpoint until a result
is
    available or a timeout is reached.

    Args:
        config (dict): The application configuration dictionary, which must
            contain the 'api_base_url'.

    Returns:
        A dictionary containing the prediction result if the job completes
        successfully. Returns None if any step fails, including API errors,
        timeouts, or file-not-found issues.
    """
    api_base_url = config.get('api_base_url')
    if not api_base_url:
        logger.error("API base URL not found in configuration. Cannot proceed.")
        return None

    prediction_endpoint = f"{api_base_url}/predictions"
    logger.info("--- Starting Prediction Workflow ---")
    
    latest_file = find_latest_data_file(DATA_FILE_PATH)
    if not latest_file:
        logger.error("No data file found. Please run the data pull process first.")
        return None

    logger.info(f"Found latest data file: {os.path.basename(latest_file)}")

    try:
        df = pd.read_csv(latest_file)
        # Send the last 600 data points as a CSV string.
        data_payload = df.tail(600).to_csv()
        request_body = {"data": data_payload}
    except Exception as e:
        logger.error(f"Error reading or processing data file: {e}", exc_info=True)
        return None

    try:
        logger.info(f"Sending data to {prediction_endpoint} to start prediction job...")
        response = requests.post(prediction_endpoint, json=request_body, timeout=15)
        response.raise_for_status()
        
        job_info = response.json()
        job_id = job_info.get('id')
        if not job_id:
            logger.error("API did not return a valid job ID.")
            return None

        logger.info(f"Successfully created prediction job. Job ID: {job_id}")
        
        result_url = f"{prediction_endpoint}/{job_id}"
        timeout_seconds = 120
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            logger.info("Polling for result...")
            result_response = requests.get(result_url, timeout=10)
            result_response.raise_for_status()
            
            status_info = result_response.json()
            if status_info.get('status') == 'completed':
                logger.info("\n--- Prediction Complete ---")
                logger.info(json.dumps(status_info.get('result'), indent=2))
                return status_info.get('result')
            elif status_info.get('status') == 'failed':
                logger.error(f"Prediction failed. Error: {status_info.get('error')}")
                return None
            
            time.sleep(5)

        logger.error("Polling timed out. The prediction job took too long.")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while communicating with the API: {e}", exc_info=True)
        return None
    except json.JSONDecodeError:
        logger.error("Could not decode the JSON response from the server.", exc_info=True)
        return None


if __name__ == "__main__":
    logger.info("Running API agent as a standalone script.")
    config = load_config()
    if config:
        send_data_and_get_prediction(config)
    else:
        logger.error("Failed to load configuration. Cannot run standalone script.")