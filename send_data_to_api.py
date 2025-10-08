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

DATA_FILE_PATH = os.path.dirname(os.path.abspath(__file__)) # Directory of this script

def find_latest_data_file(directory: str) -> str | None:
    """Finds the most recently created coffee data CSV in the specified directory."""
    try:
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.startswith("coffee_futures_data_") and f.endswith(".csv")
        ]
        if not files:
            return None
        # This is more robust as it finds the latest file, even if not from today
        return max(files, key=os.path.getmtime)
    except OSError as e:
        print(f"Error accessing directory {directory}: {e}")
        return None

def send_data_and_get_prediction(config: dict):
    """
    Finds the latest data, sends it to the prediction API, and polls for the result.
    The API endpoint is determined by the 'api_base_url' in the config.
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

