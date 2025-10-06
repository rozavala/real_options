import requests
import time
import os
import pandas as pd
import json
from datetime import datetime

# --- Configuration ---
# Using the specific API URL from your original script
API_BASE_URL = "http://franciscos-mac-studio.tailccaee.ts.net:8000"
PREDICTION_ENDPOINT = f"{API_BASE_URL}/predictions"
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

def send_data_and_get_prediction():
    """
    Finds the latest data, sends it to the prediction API to start a job,
    and then polls for the result.
    """
    print("--- Starting Prediction Workflow ---")
    
    # 1. Find the latest data file
    latest_file = find_latest_data_file(DATA_FILE_PATH)
    if not latest_file:
        print("Error: No data file found. Please run 'coffee_factors_data_pull.py' first.")
        return

    print(f"Found latest data file: {os.path.basename(latest_file)}")

    # 2. Read data and format for API
    try:
        df = pd.read_csv(latest_file)
        # Convert the last N rows to JSON. The model might only need recent data.
        # This is more efficient than sending the whole file.
        data_payload = df.tail(600).to_json(orient='split')
        request_body = {"data": data_payload}
    except Exception as e:
        print(f"Error reading or processing data file: {e}")
        return

    # 3. Post the job to the API
    try:
        print(f"Sending data to {PREDICTION_ENDPOINT} to start prediction job...")
        response = requests.post(PREDICTION_ENDPOINT, json=request_body, timeout=15)
        response.raise_for_status()
        
        job_info = response.json()
        job_id = job_info.get('id')
        
        if not job_id:
            print("Error: API did not return a valid job ID.")
            return

        print(f"Successfully created prediction job. Job ID: {job_id}")
        
        # 4. Poll for the result (retained from the new version)
        result_url = f"{PREDICTION_ENDPOINT}/{job_id}"
        timeout_seconds = 300  # Max time to wait for a result
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            print("Polling for result...")
            result_response = requests.get(result_url, timeout=10)
            result_response.raise_for_status()
            
            status_info = result_response.json()
            job_status = status_info.get('status')
            
            if job_status == 'completed':
                print("\n--- Prediction Complete ---")
                print(json.dumps(status_info.get('result'), indent=2))
                return status_info.get('result')
            elif job_status == 'failed':
                print("\n--- Prediction Failed ---")
                print(f"Error message: {status_info.get('error')}")
                return None
            
            time.sleep(5) # Wait before polling again

        print("Error: Polling timed out. The prediction job took too long.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while communicating with the API: {e}")
    except json.JSONDecodeError:
        print("\nError: Could not decode the JSON response from the server.")
        print(f"Raw Response Text: {response.text}")

if __name__ == "__main__":
    send_data_and_get_prediction()

