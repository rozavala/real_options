import requests
import os
from datetime import datetime
import json

def send_data_to_prediction_api():
    """
    Reads the coffee factors data from the CSV file for the current day,
    and sends it to the prediction API.
    """
    # --- 1. Define API and File Details ---
    api_url = "http://100.125.76.78:8000/predictions"
    
    # Dynamically generate the filename for today's date
    today_str = datetime.now().strftime('%Y-%m-%d')
    csv_filename = f"coffee_futures_data_{today_str}.csv"
    
    # Get the absolute path to the CSV file (assuming it's in the same directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filepath = os.path.join(script_dir, csv_filename)

    # --- 2. Check if the CSV File Exists ---
    if not os.path.exists(csv_filepath):
        print(f"Error: The file '{csv_filename}' was not found.")
        print("Please ensure you have run the 'coffee_factors_data_pull.py' script today.")
        return

    # --- 3. Read the CSV Data ---
    print(f"Reading data from '{csv_filename}'...")
    try:
        with open(csv_filepath, 'r') as f:
            csv_content = f.read()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- 4. Prepare and Send the API Request ---
    payload = {
        "data": csv_content
    }

    print(f"Sending data to the prediction API at {api_url}...")
    try:
        response = requests.post(api_url, json=payload)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # --- 5. Process the Response ---
        response_data = response.json()
        job_id = response_data.get('id')
        status = response_data.get('status')
        
        print("\n--- Successfully submitted prediction job ---")
        print(f"  Status: {status}")
        print(f"  Job ID: {job_id}")
        print("---------------------------------------------")
        print(f"\nYou can check the result later using: curl {api_url}/{job_id}")

    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while sending the request to the API: {e}")
    except json.JSONDecodeError:
        print("\nError: Could not decode the JSON response from the server.")
        print(f"Raw Response Text: {response.text}")


if __name__ == "__main__":
    send_data_to_prediction_api()
