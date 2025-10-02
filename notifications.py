# Filename: notifications.py
import requests
import json

def send_pushover_notification(config, title, message):
    """
    Sends a notification via the Pushover API.

    Args:
        config (dict): A dictionary containing Pushover API credentials.
                       Expected keys: 'pushover_user_key', 'pushover_api_token'.
        title (str): The title of the notification.
        message (str): The body of the notification.

    Returns:
        bool: True if the notification was sent successfully, False otherwise.
    """
    user_key = config.get('pushover_user_key')
    api_token = config.get('pushover_api_token')

    if not user_key or not api_token or user_key == "YOUR_USER_KEY" or api_token == "YOUR_API_TOKEN":
        print("Pushover credentials are not configured. Skipping notification.")
        return False

    url = "https://api.pushover.net/1/messages.json"
    payload = {
        'token': api_token,
        'user': user_key,
        'title': title,
        'message': message,
        'html': 1  # Enable HTML formatting
    }
    
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Pushover notification sent successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Pushover notification: {e}")
        return False
