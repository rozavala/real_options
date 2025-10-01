import requests
import json
import os

def send_notification(title: str, message: str, sound: str = "falling"):
    """
    Sends a push notification via Pushover using credentials from config.json.
    'sound' can be changed for different alert types, e.g., 'cashregister' for success.
    """
    try:
        # Assumes config.json is in the same directory as the script calling this function
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        notif_config = config.get('notifications', {})
        if not notif_config.get('enabled'):
            return

        user_key = notif_config.get('pushover_user_key')
        api_token = notif_config.get('pushover_api_token')

        if not user_key or not api_token or 'YOUR_KEY' in user_key or 'YOUR_TOKEN' in api_token:
            print("Notification sending skipped: Pushover keys not configured.")
            return

        response = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": api_token,
            "user": user_key,
            "title": f"Coffee Trader Alert: {title}",
            "message": message,
            "sound": sound
        }, timeout=10)
        response.raise_for_status()
        print(f"Successfully sent notification: '{title}'")

    except FileNotFoundError:
        print("Error: Could not find config.json to load notification settings.")
    except Exception as e:
        print(f"An error occurred while sending notification: {e}")

