# Filename: notifications.py
import http.client
import urllib.parse
import logging

# --- Logging Setup ---
logger = logging.getLogger(__name__)


def send_pushover_notification(config: dict, title: str, message: str):
    """
    Sends a notification via Pushover if the configuration is present and enabled.
    """
    if not config or not config.get('enabled', False):
        logger.info("Notifications are disabled. Skipping.")
        return

    user_key = config.get('pushover_user_key')
    api_token = config.get('pushover_api_token')

    if not all([user_key, api_token]):
        logger.warning("Pushover notification not sent. API key or user key is missing.")
        return

    try:
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
                     urllib.parse.urlencode({
                         "token": api_token,
                         "user": user_key,
                         "title": title,
                         "message": message,
                         "html": 1,
                     }), {"Content-type": "application/x-www-form-urlencoded"})

        response = conn.getresponse()
        response_data = response.read()

        if response.status == 200:
            logger.info("Pushover notification sent successfully.")
        else:
            logger.error(f"Failed to send Pushover notification. Status: {response.status}, Response: {response_data.decode()}")

        conn.close()
    except Exception as e:
        logger.error(f"An error occurred while sending a Pushover notification: {e}", exc_info=True)