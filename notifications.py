"""Handles sending push notifications via the Pushover service.

This module provides a function to send customized messages to a specified
Pushover user. It is used throughout the application to alert the user of
important events, such as the start and stop of services, critical errors,
and trade execution summaries.
"""

import http.client
import urllib.parse
import logging

# --- Logging Setup ---
logger = logging.getLogger(__name__)


def send_pushover_notification(config: dict, title: str, message: str):
    """Sends a notification via the Pushover API.

    The function checks for a 'notifications' section in the provided
    configuration dictionary. If the section exists, is enabled, and contains
    the necessary Pushover user key and API token, it sends the specified

    Args:
        config (dict): A dictionary containing the notification settings,
            typically from the 'notifications' section of the main config.
            It should contain 'enabled', 'pushover_user_key', and
            'pushover_api_token'.
        title (str): The title of the notification.
        message (str): The main body of the notification. HTML is enabled.
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