"""Handles sending push notifications via the Pushover service."""

import logging
import requests
import os

logger = logging.getLogger(__name__)

def send_pushover_notification(config: dict, title: str, message: str, attachment_path: str = None, monospace: bool = False):
    """Sends a notification with an optional image attachment via the Pushover API.

    Args:
        config (dict): Notification settings from the main config.
        title (str): The title of the notification.
        message (str): The main body of the notification. HTML is enabled by default.
        attachment_path (str, optional): The file path to an image to attach.
        monospace (bool, optional): If True, formats the message as monospaced. Defaults to False.
    """
    if not config or not config.get('enabled', False):
        logger.info("Notifications are disabled. Skipping.")
        return

    user_key = config.get('pushover_user_key')
    api_token = config.get('pushover_api_token')
    if not all([user_key, api_token]):
        logger.warning("Pushover credentials missing. Notification not sent.")
        return

    # Base payload
    payload = {
        "token": api_token,
        "user": user_key,
        "title": title,
        "message": message,
    }

    # Set either HTML or Monospace, but not both
    if monospace:
        payload['monospace'] = 1
    else:
        payload['html'] = 1


    files = {}
    if attachment_path and os.path.exists(attachment_path):
        try:
            # Prepare the file for multipart upload
            files['attachment'] = (os.path.basename(attachment_path), open(attachment_path, 'rb'), 'image/png')
            logger.info(f"Attaching file: {attachment_path}")
        except IOError as e:
            logger.error(f"Error opening attachment file {attachment_path}: {e}")
            attachment_path = None # Clear attachment if file can't be opened

    try:
        # The 'files' parameter in requests automatically handles multipart/form-data encoding
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data=payload,
            files=files if files else None,
            timeout=15 # Add a timeout for robustness
        )
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)

        logger.info(f"Pushover notification sent successfully (Attachment: {bool(attachment_path)}).")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Pushover notification: {e}")
    finally:
        # Ensure the file is closed if it was opened
        if 'attachment' in files and files['attachment']:
            files['attachment'][1].close()
