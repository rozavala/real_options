"""Handles sending push notifications via the Pushover service.

This module provides a function to send customized messages to a specified
Pushover user. It is used throughout the application to alert the user of
important events, such as the start and stop of services, critical errors,
and trade execution summaries.
"""

import http.client
import urllib.parse
import logging
import os
import mimetypes

# --- Logging Setup ---
logger = logging.getLogger(__name__)


def send_pushover_notification(config: dict, title: str, message: str, attachment_path: str = None):
    """Sends a notification with an optional image attachment via the Pushover API.

    Args:
        config (dict): Notification settings from the main config.
        title (str): The title of the notification.
        message (str): The main body of the notification. HTML is enabled.
        attachment_path (str, optional): The file path to an image to attach.
    """
    if not config or not config.get('enabled', False):
        logger.info("Notifications are disabled. Skipping.")
        return

    user_key = config.get('pushover_user_key')
    api_token = config.get('pushover_api_token')
    if not all([user_key, api_token]):
        logger.warning("Pushover credentials missing. Notification not sent.")
        return

    data = {
        "token": api_token,
        "user": user_key,
        "title": title,
        "message": message,
        "html": 1
    }

    try:
        conn = http.client.HTTPSConnection("api.pushover.net:443")

        if attachment_path and os.path.exists(attachment_path):
            # Use multipart/form-data for file uploads
            content_type, body = _encode_multipart_formdata(data, attachment_path)
            headers = {"Content-Type": content_type}
        else:
            # Use standard urlencoded for simple messages
            body = urllib.parse.urlencode(data)
            headers = {"Content-type": "application/x-www-form-urlencoded"}

        conn.request("POST", "/1/messages.json", body, headers)
        response = conn.getresponse()
        response_data = response.read()

        if response.status == 200:
            logger.info(f"Pushover notification sent successfully (Attachment: {bool(attachment_path)}).")
        else:
            logger.error(f"Failed to send Pushover notification. Status: {response.status}, Response: {response_data.decode()}")

        conn.close()
    except Exception as e:
        logger.error(f"An error occurred while sending Pushover notification: {e}", exc_info=True)

def _encode_multipart_formdata(fields: dict, file_path: str) -> tuple[str, bytes]:
    """Encodes form fields and a file for a multipart/form-data request."""
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = bytearray()

    for key, value in fields.items():
        body.extend(b'--' + boundary.encode('ascii') + b'\r\n')
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode('utf-8'))
        body.extend(str(value).encode('utf-8') + b'\r\n')

    filename = os.path.basename(file_path)
    content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

    body.extend(b'--' + boundary.encode('ascii') + b'\r\n')
    body.extend(f'Content-Disposition: form-data; name="attachment"; filename="{filename}"\r\n'.encode('utf-8'))
    body.extend(f'Content-Type: {content_type}\r\n\r\n'.encode('utf-8'))

    with open(file_path, 'rb') as f:
        body.extend(f.read())

    body.extend(b'\r\n--' + boundary.encode('ascii') + b'--\r\n')

    content_type_header = f'multipart/form-data; boundary={boundary}'
    return content_type_header, bytes(body)