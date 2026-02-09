"""Handles sending push notifications via the Pushover service."""

import logging
import requests
import os

logger = logging.getLogger(__name__)

# Pushover API limits
PUSHOVER_TITLE_LIMIT = 250
PUSHOVER_MESSAGE_LIMIT = 1024

def _split_message(message: str, limit: int) -> list[str]:
    """Splits a long message into chunks that respect the limit, preserving lines."""
    if len(message) <= limit:
        return [message]

    chunks = []
    current_chunk = ""
    for line in message.splitlines(keepends=True):
        if len(current_chunk) + len(line) > limit:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += line
    chunks.append(current_chunk) # Add the last part
    return chunks

def send_pushover_notification(config: dict, title: str, message: str, attachment_path: str = None, monospace: bool = False):
    """
    Sends a notification via Pushover, splitting long messages into multiple parts.
    Only the first part will contain an attachment.
    """
    if not config or not config.get('enabled', False):
        logger.info("Notifications are disabled. Skipping.")
        return

    user_key = config.get('pushover_user_key')
    api_token = config.get('pushover_api_token')
    if not all([user_key, api_token]):
        logger.warning("Pushover credentials missing. Notification not sent.")
        return

    # Load the environment name, defaulting to nothing if not set
    env_prefix = os.getenv("ENV_NAME", "")

    # Prepend OFF indicator if trading is disabled
    if os.getenv("TRADING_MODE", "LIVE").upper().strip() == "OFF":
        env_prefix = f"OFF {env_prefix}".strip() if env_prefix else "OFF"

    # Only add the prefix if it exists
    if env_prefix:
        title = f"{env_prefix} - {title}"
    
    # Truncate title if it's too long
    if len(title) > PUSHOVER_TITLE_LIMIT:
        title = title[:PUSHOVER_TITLE_LIMIT - 3] + "..."
        logger.warning("Notification title was truncated.")

    message_chunks = _split_message(message, PUSHOVER_MESSAGE_LIMIT)
    total_chunks = len(message_chunks)

    for i, chunk in enumerate(message_chunks):
        part_title = f"{title} ({i+1}/{total_chunks})" if total_chunks > 1 else title
        is_first_chunk = (i == 0)

        # --- Send the actual notification for the chunk ---
        _send_single_pushover_chunk(
            api_token,
            user_key,
            part_title,
            chunk,
            attachment_path if is_first_chunk else None, # Only attach to the first part
            monospace
        )

def _send_single_pushover_chunk(api_token: str, user_key: str, title: str, message: str, attachment_path: str, monospace: bool):
    """Helper function to send one part of a potentially split notification."""
    payload = {
        "token": api_token, "user": user_key, "title": title, "message": message
    }
    if monospace:
        payload['monospace'] = 1
    else:
        payload['html'] = 1

    files = {}
    if attachment_path and os.path.exists(attachment_path):
        try:
            files['attachment'] = (os.path.basename(attachment_path), open(attachment_path, 'rb'), 'image/png')
        except IOError as e:
            logger.error(f"Error opening attachment file {attachment_path}: {e}")
            attachment_path = None

    try:
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data=payload,
            files=files or None,
            timeout=15
        )
        response.raise_for_status()
        logger.info(f"Pushover chunk sent successfully (Attachment: {bool(attachment_path)}).")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Pushover chunk: {e}")
    finally:
        if 'attachment' in files:
            files['attachment'][1].close()
