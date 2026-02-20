"""Configures the logging settings for the entire application.

This module provides a centralized function to set up a consistent
logging format and level, ensuring that all parts of the application
produce uniform and readable log messages.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def sanitize_log_message(record_msg: str) -> str:
    """Escapes newlines in log messages to prevent log injection attacks."""
    if isinstance(record_msg, str):
        return record_msg.replace('\n', '\\n').replace('\r', '\\r')
    return record_msg


class SanitizedFormatter(logging.Formatter):
    """Formatter that sanitizes log messages to prevent injection."""

    def formatMessage(self, record):
        # Save original message to avoid side effects
        original_message = record.message

        # Sanitize the message content
        record.message = sanitize_log_message(record.message)

        # Call parent to format the final string (e.g. "%(asctime)s ...")
        s = super().formatMessage(record)

        # Restore original message
        record.message = original_message

        return s


def setup_logging(log_file: str = None):
    """Sets up a centralized logging configuration for the application.

    Args:
        log_file: Optional path to log file. If None, logs only to stdout.
                  Examples: "logs/orchestrator.log", "logs/dashboard.log"

    This function configures the root logger to output messages of level
    INFO and higher. If log_file is provided, uses RotatingFileHandler.
    To prevent log duplication in environments where stdout is redirected
    to the same log file (e.g. via nohup/deploy.sh), StreamHandler is
    only attached if the session is interactive or no log_file is set.
    """

    handlers = []

    # Use SanitizedFormatter for security
    formatter = SanitizedFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add file handler if requested
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            # Rotating File Handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB per file
                backupCount=5,               # Keep 5 rotated files
                encoding='utf-8',
            )

            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        except (PermissionError, OSError) as e:
            # Fallback to stdout only if we can't write to the log file
            sys.stderr.write(f"WARNING: Could not create log file handler at {log_file}: {e}\n")
            sys.stderr.write("Logging will continue to stdout only.\n")
            fallback_handler = logging.StreamHandler(sys.stdout)
            fallback_handler.setFormatter(formatter)
            handlers.append(fallback_handler)

    # Determine whether to add StreamHandler (Stdout)
    # 1. Always add if no log_file is provided (default behavior)
    # 2. If log_file IS provided, only add StreamHandler if we are in an interactive terminal.
    #    This avoids duplication when running via deploy.sh/nohup where stdout is redirected to the log file.
    should_add_stream = (log_file is None) or sys.stdout.isatty()

    if should_add_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    # Basic Config with handlers
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True  # Override any existing configuration
    )

    # --- Quieter Logging for Third-Party Libs ---
    logging.getLogger('ib_insync').setLevel(logging.WARNING)
    logging.getLogger('google_genai').setLevel(logging.WARNING)
    logging.getLogger('google_genai.models').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
