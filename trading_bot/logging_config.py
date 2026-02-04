"""Configures the logging settings for the entire application.

This module provides a centralized function to set up a consistent
logging format and level, ensuring that all parts of the application
produce uniform and readable log messages.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def setup_logging():
    """Sets up a centralized logging configuration for the application.

    This function configures the root logger to output messages of level
    INFO and higher to standard output AND rotating log file. The log format includes the timestamp,
    logger name, log level, and the message, providing clear and consistent
    context for all log entries.
    """

    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "orchestrator.log")

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB per file
        backupCount=5,               # Keep 5 rotated files
        encoding='utf-8',
    )

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Basic Config with handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler
        ]
    )
    # --- Quieter Logging for Third-Party Libs ---
    logging.getLogger('ib_insync').setLevel(logging.WARNING)

    # Silence noisy GenAI and HTTP libs (Fix P1)
    logging.getLogger('google_genai').setLevel(logging.WARNING)
    logging.getLogger('google_genai.models').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
