"""Configures the logging settings for the entire application.

This module provides a centralized function to set up a consistent
logging format and level, ensuring that all parts of the application
produce uniform and readable log messages.
"""

import logging
import sys


def setup_logging():
    """Sets up a centralized logging configuration for the application.

    This function configures the root logger to output messages of level
    INFO and higher to standard output. The log format includes the timestamp,
    logger name, log level, and the message, providing clear and consistent
    context for all log entries.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout  # Direct logs to standard output
    )
    # --- Quieter Logging for ib_insync ---
    # The ib_insync library is very verbose at the INFO level. To avoid
    # spamming the logs with routine messages (like 'updatePortfolio'),
    # we set its specific logger to the WARNING level.
    logging.getLogger('ib_insync').setLevel(logging.WARNING)