import logging
import sys

def setup_logging():
    """
    Sets up a centralized logging configuration for the entire application.
    Logs will be directed to standard output with a consistent format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout  # Direct logs to standard output
    )