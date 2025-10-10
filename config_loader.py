"""Handles loading the application's configuration from a JSON file.

This module provides a centralized way to access configuration settings,
ensuring that other modules can easily retrieve necessary parameters like
API keys, connection details, and trading settings.
"""

import json
import os
import logging

logger = logging.getLogger("ConfigLoader")


def load_config() -> dict | None:
    """Loads the application configuration from the `config.json` file.

    The file is expected to be in the same directory as this script.
    This function is critical for initializing the application with the
    correct parameters.

    Returns:
        A dictionary containing the configuration settings if the file is
        found and successfully parsed. Returns None if the file cannot be
        found or if an error occurs during parsing.
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        return None