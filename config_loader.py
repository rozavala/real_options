import json
import os
import logging

logger = logging.getLogger("ConfigLoader")

def load_config():
    """Loads the configuration from config.json."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        return None