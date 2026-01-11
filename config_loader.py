"""Handles loading the application's configuration from a JSON file.

This module provides a centralized way to access configuration settings,
ensuring that other modules can easily retrieve necessary parameters like
API keys, connection details, and trading settings.
"""

import json
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger("ConfigLoader")

def load_config() -> dict | None:
    """
    Loads config.json and overrides specific values from .env file (if present).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.json')
    env_path = os.path.join(base_dir, '.env')

    # 1. Load the Base JSON (Shared Settings)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        return None

    # 2. Load .env file (Environment Specifics)
    load_dotenv(env_path)

    # 3. OVERRIDE: Connection Settings
    # If .env has IB_PORT, use it. Otherwise keep config.json value.
    if os.getenv("IB_PORT"):
        config['connection']['port'] = int(os.getenv("IB_PORT"))
    
    if os.getenv("IB_CLIENT_ID"):
        config['connection']['clientId'] = int(os.getenv("IB_CLIENT_ID"))

    # 4. OVERRIDE: Flex Query (Secrets)
    if os.getenv("FLEX_TOKEN"):
        config['flex_query']['token'] = os.getenv("FLEX_TOKEN")
    
    if os.getenv("FLEX_QUERY_ID"):
        # Assuming single ID or comma-separated in env
        config['flex_query']['query_ids'] = os.getenv("FLEX_QUERY_ID").split(',')
        
    if os.getenv("FLEX_POSITIONS_ID"):
        config['flex_query']['active_positions_query_id'] = os.getenv("FLEX_POSITIONS_ID")

    # 5. OVERRIDE: Strategy Sizing (Safety)
    if os.getenv("STRATEGY_QTY"):
        config['strategy']['quantity'] = int(os.getenv("STRATEGY_QTY"))

    return config
