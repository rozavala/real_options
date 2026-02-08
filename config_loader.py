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
        raise

    # 2. Load .env file (Environment Specifics)
    load_dotenv(env_path)

    # 3. OVERRIDE: Connection Settings
    # If .env has IB_PORT, use it. Otherwise keep config.json value.
    if os.getenv("IB_PORT"):
        config['connection']['port'] = int(os.getenv("IB_PORT"))
    
    if os.getenv("IB_CLIENT_ID"):
        config['connection']['clientId'] = int(os.getenv("IB_CLIENT_ID"))

    # Validate Connection Settings
    if 'connection' in config:
        if 'port' not in config['connection']:
            raise ValueError("Config validation: 'connection.port' is missing!")
        if 'clientId' not in config['connection']:
            raise ValueError("Config validation: 'connection.clientId' is missing!")

        if not isinstance(config['connection']['port'], int):
            raise TypeError(f"Config validation: 'connection.port' must be an integer, got {type(config['connection']['port']).__name__}")
        if not isinstance(config['connection']['clientId'], int):
            raise TypeError(f"Config validation: 'connection.clientId' must be an integer, got {type(config['connection']['clientId']).__name__}")

    # 4. OVERRIDE: Flex Query (Secrets)
    if os.getenv("FLEX_TOKEN"):
        config['flex_query']['token'] = os.getenv("FLEX_TOKEN")
    
    if os.getenv("FLEX_QUERY_ID"):
        # Assuming single ID or comma-separated in env
        config['flex_query']['query_ids'] = os.getenv("FLEX_QUERY_ID").split(',')
        
    if os.getenv("FLEX_POSITIONS_ID"):
        config['flex_query']['active_positions_query_id'] = os.getenv("FLEX_POSITIONS_ID")

    if os.getenv("FLEX_EQUITY_ID"):
        config['flex_query']['equity_query_id'] = os.getenv("FLEX_EQUITY_ID")

    # 5. OVERRIDE: Strategy Sizing (Safety)
    if os.getenv("STRATEGY_QTY"):
        config['strategy']['quantity'] = int(os.getenv("STRATEGY_QTY"))

    # 6. OVERRIDE: Notifications (Secrets)
    if os.getenv("PUSHOVER_USER_KEY"):
        if 'notifications' not in config:
            config['notifications'] = {}
        config['notifications']['pushover_user_key'] = os.getenv("PUSHOVER_USER_KEY")

    if os.getenv("PUSHOVER_API_TOKEN"):
        if 'notifications' not in config:
            config['notifications'] = {}
        config['notifications']['pushover_api_token'] = os.getenv("PUSHOVER_API_TOKEN")

    # Validate Notifications
    notifications = config.get('notifications', {})
    if notifications.get('enabled'):
        user_key = notifications.get('pushover_user_key')
        api_token = notifications.get('pushover_api_token')

        # Check if values are still the placeholders (meaning env var was missing)
        # or if they are missing/empty

        missing_creds = []
        if not user_key or user_key == "LOADED_FROM_ENV":
            # If overridden by env above, it would have the value.
            # If not overridden, it stays "LOADED_FROM_ENV" (if config.json has that).
            # So if it is "LOADED_FROM_ENV", it means env var was missing.
            missing_creds.append("PUSHOVER_USER_KEY")

        if not api_token or api_token == "LOADED_FROM_ENV":
            missing_creds.append("PUSHOVER_API_TOKEN")

        if missing_creds:
            # MECHANIC FIX: Fail fast if notifications are enabled but credentials are missing
            raise ValueError(f"Config validation: Notifications enabled but credentials missing: {', '.join(missing_creds)}")

    # 7. OVERRIDE: Data Providers (Secrets)
    if os.getenv("FRED_API_KEY"):
        config['fred_api_key'] = os.getenv("FRED_API_KEY")

    if config.get('fred_api_key') == "LOADED_FROM_ENV" or not config.get('fred_api_key'):
        logger.warning("WARNING: FRED_API_KEY not found in environment! Macro features may be limited.")

    if os.getenv("NASDAQ_API_KEY"):
        config['nasdaq_api_key'] = os.getenv("NASDAQ_API_KEY")

    if config.get('nasdaq_api_key') == "LOADED_FROM_ENV" or not config.get('nasdaq_api_key'):
        logger.warning("WARNING: NASDAQ_API_KEY not found in environment! Data feeds may be limited.")

    # Check X API Bearer Token
    if os.getenv("X_BEARER_TOKEN"):
        if 'x_api' not in config:
            config['x_api'] = {}
        config['x_api']['bearer_token'] = os.getenv("X_BEARER_TOKEN")

    x_api = config.get('x_api', {})
    if x_api.get('bearer_token') == "LOADED_FROM_ENV" or not x_api.get('bearer_token'):
        logger.warning("WARNING: X_BEARER_TOKEN not found! XSentimentSentinel will be disabled.")

    # 8. Models
    if config['gemini']['api_key'] == "LOADED_FROM_ENV":
        config['gemini']['api_key'] = os.getenv("GEMINI_API_KEY")

    if config['anthropic']['api_key'] == "LOADED_FROM_ENV":
        config['anthropic']['api_key'] = os.getenv("ANTHROPIC_API_KEY")

    if config['openai']['api_key'] == "LOADED_FROM_ENV":
        config['openai']['api_key'] = os.getenv("OPENAI_API_KEY")

    if config['xai']['api_key'] == "LOADED_FROM_ENV":
        config['xai']['api_key'] = os.getenv("XAI_API_KEY")
        
    # Safety Check (Optional but recommended)
    # Check for at least one LLM key
    llm_keys = [
        config['gemini']['api_key'],
        config['anthropic']['api_key'],
        config['openai']['api_key'],
        config['xai']['api_key']
    ]

    if not any(llm_keys):
        raise RuntimeError("CRITICAL: No LLM API keys found! Please set at least one of GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or XAI_API_KEY.")

    if not config['gemini']['api_key']:
        logger.warning("WARNING: GEMINI_API_KEY not found in environment!")

    return config
