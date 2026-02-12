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

    if os.getenv("IB_HOST"):
        config['connection']['host'] = os.getenv("IB_HOST").strip()

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

        # Ensure host is present (default to localhost)
        if 'host' not in config['connection']:
            config['connection']['host'] = '127.0.0.1'

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

    # Validate Strategy
    if 'strategy' not in config:
        raise ValueError("Config validation: 'strategy' section is missing!")

    if 'quantity' not in config['strategy']:
        raise ValueError("Config validation: 'strategy.quantity' is missing!")

    if not isinstance(config['strategy']['quantity'], int) or config['strategy']['quantity'] <= 0:
        raise ValueError(f"Config validation: 'strategy.quantity' must be a positive integer, got {config['strategy']['quantity']}")

    # Validate Risk Management
    if 'risk_management' not in config:
         raise ValueError("Config validation: 'risk_management' section is missing!")

    if 'min_confidence_threshold' not in config['risk_management']:
         raise ValueError("Config validation: 'risk_management.min_confidence_threshold' is missing!")

    threshold = config['risk_management']['min_confidence_threshold']
    if not isinstance(threshold, float) or not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Config validation: 'risk_management.min_confidence_threshold' must be a float between 0.0 and 1.0, got {threshold}")

    # 6. OVERRIDE: Notifications (Secrets)
    notifications = config.get('notifications', {})

    # Load credentials from env if available
    for key, env_var in [('pushover_user_key', 'PUSHOVER_USER_KEY'), ('pushover_api_token', 'PUSHOVER_API_TOKEN')]:
        if os.getenv(env_var):
            notifications[key] = os.getenv(env_var)
        elif notifications.get(key) == "LOADED_FROM_ENV":
            # If not in env, and config has placeholder, clear it to fail validation below
            notifications[key] = None

    config['notifications'] = notifications

    # Validate Notifications
    if notifications.get('enabled'):
        missing_creds = []
        if not notifications.get('pushover_user_key'):
            missing_creds.append("PUSHOVER_USER_KEY")
        if not notifications.get('pushover_api_token'):
            missing_creds.append("PUSHOVER_API_TOKEN")

        if missing_creds:
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
    # Helper to load LLM keys (Loop to handle LOADED_FROM_ENV replacements)
    for provider in ['gemini', 'anthropic', 'openai', 'xai']:
        if provider in config and 'api_key' in config[provider]:
            if config[provider]['api_key'] == "LOADED_FROM_ENV":
                config[provider]['api_key'] = os.getenv(f"{provider.upper()}_API_KEY")

    # Safety Check (Optional but recommended)
    # Check for at least one LLM key
    llm_keys = [
        config.get('gemini', {}).get('api_key'),
        config.get('anthropic', {}).get('api_key'),
        config.get('openai', {}).get('api_key'),
        config.get('xai', {}).get('api_key')
    ]

    # Filter out empty strings or None
    valid_keys = [k for k in llm_keys if k and k != "LOADED_FROM_ENV"]

    if not valid_keys:
        raise ValueError("CRITICAL: No LLM API keys found! Please set at least one of GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or XAI_API_KEY.")

    if not config['gemini']['api_key']:
        logger.warning("WARNING: GEMINI_API_KEY not found in environment!")

    # 9. TRADING MODE: LIVE (default, backward compatible) or OFF (training/observation)
    trading_mode = os.getenv("TRADING_MODE", "LIVE").upper().strip()
    if trading_mode not in ("LIVE", "OFF"):
        logger.warning(f"Invalid TRADING_MODE '{trading_mode}', defaulting to LIVE")
        trading_mode = "LIVE"
    config['trading_mode'] = trading_mode
    if trading_mode == "OFF":
        logger.warning("*** TRADING MODE OFF — No real orders will be placed ***")

    # Log successful load
    loaded_providers = [p for p in ['gemini', 'anthropic', 'openai', 'xai'] if config.get(p, {}).get('api_key')]
    logger.info(f"Config loaded successfully. Mode: {trading_mode}. Providers: {', '.join(loaded_providers)}")

    return config
