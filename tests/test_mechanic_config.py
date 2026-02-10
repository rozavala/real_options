import pytest
from unittest.mock import patch, mock_open
import config_loader
import os

@pytest.fixture
def base_config():
    return {
        "strategy": {"quantity": 1},
        "risk_management": {"min_confidence_threshold": 0.7},
        "connection": {"port": 7497, "clientId": 55},
        "notifications": {"enabled": False},
        "flex_query": {},
        "gemini": {"api_key": "dummy"},
        "anthropic": {"api_key": "dummy"},
        "openai": {"api_key": "dummy"},
        "xai": {"api_key": "dummy"}
    }

# Env vars that previous tests may leak via load_dotenv on the real .env file.
# Patching load_dotenv prevents THIS test from loading more, but vars already
# in the process env from earlier tests still need to be neutralized.
_CLEAN_ENV = {
    "GEMINI_API_KEY": "dummy",
    "FLEX_TOKEN": "",
    "STRATEGY_QTY": "",
    "IB_PORT": "",
    "IB_CLIENT_ID": "",
}

def test_strategy_quantity_validation(base_config):
    with patch("builtins.open", mock_open(read_data="{}")), \
         patch("json.load", return_value=base_config), \
         patch("config_loader.load_dotenv"), \
         patch.dict(os.environ, _CLEAN_ENV):

        # Valid
        config = config_loader.load_config()
        assert config['strategy']['quantity'] == 1

        # Invalid
        base_config['strategy']['quantity'] = 0
        with pytest.raises(ValueError, match="strategy.quantity"):
            config_loader.load_config()

def test_risk_threshold_validation(base_config):
    with patch("builtins.open", mock_open(read_data="{}")), \
         patch("json.load", return_value=base_config), \
         patch("config_loader.load_dotenv"), \
         patch.dict(os.environ, _CLEAN_ENV):

        # Valid
        config = config_loader.load_config()
        assert config['risk_management']['min_confidence_threshold'] == 0.7

        # Invalid
        base_config['risk_management']['min_confidence_threshold'] = 1.1
        with pytest.raises(ValueError, match="risk_management.min_confidence_threshold"):
            config_loader.load_config()

def test_connection_host_default(base_config):
    # Ensure host is missing in base config for this test
    if 'host' in base_config['connection']:
        del base_config['connection']['host']

    with patch("builtins.open", mock_open(read_data="{}")), \
         patch("json.load", return_value=base_config), \
         patch("config_loader.load_dotenv"), \
         patch.dict(os.environ, _CLEAN_ENV):

        config = config_loader.load_config()
        assert config['connection']['host'] == '127.0.0.1'
