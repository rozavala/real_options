import pytest
from unittest.mock import patch, mock_open
import os
import json
import sys

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_loader

# Sample config structure to mock file read
SAMPLE_CONFIG = {
    "gemini": {"api_key": "LOADED_FROM_ENV"},
    "anthropic": {"api_key": "LOADED_FROM_ENV"},
    "openai": {"api_key": "LOADED_FROM_ENV"},
    "xai": {"api_key": "LOADED_FROM_ENV"},
    "notifications": {
        "enabled": True,
        "pushover_user_key": "LOADED_FROM_ENV",
        "pushover_api_token": "LOADED_FROM_ENV"
    },
    "connection": {
        "port": 7497,
        "clientId": 1
    },
    "strategy": {
        "quantity": 1
    },
    "risk_management": {
        "min_confidence_threshold": 0.6
    }
}

@pytest.fixture
def mock_config_file():
    with patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_CONFIG))):
        yield

def test_missing_llm_keys_raises_value_error(mock_config_file):
    # Pass notification keys so we can reach LLM check
    with patch.dict(os.environ, {"PUSHOVER_USER_KEY": "dummy", "PUSHOVER_API_TOKEN": "dummy"}, clear=True):
        with patch("config_loader.load_dotenv"):
            with pytest.raises(ValueError, match="CRITICAL: No LLM API keys found"):
                config_loader.load_config()

def test_valid_llm_key_loads(mock_config_file):
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_gemini_key"}, clear=True):
        with patch("config_loader.load_dotenv"):
            # We must also provide notification keys if enabled, or disable them, or expect failure
            # Let's mock env for notifications too to pass that check
            with patch.dict(os.environ, {"PUSHOVER_USER_KEY": "u", "PUSHOVER_API_TOKEN": "a"}):
                config = config_loader.load_config()
                assert config["gemini"]["api_key"] == "test_gemini_key"

def test_notifications_missing_creds_raises_value_error(mock_config_file):
    # Provide an LLM key so we don't fail there
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True):
        with patch("config_loader.load_dotenv"):
             # Missing notification keys (cleared env)
            with pytest.raises(ValueError, match="Notifications enabled but credentials missing"):
                config_loader.load_config()

def test_notifications_valid_creds_loads(mock_config_file):
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "test_key",
        "PUSHOVER_USER_KEY": "user_key",
        "PUSHOVER_API_TOKEN": "api_token"
    }, clear=True):
        with patch("config_loader.load_dotenv"):
            config = config_loader.load_config()
            assert config["notifications"]["pushover_user_key"] == "user_key"
            assert config["notifications"]["pushover_api_token"] == "api_token"


def test_ib_host_env_override(mock_config_file):
    """IB_HOST env var should override the default connection host."""
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "test_key",
        "PUSHOVER_USER_KEY": "u",
        "PUSHOVER_API_TOKEN": "a",
        "IB_HOST": "100.64.0.1",
    }, clear=True):
        with patch("config_loader.load_dotenv"):
            config = config_loader.load_config()
            assert config["connection"]["host"] == "100.64.0.1"


def test_ib_paper_env_override(mock_config_file):
    """IB_PAPER=true should set connection.paper to True."""
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "test_key",
        "PUSHOVER_USER_KEY": "u",
        "PUSHOVER_API_TOKEN": "a",
        "IB_PAPER": "true",
    }, clear=True):
        with patch("config_loader.load_dotenv"):
            config = config_loader.load_config()
            assert config["connection"]["paper"] is True


def test_ib_paper_not_set_by_default(mock_config_file):
    """Without IB_PAPER, connection.paper should not be set."""
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "test_key",
        "PUSHOVER_USER_KEY": "u",
        "PUSHOVER_API_TOKEN": "a",
    }, clear=True):
        with patch("config_loader.load_dotenv"):
            config = config_loader.load_config()
            assert config["connection"].get("paper") is None


def test_ib_host_default_when_not_set(mock_config_file):
    """Without IB_HOST, connection.host should default to 127.0.0.1."""
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "test_key",
        "PUSHOVER_USER_KEY": "u",
        "PUSHOVER_API_TOKEN": "a",
    }, clear=True):
        with patch("config_loader.load_dotenv"):
            config = config_loader.load_config()
            assert config["connection"]["host"] == "127.0.0.1"
