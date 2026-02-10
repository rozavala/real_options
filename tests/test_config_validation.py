import unittest
from unittest.mock import patch, MagicMock
import os
import json
import logging

# Ensure we can import config_loader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_loader

class TestConfigValidation(unittest.TestCase):

    def setUp(self):
        self.base_config = {
            "connection": {"port": 7497, "clientId": 999},
            "notifications": {
                "enabled": False,
                "pushover_user_key": "LOADED_FROM_ENV",
                "pushover_api_token": "LOADED_FROM_ENV"
            },
            "fred_api_key": "LOADED_FROM_ENV",
            "nasdaq_api_key": "LOADED_FROM_ENV",
            "x_api": {"bearer_token": "LOADED_FROM_ENV"},
            "gemini": {"api_key": "LOADED_FROM_ENV"},
            "anthropic": {"api_key": "LOADED_FROM_ENV"},
            "openai": {"api_key": "LOADED_FROM_ENV"},
            "xai": {"api_key": "LOADED_FROM_ENV"}
        }

    @patch('config_loader.load_dotenv')
    @patch('builtins.open')
    @patch('json.load')
    @patch('os.path.exists')
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake_gemini_key"}, clear=True) # Valid LLM key
    def test_notifications_validation_failure(self, mock_exists, mock_json_load, mock_open, mock_load_dotenv):
        # Enable notifications but don't set env vars
        config = self.base_config.copy()
        config['notifications']['enabled'] = True

        mock_json_load.return_value = config

        with self.assertRaises(ValueError) as cm:
            config_loader.load_config()

        self.assertIn("Config validation: Notifications enabled but credentials missing", str(cm.exception))

    @patch('config_loader.load_dotenv')
    @patch('builtins.open')
    @patch('json.load')
    @patch('os.path.exists')
    @patch('config_loader.logger')
    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "fake_gemini_key",
        "PUSHOVER_USER_KEY": "fake_user",
        "PUSHOVER_API_TOKEN": "fake_token"
    }, clear=True)
    def test_notifications_validation_success(self, mock_logger, mock_exists, mock_json_load, mock_open, mock_load_dotenv):
        # Enable notifications and set env vars
        config = self.base_config.copy()
        config['notifications']['enabled'] = True

        mock_json_load.return_value = config

        # Should not raise
        loaded_config = config_loader.load_config()
        self.assertEqual(loaded_config['notifications']['pushover_user_key'], "fake_user")

    @patch('config_loader.load_dotenv')
    @patch('builtins.open')
    @patch('json.load')
    @patch('os.path.exists')
    @patch('config_loader.logger')
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake_gemini_key"}, clear=True)
    def test_missing_data_keys_warnings(self, mock_logger, mock_exists, mock_json_load, mock_open, mock_load_dotenv):
        # Don't set FRED/NASDAQ/X keys
        config = self.base_config.copy()
        mock_json_load.return_value = config

        config_loader.load_config()

        # Check for warnings
        warnings = [call.args[0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(any("FRED_API_KEY not found" in w for w in warnings))
        self.assertTrue(any("NASDAQ_API_KEY not found" in w for w in warnings))
        self.assertTrue(any("X_BEARER_TOKEN not found" in w for w in warnings))

if __name__ == '__main__':
    unittest.main()
