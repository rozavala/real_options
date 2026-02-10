import pytest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config

class TestSecretLoading:
    @patch.dict(os.environ, {
        "PUSHOVER_USER_KEY": "env_pushover_user",
        "PUSHOVER_API_TOKEN": "env_pushover_token",
        "FRED_API_KEY": "env_fred_key",
        "NASDAQ_API_KEY": "env_nasdaq_key",
        "GEMINI_API_KEY": "fake_gemini_key"
    }, clear=False)
    def test_load_config_secrets_from_env(self):
        """Test that secrets are loaded from environment variables."""
        config = load_config()

        # Verify Notifications secrets
        assert config['notifications']['pushover_user_key'] == "env_pushover_user"
        assert config['notifications']['pushover_api_token'] == "env_pushover_token"

        # Verify FRED and Nasdaq secrets
        assert config['fred_api_key'] == "env_fred_key"
        assert config['nasdaq_api_key'] == "env_nasdaq_key"

    def test_load_config_defaults_without_env(self):
        """Test that validation rejects config when notifications enabled but creds missing."""
        with patch('config_loader.load_dotenv'):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}, clear=True):
                with pytest.raises(ValueError, match="credentials missing"):
                    load_config()
