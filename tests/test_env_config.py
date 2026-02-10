import pytest
import os
from unittest.mock import MagicMock, patch
from trading_bot.utils import configure_market_data_type

class TestEnvConfig:
    @patch.dict(os.environ, {"ENV_NAME": "PROD"}, clear=True)
    def test_configure_market_data_type_prod(self):
        """PROD mode should use Live data (Type 1)"""
        ib_mock = MagicMock()
        configure_market_data_type(ib_mock)
        ib_mock.reqMarketDataType.assert_called_with(1)

    @patch.dict(os.environ, {"ENV_NAME": "DEV"}, clear=True)
    def test_configure_market_data_type_dev_default(self):
        """DEV mode should now use Live data (Type 1) by default"""
        ib_mock = MagicMock()
        configure_market_data_type(ib_mock)
        ib_mock.reqMarketDataType.assert_called_with(1)  # Changed from 3 to 1

    @patch.dict(os.environ, {"ENV_NAME": "DEV", "FORCE_DELAYED_DATA": "1"}, clear=True)
    def test_configure_market_data_type_dev_forced_delayed(self):
        """DEV mode with FORCE_DELAYED_DATA should use Delayed (Type 3)"""
        ib_mock = MagicMock()
        configure_market_data_type(ib_mock)
        ib_mock.reqMarketDataType.assert_called_with(3)

    @patch.dict(os.environ, {}, clear=True)
    def test_configure_market_data_type_missing_env(self):
        """Missing ENV_NAME should default to Live (Type 1)"""
        ib_mock = MagicMock()
        configure_market_data_type(ib_mock)
        ib_mock.reqMarketDataType.assert_called_with(1)  # Changed from 3 to 1
