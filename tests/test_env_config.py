import pytest
import os
from unittest.mock import MagicMock, patch
from trading_bot.utils import configure_market_data_type

class TestEnvConfig:
    @patch.dict(os.environ, {"ENV_NAME": "PROD"}, clear=True)
    def test_configure_market_data_type_prod(self):
        ib_mock = MagicMock()
        configure_market_data_type(ib_mock)
        ib_mock.reqMarketDataType.assert_called_with(1)

    @patch.dict(os.environ, {"ENV_NAME": "DEV"}, clear=True)
    def test_configure_market_data_type_dev(self):
        ib_mock = MagicMock()
        configure_market_data_type(ib_mock)
        ib_mock.reqMarketDataType.assert_called_with(3)

    @patch.dict(os.environ, {}, clear=True) # ENV_NAME missing
    def test_configure_market_data_type_missing_env(self):
        ib_mock = MagicMock()
        configure_market_data_type(ib_mock)
        # Should default to DEV (3)
        ib_mock.reqMarketDataType.assert_called_with(3)

    @patch.dict(os.environ, {"ENV_NAME": "PROD 🚀"}, clear=True)
    def test_configure_market_data_type_with_emoji(self):
        # Even if the old emoji value is present, it is NOT "PROD", so it should act as DEV
        ib_mock = MagicMock()
        configure_market_data_type(ib_mock)
        ib_mock.reqMarketDataType.assert_called_with(3)
