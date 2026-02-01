import pytest
from unittest.mock import MagicMock, patch
from trading_bot.utils import (
    get_contract_multiplier, get_dollar_multiplier,
    get_tick_size, get_ibkr_exchange, CENTS_INDICATORS
)
from config.commodity_profiles import CommodityProfile, ContractSpec

# Mock profiles
@pytest.fixture
def mock_get_profile():
    with patch('config.commodity_profiles.get_commodity_profile') as mock:
        yield mock

def test_coffee_profile(mock_get_profile):
    # Setup KC Profile
    kc_profile = MagicMock(spec=CommodityProfile)
    kc_profile.contract = MagicMock(spec=ContractSpec)
    kc_profile.contract.contract_size = 37500.0
    kc_profile.contract.unit = "cents/lb"
    kc_profile.contract.tick_size = 0.05
    kc_profile.contract.exchange = "ICE"

    mock_get_profile.return_value = kc_profile
    config = {'commodity': {'ticker': 'KC'}}

    assert get_contract_multiplier(config) == 37500.0
    # 37500 / 100 because 'cents' in unit
    assert get_dollar_multiplier(config) == 375.0
    assert get_tick_size(config) == 0.05
    assert get_ibkr_exchange(config) == 'NYBOT'

def test_cocoa_profile(mock_get_profile):
    # Setup CC Profile
    cc_profile = MagicMock(spec=CommodityProfile)
    cc_profile.contract = MagicMock(spec=ContractSpec)
    cc_profile.contract.contract_size = 10.0
    cc_profile.contract.unit = "$/metric ton"
    cc_profile.contract.tick_size = 1.0
    cc_profile.contract.exchange = "ICE"

    mock_get_profile.return_value = cc_profile
    config = {'commodity': {'ticker': 'CC'}}

    assert get_contract_multiplier(config) == 10.0
    # 10.0 because '$' (no cents) in unit
    assert get_dollar_multiplier(config) == 10.0
    assert get_tick_size(config) == 1.0
    assert get_ibkr_exchange(config) == 'NYBOT'

class TestDollarMultiplierEdgeCases:
    def test_cent_symbol(self, mock_get_profile):
        """Profile with ¢/lb unit still divides by 100."""
        profile = MagicMock(spec=CommodityProfile)
        profile.contract = MagicMock(spec=ContractSpec)
        profile.contract.contract_size = 50000.0
        profile.contract.unit = "¢/lb" # Unicode symbol

        mock_get_profile.return_value = profile
        config = {'commodity': {'ticker': 'TEST'}}

        assert get_dollar_multiplier(config) == 500.0 # 50000 / 100

    def test_usc_symbol(self, mock_get_profile):
        """Profile with USc/lb unit still divides by 100."""
        profile = MagicMock(spec=CommodityProfile)
        profile.contract = MagicMock(spec=ContractSpec)
        profile.contract.contract_size = 50000.0
        profile.contract.unit = "USc/lb"

        mock_get_profile.return_value = profile
        config = {'commodity': {'ticker': 'TEST'}}

        assert get_dollar_multiplier(config) == 500.0

    def test_indicators_constant(self):
        assert '¢' in CENTS_INDICATORS
        assert 'cent' in CENTS_INDICATORS
        assert 'usc' in CENTS_INDICATORS
