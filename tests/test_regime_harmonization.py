"""Tests for regime label harmonization."""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from trading_bot.weighted_voting import harmonize_regime

class TestHarmonizeRegime:
    def test_canonical_passthrough(self):
        for regime in ['HIGH_VOLATILITY', 'TRENDING', 'RANGE_BOUND', 'UNKNOWN']:
            assert harmonize_regime(regime) == regime

    def test_alias_mapping(self):
        assert harmonize_regime('HIGH_VOL') == 'HIGH_VOLATILITY'
        assert harmonize_regime('LOW_VOL') == 'RANGE_BOUND'
        assert harmonize_regime('VOLATILE') == 'HIGH_VOLATILITY'
        assert harmonize_regime('MEAN_REVERTING') == 'RANGE_BOUND'

    def test_unknown_fallback(self):
        assert harmonize_regime('NONSENSE') == 'UNKNOWN'
        assert harmonize_regime('') == 'UNKNOWN'
        assert harmonize_regime(None) == 'UNKNOWN'

    def test_case_insensitivity(self):
        assert harmonize_regime('high_volatility') == 'HIGH_VOLATILITY'
        assert harmonize_regime('Trending') == 'TRENDING'
        assert harmonize_regime('range_bound') == 'RANGE_BOUND'
        assert harmonize_regime('high_vol') == 'HIGH_VOLATILITY'
