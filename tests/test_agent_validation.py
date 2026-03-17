"""Tests for agent output validation: direction-evidence matching and number whitelisting."""

import pytest
from unittest.mock import MagicMock, patch
from trading_bot.observability import count_directional_evidence, BULLISH_WORDS, BEARISH_WORDS


class TestDirectionalEvidenceCounting:
    """v9.0: Updated after removing ambiguous logistics words from BEARISH_WORDS.

    Words like 'building', 'inflows', 'arrivals', 'restocking', 'replenishment'
    were removed because they triggered false direction-evidence mismatches
    on bullish supply-recovery reports (e.g., "demand building" is bullish,
    not bearish). Only unambiguously bearish words remain.
    """

    def test_unambiguous_oversupply_language_counted_bearish(self):
        """Clear oversupply terms should register as bearish evidence."""
        text = "Surplus bumper crop leads to glut and oversupply in warehouses"
        bullish, bearish = count_directional_evidence(text)
        assert bearish >= 3, f"Expected >=3 bearish words, got {bearish}"

    def test_existing_bearish_words_unchanged(self):
        """Core bearish vocabulary still works."""
        text = "Surplus bumper crop leads to glut and selloff in warehouses"
        bullish, bearish = count_directional_evidence(text)
        assert bearish >= 3

    def test_existing_bullish_words_unchanged(self):
        """Existing bullish vocabulary still works."""
        text = "Severe drought causes shortage and deficit with tight supply"
        bullish, bearish = count_directional_evidence(text)
        assert bullish >= 4

    def test_ambiguous_logistics_words_no_longer_bearish(self):
        """Ambiguous logistics words should NOT count as bearish evidence.

        These words caused false direction-evidence mismatches on bullish
        supply-recovery reports, contributing to systematic bearish bias.
        """
        for word in ('building', 'inflows', 'arrivals', 'replenishment', 'restocking',
                     'easing', 'normalizing', 'resolved', 'resolution'):
            assert word not in BEARISH_WORDS, f"'{word}' should have been removed from BEARISH_WORDS"

    def test_negation_still_works(self):
        """Negation flips direction."""
        text = "There is not a surplus at current levels"
        bullish, bearish = count_directional_evidence(text)
        # "not surplus" → bullish (negated bearish)
        assert bullish >= 1

    def test_inventory_report_now_balanced(self):
        """A typical inventory report with mixed language should not be heavily bearish."""
        text = (
            "ICE certified stocks rose to 435,494 bags. "
            "Shortage concerns persist as supply tightness continues."
        )
        bullish, bearish = count_directional_evidence(text)
        # rose + shortage + tightness = bullish; stocks is neutral
        assert bullish >= 2, f"Expected >=2 bullish, got {bullish}"


class TestNumberWhitelistArithmetic:
    """Fix A: Arithmetic derivation whitelist for inventory/supply_chain agents."""

    @pytest.fixture
    def council(self):
        """Minimal CoffeeCouncil mock with _validate_agent_output accessible."""
        with patch("google.genai.Client"):
            from trading_bot.agents import CoffeeCouncil
            config = {"gemini": {"api_key": "TEST", "personas": {}}}
            c = CoffeeCouncil(config)
            return c

    def test_inventory_sum_whitelisted(self, council):
        """Numbers that are sums of source numbers should not be flagged."""
        # Source has 393759 and 41735 → sum = 435494
        grounded = "Regional stocks: 393759 bags in Europe, 41735 bags in US warehouses."
        analysis = (
            '{"sentiment": "BEARISH", "confidence": 0.65, '
            '"evidence": "Total certified stocks reached 435494 bags, '
            'combining 393759 European and 41735 US warehouse figures."}'
        )
        is_valid, issues, _ = council._validate_agent_output('inventory', analysis, grounded)
        hallucination_issues = [i for i in issues if 'hallucination' in i.lower()]
        assert len(hallucination_issues) == 0, f"Sum 435494 should be whitelisted: {issues}"

    def test_inventory_difference_whitelisted(self, council):
        """Numbers that are differences of source numbers should not be flagged."""
        grounded = "Current stocks 477229 bags, previous week 435494 bags."
        analysis = (
            '{"sentiment": "BEARISH", "confidence": 0.65, '
            '"evidence": "Weekly build of 41735 bags, from 435494 to 477229."}'
        )
        is_valid, issues, _ = council._validate_agent_output('inventory', analysis, grounded)
        hallucination_issues = [i for i in issues if 'hallucination' in i.lower()]
        assert len(hallucination_issues) == 0, f"Difference 41735 should be whitelisted: {issues}"

    def test_inventory_proximity_whitelisted(self, council):
        """Numbers within 5% of source should not be flagged (rounding diffs)."""
        grounded = "Certified stocks approximately 435000 bags."
        analysis = (
            '{"sentiment": "BEARISH", "confidence": 0.65, '
            '"evidence": "Stocks at 435494 bags according to ICE report."}'
        )
        is_valid, issues, _ = council._validate_agent_output('inventory', analysis, grounded)
        hallucination_issues = [i for i in issues if 'hallucination' in i.lower()]
        assert len(hallucination_issues) == 0, f"435494 ~= 435000 should be whitelisted: {issues}"

    def test_truly_fabricated_numbers_still_flagged(self, council):
        """Numbers with no derivation path should still be flagged."""
        grounded = "Stocks at 100000 bags, weekly change 5000 bags."
        analysis = (
            '{"sentiment": "BEARISH", "confidence": 0.65, '
            '"evidence": "Total global supply reached 9999999 bags with 7777777 in transit '
            'and 5555555 pending and 3333333 certified."}'
        )
        is_valid, issues, _ = council._validate_agent_output('inventory', analysis, grounded)
        hallucination_issues = [i for i in issues if 'hallucination' in i.lower()]
        assert len(hallucination_issues) > 0, "Truly fabricated numbers should be flagged"

    def test_supply_chain_also_whitelisted(self, council):
        """Supply chain agent gets the same arithmetic whitelist."""
        grounded = "Port throughput 150000 tons, vessel capacity 50000 tons."
        analysis = (
            '{"sentiment": "BEARISH", "confidence": 0.65, '
            '"evidence": "Combined capacity 200000 tons available."}'
        )
        is_valid, issues, _ = council._validate_agent_output('supply_chain', analysis, grounded)
        hallucination_issues = [i for i in issues if 'hallucination' in i.lower()]
        assert len(hallucination_issues) == 0, f"Sum 200000 should be whitelisted: {issues}"

    def test_technical_agent_unchanged(self, council):
        """Technical agent still uses price-proximity, not arithmetic."""
        grounded = "Current price 283.20, SMA 329.19."
        analysis = (
            '{"sentiment": "BEARISH", "confidence": 0.65, '
            '"evidence": "Support at 265.00, resistance at 310.00."}'
        )
        is_valid, issues, _ = council._validate_agent_output('technical', analysis, grounded)
        # 265 and 310 are within 30% of 283.20 and 329.19 → whitelisted
        hallucination_issues = [i for i in issues if 'hallucination' in i.lower()]
        assert len(hallucination_issues) == 0
