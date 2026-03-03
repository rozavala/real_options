"""Tests for agent output validation: direction-evidence matching and number whitelisting."""

import pytest
from unittest.mock import MagicMock, patch
from trading_bot.observability import count_directional_evidence, BULLISH_WORDS, BEARISH_WORDS


class TestDirectionalEvidenceCounting:
    """Fix B: Inventory-domain vocabulary in BEARISH_WORDS."""

    def test_inventory_building_language_counted_bearish(self):
        """Stock-building terms should register as bearish evidence."""
        text = "Certified stocks building steadily with fresh inflows and arrivals at port"
        bullish, bearish = count_directional_evidence(text)
        assert bearish >= 3, f"Expected >=3 bearish words, got {bearish} (building, inflows, arrivals)"

    def test_restocking_replenishment_bearish(self):
        text = "Warehouse replenishment cycle underway, restocking from Brazil origins"
        bullish, bearish = count_directional_evidence(text)
        assert bearish >= 2, f"Expected >=2 bearish words, got {bearish}"

    def test_new_words_in_bearish_set(self):
        for word in ('building', 'inflows', 'arrivals', 'replenishment', 'restocking'):
            assert word in BEARISH_WORDS, f"'{word}' missing from BEARISH_WORDS"

    def test_inventory_report_balanced_evidence(self):
        """A typical inventory report should now show balanced or bearish-leaning evidence."""
        text = (
            "ICE certified stocks rose to 435,494 bags, building from prior tight conditions. "
            "Shortage concerns easing as arrivals increase. Fresh inflows from Brazil."
        )
        bullish, bearish = count_directional_evidence(text)
        # Before fix: 4+ bullish (rose/shortage/tight/increase), 0-1 bearish
        # After fix: also counts building, arrivals, inflows, easing → bearish >= 4
        assert bearish >= 3, f"Expected >=3 bearish, got {bearish}"

    def test_existing_bearish_words_unchanged(self):
        """Existing bearish vocabulary still works."""
        text = "Surplus bumper crop leads to glut and selloff in warehouses"
        bullish, bearish = count_directional_evidence(text)
        assert bearish >= 3

    def test_existing_bullish_words_unchanged(self):
        """Existing bullish vocabulary still works."""
        text = "Severe drought causes shortage and deficit with tight supply"
        bullish, bearish = count_directional_evidence(text)
        assert bullish >= 4

    def test_negation_still_works(self):
        """'not building' should flip to bullish."""
        text = "Stocks are not building at current levels"
        bullish, bearish = count_directional_evidence(text)
        # "not building" → bullish (negated bearish)
        assert bullish >= 1

    def test_unnegated_arrivals_bearish(self):
        """Arrivals without negation context should count as bearish."""
        text = "Port arrivals and warehouse inflows continued this week"
        bullish, bearish = count_directional_evidence(text)
        assert bearish >= 2


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
