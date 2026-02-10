# tests/test_compliance_risk.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from ib_insync import Bag, ComboLeg, LimitOrder, Contract

@pytest.fixture(autouse=True)
def clear_cache():
    from trading_bot.compliance import _CONTRACT_DETAILS_CACHE
    _CONTRACT_DETAILS_CACHE.clear()
    yield

@pytest.mark.asyncio
async def test_debit_spread_risk_true_width():
    """Verify correct risk calculation using true strike width."""
    from trading_bot.compliance import calculate_spread_max_risk

    # Mock IB connection
    mock_ib = AsyncMock()

    # Mock leg details (Bull Call Spread: Buy 350C, Sell 360C)
    # Note: reqContractDetailsAsync returns a list of ContractDetails objects
    mock_ib.reqContractDetailsAsync.side_effect = [
        [MagicMock(contract=MagicMock(strike=350.0, right='C'))],
        [MagicMock(contract=MagicMock(strike=360.0, right='C'))]
    ]

    bag = Bag(symbol='KC')
    bag.comboLegs = [ComboLeg(conId=1), ComboLeg(conId=2)]
    order = LimitOrder('BUY', 1, 0.50)  # 50 cents/lb debit
    config = {}

    risk = await calculate_spread_max_risk(mock_ib, bag, order, config)

    # Debit spread: risk = premium paid = 0.50 * 375 = $187.50
    # Note: Multiplier is 375 for cents/lb (37500/100).
    assert risk == 187.50, f"Expected $187.50, got ${risk}"


@pytest.mark.asyncio
async def test_credit_spread_risk_narrow_wings():
    """
    Verify Iron Condor with NARROW wings (account-appropriate).

    FLIGHT DIRECTOR WARNING - "Risk Explosion":
    Wide wings (10+ points) create $300K+ risk per lot, making Iron Condors
    impossible for a $50K account. This test uses realistic narrow wings.
    """
    from trading_bot.compliance import calculate_spread_max_risk

    mock_ib = AsyncMock()

    # NARROW Iron Condor (0.5 point wings - account appropriate):
    # Buy 349.5P, Sell 350P, Sell 360C, Buy 360.5C
    # Put wing: 350-349.5 = 0.5 points
    # Call wing: 360.5-360 = 0.5 points
    mock_ib.reqContractDetailsAsync.side_effect = [
        [MagicMock(contract=MagicMock(strike=349.5, right='P'))],
        [MagicMock(contract=MagicMock(strike=350.0, right='P'))],
        [MagicMock(contract=MagicMock(strike=360.0, right='C'))],
        [MagicMock(contract=MagicMock(strike=360.5, right='C'))]
    ]

    bag = Bag(symbol='KC')
    bag.comboLegs = [ComboLeg(conId=i) for i in range(4)]
    order = LimitOrder('SELL', 1, 0.30)  # 30 cents credit
    config = {}

    risk = await calculate_spread_max_risk(mock_ib, bag, order, config)

    # Credit spread: risk = (wing_width - credit) * multiplier
    # = (0.5 - 0.30) * 375 = 0.2 * 375 = $75.00
    expected = (0.5 - 0.30) * 375
    assert risk == expected, f"Expected ${expected:,.2f}, got ${risk:,.2f}"

    # This $75.00 risk is 0.15% of $50K equity - PASSABLE at 40% limit


@pytest.mark.asyncio
async def test_credit_spread_risk_wide_wings_blocked():
    """
    Demonstrate that WIDE wings correctly produce massive risk.

    FLIGHT DIRECTOR WARNING: This test documents expected behavior where
    wide-wing Iron Condors are blocked due to tail risk exposure.
    """
    from trading_bot.compliance import calculate_spread_max_risk

    mock_ib = AsyncMock()

    # WIDE Iron Condor (100 point wings - TOO RISKY for $50K):
    # Width 100 * 375 = $37,500 > $20,000 limit
    mock_ib.reqContractDetailsAsync.side_effect = [
        [MagicMock(contract=MagicMock(strike=250.0, right='P'))],
        [MagicMock(contract=MagicMock(strike=350.0, right='P'))],
        [MagicMock(contract=MagicMock(strike=370.0, right='C'))],
        [MagicMock(contract=MagicMock(strike=470.0, right='C'))]
    ]

    bag = Bag(symbol='KC')
    bag.comboLegs = [ComboLeg(conId=i) for i in range(4)]
    order = LimitOrder('SELL', 1, 0.30)  # 30 cents credit
    config = {}

    risk = await calculate_spread_max_risk(mock_ib, bag, order, config)

    # Credit spread: risk = (100 - 0.30) * 375 = $37,387.50
    # This EXCEEDS 40% of $50K ($20K) and will be BLOCKED
    expected = (100.0 - 0.30) * 375
    assert risk == expected, f"Expected ${expected:,.2f}, got ${risk:,.2f}"
    assert risk > 50000 * 0.40, "Wide wings should exceed max_position_pct limit"
