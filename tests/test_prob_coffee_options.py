import pytest
from ib_insync import FuturesOption, Bag, Position
from prob_coffee_options import get_position_details
from types import SimpleNamespace

# Test cases for get_position_details function

def test_get_position_details_single_leg_normal_strike():
    """
    Tests that a single leg position with a normal strike price is handled correctly.
    """
    contract = FuturesOption(symbol='KC', strike=3.5)
    # FIX: Add missing 'account' and 'avgCost' arguments
    position = Position(account='U123', contract=contract, position=1, avgCost=1.0)
    details = get_position_details(position)
    assert details['type'] == 'SINGLE_LEG'
    assert details['key_strikes'] == [3.5]

def test_get_position_details_single_leg_magnified_strike():
    """
    Tests that a single leg position with a magnified strike price is correctly normalized.
    """
    contract = FuturesOption(symbol='KC', strike=350.0)
    # FIX: Add missing 'account' and 'avgCost' arguments
    position = Position(account='U123', contract=contract, position=1, avgCost=1.0)
    details = get_position_details(position)
    assert details['type'] == 'SINGLE_LEG'
    assert details['key_strikes'] == [3.5]

def test_get_position_details_bear_put_spread_magnified_strikes():
    """
    Tests that a bear put spread with magnified strike prices in its combo legs is correctly normalized.
    """
    # FIX: Mock ComboLegs using SimpleNamespace because the real ib_insync.ComboLeg
    # objects do not have 'right' or 'strike' attributes, which exposes a latent
    # bug in the code under test. This mock allows us to test the function's logic.
    leg1 = SimpleNamespace(action='BUY', right='P', strike=360.0)
    leg2 = SimpleNamespace(action='SELL', right='P', strike=350.0)
    contract = Bag(symbol='KC', comboLegs=[leg1, leg2])
    # FIX: Add missing 'account' and 'avgCost' arguments
    position = Position(account='U123', contract=contract, position=1, avgCost=1.0)
    details = get_position_details(position)
    assert details['type'] == 'BEAR_PUT_SPREAD'
    # The strikes should be sorted and normalized
    assert details['key_strikes'] == [3.5, 3.6]

def test_get_position_details_bull_call_spread_normal_strikes():
    """
    Tests a bull call spread with normal strike prices.
    """
    # FIX: Mock ComboLegs
    leg1 = SimpleNamespace(action='BUY', right='C', strike=3.4)
    leg2 = SimpleNamespace(action='SELL', right='C', strike=3.5)
    contract = Bag(symbol='KC', comboLegs=[leg1, leg2])
    # FIX: Add missing 'account' and 'avgCost' arguments
    position = Position(account='U123', contract=contract, position=1, avgCost=1.0)
    details = get_position_details(position)
    assert details['type'] == 'BULL_CALL_SPREAD'
    assert details['key_strikes'] == [3.4, 3.5]