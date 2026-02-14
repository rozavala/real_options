import pandas as pd
import sys
from unittest.mock import MagicMock
sys.path.insert(0, '.')

# Mock streamlit before importing dashboard_utils
sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit'].cache_data = lambda func=None, ttl=None: (lambda f: f) if func is None else func
sys.modules['streamlit'].error = MagicMock()

# Mock matplotlib
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.dates'] = MagicMock()
sys.modules['matplotlib.ticker'] = MagicMock()

from dashboard_utils import grade_decision_quality

def test_accuracy_fixes():
    """Comprehensive test of accuracy tracking fixes."""

    print("=" * 60)
    print("ACCURACY TRACKING FIX VERIFICATION")
    print("=" * 60)

    # Test 1: LONG_STRADDLE with 2.4% move (above 1.8%) = WIN
    print("\nTest 1: LONG_STRADDLE 2.4% move (threshold 1.8%)")
    test_df = pd.DataFrame([{
        'timestamp': '2026-01-19 13:53:20',
        'contract': 'KCK6 (202605)',
        'master_decision': 'NEUTRAL',
        'master_confidence': 0.70,
        'prediction_type': 'VOLATILITY',
        'strategy_type': 'LONG_STRADDLE',
        'entry_price': 337.5,
        'exit_price': 329.4,  # -2.4% move
        'pnl_realized': 2.03,  # Net: (2.4% - 1.8%) × 337.5 = 2.025
        'volatility_outcome': 'BIG_MOVE'
    }])
    result = grade_decision_quality(test_df)
    outcome = result.iloc[0]['outcome']
    print(f"  Move: {abs((329.4 - 337.5) / 337.5):.2%}")
    print(f"  Expected: WIN | Got: {outcome}")
    assert outcome == 'WIN', f"Failed: 2.4% move should be WIN"
    print("  ✅ PASSED")

    # Test 2: LONG_STRADDLE with 1.0% move (below 1.8%) = LOSS
    print("\nTest 2: LONG_STRADDLE 1.0% move (threshold 1.8%)")
    test_df = pd.DataFrame([{
        'timestamp': '2026-01-19 13:53:20',
        'contract': 'KCK6 (202605)',
        'master_decision': 'NEUTRAL',
        'master_confidence': 0.70,
        'prediction_type': 'VOLATILITY',
        'strategy_type': 'LONG_STRADDLE',
        'entry_price': 337.5,
        'exit_price': 334.125,  # -1.0% move
        'pnl_realized': -6.075,  # Lost premium: -1.8% × 337.5 = -6.075
        'volatility_outcome': 'STAYED_FLAT'
    }])
    result = grade_decision_quality(test_df)
    outcome = result.iloc[0]['outcome']
    print(f"  Move: {abs((334.125 - 337.5) / 337.5):.2%}")
    print(f"  Expected: LOSS | Got: {outcome}")
    assert outcome == 'LOSS', f"Failed: 1.0% move should be LOSS"
    print("  ✅ PASSED")

    # Test 3: IRON_CONDOR with 1.0% move (within 1.5%) = WIN
    print("\nTest 3: IRON_CONDOR 1.0% move (threshold 1.5%)")
    test_df = pd.DataFrame([{
        'timestamp': '2026-01-19 13:53:20',
        'contract': 'KCK6 (202605)',
        'master_decision': 'NEUTRAL',
        'master_confidence': 0.70,
        'prediction_type': 'VOLATILITY',
        'strategy_type': 'IRON_CONDOR',
        'entry_price': 337.5,
        'exit_price': 334.125,  # -1.0% move
        'pnl_realized': 3.375,  # Premium kept: 1% × 337.5 = 3.375
        'volatility_outcome': 'STAYED_FLAT'
    }])
    result = grade_decision_quality(test_df)
    outcome = result.iloc[0]['outcome']
    print(f"  Move: {abs((334.125 - 337.5) / 337.5):.2%}")
    print(f"  Expected: WIN | Got: {outcome}")
    assert outcome == 'WIN', f"Failed: 1.0% move should be WIN for condor"
    print("  ✅ PASSED")

    # Test 4: IRON_CONDOR with 2.0% move (exceeds 1.5%) = LOSS
    print("\nTest 4: IRON_CONDOR 2.0% move (threshold 1.5%)")
    test_df = pd.DataFrame([{
        'timestamp': '2026-01-19 13:53:20',
        'contract': 'KCK6 (202605)',
        'master_decision': 'NEUTRAL',
        'master_confidence': 0.70,
        'prediction_type': 'VOLATILITY',
        'strategy_type': 'IRON_CONDOR',
        'entry_price': 337.5,
        'exit_price': 330.75,  # -2.0% move
        'pnl_realized': -1.6875,  # Net loss: -(2.0% - 1.5%) × 337.5 = -1.6875
        'volatility_outcome': 'BIG_MOVE'
    }])
    result = grade_decision_quality(test_df)
    outcome = result.iloc[0]['outcome']
    print(f"  Move: {abs((330.75 - 337.5) / 337.5):.2%}")
    print(f"  Expected: LOSS | Got: {outcome}")
    assert outcome == 'LOSS', f"Failed: 2.0% move should be LOSS for condor"
    print("  ✅ PASSED")

    # Test 5: Directional BULLISH with positive P&L = WIN
    print("\nTest 5: Directional BULLISH +5 P&L")
    test_df = pd.DataFrame([{
        'timestamp': '2026-01-19 13:53:20',
        'contract': 'TEST',
        'master_decision': 'BULLISH',
        'master_confidence': 0.80,
        'prediction_type': 'DIRECTIONAL',
        'strategy_type': 'BULL_CALL_SPREAD',
        'entry_price': 100.0,
        'exit_price': 105.0,
        'pnl_realized': 5.0
    }])
    result = grade_decision_quality(test_df)
    outcome = result.iloc[0]['outcome']
    print(f"  P&L: +5.0")
    print(f"  Expected: WIN | Got: {outcome}")
    assert outcome == 'WIN', f"Failed: Positive P&L should be WIN"
    print("  ✅ PASSED")

    # Test 6: Directional with 0.0 P&L = PENDING (not LOSS)
    print("\nTest 6: Directional 0.0 P&L = PENDING")
    test_df = pd.DataFrame([{
        'timestamp': '2026-01-19 13:53:20',
        'contract': 'TEST',
        'master_decision': 'BULLISH',
        'master_confidence': 0.80,
        'prediction_type': 'DIRECTIONAL',
        'strategy_type': 'BULL_CALL_SPREAD',
        'entry_price': 100.0,
        'exit_price': 100.0,
        'pnl_realized': 0.0
    }])
    result = grade_decision_quality(test_df)
    outcome = result.iloc[0]['outcome']
    print(f"  P&L: 0.0")
    print(f"  Expected: PENDING | Got: {outcome}")
    assert outcome == 'PENDING', f"Failed: Zero P&L should be PENDING"
    print("  ✅ PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)

if __name__ == "__main__":
    test_accuracy_fixes()
