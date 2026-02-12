
import sys
import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

# Mock streamlit before importing dashboard_utils
sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit'].cache_data = lambda func=None, ttl=None: (lambda f: f) if func is None else func
sys.modules['streamlit'].error = MagicMock()

# Mock matplotlib to avoid import errors
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.dates'] = MagicMock()
sys.modules['matplotlib.ticker'] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard_utils import grade_decision_quality

def test_grade_decision_volatility():
    data = {
        'timestamp': [pd.Timestamp('2024-01-01')],
        'master_decision': ['NEUTRAL'],
        'prediction_type': ['VOLATILITY'],
        'strategy_type': ['LONG_STRADDLE'],
        'volatility_outcome': ['BIG_MOVE'],
        'pnl_realized': [0.0],
        'actual_trend_direction': ['UP']
    }
    df = pd.DataFrame(data)

    graded = grade_decision_quality(df)
    assert graded.iloc[0]['outcome'] == 'WIN'

    # Test Loss case
    data['strategy_type'] = ['IRON_CONDOR']
    df = pd.DataFrame(data)
    graded = grade_decision_quality(df)
    assert graded.iloc[0]['outcome'] == 'LOSS'

def test_grade_decision_directional_pnl():
    data = {
        'timestamp': [pd.Timestamp('2024-01-01')],
        'master_decision': ['BULLISH'],
        'prediction_type': ['DIRECTIONAL'],
        'strategy_type': [''],
        'volatility_outcome': [None],
        'pnl_realized': [100.0],
        'actual_trend_direction': ['DOWN'] # Should be ignored if PnL is present
    }
    df = pd.DataFrame(data)

    graded = grade_decision_quality(df)
    assert graded.iloc[0]['outcome'] == 'WIN'

    # Test Loss case
    data['pnl_realized'] = [-50.0]
    data['actual_trend_direction'] = ['UP'] # Should be ignored
    df = pd.DataFrame(data)
    graded = grade_decision_quality(df)
    assert graded.iloc[0]['outcome'] == 'LOSS'

def test_grade_decision_directional_trend():
    data = {
        'timestamp': [pd.Timestamp('2024-01-01')],
        'master_decision': ['BULLISH'],
        'prediction_type': ['DIRECTIONAL'],
        'strategy_type': [''],
        'volatility_outcome': [None],
        'pnl_realized': [0.0], # No PnL, fallback to trend
        'actual_trend_direction': ['UP']
    }
    df = pd.DataFrame(data)

    graded = grade_decision_quality(df)
    assert graded.iloc[0]['outcome'] == 'WIN'

    # Test Loss case
    data['actual_trend_direction'] = ['DOWN']
    df = pd.DataFrame(data)
    graded = grade_decision_quality(df)
    assert graded.iloc[0]['outcome'] == 'LOSS'

def test_grade_decision_neutral_filtering():
    data = {
        'timestamp': [pd.Timestamp('2024-01-01')],
        'master_decision': ['NEUTRAL'],
        'prediction_type': ['DIRECTIONAL'],
        'strategy_type': [''],
        'volatility_outcome': [None],
        'pnl_realized': [100.0], # PnL exists but decision is Neutral
        'actual_trend_direction': ['UP']
    }
    df = pd.DataFrame(data)

    # Neutral decisions should be filtered out unless they are Volatility trades (checked separately)
    # The function grade_decision_quality filters out NEUTRAL directional decisions that are PENDING.
    # But wait, original code: if decision == 'NEUTRAL': continue (outcome stays PENDING).
    # Then filtered out: graded_df['outcome'] != 'PENDING'

    graded = grade_decision_quality(df)
    assert graded.empty

def test_grade_decision_mixed_batch():
    # Test a batch with all types
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=5),
        'master_decision': ['BULLISH', 'BEARISH', 'NEUTRAL', 'NEUTRAL', 'BULLISH'],
        'prediction_type': ['DIRECTIONAL', 'DIRECTIONAL', 'VOLATILITY', 'DIRECTIONAL', 'DIRECTIONAL'],
        'strategy_type': ['', '', 'LONG_STRADDLE', '', ''],
        'volatility_outcome': [None, None, 'BIG_MOVE', None, None],
        'pnl_realized': [100.0, -50.0, 0.0, 0.0, 0.0],
        'actual_trend_direction': ['UP', 'UP', 'UP', 'UP', 'DOWN']
    }
    df = pd.DataFrame(data)

    # Expected:
    # 0: Bullish + PnL>0 -> WIN
    # 1: Bearish + PnL<0 -> LOSS
    # 2: Volatility + Straddle + BigMove -> WIN
    # 3: Neutral Directional -> Filtered Out
    # 4: Bullish + PnL=0 + Down -> LOSS

    graded = grade_decision_quality(df)

    assert len(graded) == 4
    assert graded.iloc[0]['outcome'] == 'WIN'
    assert graded.iloc[1]['outcome'] == 'LOSS'
    assert graded.iloc[2]['outcome'] == 'WIN' # Row 2 (Volatility) became index 2? No, filtered row 3 is gone.
    # Wait, indices reset? "graded_df = graded_df[...]" preserves index unless reset_index is called?
    # Function returns graded_df without reset_index.

    # Let's check by timestamp or reset index locally
    graded = graded.reset_index(drop=True)
    assert graded.iloc[0]['outcome'] == 'WIN'
    assert graded.iloc[1]['outcome'] == 'LOSS'
    assert graded.iloc[2]['outcome'] == 'WIN' # The volatility trade
    assert graded.iloc[3]['outcome'] == 'LOSS' # The last directional trade

if __name__ == "__main__":
    pytest.main([__file__])
