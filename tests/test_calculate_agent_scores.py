
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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard_utils import calculate_agent_scores

def test_calculate_agent_scores_empty():
    df = pd.DataFrame()
    scores = calculate_agent_scores(df)
    assert scores['master_decision']['total'] == 0
    assert scores['master_decision']['correct'] == 0

def test_calculate_agent_scores_volatility():
    data = {
        'prediction_type': ['VOLATILITY', 'VOLATILITY', 'VOLATILITY', 'VOLATILITY'],
        'strategy_type': ['LONG_STRADDLE', 'IRON_CONDOR', 'LONG_STRADDLE', 'IRON_CONDOR'],
        'volatility_outcome': ['BIG_MOVE', 'STAYED_FLAT', 'STAYED_FLAT', 'BIG_MOVE'],
        'volatility_sentiment': ['HIGH', 'LOW', 'LOW', 'HIGH'],
        'master_decision': ['NEUTRAL', 'NEUTRAL', 'NEUTRAL', 'NEUTRAL']
    }
    df = pd.DataFrame(data)

    scores = calculate_agent_scores(df)

    # Master Decision
    # Row 0: LS + BIG_MOVE -> Win
    # Row 1: IC + FLAT -> Win
    # Row 2: LS + FLAT -> Loss
    # Row 3: IC + BIG_MOVE -> Loss
    assert scores['master_decision']['total'] == 4
    assert scores['master_decision']['correct'] == 2

    # Volatility Sentiment
    # Row 0: HIGH + BIG_MOVE -> Correct
    # Row 1: LOW + FLAT -> Correct
    # Row 2: LOW + FLAT -> Correct (Wait, outcome is FLAT, sentiment LOW -> correct)
    # Row 3: HIGH + BIG_MOVE -> Correct (Wait, outcome is BIG_MOVE, sentiment HIGH -> correct)

    # Let's recheck logic:
    # predicted_high = sent.isin(['HIGH', 'BULLISH', 'VOLATILE'])
    # predicted_low = sent.isin(['LOW', 'BEARISH', 'QUIET', 'RANGE_BOUND'])

    # Row 0: HIGH (pred_high) + BIG_MOVE -> Correct
    # Row 1: LOW (pred_low) + STAYED_FLAT -> Correct
    # Row 2: LOW (pred_low) + STAYED_FLAT -> Correct
    # Row 3: HIGH (pred_high) + BIG_MOVE -> Correct (Wait, outcome is BIG_MOVE)

    # Row 3 outcome is BIG_MOVE. Row 3 sentiment is HIGH. Logic:
    # ((vol_outcome == 'BIG_MOVE' and predicted_high) or ...)
    # So Row 3 is correct.

    # Wait, my test data setup:
    # Row 2: LS + STAYED_FLAT. Outcome FLAT. Sentiment LOW.
    # Logic: outcome FLAT + pred_low -> Correct.

    # So all 4 should be correct for volatility_sentiment.
    assert scores['volatility_sentiment']['total'] == 4
    assert scores['volatility_sentiment']['correct'] == 4

def test_calculate_agent_scores_directional():
    data = {
        'prediction_type': ['DIRECTIONAL', 'DIRECTIONAL', 'DIRECTIONAL'],
        'actual_trend_direction': ['UP', 'DOWN', 'UP'],
        'meteorologist_sentiment': ['BULLISH', 'BEARISH', 'BEARISH'],
        'master_decision': ['BULLISH', 'BEARISH', 'BULLISH'] # Not used for master scoring here but for context
    }
    df = pd.DataFrame(data)

    scores = calculate_agent_scores(df)

    # Meteorologist
    # Row 0: BULLISH + UP -> Correct
    # Row 1: BEARISH + DOWN -> Correct
    # Row 2: BEARISH + UP -> Incorrect
    assert scores['meteorologist_sentiment']['total'] == 3
    assert scores['meteorologist_sentiment']['correct'] == 2

def test_calculate_agent_scores_live_price_fallback():
    data = {
        'prediction_type': ['DIRECTIONAL', 'DIRECTIONAL'],
        'actual_trend_direction': [None, np.nan],
        'entry_price': [100.0, 100.0],
        'meteorologist_sentiment': ['BULLISH', 'BEARISH']
    }
    df = pd.DataFrame(data)

    # Live price 101 (> 100 * 1.005) -> UP
    # Row 0: BULLISH + UP -> Correct
    # Row 1: BEARISH + UP -> Incorrect

    scores = calculate_agent_scores(df, live_price=101.0)

    assert scores['meteorologist_sentiment']['total'] == 2
    assert scores['meteorologist_sentiment']['correct'] == 1

if __name__ == "__main__":
    pytest.main([__file__])
