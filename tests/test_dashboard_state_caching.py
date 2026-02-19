
import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

# Mock streamlit before importing dashboard_utils
sys.modules['streamlit'] = MagicMock()
# Simple mock for cache_data that just returns the function but records calls
def mock_cache_data(ttl=None):
    def decorator(func):
        func.ttl = ttl
        return func
    return decorator

sys.modules['streamlit'].cache_data = mock_cache_data
sys.modules['streamlit'].error = MagicMock()

# Mock matplotlib to avoid import errors
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.dates'] = MagicMock()
sys.modules['matplotlib.ticker'] = MagicMock()

# Mock dependencies
mock_ib = MagicMock()
sys.modules['ib_insync'] = mock_ib
mock_ib.IB = MagicMock() # Ensure IB class exists

sys.modules['yfinance'] = MagicMock()
sys.modules['chromadb'] = MagicMock()

# Mock trading_bot.utils to avoid ImportErrors or dependency issues
mock_utils = MagicMock()
sys.modules['trading_bot.utils'] = mock_utils
mock_utils.configure_market_data_type = MagicMock()

# Mock trading_bot.timestamps
sys.modules['trading_bot.timestamps'] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to ensure that when dashboard_utils imports things, they exist.
# dashboard_utils imports:
# from ib_insync import IB -> Handled by mock_ib
# from trading_bot.utils import configure_market_data_type -> Handled by mock_utils
# from trading_bot.timestamps import parse_ts_column -> Handled by mock_timestamps

# Import AFTER mocks
try:
    from dashboard_utils import _resolve_data_path, _load_shared_state, load_deduplicator_metrics, get_sentinel_status
except ImportError:
    # If import fails, we can't run tests, but we'll try to let individual tests fail
    pass

def test_load_deduplicator_metrics_caching():
    """Verify that load_deduplicator_metrics is cached with TTL=10."""
    # Re-import to be safe
    from dashboard_utils import load_deduplicator_metrics

    # Check if decorated
    assert hasattr(load_deduplicator_metrics, 'ttl'), "load_deduplicator_metrics is not cached"
    assert load_deduplicator_metrics.ttl == 10

def test_load_shared_state_caching():
    """Verify that _load_shared_state is cached with TTL=2."""
    from dashboard_utils import _load_shared_state

    assert hasattr(_load_shared_state, 'ttl'), "_load_shared_state is not cached"
    assert _load_shared_state.ttl == 2

def test_load_shared_state_calls_statemanager():
    """Verify _load_shared_state calls StateManager by default."""
    from dashboard_utils import _load_shared_state

    # Since StateManager is imported INSIDE the function, we need to patch it where it is imported.
    # But sys.modules cache might interfere.
    # The function does `from trading_bot.state_manager import StateManager`.
    # We can patch `trading_bot.state_manager` in sys.modules.

    mock_sm_module = MagicMock()
    mock_sm_class = MagicMock()
    mock_sm_module.StateManager = mock_sm_class
    mock_sm_class._load_raw_sync.return_value = {'sentinels': {'status': 'OK'}}

    with patch.dict(sys.modules, {'trading_bot.state_manager': mock_sm_module}):
        result = _load_shared_state()
        assert result == {'sentinels': {'status': 'OK'}}
        mock_sm_class._load_raw_sync.assert_called_once()

def test_load_shared_state_fallback():
    """Verify _load_shared_state falls back to file read if StateManager fails."""
    from dashboard_utils import _load_shared_state

    # Mock StateManager to raise Exception
    mock_sm_module = MagicMock()
    mock_sm_class = MagicMock()
    mock_sm_module.StateManager = mock_sm_class
    mock_sm_class._load_raw_sync.side_effect = Exception("Mock Error")

    # Mock file read
    mock_data = json.dumps({'sentinels': {'status': 'FALLBACK'}})

    with patch.dict(sys.modules, {'trading_bot.state_manager': mock_sm_module}):
        with patch('builtins.open', mock_open(read_data=mock_data)):
            with patch('os.path.exists', return_value=True):
                 result = _load_shared_state()
                 assert result == {'sentinels': {'status': 'FALLBACK'}}

def test_get_sentinel_status_uses_shared_state():
    """Verify get_sentinel_status calls _load_shared_state."""
    from dashboard_utils import get_sentinel_status

    # We need to patch _load_shared_state in dashboard_utils.
    # Since we imported it, we can patch it on the module.

    with patch('dashboard_utils._load_shared_state') as mock_load:
        mock_load.return_value = {
            'sentinel_health': {
                'WeatherSentinel': {'data': {'status': 'OK', 'timestamp': 1000}}
            }
        }

        status = get_sentinel_status()
        assert 'WeatherSentinel' in status
        assert status['WeatherSentinel']['status'] == 'OK'
        mock_load.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
