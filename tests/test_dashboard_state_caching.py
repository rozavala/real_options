
import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

def test_caching_configuration():
    """Verify TTL configuration for cached functions."""
    import importlib

    mock_st = MagicMock()
    def cache_data(ttl=None):
        def decorator(func):
            func.ttl = ttl
            return func
        return decorator
    mock_st.cache_data = cache_data
    mock_st.error = MagicMock()

    # Mock EVERYTHING that dashboard_utils imports to avoid side effects
    with patch.dict(sys.modules, {
        'streamlit': mock_st,
        'ib_insync': MagicMock(),
        'yfinance': MagicMock(),
        'matplotlib': MagicMock(),
        'matplotlib.pyplot': MagicMock(),
        'chromadb': MagicMock(),
        'performance_analyzer': MagicMock(),
        'trading_bot.decision_signals': MagicMock(),
        'trading_bot.brier_bridge': MagicMock(),
        'trading_bot.utils': MagicMock(),
        'trading_bot.timestamps': MagicMock(),
        'config_loader': MagicMock(),
    }):
        try:
            if 'dashboard_utils' in sys.modules:
                 import dashboard_utils
                 importlib.reload(dashboard_utils)
            else:
                 import dashboard_utils
        except ImportError:
            pytest.skip("Could not import dashboard_utils")
            return

        from dashboard_utils import load_deduplicator_metrics, _load_shared_state

        assert hasattr(load_deduplicator_metrics, 'ttl'), "load_deduplicator_metrics should be cached"
        assert load_deduplicator_metrics.ttl == 10

        assert hasattr(_load_shared_state, 'ttl'), "_load_shared_state should be cached"
        assert _load_shared_state.ttl == 2

def test_load_shared_state_logic():
    """Verify logic of _load_shared_state."""
    import importlib

    mock_st = MagicMock()
    def cache_data(ttl=None):
        def decorator(func):
            func.ttl = ttl
            return func
        return decorator
    mock_st.cache_data = cache_data
    mock_st.error = MagicMock()

    with patch.dict(sys.modules, {
        'streamlit': mock_st,
        'ib_insync': MagicMock(),
        'yfinance': MagicMock(),
        'matplotlib': MagicMock(),
        'matplotlib.pyplot': MagicMock(),
        'chromadb': MagicMock(),
        'performance_analyzer': MagicMock(),
        'trading_bot.decision_signals': MagicMock(),
        'trading_bot.brier_bridge': MagicMock(),
        'trading_bot.utils': MagicMock(),
        'trading_bot.timestamps': MagicMock(),
        'config_loader': MagicMock(),
    }):
        try:
            if 'dashboard_utils' in sys.modules:
                 import dashboard_utils
                 importlib.reload(dashboard_utils)
            else:
                 import dashboard_utils
        except ImportError:
            pytest.skip("Could not import dashboard_utils")
            return

        from dashboard_utils import _load_shared_state

        # Test StateManager path
        mock_sm_module = MagicMock()
        mock_sm_class = MagicMock()
        mock_sm_module.StateManager = mock_sm_class
        mock_sm_class._load_raw_sync.return_value = {'status': 'ok'}

        with patch.dict(sys.modules, {'trading_bot.state_manager': mock_sm_module}):
            assert _load_shared_state() == {'status': 'ok'}
            mock_sm_class._load_raw_sync.assert_called_once()

        # Test Fallback path
        mock_sm_class._load_raw_sync.side_effect = ImportError

        with patch.dict(sys.modules, {'trading_bot.state_manager': mock_sm_module}):
            with patch('builtins.open', mock_open(read_data='{"status": "fallback"}')):
                with patch('os.path.exists', return_value=True):
                    assert _load_shared_state() == {'status': 'fallback'}

        # Test Failure path
        mock_sm_class._load_raw_sync.side_effect = Exception

        with patch.dict(sys.modules, {'trading_bot.state_manager': mock_sm_module}):
            with patch('os.path.exists', return_value=False):
                assert _load_shared_state() == {}

if __name__ == "__main__":
    pytest.main([__file__])
