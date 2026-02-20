
import ast
import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

# Mock streamlit and matplotlib before importing dashboard_utils.
# Only mock these two — everything else (ib_insync, chromadb, etc.) is installed.
# NEVER mock ib_insync, trading_bot.utils, etc. at module level — it
# pollutes sys.modules and breaks downstream tests.
if 'streamlit' not in sys.modules:
    sys.modules['streamlit'] = MagicMock()
if 'matplotlib' not in sys.modules:
    sys.modules['matplotlib'] = MagicMock()
    sys.modules['matplotlib.pyplot'] = MagicMock()
    sys.modules['matplotlib.dates'] = MagicMock()
    sys.modules['matplotlib.ticker'] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard_utils import get_sentinel_status, get_ib_connection_health

# Path to dashboard_utils source for AST-based decorator checks
_DASHBOARD_UTILS_PATH = os.path.join(os.path.dirname(__file__), '..', 'dashboard_utils.py')


def _get_cache_ttl(func_name: str) -> int | None:
    """Parse dashboard_utils.py AST to find @st.cache_data(ttl=N) for a function."""
    with open(_DASHBOARD_UTILS_PATH) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and any(
                    kw.arg == 'ttl' and isinstance(kw.value, ast.Constant)
                    for kw in dec.keywords
                ):
                    return next(
                        kw.value.value for kw in dec.keywords if kw.arg == 'ttl'
                    )
    return None


def test_load_shared_state_has_cache_ttl_2():
    """Verify that _load_shared_state has @st.cache_data(ttl=2)."""
    ttl = _get_cache_ttl('_load_shared_state')
    assert ttl == 2, f"Expected TTL=2, got {ttl}"


def test_load_deduplicator_metrics_has_cache_ttl_10():
    """Verify that load_deduplicator_metrics has @st.cache_data(ttl=10)."""
    ttl = _get_cache_ttl('load_deduplicator_metrics')
    assert ttl == 10, f"Expected TTL=10, got {ttl}"


def test_get_sentinel_status_uses_shared_state():
    """Verify get_sentinel_status delegates to _load_shared_state."""
    with patch('dashboard_utils._load_shared_state') as mock_load:
        mock_load.return_value = {
            'sentinel_health': {
                'WeatherSentinel': {
                    'data': {'status': 'OK', 'interval_seconds': 300, 'last_check_utc': '2026-01-01T00:00:00'},
                    'timestamp': 1000,
                }
            }
        }

        status = get_sentinel_status()
        mock_load.assert_called_once()
        assert 'WeatherSentinel' in status
        assert status['WeatherSentinel']['status'] == 'OK'


def test_get_ib_connection_health_uses_shared_state():
    """Verify get_ib_connection_health delegates to _load_shared_state."""
    with patch('dashboard_utils._load_shared_state') as mock_load:
        mock_load.return_value = {
            'sensors': {
                'ib_heartbeat': {
                    'data': {'connected': True, 'last_heartbeat': '2026-01-01T00:00:00'},
                    'timestamp': 1000,
                }
            }
        }

        health = get_ib_connection_health()
        mock_load.assert_called_once()
        assert isinstance(health, dict)


def test_shared_state_source_uses_statemanager():
    """Verify _load_shared_state source code calls StateManager._load_raw_sync."""
    with open(_DASHBOARD_UTILS_PATH) as f:
        source = f.read()

    # Find the _load_shared_state function and check it calls StateManager
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_load_shared_state':
            func_source = ast.get_source_segment(source, node)
            assert 'StateManager._load_raw_sync' in func_source, \
                "_load_shared_state should call StateManager._load_raw_sync"
            assert 'STATE_FILE_PATH' in func_source, \
                "_load_shared_state should have a fallback using STATE_FILE_PATH"
            return

    pytest.fail("_load_shared_state function not found in dashboard_utils.py")
