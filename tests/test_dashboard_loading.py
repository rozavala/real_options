
import sys
import os
import importlib
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock matplotlib to avoid import errors
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.dates'] = MagicMock()
sys.modules['matplotlib.ticker'] = MagicMock()


@pytest.fixture
def mixed_timestamp_csv(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "council_history.csv"
    csv_content = """timestamp,contract,master_decision
2024-01-01 12:00:00,KC H4,BULLISH
2024-01-01 12:00:00.374747+00:00,KC H4,BEARISH
"""
    p.write_text(csv_content)
    return str(p)


def test_load_council_history_mixed_format(mixed_timestamp_csv):
    """Test that load_council_history handles mixed timestamp formats correctly."""
    # Ensure streamlit mock is in place before (re)importing dashboard_utils
    mock_st = MagicMock()
    mock_st.cache_data = lambda func=None, ttl=None: (lambda f: f) if func is None else func
    mock_st.error = MagicMock()

    with patch.dict(sys.modules, {
        'streamlit': mock_st,
    }):
        # Force reimport to pick up our streamlit mock
        if 'dashboard_utils' in sys.modules:
            importlib.reload(sys.modules['dashboard_utils'])
        import dashboard_utils

        with patch.object(dashboard_utils, '_resolve_data_path_for', return_value=mixed_timestamp_csv), \
             patch.object(dashboard_utils, '_load_legacy_council_history', return_value=pd.DataFrame()):

            df = dashboard_utils.load_council_history()

            # Verify result
            assert not df.empty
            assert len(df) == 2

            # Check that timestamps are correctly parsed
            assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

            # load_council_history sorts descending by timestamp
            # Row with microseconds (BEARISH) is later than row without (BULLISH)
            assert df.iloc[0]['master_decision'] == 'BEARISH'
            assert df.iloc[0]['timestamp'].microsecond == 374747

            assert df.iloc[1]['master_decision'] == 'BULLISH'
            assert df.iloc[1]['timestamp'].microsecond == 0


if __name__ == "__main__":
    pytest.main([__file__])
