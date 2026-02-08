
import sys
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

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

from dashboard_utils import load_council_history, COUNCIL_HISTORY_PATH

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
    # Patch the COUNCIL_HISTORY_PATH to point to our temp file
    # We also need to patch os.path.dirname inside the function if it uses it to find legacy files,
    # but the mocked path is in a 'data' subdir so it should work.

    with patch('dashboard_utils.COUNCIL_HISTORY_PATH', mixed_timestamp_csv):
        # We also need to patch os.path.dirname to avoid looking for legacy files in the wrong place
        # Actually, since we created a 'data' dir in tmp_path, it's fine.

        # Call the function
        df = load_council_history()

        # Verify result
        assert not df.empty
        assert len(df) == 2

        # Check that timestamps are correctly parsed (should be datetime objects)
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

        # Verify values (ignoring timezone for simple comparison or ensuring equality)
        # Note: pd.to_datetime(..., utc=True) converts everything to UTC.
        # First row: 2024-01-01 12:00:00 -> 2024-01-01 12:00:00+00:00 (since we treat as UTC or naive)
        # Wait, if input is naive "2024-01-01 12:00:00" and we say utc=True, pandas assumes it's UTC.

        ts1 = df['timestamp'].iloc[1] # Because we sort descending in the function!
        # wait, load_council_history sorts descending.
        # Row 1 (BEARISH) is 12:00:00.374... which is LATER than Row 0 (BULLISH) 12:00:00
        # So Row 0 in DF should be the one with microseconds.

        assert df.iloc[0]['master_decision'] == 'BEARISH'
        assert df.iloc[0]['timestamp'].microsecond == 374747

        assert df.iloc[1]['master_decision'] == 'BULLISH'
        assert df.iloc[1]['timestamp'].microsecond == 0

if __name__ == "__main__":
    pytest.main([__file__])
