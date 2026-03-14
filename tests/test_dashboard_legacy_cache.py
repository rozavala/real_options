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

# Import AFTER patching
from dashboard_utils import load_council_history, _load_legacy_council_history

def test_load_council_history_legacy_integration(tmp_path):
    """
    Verifies that load_council_history correctly combines the main CSV
    with legacy CSVs loaded via _load_legacy_council_history.
    """
    # Setup directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create main history file
    main_csv = data_dir / "council_history.csv"
    main_content = """timestamp,contract,master_decision
2024-02-01 12:00:00,KC H4,BULLISH
"""
    main_csv.write_text(main_content)

    # Create legacy file 1
    legacy1_csv = data_dir / "council_history_legacy_2023.csv"
    legacy1_content = """timestamp,contract,master_decision
2023-12-01 12:00:00,KC Z3,BEARISH
"""
    legacy1_csv.write_text(legacy1_content)

    # Create legacy file 2
    legacy2_csv = data_dir / "council_history_legacy_archive.csv"
    legacy2_content = """timestamp,contract,master_decision
2023-11-01 12:00:00,KC Z3,NEUTRAL
"""
    legacy2_csv.write_text(legacy2_content)

    # Patch _resolve_data_path_for to point to our temp main file
    # os.path.dirname(council_path) resolves to data_dir for legacy file discovery
    with patch('dashboard_utils._resolve_data_path_for', return_value=str(main_csv)):

        # 1. Test _load_legacy_council_history directly first
        legacy_df = _load_legacy_council_history(str(data_dir))
        assert len(legacy_df) == 2
        assert 'BEARISH' in legacy_df['master_decision'].values
        assert 'NEUTRAL' in legacy_df['master_decision'].values

        # 2. Test full load_council_history
        combined_df = load_council_history()

        assert len(combined_df) == 3
        decisions = combined_df['master_decision'].tolist()
        assert 'BULLISH' in decisions
        assert 'BEARISH' in decisions
        assert 'NEUTRAL' in decisions

        # Verify sorting (descending timestamp)
        # 2024-02-01 (BULLISH) > 2023-12-01 (BEARISH) > 2023-11-01 (NEUTRAL)
        assert combined_df.iloc[0]['master_decision'] == 'BULLISH'
        assert combined_df.iloc[1]['master_decision'] == 'BEARISH'
        assert combined_df.iloc[2]['master_decision'] == 'NEUTRAL'

if __name__ == "__main__":
    pytest.main([__file__])
