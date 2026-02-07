
import sys
import os
import time
import pytest
from unittest.mock import MagicMock

# Mock streamlit before importing dashboard_utils
sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit'].cache_data = lambda func=None, ttl=None: (lambda f: f) if func is None else func
sys.modules['streamlit'].error = MagicMock()

# Mock matplotlib to avoid import errors if not installed in test env
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.dates'] = MagicMock()
sys.modules['matplotlib.ticker'] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard_utils import tail_file

@pytest.fixture
def large_log_file(tmp_path):
    """Creates a temporary large log file (approx 10MB)."""
    p = tmp_path / "test_large.log"

    # Write 100,000 lines
    with open(p, 'w', encoding='utf-8') as f:
        for i in range(100000):
            f.write(f"This is log line number {i} with some extra padding text to make it longer.\n")

    return str(p)

def test_tail_file_correctness(large_log_file):
    """Verifies that tail_file returns exactly the same last N lines as naive readlines()."""
    n_lines = 50

    # Naive approach
    start_naive = time.time()
    with open(large_log_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        expected_lines = all_lines[-n_lines:]
    time_naive = time.time() - start_naive

    # Optimized approach
    start_opt = time.time()
    result_lines = tail_file(large_log_file, n_lines=n_lines)
    time_opt = time.time() - start_opt

    print(f"\nPerformance Comparison (100k lines, ~7MB):")
    print(f"Naive readlines(): {time_naive:.6f}s")
    print(f"Optimized tail_file(): {time_opt:.6f}s")
    print(f"Speedup: {time_naive / time_opt:.2f}x")

    # Verify correctness
    assert len(result_lines) == n_lines
    assert result_lines == expected_lines

def test_tail_file_small_file(tmp_path):
    """Test with a file smaller than block size."""
    p = tmp_path / "small.log"
    lines = ["Line 1\n", "Line 2\n", "Line 3"]
    p.write_text("".join(lines), encoding='utf-8')

    result = tail_file(str(p), n_lines=2)
    assert result == ["Line 2\n", "Line 3"]

def test_tail_file_empty_file(tmp_path):
    """Test with an empty file."""
    p = tmp_path / "empty.log"
    p.touch()

    result = tail_file(str(p), n_lines=50)
    assert result == []

def test_tail_file_nonexistent():
    """Test with a non-existent file."""
    result = tail_file("nonexistent_file.log")
    assert len(result) == 1
    assert result[0].startswith("Error: File")

if __name__ == "__main__":
    pytest.main([__file__])
