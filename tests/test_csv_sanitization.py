import pytest
from trading_bot.utils import sanitize_for_csv

def test_sanitize_normal_string():
    """Test that normal strings are left untouched."""
    assert sanitize_for_csv("hello world") == "hello world"
    assert sanitize_for_csv("123") == "123"
    assert sanitize_for_csv("") == ""

def test_sanitize_injection_patterns():
    """Test that injection patterns are escaped."""
    assert sanitize_for_csv("=1+1") == "'=1+1"
    assert sanitize_for_csv("+1+1") == "'+1+1"
    assert sanitize_for_csv("-1+1") == "'-1+1"
    assert sanitize_for_csv("@SUM(1,1)") == "'@SUM(1,1)"

def test_sanitize_non_string():
    """Test that non-string inputs are handled gracefully."""
    assert sanitize_for_csv(123) == 123
    assert sanitize_for_csv(None) is None
    assert sanitize_for_csv(1.23) == 1.23

def test_sanitize_whitespace_prefix():
    """Test that whitespace before injection characters is handled."""
    # Ensure aggressive sanitization even with leading whitespace
    assert sanitize_for_csv(" =1+1") == "' =1+1"
