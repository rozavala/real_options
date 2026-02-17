import pytest
from trading_bot.utils import sanitize_log_message

def test_sanitize_log_message_basic():
    """Test basic sanitization of single-line strings."""
    assert sanitize_log_message("Hello World") == "Hello World"
    assert sanitize_log_message("") == ""
    assert sanitize_log_message(None) == "None"

def test_sanitize_log_message_newlines():
    """Test escaping of newline characters."""
    assert sanitize_log_message("Hello\nWorld") == "Hello\\nWorld"
    assert sanitize_log_message("Hello\r\nWorld") == "Hello\\r\\nWorld"
    assert sanitize_log_message("Hello\rWorld") == "Hello\\rWorld"

def test_sanitize_log_message_non_string():
    """Test sanitization of non-string inputs."""
    assert sanitize_log_message(123) == "123"
    assert sanitize_log_message(3.14) == "3.14"
    assert sanitize_log_message(True) == "True"
    assert sanitize_log_message({"key": "value"}) == "{'key': 'value'}"

def test_sanitize_log_message_malicious():
    """Test malicious input attempting log injection."""
    malicious_input = "User input\nERROR: Something bad happened"
    expected_output = "User input\\nERROR: Something bad happened"
    assert sanitize_log_message(malicious_input) == expected_output
