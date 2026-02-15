"""Tests for CSV formula injection sanitization."""

from trading_bot.utils import sanitize_for_csv


def test_normal_string_unchanged():
    assert sanitize_for_csv("Coffee prices rose") == "Coffee prices rose"


def test_equals_prefix_escaped():
    assert sanitize_for_csv("=CMD()") == "'=CMD()"


def test_plus_prefix_escaped():
    assert sanitize_for_csv("+1234") == "'+1234"


def test_at_prefix_escaped():
    assert sanitize_for_csv("@SUM(A1)") == "'@SUM(A1)"


def test_dash_prefix_not_escaped():
    """Dash is NOT a trigger — it would corrupt negative numbers and markdown bullets."""
    assert sanitize_for_csv("-0.35") == "-0.35"
    assert sanitize_for_csv("- Coffee supply disrupted") == "- Coffee supply disrupted"
    assert sanitize_for_csv("-123.45") == "-123.45"


def test_whitespace_prefix_escaped():
    """Leading whitespace before a trigger character is still caught."""
    assert sanitize_for_csv("  =CMD()") == "'  =CMD()"
    assert sanitize_for_csv(" +1234") == "' +1234"


def test_non_string_passthrough():
    """Non-string values pass through unchanged."""
    assert sanitize_for_csv(42) == 42
    assert sanitize_for_csv(3.14) == 3.14
    assert sanitize_for_csv(None) is None
    assert sanitize_for_csv(True) is True


def test_empty_string_unchanged():
    assert sanitize_for_csv("") == ""
