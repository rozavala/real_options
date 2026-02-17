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


def test_dash_smart_handling():
    """Dash handling is contextual: negative numbers/bullets allowed, potential formulas escaped."""
    # Allowed (Negative Numbers)
    assert sanitize_for_csv("-0.35") == "-0.35"
    assert sanitize_for_csv("-123.45") == "-123.45"
    assert sanitize_for_csv("-5") == "-5"
    assert sanitize_for_csv("-.5") == "-.5"
    assert sanitize_for_csv("-1.2E-4") == "-1.2E-4"
    assert sanitize_for_csv("-1e10") == "-1e10"

    # Allowed (Bullet Points)
    assert sanitize_for_csv("- Coffee supply disrupted") == "- Coffee supply disrupted"
    assert sanitize_for_csv("-  Indented bullet") == "-  Indented bullet"

    # Escaped (Potential Formulas)
    assert sanitize_for_csv("-cmd|'/C calc'!A0") == "'-cmd|'/C calc'!A0"
    assert sanitize_for_csv("-func(1,2)") == "'-func(1,2)"
    assert sanitize_for_csv("-abc") == "'-abc"

    # CRITICAL: Bypass attempt (Number prefixing a formula)
    # The regex must be anchored to the end ($) to catch this
    assert sanitize_for_csv("-5+cmd|'/C calc'!A0") == "'-5+cmd|'/C calc'!A0"
    assert sanitize_for_csv("-1.5=HYPERLINK(...)") == "'-1.5=HYPERLINK(...)"


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
