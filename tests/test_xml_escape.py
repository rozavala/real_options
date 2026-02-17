import pytest
from trading_bot.utils import escape_xml

def test_escape_xml_basic():
    """Test basic alphanumeric text."""
    input_text = "Hello World 123"
    assert escape_xml(input_text) == input_text

def test_escape_xml_special_chars():
    """Test escaping of XML special characters."""
    input_text = "Use <tags> & 'quotes' \""
    expected = "Use &lt;tags&gt; &amp; 'quotes' \""
    assert escape_xml(input_text) == expected

def test_escape_xml_control_chars():
    """Test stripping of invalid XML control characters."""
    # \x00 is null, \x1F is unit separator - invalid in XML 1.0 (except tab/cr/lf)
    input_text = "Null\x00Byte and \x1FUnitSeparator"
    expected = "NullByte and UnitSeparator"
    assert escape_xml(input_text) == expected

def test_escape_xml_allowed_control_chars():
    """Test that allowed whitespace characters are preserved."""
    input_text = "Line\nFeed\tTab\rCarriageReturn"
    assert escape_xml(input_text) == input_text

def test_escape_xml_mixed():
    """Test mixed content."""
    input_text = "<div>Bad\x00Tag</div> & more"
    expected = "&lt;div&gt;BadTag&lt;/div&gt; &amp; more"
    assert escape_xml(input_text) == expected

def test_escape_xml_non_string():
    """Test that non-string inputs are converted safely."""
    assert escape_xml(123) == "123"
    assert escape_xml(None) == "None"
