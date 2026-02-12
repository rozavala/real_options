import unittest
from trading_bot.utils import sanitize_log_message

class TestLogSanitization(unittest.TestCase):
    def test_sanitize_log_message(self):
        # Test 1: Basic string
        self.assertEqual(sanitize_log_message("Hello World"), "Hello World")

        # Test 2: Newlines
        self.assertEqual(sanitize_log_message("Line 1\nLine 2"), "Line 1\\nLine 2")
        self.assertEqual(sanitize_log_message("Line 1\rLine 2"), "Line 1\\rLine 2")
        self.assertEqual(sanitize_log_message("Line 1\r\nLine 2"), "Line 1\\r\\nLine 2")

        # Test 3: Truncation
        long_msg = "A" * 1005
        sanitized = sanitize_log_message(long_msg, max_length=1000)
        self.assertTrue(sanitized.endswith("...[TRUNCATED]"))
        # Implementation: message[:max_length] + "...[TRUNCATED]"
        # So "A"*1000 + "...[TRUNCATED]"
        expected_len = 1000 + len("...[TRUNCATED]")
        self.assertEqual(len(sanitized), expected_len)

        # Test 4: Control characters
        # \x07 is Bell, \x1b is Esc
        self.assertEqual(sanitize_log_message("Alert\x07"), "Alert")
        self.assertEqual(sanitize_log_message("Color\x1b[31mRed"), "Color[31mRed")

        # Test 5: Empty/None
        self.assertEqual(sanitize_log_message(""), "")
        self.assertEqual(sanitize_log_message(None), "")

        # Test 6: Non-string input
        self.assertEqual(sanitize_log_message(123), "123")

if __name__ == '__main__':
    unittest.main()
