import unittest
import logging
from trading_bot.logging_config import sanitize_log_message, SanitizedFormatter

class TestLogRedaction(unittest.TestCase):
    def test_redaction_patterns(self):
        """Test that sensitive API keys are redacted from logs."""

        test_cases = [
            ("Error with key sk-1234567890abcdef1234567890abcdef", "Error with key [REDACTED]"),
            ("Gemini key AIzaSyD-1234567890abcdef1234567890abcde used", "Gemini key [REDACTED] used"),
            ("xAI token xai-1234567890abcdef1234567890abcdef1234567890abcdef failed", "xAI token [REDACTED] failed"),
            ("Anthropic sk-ant-api03-1234567890abcdef1234567890abcdef-123456AA expired", "Anthropic [REDACTED] expired"),
            ("Safe message\nwith newline", "Safe message\\nwith newline"),  # Existing functionality check
            # False positive check:
            ("This is a sketch of a skeleton key", "This is a sketch of a skeleton key"),
            ("mask-12345 is safe", "mask-12345 is safe"),
            # Boundary checks:
            ("key=sk-1234567890abcdef1234567890abcdef", "key=[REDACTED]"), # Equals sign is boundary
            ("(sk-1234567890abcdef1234567890abcdef)", "([REDACTED])"),     # Parentheses are boundaries
        ]

        for original, expected in test_cases:
            with self.subTest(original=original):
                sanitized = sanitize_log_message(original)
                self.assertEqual(sanitized, expected)

    def test_formatter_integration(self):
        """Verify the formatter applies the redaction."""
        formatter = SanitizedFormatter()
        # Initialize LogRecord properly
        record = logging.LogRecord("test", logging.INFO, "path", 1, "Leaked sk-1234567890abcdef1234567890abcdef key", (), None)
        # Manually set message as logging system does before calling formatMessage
        record.message = record.getMessage()

        formatted = formatter.formatMessage(record)
        self.assertIn("[REDACTED]", formatted)
        self.assertNotIn("sk-12345", formatted)

if __name__ == "__main__":
    unittest.main()
