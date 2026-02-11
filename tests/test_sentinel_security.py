import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from trading_bot.sentinels import Sentinel

class TestSentinelSecurity:
    @pytest.fixture
    def sentinel(self):
        # Initialize with dummy config to avoid errors
        return Sentinel({"sentinels": {"price": {}, "weather": {}, "logistics": {}, "news": {}, "x_sentiment": {}, "prediction_markets": {}}})

    @pytest.mark.asyncio
    async def test_fetch_rss_safe_enforces_size_limit(self, sentinel):
        """
        Verify that _fetch_rss_safe stops reading after 5MB.
        Current implementation uses response.text() which reads everything.
        New implementation should use iter_chunked and stop early.
        """
        mock_response = AsyncMock()
        mock_response.status = 200

        # Simulate a stream that yields 1MB chunks for 6MB total
        async def chunk_generator(n):
            for _ in range(6):
                yield b"A" * (1024 * 1024) # 1MB chunk

        # Mock content.iter_chunked for new implementation
        mock_response.content.iter_chunked = chunk_generator

        # Mock text() for old implementation (simulating full read)
        mock_response.text.return_value = "A" * (6 * 1024 * 1024)

        # Patch aiohttp
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            # Run the method
            # We expect it to return [] because it hits the limit and logs warning (in new impl)
            # In old impl, it reads 6MB and tries to parse it (likely returns [] or garbage titles)
            result = await sentinel._fetch_rss_safe("http://example.com/rss", set())

            # Verify we did NOT call .text() (which reads everything into memory)
            # This checks that we migrated away from the unsafe .text() method
            if mock_response.text.call_count > 0:
                pytest.fail("Security Vulnerability: _fetch_rss_safe used .text() which reads unlimited content into memory.")

            # Verify result is empty (due to limit logic returning early)
            assert result == []

    def test_escape_xml_removes_control_chars(self, sentinel):
        """Verify _escape_xml removes invalid XML control characters."""
        # \x00 is null byte, \x08 is backspace - invalid in XML
        # \x09 (tab), \x0A (LF), \x0D (CR) are valid
        input_text = "Safe\x09Text\x0A\x00\x08Bad<Tag>"

        # Expected: Control chars gone, < escaped
        expected = "Safe\x09Text\x0ABad&lt;Tag&gt;"

        actual = sentinel._escape_xml(input_text)

        if actual != expected:
            pytest.fail(f"Security Vulnerability: _escape_xml failed to sanitize input.\nExpected: {repr(expected)}\nActual:   {repr(actual)}")
