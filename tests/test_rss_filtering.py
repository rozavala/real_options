import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
import time
from trading_bot.sentinels import Sentinel

# Mock Sentinel that exposes _fetch_rss_safe
class MockSentinel(Sentinel):
    def __init__(self):
        super().__init__({})

    async def check(self):
        return None

@pytest.mark.asyncio
async def test_rss_date_filtering_logic():
    """
    Test that _fetch_rss_safe correctly filters old articles.
    """
    sentinel = MockSentinel()
    seen_cache = set()

    # Current time
    now = datetime.now(timezone.utc)

    # Create mock feed entries
    # 1. New article (1 hour ago)
    entry_new = MagicMock()
    entry_new.title = "New Article"
    entry_new.link = "http://new.com"
    entry_new.published_parsed = (now - timedelta(hours=1)).timetuple()

    # 2. Old article (3 days ago) - Should be filtered
    entry_old = MagicMock()
    entry_old.title = "Old Article"
    entry_old.link = "http://old.com"
    entry_old.published_parsed = (now - timedelta(days=3)).timetuple()

    # 3. No date article - Should be filtered (per new requirements)
    entry_nodate = MagicMock()
    entry_nodate.title = "No Date Article"
    entry_nodate.link = "http://nodate.com"
    # Ensure all date fields are missing
    del entry_nodate.published_parsed
    del entry_nodate.published
    del entry_nodate.updated_parsed

    # Mock feedparser return
    mock_feed = MagicMock()
    mock_feed.bozo = 0
    mock_feed.entries = [entry_new, entry_old, entry_nodate]

    # Mock aiohttp response and feedparser
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status = 200
        # Mock text() as an async method
        mock_response.text = AsyncMock(return_value="dummy content")

        # Async context manager mock
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_get.return_value = mock_ctx

        with patch('feedparser.parse', return_value=mock_feed):
            # Call with current signature (no max_age_hours)
            headlines = await sentinel._fetch_rss_safe("http://dummy.url", seen_cache)

    # Assertions
    print(f"Headlines found: {headlines}")

    # This assertion will FAIL currently because filtering is not implemented
    # It returns ALL headlines currently
    assert "New Article" in headlines
    assert "Old Article" not in headlines  # This will fail
    assert "No Date Article" not in headlines # This will fail
