import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from trading_bot.sentinels import FundamentalRegimeSentinel

@pytest.mark.asyncio
async def test_fundamental_regime_sentinel_logic():
    # Mock config and profile
    config = {
        'commodity': {'ticker': 'KC', 'name': 'Coffee'},
        'data_dir': '/tmp',
        'sentinels': {'fundamental': {}}
    }

    mock_profile = MagicMock()
    mock_profile.name = 'Coffee'

    with patch('trading_bot.sentinels.get_commodity_profile', return_value=mock_profile):
        sentinel = FundamentalRegimeSentinel(config)

        # Override check_ice_stocks_trend to isolate news sentiment test
        sentinel.check_ice_stocks_trend = MagicMock(return_value="BALANCED")
        # Initialize last_check to force execution
        sentinel.last_check = 0
        # Initialize current_regime
        sentinel.current_regime = {'regime': 'UNKNOWN'}

        # Case 1: SURPLUS (10 surplus, 2 deficit)
        # We need to mock _fetch_rss_count or whatever the implementation uses.
        # But since we are testing the public interface `check`, we should mock the network calls.

        # However, checking `check` which calls `check_news_sentiment` internally is better.
        # If I change `check_news_sentiment` to be async, I'll need to mock the underlying calls.

        # Let's mock `_get_session` to return a mock session
        mock_session = MagicMock() # Not AsyncMock for session itself if attributes need to be MagicMock
        # But _get_session is async, so it returns the session.
        sentinel._get_session = AsyncMock(return_value=mock_session)

        # Define mock responses for aiohttp
        # These act as the context manager AND the response
        mock_resp_surplus = MagicMock()
        mock_resp_surplus.status = 200
        mock_resp_surplus.read = AsyncMock(return_value=b"<rss>surplus_content</rss>")
        mock_resp_surplus.__aenter__ = AsyncMock(return_value=mock_resp_surplus)
        mock_resp_surplus.__aexit__ = AsyncMock(return_value=None)

        mock_resp_deficit = MagicMock()
        mock_resp_deficit.status = 200
        mock_resp_deficit.read = AsyncMock(return_value=b"<rss>deficit_content</rss>")
        mock_resp_deficit.__aenter__ = AsyncMock(return_value=mock_resp_deficit)
        mock_resp_deficit.__aexit__ = AsyncMock(return_value=None)

        def get_side_effect(url, **kwargs):
            if "surplus" in url:
                return mock_resp_surplus
            elif "deficit" in url:
                return mock_resp_deficit
            m = MagicMock()
            m.status = 404
            m.__aenter__ = AsyncMock(return_value=m)
            m.__aexit__ = AsyncMock(return_value=None)
            return m

        mock_session.get = MagicMock(side_effect=get_side_effect)

        # Mock feedparser.parse
        with patch('feedparser.parse') as mock_parse:
            def parse_side_effect(content):
                m = MagicMock()
                m.bozo = False
                if content == b"<rss>surplus_content</rss>":
                    m.entries = [1] * 10
                elif content == b"<rss>deficit_content</rss>":
                    m.entries = [1] * 2
                else:
                    m.entries = []
                return m

            mock_parse.side_effect = parse_side_effect

            # --- EXECUTE ---
            # This relies on `check` awaiting `check_news_sentiment` (async).
            # If the code is not yet modified, this test might fail or hang if `check` expects sync.
            # But `pytest-asyncio` handles async tests.

            # We are writing the test *before* the change, but asserting the *future* behavior.
            # So this test is expected to fail or error out on the *unmodified* code
            # (because `check` uses `run_in_executor` on a method we expect to use aiohttp/async).

            # Actually, the unmodified code uses `feedparser.parse(url)` which blocks.
            # And `check` calls `loop.run_in_executor(None, self.check_news_sentiment)`.
            # If we run this test against unmodified code, `check` will run `check_news_sentiment` in executor.
            # `check_news_sentiment` (unmodified) calls `feedparser.parse(url)`.
            # `feedparser.parse` (mocked) receives the URL string.
            # Our mock `parse_side_effect` expects bytes content.
            # So the test will fail on unmodified code.

            # This confirms the test effectively guards the new implementation.

            # Verify the regime logic
            trigger = await sentinel.check()

            assert trigger is not None
            # Stocks (BALANCED) wins the vote (2 vs 1), but we verify News detected SURPLUS
            assert trigger.payload['regime'] == "BALANCED"
            assert trigger.payload['evidence']['news_sentiment'] == "SURPLUS"
