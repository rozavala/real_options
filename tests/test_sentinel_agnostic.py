import pytest
from unittest.mock import patch, AsyncMock
from trading_bot.sentinels import NewsSentinel
from config.commodity_profiles import CommodityProfile, ContractSpec, CommodityType

@pytest.mark.asyncio
async def test_news_sentinel_uses_commodity_profile():
    """Verify NewsSentinel prompt uses commodity profile, not hardcoded 'Coffee'."""
    config = {
        'sentinels': {'news': {'rss_urls': ['http://test.rss'], 'sentiment_magnitude_threshold': 8, 'model': 'test'}},
        'gemini': {'api_key': 'TEST'},
        'commodity': {'ticker': 'CT'}  # Cotton, not coffee
    }

    mock_profile = CommodityProfile(
        name="Cotton #2",
        ticker="CT",
        commodity_type=CommodityType.SOFT,
        contract=ContractSpec(
            symbol="CT",
            exchange="ICE",
            contract_months=[],
            tick_size=0.01,
            tick_value=500,
            contract_size=50000,
            unit="lbs",
            trading_hours_utc=""
        ),
        sentiment_search_queries=["cotton futures"]
    )

    with patch('trading_bot.sentinels.get_commodity_profile', return_value=mock_profile), \
         patch.object(NewsSentinel, '_fetch_rss_safe', new_callable=AsyncMock) as mock_rss, \
         patch('google.genai.Client'):

        mock_rss.return_value = ["Cotton prices surge on drought"]
        sentinel = NewsSentinel(config)

        # Verify profile loaded correctly
        assert sentinel.profile.ticker == 'CT'
        assert 'Coffee' not in sentinel.profile.name

        # Capture the prompt sent to AI
        prompts_sent = []
        async def capture_prompt(prompt):
            prompts_sent.append(prompt)
            return {"score": 7, "summary": "Cotton surge"}

        sentinel._analyze_with_ai = capture_prompt
        await sentinel.check()

        # Verify prompt uses Cotton, not Coffee
        assert len(prompts_sent) == 1
        assert 'Coffee' not in prompts_sent[0]
        assert sentinel.profile.name in prompts_sent[0]
        assert "Cotton #2" in prompts_sent[0]
