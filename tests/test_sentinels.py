import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from trading_bot.sentinels import PriceSentinel, WeatherSentinel, LogisticsSentinel, NewsSentinel, SentinelTrigger
from datetime import datetime, timezone
import time

# --- Price Sentinel Tests ---
@pytest.mark.asyncio
async def test_price_sentinel_trigger():
    mock_ib = MagicMock()
    # Mock reqHistoricalDataAsync instead of reqMktData
    mock_ib.reqHistoricalDataAsync = AsyncMock()

    # Create mock bars
    mock_bar = MagicMock()
    mock_bar.open = 100.0
    mock_bar.close = 102.0 # 2% change
    mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

    # Mock datetime to be Mon 10 AM NY (14:00 UTC during EDST)
    with patch('trading_bot.sentinels.datetime') as mock_datetime:
        mock_now = datetime(2023, 10, 23, 14, 0, 0, tzinfo=timezone.utc) # Monday
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        # Mock get_active_futures
        with patch('trading_bot.ib_interface.get_active_futures', new_callable=AsyncMock) as mock_futures:
            mock_contract = MagicMock()
            mock_contract.localSymbol = "KC_TEST"
            mock_futures.return_value = [mock_contract]

            config = {'sentinels': {'price': {'pct_change_threshold': 1.5}}, 'symbol': 'KC', 'exchange': 'NYBOT'}
            sentinel = PriceSentinel(config, mock_ib)

            # First trigger
            trigger = await sentinel.check()
            assert trigger is not None
            assert trigger.source == "PriceSentinel"
            assert trigger.payload['change'] == 2.0

            # Cooldown check: Immediate second call should return None
            trigger_cooldown = await sentinel.check()
            assert trigger_cooldown is None

@pytest.mark.asyncio
async def test_price_sentinel_no_trigger():
    mock_ib = MagicMock()
    mock_ib.reqHistoricalDataAsync = AsyncMock()

    mock_bar = MagicMock()
    mock_bar.open = 100.0
    mock_bar.close = 100.5 # 0.5% change
    mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

    with patch('trading_bot.sentinels.datetime') as mock_datetime:
        # Mon 10 AM NY (14:00 UTC)
        mock_now = datetime(2023, 10, 23, 14, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        with patch('trading_bot.ib_interface.get_active_futures', new_callable=AsyncMock) as mock_futures:
            mock_contract = MagicMock()
            mock_futures.return_value = [mock_contract]

            config = {'sentinels': {'price': {'pct_change_threshold': 1.5}}}
            sentinel = PriceSentinel(config, mock_ib)

            trigger = await sentinel.check()
            assert trigger is None

@pytest.mark.asyncio
async def test_price_sentinel_market_closed():
    mock_ib = MagicMock()

    # Mock datetime to be Sunday
    with patch('trading_bot.sentinels.datetime') as mock_datetime:
        mock_now = datetime(2023, 10, 22, 14, 0, 0, tzinfo=timezone.utc) # Sunday
        mock_datetime.now.return_value = mock_now

        config = {'sentinels': {'price': {'pct_change_threshold': 1.5}}}
        sentinel = PriceSentinel(config, mock_ib)

        trigger = await sentinel.check()
        assert trigger is None

# --- Weather Sentinel Tests ---
@pytest.mark.asyncio
async def test_weather_sentinel_frost():
    config = {
        'sentinels': {
            'weather': {
                'api_url': 'http://test-weather.com',
                'locations': [{'name': 'TestLoc', 'lat': 0, 'lon': 0}],
                'triggers': {'min_temp_c': 4.0}
            }
        }
    }

    sentinel = WeatherSentinel(config)
    sentinel._active_alerts = {}

    # Mock aiohttp session
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.json = AsyncMock(return_value={
        'daily': {
            'time': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'temperature_2m_min': [10.0, 1.0, 12.0],
            'precipitation_sum': [0.0, 0.0, 0.0]
        }
    })

    # Setup context manager for session.get
    mock_get_ctx = MagicMock()
    mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_session.get.return_value = mock_get_ctx

    # Inject mock session
    sentinel._get_session = AsyncMock(return_value=mock_session)

    trigger = await sentinel.check()

    assert trigger is not None
    assert trigger.source == "WeatherSentinel"
    assert trigger.payload['type'] == "FROST"
    assert trigger.payload['min_temp_c'] == 1.0

# --- Logistics Sentinel Tests ---
@pytest.mark.asyncio
async def test_logistics_sentinel_trigger():
    config = {
        'sentinels': {
            'logistics': {
                'rss_urls': ['http://test.rss'],
                'model': 'test-model'
            }
        },
        'gemini': {'api_key': 'TEST'}
    }

    # Mock _fetch_rss_safe instead of feedparser directly to bypass network/aiohttp
    with patch.object(LogisticsSentinel, '_fetch_rss_safe', new_callable=AsyncMock) as mock_rss:
        mock_rss.return_value = ["Strike at Port of Santos"]

        with patch('google.genai.Client') as MockClient:
            mock_client_instance = MockClient.return_value
            mock_client_instance.aio.models.generate_content = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = '{"score": 8, "summary": "Port strike detected at Santos"}'
            mock_client_instance.aio.models.generate_content.return_value = mock_response

            sentinel = LogisticsSentinel(config)
            trigger = await sentinel.check()

            assert trigger is not None
            assert trigger.source == "LogisticsSentinel"
            assert "Strike at Port of Santos" in trigger.payload['headlines']
            assert sentinel.model == 'test-model'

# --- News Sentinel Tests ---
@pytest.mark.asyncio
async def test_news_sentinel_trigger():
    config = {
        'sentinels': {
            'news': {
                'rss_urls': ['http://test.rss'],
                'sentiment_magnitude_threshold': 8,
                'model': 'test-model'
            }
        },
        'gemini': {'api_key': 'TEST'}
    }

    with patch.object(NewsSentinel, '_fetch_rss_safe', new_callable=AsyncMock) as mock_rss:
        mock_rss.return_value = ["Coffee Market Crash"]

        with patch('google.genai.Client') as MockClient:
            mock_client_instance = MockClient.return_value
            mock_client_instance.aio.models.generate_content = AsyncMock()
            mock_response = MagicMock()
            # Return valid JSON
            mock_response.text = '{"score": 9, "summary": "Panic selling"}'
            mock_client_instance.aio.models.generate_content.return_value = mock_response

            sentinel = NewsSentinel(config)
            trigger = await sentinel.check()

            assert trigger is not None
            assert trigger.source == "NewsSentinel"
            assert trigger.payload['score'] == 9
            assert sentinel.model == 'test-model'
