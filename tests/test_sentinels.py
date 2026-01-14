import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from trading_bot.sentinels import PriceSentinel, WeatherSentinel, LogisticsSentinel, NewsSentinel, SentinelTrigger
from datetime import datetime

# --- Price Sentinel Tests ---
@pytest.mark.asyncio
async def test_price_sentinel_trigger():
    mock_ib = MagicMock()
    mock_ib.reqMktData = MagicMock()
    mock_ticker = MagicMock()
    mock_ticker.close = 100.0
    mock_ticker.last = 102.0 # 2% change
    mock_ib.reqMktData.return_value = mock_ticker

    # Mock datetime to be Mon 10 AM EST
    # We patch the datetime class in the module where it is used (trading_bot.sentinels)
    with patch('trading_bot.sentinels.datetime') as mock_datetime:
        mock_now = datetime(2023, 10, 23, 10, 0, 0) # Monday
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        # Mock get_active_futures
        with patch('trading_bot.ib_interface.get_active_futures', new_callable=AsyncMock) as mock_futures:
            mock_contract = MagicMock()
            mock_contract.localSymbol = "KC_TEST"
            mock_futures.return_value = [mock_contract]

            config = {'sentinels': {'price': {'pct_change_threshold': 1.5}}, 'symbol': 'KC', 'exchange': 'NYBOT'}
            sentinel = PriceSentinel(config, mock_ib)

            trigger = await sentinel.check()

            assert trigger is not None
            assert trigger.source == "PriceSentinel"
            assert trigger.payload['change'] == 2.0

@pytest.mark.asyncio
async def test_price_sentinel_no_trigger():
    mock_ib = MagicMock()
    mock_ticker = MagicMock()
    mock_ticker.close = 100.0
    mock_ticker.last = 100.5 # 0.5% change
    mock_ib.reqMktData.return_value = mock_ticker

    with patch('trading_bot.sentinels.datetime') as mock_datetime:
        mock_now = datetime(2023, 10, 23, 10, 0, 0) # Monday 10 AM
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
        mock_now = datetime(2023, 10, 22, 10, 0, 0) # Sunday
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
                'locations': [{'name': 'TestLoc', 'lat': 0, 'lon': 0}],
                'triggers': {'min_temp_c': 4.0}
            }
        }
    }

    # Mock requests.get
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        # Mock API response: 3.0 degrees (Frost)
        mock_response.json.return_value = {
            'daily': {
                'temperature_2m_min': [10.0, 3.0, 12.0],
                'precipitation_sum': [0.0, 0.0, 0.0]
            }
        }
        mock_get.return_value = mock_response

        sentinel = WeatherSentinel(config)
        trigger = await sentinel.check()

        assert trigger is not None
        assert trigger.source == "WeatherSentinel"
        assert trigger.payload['type'] == "FROST"
        assert trigger.payload['value'] == 3.0

# --- Logistics Sentinel Tests ---
@pytest.mark.asyncio
async def test_logistics_sentinel_trigger():
    config = {
        'sentinels': {'logistics': {'rss_urls': ['http://test.rss']}},
        'gemini': {'api_key': 'TEST'}
    }

    with patch('feedparser.parse') as mock_feed:
        entry = MagicMock()
        entry.title = "Strike at Port of Santos"
        mock_feed.return_value.entries = [entry]

        with patch('google.genai.Client') as MockClient:
            mock_client_instance = MockClient.return_value
            mock_client_instance.aio.models.generate_content = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = "YES"
            mock_client_instance.aio.models.generate_content.return_value = mock_response

            sentinel = LogisticsSentinel(config)
            trigger = await sentinel.check()

            assert trigger is not None
            assert trigger.source == "LogisticsSentinel"
            assert "Strike at Port of Santos" in trigger.payload['headlines']

# --- News Sentinel Tests ---
@pytest.mark.asyncio
async def test_news_sentinel_trigger():
    config = {
        'sentinels': {'news': {'rss_urls': ['http://test.rss'], 'sentiment_magnitude_threshold': 8}},
        'gemini': {'api_key': 'TEST'}
    }

    with patch('feedparser.parse') as mock_feed:
        entry = MagicMock()
        entry.title = "Coffee Market Crash"
        mock_feed.return_value.entries = [entry]

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
