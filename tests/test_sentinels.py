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

    # Mock aiohttp session for _fetch_weather
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        'daily': {
            'time': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'temperature_2m_min': [10.0, 1.0, 12.0],
            'precipitation_sum': [0.0, 0.0, 0.0]
        }
    })
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_response)

    sentinel = WeatherSentinel(config)
    sentinel._active_alerts = {}  # Clear loaded state to avoid cooldown from production data
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


# --- MacroContagionSentinel Tests ---

@pytest.mark.asyncio
async def test_macro_get_history_handles_yfinance_crash():
    """_get_history returns empty DataFrame when yfinance raises internally."""
    import pandas as pd
    from trading_bot.sentinels import MacroContagionSentinel

    config = {
        'sentinels': {'macro_contagion': {}},
        'symbol': 'KC',
    }
    with patch('trading_bot.sentinels.genai'):
        sentinel = MacroContagionSentinel(config)

    # Simulate yfinance internal crash (NoneType subscript error)
    with patch.object(sentinel, '_get_history', wraps=sentinel._get_history):
        import asyncio
        loop = asyncio.get_running_loop()
        with patch('asyncio.get_running_loop', return_value=loop):
            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker.return_value.history.side_effect = TypeError(
                    "'NoneType' object is not subscriptable"
                )
                result = await sentinel._get_history("DX-Y.NYB")
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0


@pytest.mark.asyncio
async def test_macro_fed_policy_unwraps_json_array():
    """check_fed_policy_shock handles Gemini returning a JSON array."""
    from trading_bot.sentinels import MacroContagionSentinel

    config = {
        'sentinels': {'macro_contagion': {'policy_check_interval': 0}},
        'symbol': 'KC',
    }
    with patch('trading_bot.sentinels.genai'):
        sentinel = MacroContagionSentinel(config)
    sentinel.last_policy_check = None

    mock_response = MagicMock()
    mock_response.text = '[{"shock_detected": false}]'
    mock_response.usage_metadata = None

    with patch.object(sentinel.client.aio.models, 'generate_content',
                      new_callable=AsyncMock, return_value=mock_response), \
         patch('trading_bot.sentinels.acquire_api_slot', new_callable=AsyncMock):
        result = await sentinel.check_fed_policy_shock()
        # Should NOT raise 'list has no attribute get' — should return None (no shock)
        assert result is None


@pytest.mark.asyncio
async def test_macro_contagion_skips_missing_tickers():
    """check_cross_commodity_contagion skips tickers with no data."""
    import pandas as pd
    from trading_bot.sentinels import MacroContagionSentinel

    config = {
        'sentinels': {'macro_contagion': {}},
        'symbol': 'KC',
    }
    with patch('trading_bot.sentinels.genai'):
        sentinel = MacroContagionSentinel(config)

    # Mock _get_history: return data for KC and gold, empty for delisted ticker
    async def mock_history(ticker, period="5d", interval="1h"):
        if ticker in ("KC=F", "GC=F", "ZW=F"):
            dates = pd.date_range("2026-01-01", periods=5, freq="D")
            return pd.DataFrame({"Close": [100, 101, 99, 102, 98]}, index=dates)
        return pd.DataFrame()  # Simulate delisted ticker

    with patch.object(sentinel, '_get_history', side_effect=mock_history):
        # Force a specific basket including a bad ticker
        sentinel.profile = MagicMock()
        sentinel.profile.ticker = "KC"
        sentinel.profile.name = "Coffee"
        sentinel.profile.cross_commodity_basket = {
            'gold': 'GC=F',
            'wheat': 'ZW=F',
            'coal': 'MTF=F',  # Will return empty
        }
        result = await sentinel.check_cross_commodity_contagion()
        # Should complete without error (coal skipped, 3 tickers remain)
        # Result is None because KC isn't correlating with gold > 0.7
        assert result is None
