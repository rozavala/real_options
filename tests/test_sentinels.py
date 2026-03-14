import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from trading_bot.sentinels import PriceSentinel, WeatherSentinel, LogisticsSentinel, NewsSentinel, XSentimentSentinel, SentinelTrigger
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

        # Mock get_market_state to return ACTIVE (sentinel now uses state resolver)
        with patch('trading_bot.utils.get_market_state', return_value='ACTIVE'):

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

        with patch('trading_bot.utils.get_market_state', return_value='ACTIVE'):
          with patch('trading_bot.ib_interface.get_active_futures', new_callable=AsyncMock) as mock_futures:
            mock_contract = MagicMock()
            mock_futures.return_value = [mock_contract]

            config = {'sentinels': {'price': {'pct_change_threshold': 1.5}}, 'symbol': 'KC', 'exchange': 'NYBOT'}
            sentinel = PriceSentinel(config, mock_ib)

            trigger = await sentinel.check()
            assert trigger is None

@pytest.mark.asyncio
async def test_price_sentinel_market_closed():
    mock_ib = MagicMock()

    # Mock get_market_state to return SLEEPING (Sunday)
    with patch('trading_bot.utils.get_market_state', return_value='SLEEPING'):
        config = {'sentinels': {'price': {'pct_change_threshold': 1.5}}, 'symbol': 'KC', 'exchange': 'NYBOT'}
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


# --- Weather Sentinel Energy Stage Tests ---
class TestWeatherSentinelEnergyStages:
    """Test that energy commodity regions use INFRASTRUCTURE stage, not crop stages."""

    def _make_sentinel(self):
        config = {
            'sentinels': {'weather': {'api_url': 'http://test.com'}},
            'commodity': {'ticker': 'NG'}
        }
        sentinel = WeatherSentinel(config)
        sentinel._active_alerts = {}
        return sentinel

    def test_drought_direction_infrastructure(self):
        sentinel = self._make_sentinel()
        assert sentinel._determine_drought_direction("INFRASTRUCTURE") == "NEUTRAL"

    def test_flood_direction_infrastructure(self):
        sentinel = self._make_sentinel()
        assert sentinel._determine_flood_direction("INFRASTRUCTURE") == "BULLISH"

    def test_drought_direction_flowering_unchanged(self):
        sentinel = self._make_sentinel()
        assert sentinel._determine_drought_direction("FLOWERING") == "BULLISH"

    def test_flood_direction_harvest_unchanged(self):
        sentinel = self._make_sentinel()
        assert sentinel._determine_flood_direction("HARVEST") == "BULLISH"

    @pytest.mark.asyncio
    async def test_energy_region_gets_infrastructure_stage(self):
        """Energy regions (no agricultural months) should use INFRASTRUCTURE stage."""
        from config.commodity_profiles import GrowingRegion
        sentinel = self._make_sentinel()

        region = GrowingRegion(
            name="Permian Basin", country="US",
            latitude_range=(31.0, 32.0), longitude_range=(-103.0, -101.0),
            production_share=0.30,
            drought_threshold_mm=10.0, flood_threshold_mm=200.0,
            frost_threshold_celsius=-5.0,
            flowering_months=[], harvest_months=[], bean_filling_months=[],
        )

        # Mock weather data: drought conditions (very low precip)
        weather_data = [
            {'min_temp_c': 20.0, 'precipitation_mm': 1.0} for _ in range(7)
        ]
        with patch.object(sentinel, '_fetch_weather', new_callable=AsyncMock,
                          return_value=weather_data):
            result = await sentinel.async_check_region_weather(region)

        assert result is not None
        assert result['type'] == 'DROUGHT'
        assert result['stage'] == 'INFRASTRUCTURE'
        assert result['direction'] == 'NEUTRAL'

    @pytest.mark.asyncio
    async def test_agricultural_region_keeps_crop_stages(self):
        """Agricultural regions should still use FLOWERING/HARVEST/etc stages."""
        from config.commodity_profiles import GrowingRegion
        sentinel = self._make_sentinel()

        region = GrowingRegion(
            name="Minas Gerais", country="Brazil",
            latitude_range=(-22.0, -18.0), longitude_range=(-48.0, -44.0),
            production_share=0.30,
            drought_threshold_mm=30.0, flood_threshold_mm=150.0,
            flowering_months=[9, 10, 11], harvest_months=[5, 6, 7],
            bean_filling_months=[12, 1, 2, 3],
        )

        weather_data = [
            {'min_temp_c': 20.0, 'precipitation_mm': 1.0} for _ in range(7)
        ]
        with patch.object(sentinel, '_fetch_weather', new_callable=AsyncMock,
                          return_value=weather_data):
            result = await sentinel.async_check_region_weather(region)

        assert result is not None
        assert result['type'] == 'DROUGHT'
        # Stage depends on current month — just verify it's NOT INFRASTRUCTURE
        assert result['stage'] != 'INFRASTRUCTURE'


# --- X Sentiment Sentinel Tests ---
class TestXSentimentSinceIdFix:
    """Test that since_id and start_time are never sent together to Twitter API."""

    def _make_sentinel(self):
        config = {
            'sentinels': {'x_sentiment': {'search_queries': ['coffee']}},
            'commodity': {'ticker': 'KC'},
            'xai': {'api_key': 'test-xai-key'},
            'x_api': {'bearer_token': 'test-bearer-token'},
        }
        sentinel = XSentimentSentinel(config)
        sentinel._since_ids = {}
        return sentinel

    @pytest.mark.asyncio
    async def test_since_id_removes_start_time(self):
        """When since_id is present, start_time must be removed from params."""
        sentinel = self._make_sentinel()
        # Pre-populate a since_id for the query hash
        import hashlib
        sid_key = hashlib.md5(b"coffee").hexdigest()[:16]
        sentinel._since_ids[sid_key] = "123456789"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": [], "meta": {"result_count": 0}})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        sentinel._get_session = AsyncMock(return_value=mock_session)

        await sentinel._fetch_x_posts("coffee", limit=10, sort_order="recency")

        # Verify the params sent to the API
        call_args = mock_session.get.call_args
        params = call_args.kwargs.get('params') or call_args[1].get('params')
        assert "since_id" in params, "since_id should be in params"
        assert "start_time" not in params, "start_time must be removed when since_id is present"

    @pytest.mark.asyncio
    async def test_no_since_id_keeps_start_time(self):
        """When no since_id exists, start_time should be present."""
        sentinel = self._make_sentinel()
        # No since_ids set — first run

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": [], "meta": {"result_count": 0}})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        sentinel._get_session = AsyncMock(return_value=mock_session)

        await sentinel._fetch_x_posts("coffee", limit=10, sort_order="recency")

        call_args = mock_session.get.call_args
        params = call_args.kwargs.get('params') or call_args[1].get('params')
        assert "start_time" in params, "start_time should be present on first run"
        assert "since_id" not in params, "since_id should not be present on first run"
