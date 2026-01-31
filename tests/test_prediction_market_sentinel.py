import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from trading_bot.sentinels import PredictionMarketSentinel, SentinelTrigger

@pytest.fixture
def mock_config():
    return {
        'sentinels': {
            'prediction_markets': {
                'enabled': True,
                'poll_interval_seconds': 0,  # Disable rate limiting for tests
                'min_liquidity_usd': 50000,
                'min_volume_usd': 10000,
                'hwm_decay_hours': 24,
                'providers': {
                    'polymarket': {
                        'api_url': 'https://gamma-api.polymarket.com/events',
                        'search_limit': 10,
                        'enabled': True
                    }
                },
                'topics_to_watch': [
                    {
                        'query': 'Federal Reserve Interest Rate',
                        'tag': 'Fed',
                        'display_name': 'Fed Policy',
                        'trigger_threshold_pct': 10.0,
                        'importance': 'macro',
                        'coffee_impact': 'USD strength'
                    }
                ]
            }
        },
        'notifications': {'enabled': False}
    }


def mock_polymarket_response(slug, title, price, liquidity=200000, volume=100000):
    """Helper to create mock Polymarket API responses."""
    return [
        {
            'slug': slug,
            'title': title,
            'markets': [{
                'outcomePrices': [str(price), str(1 - price)],
                'volume': str(volume),
                'liquidity': str(liquidity)
            }]
        }
    ]


@pytest.mark.asyncio
async def test_initialization(mock_config):
    """Test sentinel initializes correctly with new v2.0 config."""
    sentinel = PredictionMarketSentinel(mock_config)
    assert len(sentinel.topics) == 1
    assert sentinel.poll_interval == 0
    assert sentinel.min_liquidity == 50000
    assert sentinel.min_volume == 10000
    assert sentinel.hwm_decay_hours == 24


@pytest.mark.asyncio
async def test_dynamic_discovery_selects_highest_liquidity(mock_config):
    """Test that _resolve_active_market selects the highest liquidity market."""
    sentinel = PredictionMarketSentinel(mock_config)

    # Mock response with multiple markets, different liquidity
    multi_market_response = [
        {'slug': 'fed-july', 'title': 'Fed July', 'markets': [
            {'outcomePrices': ['0.60'], 'volume': '50000', 'liquidity': '100000'}
        ]},
        {'slug': 'fed-june', 'title': 'Fed June', 'markets': [
            {'outcomePrices': ['0.70'], 'volume': '80000', 'liquidity': '300000'}  # HIGHEST
        ]},
        {'slug': 'fed-sept', 'title': 'Fed Sept', 'markets': [
            {'outcomePrices': ['0.55'], 'volume': '60000', 'liquidity': '150000'}
        ]},
    ]

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=multi_market_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        result = await sentinel._resolve_active_market("Federal Reserve")

    # Should select fed-june (highest liquidity)
    assert result is not None
    assert result['slug'] == 'fed-june'
    assert result['liquidity'] == 300000


@pytest.mark.asyncio
async def test_slug_consistency_check_resets_baseline(mock_config):
    """Test that market rollover (June→July) resets baseline without triggering."""
    sentinel = PredictionMarketSentinel(mock_config)

    # Seed cache with June market at 90%
    sentinel.state_cache['Federal Reserve Interest Rate'] = {
        'slug': 'fed-june-2025',  # OLD slug
        'price': 0.90,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'severity_hwm': 0,
        'hwm_timestamp': None
    }

    # New market is July at 40% (appears like a -50% crash!)
    mock_response = mock_polymarket_response('fed-july-2025', 'Fed July', 0.40)

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        with patch('trading_bot.sentinels.send_pushover_notification') as mock_notify:
            trigger = await sentinel.check()

    # Should NOT trigger (rollover detected)
    assert trigger is None

    # Cache should be reset to new market
    assert sentinel.state_cache['Federal Reserve Interest Rate']['slug'] == 'fed-july-2025'
    assert sentinel.state_cache['Federal Reserve Interest Rate']['price'] == 0.40

    # Should have sent informational notification about rollover
    assert mock_notify.called


@pytest.mark.asyncio
async def test_high_water_mark_prevents_flapping(mock_config):
    """
    Test that HWM prevents alert flapping on severity oscillation.

    SCENARIO:
    1. Price +15% (severity 6) → triggers
    2. Price +21% (severity 7) → triggers (escalation)
    3. Price +19% (severity 6) → SUPPRESSED (de-escalation)
    """
    sentinel = PredictionMarketSentinel(mock_config)

    query = 'Federal Reserve Interest Rate'

    # === Move 1: 50% → 65% (+15%, severity 6) ===
    sentinel.state_cache[query] = {
        'slug': 'fed-test',
        'price': 0.50,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'severity_hwm': 0,
        'hwm_timestamp': None
    }

    mock_response = mock_polymarket_response('fed-test', 'Fed Test', 0.65)

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        trigger1 = await sentinel.check()

    assert trigger1 is not None
    assert trigger1.severity == 6
    assert sentinel.state_cache[query]['severity_hwm'] == 6

    # === Move 2: 65% → 86% (+21%, severity 7) - ESCALATION ===
    sentinel._last_poll_time = 0
    mock_response = mock_polymarket_response('fed-test', 'Fed Test', 0.86)

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        trigger2 = await sentinel.check()

    assert trigger2 is not None
    assert trigger2.severity == 7  # Escalation alerts
    assert sentinel.state_cache[query]['severity_hwm'] == 7

    # === Move 3: 86% → 67% (-19%, severity 6) - DE-ESCALATION ===
    sentinel._last_poll_time = 0
    mock_response = mock_polymarket_response('fed-test', 'Fed Test', 0.67)

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        trigger3 = await sentinel.check()

    # Should be SUPPRESSED (severity 6 <= HWM 7)
    assert trigger3 is None


@pytest.mark.asyncio
async def test_hwm_decay_allows_re_alerting(mock_config):
    """Test that HWM decays after 24h, allowing re-alerting."""
    sentinel = PredictionMarketSentinel(mock_config)

    query = 'Federal Reserve Interest Rate'

    # Set HWM from 25 hours ago (should decay)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()

    sentinel.state_cache[query] = {
        'slug': 'fed-test',
        'price': 0.50,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'severity_hwm': 7,  # Previous high severity
        'hwm_timestamp': old_timestamp  # OLD - should decay
    }

    # Move that would normally be suppressed (severity 6 < HWM 7)
    mock_response = mock_polymarket_response('fed-test', 'Fed Test', 0.65)

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        trigger = await sentinel.check()

    # Should trigger because HWM decayed
    assert trigger is not None
    assert trigger.severity == 6


@pytest.mark.asyncio
async def test_liquidity_filter_blocks_thin_markets(mock_config):
    """Test that low liquidity markets are filtered out."""
    sentinel = PredictionMarketSentinel(mock_config)

    # Mock response with HUGE move but LOW liquidity
    thin_market_response = [
        {'slug': 'thin-market', 'title': 'Thin', 'markets': [
            {'outcomePrices': ['0.90'], 'volume': '5000', 'liquidity': '10000'}  # Below threshold
        ]}
    ]

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=thin_market_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        result = await sentinel._resolve_active_market("Test Query")

    # Should return None (filtered out)
    assert result is None


@pytest.mark.asyncio
async def test_no_trigger_on_small_move(mock_config):
    """Test that small moves don't trigger."""
    sentinel = PredictionMarketSentinel(mock_config)

    query = 'Federal Reserve Interest Rate'

    sentinel.state_cache[query] = {
        'slug': 'fed-test',
        'price': 0.50,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'severity_hwm': 0,
        'hwm_timestamp': None
    }

    # 5% move (below 10% threshold)
    mock_response = mock_polymarket_response('fed-test', 'Fed Test', 0.55)

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        trigger = await sentinel.check()

    assert trigger is None


@pytest.mark.asyncio
async def test_severity_scaling():
    """Test severity calculation at different thresholds."""
    mock_cfg = {'sentinels': {'prediction_markets': {}}}
    sentinel = PredictionMarketSentinel(mock_cfg)

    assert sentinel._calculate_severity(15) == 6   # 10-20%
    assert sentinel._calculate_severity(25) == 7   # 20-30%
    assert sentinel._calculate_severity(35) == 9   # 30%+
    assert sentinel._calculate_severity(50) == 9   # 30%+


@pytest.mark.asyncio
async def test_hwm_decay_check():
    """Test HWM decay time calculation."""
    mock_cfg = {'sentinels': {'prediction_markets': {'hwm_decay_hours': 24}}}
    sentinel = PredictionMarketSentinel(mock_cfg)

    # 23 hours ago - should NOT decay
    recent = (datetime.now(timezone.utc) - timedelta(hours=23)).isoformat()
    assert sentinel._should_decay_hwm(recent) is False

    # 25 hours ago - should decay
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    assert sentinel._should_decay_hwm(old) is True

    # None - should not decay
    assert sentinel._should_decay_hwm(None) is False
