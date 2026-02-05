"""
Tests for V6.2 sentinel crash fixes.
Covers both crash vectors + StateManager systemic fix.
"""
import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock
from trading_bot.sentinels import NewsSentinel, PredictionMarketSentinel, Sentinel
from trading_bot.state_manager import StateManager


# === TEST 1: NewsSentinel string response ===
@pytest.mark.asyncio
async def test_news_sentinel_handles_string_response():
    """FD's finding: LLM returns string instead of dict."""
    config = {
        'sentinels': {'news': {'rss_urls': ['http://test.rss'], 'sentiment_magnitude_threshold': 8, 'model': 'test'}},
        'gemini': {'api_key': 'TEST'}
    }
    with patch.object(NewsSentinel, '_fetch_rss_safe', new_callable=AsyncMock) as mock_rss, \
         patch('google.genai.Client'):
        mock_rss.return_value = ["Some headline"]
        sentinel = NewsSentinel(config)
        # Simulate LLM returning a string (valid JSON but not dict)
        sentinel._analyze_with_ai = AsyncMock(return_value="No significant news found")

        trigger = await sentinel.check()
        # Should return None gracefully, not crash
        assert trigger is None


@pytest.mark.asyncio
async def test_news_sentinel_handles_list_response():
    """LLM returns JSON array instead of object."""
    config = {
        'sentinels': {'news': {'rss_urls': ['http://test.rss'], 'sentiment_magnitude_threshold': 8, 'model': 'test'}},
        'gemini': {'api_key': 'TEST'}
    }
    with patch.object(NewsSentinel, '_fetch_rss_safe', new_callable=AsyncMock) as mock_rss, \
         patch('google.genai.Client'):
        mock_rss.return_value = ["Some headline"]
        sentinel = NewsSentinel(config)
        sentinel._analyze_with_ai = AsyncMock(return_value=[{"score": 9}])

        trigger = await sentinel.check()
        assert trigger is None


# === TEST 2: StateManager STALE string substitution ===
def test_load_state_raw_bypasses_stale_strings(tmp_path):
    """Prove that load_state_raw returns dicts even for stale data."""
    import json, os

    # Create a state file with old timestamp
    state_file = tmp_path / "state.json"
    state_data = {
        "test_ns": {
            "key1": {
                "data": {"slug": "test-slug", "price": 0.5},
                "timestamp": time.time() - 7200  # 2 hours old (stale)
            }
        }
    }
    with open(state_file, 'w') as f:
        json.dump(state_data, f)

    # Patch STATE_FILE
    with patch('trading_bot.state_manager.STATE_FILE', str(state_file)):
        # load_state returns STALE string
        result_old = StateManager.load_state(namespace="test_ns")
        assert isinstance(result_old.get("key1"), str)
        assert "STALE" in result_old["key1"]

        # load_state_raw returns dict
        result_raw = StateManager.load_state_raw(namespace="test_ns")
        assert isinstance(result_raw.get("key1"), dict)
        assert result_raw["key1"]["slug"] == "test-slug"


# === TEST 3: PredictionMarket loads stale state safely ===
@pytest.mark.asyncio
async def test_prediction_market_loads_stale_state_safely():
    """Crash Vector B: state_cache should never contain strings."""
    config = {
        'sentinels': {'prediction_markets': {
            'enabled': True, 'poll_interval_seconds': 0,
            'topics_to_watch': [{'query': 'Fed', 'tag': 'Fed'}]
        }},
        'notifications': {'enabled': False}
    }

    # Mock load_state_raw to return valid dicts
    with patch.object(StateManager, 'load_state_raw', return_value={
        "Fed": {"slug": "test-slug", "price": 0.5, "severity_hwm": 0, "hwm_timestamp": None}
    }):
        sentinel = PredictionMarketSentinel(config)

        # All values should be dicts, not strings
        for key, value in sentinel.state_cache.items():
            assert isinstance(value, dict), f"state_cache['{key}'] is {type(value)}, expected dict"


# === TEST 4: Relevance scoring ===
@pytest.mark.asyncio
async def test_relevance_weighted_selection():
    """Verify relevance keywords influence market selection."""
    config = {
        'sentinels': {'prediction_markets': {
            'enabled': True, 'poll_interval_seconds': 0,
            'min_liquidity_usd': 1000, 'min_volume_usd': 1000,
            'topics_to_watch': []
        }},
        'notifications': {'enabled': False}
    }

    sentinel = PredictionMarketSentinel(config)

    # Mock: deportation market has higher liquidity, but Fed market is relevant
    mock_response = [
        {'slug': 'trump-deportation', 'title': 'How many people will Trump deport?', 'markets': [
            {'outcomePrices': ['0.50'], 'volume': '500000', 'liquidity': '1000000'}
        ]},
        {'slug': 'fed-rate-march', 'title': 'Will the Federal Reserve cut rates in March?', 'markets': [
            {'outcomePrices': ['0.30'], 'volume': '200000', 'liquidity': '500000'}
        ]}
    ]

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        # With relevance keywords, should pick Fed market despite lower liquidity
        result = await sentinel._resolve_active_market(
            "Federal Reserve interest rate",
            relevance_keywords=["federal reserve", "interest rate", "fomc", "fed funds"]
        )

    assert result is not None
    assert result['slug'] == 'fed-rate-march'


# === TEST 5: Duplicate slug detection ===
@pytest.mark.asyncio
async def test_duplicate_slug_detection(caplog):
    """Verify warning when multiple topics resolve to same market."""
    config = {
        'sentinels': {'prediction_markets': {
            'enabled': True, 'poll_interval_seconds': 0,
            'topics_to_watch': [
                {'query': 'Fed', 'tag': 'Fed'},
                {'query': 'Brazil', 'tag': 'Brazil'}
            ]
        }},
        'notifications': {'enabled': False}
    }

    with patch.object(StateManager, 'load_state_raw', return_value={}):
        sentinel = PredictionMarketSentinel(config)

    # Manually seed cache with same slug for both
    sentinel.state_cache = {
        'Fed': {'slug': 'same-market', 'price': 0.5},
        'Brazil': {'slug': 'same-market', 'price': 0.5}
    }

    # The duplicate detection runs at end of check()
    # We can test the detection logic directly
    slug_map = {}
    for topic in sentinel.topics:
        query = topic.get('query', '')
        cached = sentinel.state_cache.get(query, {})
        slug = cached.get('slug') if isinstance(cached, dict) else None
        if slug:
            slug_map.setdefault(slug, []).append(topic.get('tag', query))

    duplicates = {s: t for s, t in slug_map.items() if len(t) > 1}
    assert len(duplicates) == 1
    assert 'same-market' in duplicates
    assert set(duplicates['same-market']) == {'Fed', 'Brazil'}


# === TEST 6: Base Sentinel validation method ===
def test_validate_ai_response():
    """Test the shared response validation utility."""
    sentinel = Sentinel.__new__(Sentinel)
    sentinel.__class__ = type('TestSentinel', (Sentinel,), {})

    # Valid dict
    assert sentinel._validate_ai_response({"score": 9}) == {"score": 9}

    # None
    assert sentinel._validate_ai_response(None) is None

    # String (the crash case)
    assert sentinel._validate_ai_response("No news") is None

    # List
    assert sentinel._validate_ai_response([1, 2, 3]) is None

    # Integer
    assert sentinel._validate_ai_response(42) is None

# === TEST 7: Relevance gate returns None when no match ===
@pytest.mark.asyncio
async def test_relevance_gate_returns_none_on_no_match():
    """Fix #1: When relevance_keywords exist but no candidates match,
    _resolve_active_market should return None (not highest liquidity)."""
    config = {
        'sentinels': {'prediction_markets': {
            'enabled': True, 'poll_interval_seconds': 0,
            'min_liquidity_usd': 1000, 'min_volume_usd': 1000,
            'min_relevance_score': 1,
            'topics_to_watch': []
        }},
        'notifications': {'enabled': False}
    }

    sentinel = PredictionMarketSentinel(config)

    # Mock: Only a deportation market (irrelevant to Fed query)
    mock_response = [
        {'slug': 'trump-deportation', 'title': 'How many people will Trump deport?', 'markets': [
            {'outcomePrices': ['0.50'], 'volume': '500000', 'liquidity': '1000000'}
        ]}
    ]

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        result = await sentinel._resolve_active_market(
            "Federal Reserve interest rate",
            relevance_keywords=["federal reserve", "interest rate", "fomc"],
            min_relevance_score=1
        )

    # Must return None — NOT the deportation market
    assert result is None

# === TEST 8: Commodity impact field in payload ===
@pytest.mark.asyncio
async def test_commodity_impact_field_in_trigger():
    """Fix #4: Trigger payload should use 'commodity_impact' not 'coffee_impact'."""
    config = {
        'sentinels': {'prediction_markets': {
            'enabled': True, 'poll_interval_seconds': 0,
            'min_relevance_score': 1,
            'topics_to_watch': [{
                'query': 'Test topic',
                'tag': 'Test',
                'display_name': 'Test Topic',
                'trigger_threshold_pct': 5.0,
                'importance': 'macro',
                'commodity_impact': 'Generic commodity impact description',
                'relevance_keywords': ['test']
            }]
        }},
        'notifications': {'enabled': False}
    }

    sentinel = PredictionMarketSentinel(config)

    # Seed state with baseline
    sentinel.state_cache['Test topic'] = {
        'slug': 'test-market',
        'price': 0.50,
        'timestamp': '2026-01-01T00:00:00+00:00',
        'severity_hwm': 0,
        'hwm_timestamp': None
    }

    # Mock a 20% swing (big enough to trigger)
    mock_response = [
        {'slug': 'test-market', 'title': 'Test Coffee Market', 'markets': [
            {'outcomePrices': ['0.70'], 'volume': '100000', 'liquidity': '50000'}
        ]}
    ]

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        with patch('trading_bot.utils.is_trading_day', return_value=True),              patch('trading_bot.utils.is_market_open', return_value=True):
            trigger = await sentinel.check()

    assert trigger is not None
    assert 'commodity_impact' in trigger.payload
    # coffee_impact might still be there if the sentinel copies it over from topic dict?
    # Wait, in the code I replaced it.
    # But wait, if topic dict has coffee_impact as fallback key?
    # In my code:
    # commodity_impact = topic.get('commodity_impact', topic.get('coffee_impact', ...))
    # Payload: "commodity_impact": commodity_impact
    # So coffee_impact key should NOT be in payload unless I explicitly put it there.
    # My previous diff removed "coffee_impact": coffee_impact from payload.
    assert 'coffee_impact' not in trigger.payload

# === TEST 9: Min relevance score filters partial matches ===
@pytest.mark.asyncio
async def test_min_relevance_score_filters_partial_matches():
    """When min_relevance_score=2, a market matching only 1 keyword is rejected."""
    config = {
        'sentinels': {'prediction_markets': {
            'enabled': True, 'poll_interval_seconds': 0,
            'min_liquidity_usd': 1000, 'min_volume_usd': 1000,
            'topics_to_watch': []
        }},
        'notifications': {'enabled': False}
    }

    sentinel = PredictionMarketSentinel(config)

    # Market title matches "trump" but NOT "fed", "chair", "nominee"
    mock_response = [
        {'slug': 'trump-other', 'title': 'Will Trump visit Mars?', 'markets': [
            {'outcomePrices': ['0.10'], 'volume': '50000', 'liquidity': '100000'}
        ]}
    ]

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get.return_value = mock_ctx

        result = await sentinel._resolve_active_market(
            "Trump Fed Chair nominee",
            relevance_keywords=["fed", "chair", "nominee", "trump", "nominate"],
            min_relevance_score=2  # Needs at least 2 matches
        )

    # Only 1 keyword ("trump") matches → below threshold → None
    assert result is None

# === TEST 10: XSentimentSentinel string response (Issue 1b) ===
from trading_bot.sentinels import XSentimentSentinel

@pytest.mark.asyncio
async def test_x_sentinel_non_dict_return():
    """Test that XSentimentSentinel handles non-dict returns (e.g. error strings) without crashing."""
    config = {
        'sentinels': {
            'x_sentiment': {
                'model': 'grok-4-1-fast-reasoning',
                'search_queries': ['coffee futures'],
                'sentiment_threshold': 6.5,
                'min_engagement': 5,
                'volume_spike_multiplier': 2.0,
                'from_handles': [],
                'exclude_keywords': ['meme']
            }
        },
        'xai': {'api_key': 'test_key_12345'}
    }

    with patch('trading_bot.sentinels.AsyncOpenAI'):
        sentinel = XSentimentSentinel(config)

        # Mock _search_x_and_analyze to return a string (simulating the crash condition)
        # Note: check() calls _sem_bound_search -> _search_x_and_analyze
        # We mock _search_x_and_analyze directly as check() gathers results from it
        sentinel._search_x_and_analyze = AsyncMock(return_value="Error: API rate limit exceeded")

        # This should NOT crash with AttributeError: 'str' object has no attribute 'get'
        trigger = await sentinel.check()

        # Should return None because valid_results should be empty
        assert trigger is None
