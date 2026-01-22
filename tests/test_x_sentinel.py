import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from trading_bot.sentinels import XSentimentSentinel

@pytest.fixture
def mock_config():
    return {
        'sentinels': {
            'x_sentiment': {
                'model': 'grok-4-1-fast-reasoning',
                'search_queries': ['coffee futures', 'arabica prices'],
                'sentiment_threshold': 6.5,
                'min_engagement': 5,
                'volume_spike_multiplier': 2.0,
                'from_handles': [],
                'exclude_keywords': ['meme']
            }
        },
        'xai': {'api_key': 'test_key_12345'}
    }

@pytest.mark.asyncio
async def test_initialization(mock_config):
    """Test sentinel initializes correctly."""
    with patch('trading_bot.sentinels.AsyncOpenAI'):
        sentinel = XSentimentSentinel(mock_config)
        assert sentinel.model == 'grok-4-1-fast-reasoning'
        assert len(sentinel.search_queries) == 2
        assert sentinel.sentiment_threshold == 6.5

@pytest.mark.asyncio
async def test_bullish_trigger(mock_config):
    """Test extremely bullish sentiment triggers correctly."""
    with patch('trading_bot.sentinels.AsyncOpenAI'):
        sentinel = XSentimentSentinel(mock_config)

        # Mock successful bullish analysis
        mock_response = {
            'sentiment_score': 8.5,
            'confidence': 0.85,
            'key_themes': ['frost', 'shortage', 'panic buying'],
            'post_volume': 50,
            'summary': 'Very bullish chatter on X',
            'sentiment_distribution': {'bullish': 70, 'neutral': 20, 'bearish': 10}
        }

        sentinel._search_x_and_analyze = AsyncMock(return_value=mock_response)

        trigger = await sentinel.check()

        assert trigger is not None
        assert 'BULLISH' in trigger.reason
        assert trigger.severity == 7
        assert trigger.payload['sentiment_score'] >= 6.5
        assert trigger.payload['confidence'] > 0.8

@pytest.mark.asyncio
async def test_bearish_trigger(mock_config):
    """Test extremely bearish sentiment triggers correctly."""
    with patch('trading_bot.sentinels.AsyncOpenAI'):
        sentinel = XSentimentSentinel(mock_config)

        mock_response = {
            'sentiment_score': 2.0,
            'confidence': 0.80,
            'key_themes': ['oversupply', 'demand crash', 'recession'],
            'post_volume': 45,
            'summary': 'Bearish sentiment dominates',
            'sentiment_distribution': {'bullish': 10, 'neutral': 20, 'bearish': 70}
        }

        sentinel._search_x_and_analyze = AsyncMock(return_value=mock_response)

        trigger = await sentinel.check()

        assert trigger is not None
        assert 'BEARISH' in trigger.reason
        assert trigger.severity == 7

@pytest.mark.asyncio
async def test_volume_spike_trigger(mock_config):
    """Test volume anomaly detection triggers even with neutral sentiment."""
    with patch('trading_bot.sentinels.AsyncOpenAI'):
        sentinel = XSentimentSentinel(mock_config)

        # Seed baseline history (low volume)
        sentinel.post_volume_history = [10, 12, 11, 9, 10, 11, 10, 9, 12, 11]
        sentinel._update_volume_stats(10)  # Update stats

        # Mock spike with neutral sentiment
        mock_response = {
            'sentiment_score': 5.0,  # Neutral
            'confidence': 0.75,
            'key_themes': ['breaking', 'developing'],
            'post_volume': 100,  # 10x normal
            'summary': 'Unusual spike in coffee discussions'
        }

        sentinel._search_x_and_analyze = AsyncMock(return_value=mock_response)

        trigger = await sentinel.check()

        assert trigger is not None
        assert 'SPIKE' in trigger.reason or 'ACTIVITY' in trigger.reason
        assert trigger.payload['volume_spike'] is True
        assert trigger.severity == 6

@pytest.mark.asyncio
async def test_no_trigger_neutral(mock_config):
    """Test that neutral sentiment with normal volume doesn't trigger."""
    with patch('trading_bot.sentinels.AsyncOpenAI'):
        sentinel = XSentimentSentinel(mock_config)

        mock_response = {
            'sentiment_score': 5.0,
            'confidence': 0.6,
            'key_themes': ['normal', 'steady'],
            'post_volume': 10,
            'summary': 'Normal market chatter'
        }

        sentinel._search_x_and_analyze = AsyncMock(return_value=mock_response)

        trigger = await sentinel.check()

        assert trigger is None

@pytest.mark.asyncio
async def test_parallel_execution(mock_config):
    """Test that multiple queries execute in parallel."""
    with patch('trading_bot.sentinels.AsyncOpenAI'), \
         patch('trading_bot.sentinels.acquire_api_slot', new_callable=AsyncMock) as mock_limit, \
         patch('numpy.random.uniform', return_value=0.0): # Remove jitter

        mock_limit.return_value = True # Grant slot immediately
        sentinel = XSentimentSentinel(mock_config)
        sentinel._request_interval = 0.0 # Remove rate limit interval

        call_count = 0

        async def mock_analyze(query):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate API latency
            return {
                'sentiment_score': 5.0,
                'confidence': 0.5,
                'key_themes': [],
                'post_volume': 10,
                'summary': 'Test'
            }

        sentinel._search_x_and_analyze = mock_analyze

        import time
        start = time.time()
        await sentinel.check()
        elapsed = time.time() - start

        assert call_count == 2  # Both queries executed
        assert elapsed < 0.5  # Parallel execution should be < 0.5s (vs 0.2s sequential)

@pytest.mark.asyncio
async def test_weighted_averaging(mock_config):
    """Test that sentiment is weighted by confidence and volume."""
    with patch('trading_bot.sentinels.AsyncOpenAI'):
        sentinel = XSentimentSentinel(mock_config)

        # Query 1: High sentiment, high confidence, high volume
        # Query 2: Low sentiment, low confidence, low volume
        # Weighted average should favor Query 1

        results = [
            {
                'sentiment_score': 8.0,
                'confidence': 0.9,
                'post_volume': 100,
                'key_themes': ['bullish'],
                'summary': 'Strong signal'
            },
            {
                'sentiment_score': 3.0,
                'confidence': 0.3,
                'post_volume': 5,
                'key_themes': ['bearish'],
                'summary': 'Weak signal'
            }
        ]

        # Mock to return different results for each query
        call_index = 0
        async def mock_analyze(query):
            nonlocal call_index
            result = results[call_index]
            call_index += 1
            return result

        sentinel._search_x_and_analyze = mock_analyze

        trigger = await sentinel.check()

        # Weighted average should be closer to 8.0 than simple average (5.5)
        assert trigger is not None
        assert trigger.payload['sentiment_score'] > 6.5  # Should trigger bullish

@pytest.mark.asyncio
async def test_error_handling(mock_config):
    """Test graceful handling of API failures."""
    with patch('trading_bot.sentinels.AsyncOpenAI'):
        sentinel = XSentimentSentinel(mock_config)

        # Mock one success, one failure
        results = [
            {'sentiment_score': 5.0, 'confidence': 0.5, 'post_volume': 10,
             'key_themes': [], 'summary': 'OK'},
            Exception("API Error")
        ]

        call_index = 0
        async def mock_analyze(query):
            nonlocal call_index
            result = results[call_index]
            call_index += 1
            if isinstance(result, Exception):
                raise result
            return result

        sentinel._search_x_and_analyze = mock_analyze

        # Should not crash, should use the one successful result
        trigger = await sentinel.check()

        # No trigger expected (neutral sentiment)
        assert trigger is None

@pytest.mark.asyncio
async def test_deduplication(mock_config):
    """Test that duplicate payloads are filtered."""
    with patch('trading_bot.sentinels.AsyncOpenAI'):
        sentinel = XSentimentSentinel(mock_config)

        mock_response = {
            'sentiment_score': 8.0,
            'confidence': 0.8,
            'key_themes': ['same', 'themes'],
            'post_volume': 50,
            'summary': 'Same signal'
        }

        sentinel._search_x_and_analyze = AsyncMock(return_value=mock_response)

        # First check should trigger
        trigger1 = await sentinel.check()
        assert trigger1 is not None

        # Second check with same data should be deduplicated
        trigger2 = await sentinel.check()
        assert trigger2 is None  # Duplicate detected
