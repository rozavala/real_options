"""Tests for trading_bot.semantic_cache — C.1 Semantic Cache Completion."""

import math
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from trading_bot.semantic_cache import SemanticCache, get_semantic_cache, _cache_instance
import trading_bot.semantic_cache as sc_module


# --- Fixtures ---

@pytest.fixture
def config():
    return {
        'semantic_cache': {
            'enabled': True,
            'similarity_threshold': 0.92,
            'ttl_minutes': 60,
            'max_entries': 100,
            'severity_bypass_threshold': 8,
        }
    }


@pytest.fixture
def disabled_config():
    return {
        'semantic_cache': {
            'enabled': False,
            'similarity_threshold': 0.92,
            'ttl_minutes': 60,
            'max_entries': 100,
        }
    }


@pytest.fixture
def cache(config):
    return SemanticCache(config)


@pytest.fixture
def base_state():
    """Baseline market state for testing."""
    return {
        'price': 350.0,
        'sma_200': 340.0,
        'price_vs_sma': 0.0294,  # (350-340)/340
        'volatility_5d': 0.025,
        'regime': 'TRENDING_UP',
        'sentiment_score': 0.3,
        'recent_alert_count': 2,
    }


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before each test."""
    sc_module._cache_instance = None
    yield
    sc_module._cache_instance = None


# --- Vectorization Sensitivity Tests ---

class TestVectorizationSensitivity:
    """Regression tests for the cosine dominance bug.

    Old 8-dim vector with regime one-hot: 5% price move → similarity 0.998 (always hits).
    New 4-dim scaled vector: 5% price move → similarity ~0.93 (correctly near threshold).
    """

    def test_5pct_price_move_below_threshold(self, cache, base_state):
        """A 5% price move should produce similarity BELOW the 0.92 threshold."""
        state_a = dict(base_state)
        state_b = dict(base_state)
        # 5% price move changes price_vs_sma significantly
        state_b['price_vs_sma'] = base_state['price_vs_sma'] + 0.05

        vec_a = cache._vectorize_market_state(state_a)
        vec_b = cache._vectorize_market_state(state_b)
        similarity = cache._cosine_similarity(vec_a, vec_b)

        # The whole point: 5% price move should NOT be a cache hit
        assert similarity < 0.97, (
            f"5% price move similarity {similarity:.4f} is too high — "
            f"cache would always hit (dominance bug not fixed)"
        )

    def test_identical_states_perfect_similarity(self, cache, base_state):
        """Identical states should have similarity 1.0."""
        vec = cache._vectorize_market_state(base_state)
        similarity = cache._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, abs=1e-9)

    def test_tiny_change_above_threshold(self, cache, base_state):
        """Very small changes should still hit the cache."""
        state_a = dict(base_state)
        state_b = dict(base_state)
        # 0.5% price move — should still be cached
        state_b['price_vs_sma'] = base_state['price_vs_sma'] + 0.005

        vec_a = cache._vectorize_market_state(state_a)
        vec_b = cache._vectorize_market_state(state_b)
        similarity = cache._cosine_similarity(vec_a, vec_b)

        assert similarity >= 0.92, (
            f"0.5% price move similarity {similarity:.4f} is below threshold — "
            f"cache too sensitive"
        )

    def test_vector_is_4_dimensions(self, cache, base_state):
        """Vector should be exactly 4 dimensions (regime removed)."""
        vec = cache._vectorize_market_state(base_state)
        assert len(vec) == 4

    def test_scaling_price_vs_sma(self, cache):
        """price_vs_sma=0.15 should map to 1.0 in the vector."""
        state = {'price_vs_sma': 0.15, 'volatility_5d': 0.02, 'sentiment_score': 0.0, 'recent_alert_count': 0}
        vec = cache._vectorize_market_state(state)
        assert vec[0] == pytest.approx(1.0)

    def test_scaling_volatility(self, cache):
        """volatility_5d=0.06 should map to 1.0 in the vector."""
        state = {'price_vs_sma': 0.0, 'volatility_5d': 0.06, 'sentiment_score': 0.0, 'recent_alert_count': 0}
        vec = cache._vectorize_market_state(state)
        assert vec[1] == pytest.approx(1.0)

    def test_none_values_use_defaults(self, cache):
        """None values should fall back to defaults, not crash."""
        state = {'price_vs_sma': None, 'volatility_5d': None, 'sentiment_score': None, 'recent_alert_count': None}
        vec = cache._vectorize_market_state(state)
        assert len(vec) == 4
        assert vec[0] == 0.0  # price_vs_sma default 0
        assert vec[1] == pytest.approx(0.02 / 0.06)  # vol default 0.02


# --- Regime Partition Tests ---

class TestRegimePartition:
    """Different regimes use different partition keys — they never cross-hit."""

    def test_different_regimes_never_cross_hit(self, cache, base_state):
        """A put in TRENDING_UP should not satisfy a get in RANGE_BOUND."""
        result = {'direction': 'BULLISH', 'confidence': 'HIGH'}
        cache.put("KCN5", "PriceSentinel", base_state, result)

        # Same market data, different regime
        rb_state = dict(base_state)
        rb_state['regime'] = 'RANGE_BOUND'
        got = cache.get("KCN5", "PriceSentinel", rb_state)
        assert got is None

    def test_same_regime_hits(self, cache, base_state):
        """Same regime with similar state should hit."""
        result = {'direction': 'BULLISH', 'confidence': 'HIGH'}
        cache.put("KCN5", "PriceSentinel", base_state, result)

        got = cache.get("KCN5", "PriceSentinel", base_state)
        assert got == result

    def test_partition_key_format(self, cache, base_state):
        """Partition key should be contract|trigger|regime."""
        key = cache._partition_key("KCN5", "PriceSentinel", base_state)
        assert key == "KCN5|PriceSentinel|TRENDING_UP"

    def test_unknown_regime_partition(self, cache):
        """Missing regime defaults to UNKNOWN in partition key."""
        state = {'price_vs_sma': 0.01}
        key = cache._partition_key("KCN5", "PriceSentinel", state)
        assert key == "KCN5|PriceSentinel|UNKNOWN"


# --- Trigger Scoping Tests ---

class TestTriggerScoping:
    """PriceSentinel puts should not satisfy WeatherSentinel gets."""

    def test_different_triggers_dont_cross_hit(self, cache, base_state):
        """A PriceSentinel cached result should NOT satisfy a WeatherSentinel get."""
        result = {'direction': 'BULLISH', 'confidence': 'MODERATE'}
        cache.put("KCN5", "PriceSentinel", base_state, result)

        got = cache.get("KCN5", "WeatherSentinel", base_state)
        assert got is None

    def test_same_trigger_hits(self, cache, base_state):
        """Same trigger source should hit."""
        result = {'direction': 'BEARISH', 'confidence': 'HIGH'}
        cache.put("KCN5", "WeatherSentinel", base_state, result)

        got = cache.get("KCN5", "WeatherSentinel", base_state)
        assert got == result


# --- Ticker Invalidation Tests ---

class TestTickerInvalidation:
    """invalidate_by_ticker should clear by ticker prefix."""

    def test_invalidate_kc_clears_kc_only(self, cache, base_state):
        """invalidate_by_ticker('KC') clears KCN5, keeps CCN5."""
        kc_result = {'direction': 'BULLISH', 'confidence': 'HIGH'}
        cc_result = {'direction': 'BEARISH', 'confidence': 'LOW'}

        cache.put("KCN5", "PriceSentinel", base_state, kc_result)
        cache.put("CCN5", "PriceSentinel", base_state, cc_result)

        cache.invalidate_by_ticker("KC")

        assert cache.get("KCN5", "PriceSentinel", base_state) is None
        assert cache.get("CCN5", "PriceSentinel", base_state) == cc_result

    def test_invalidate_ticker_with_multiple_triggers(self, cache, base_state):
        """Invalidation clears all triggers for the given ticker."""
        r1 = {'direction': 'BULLISH'}
        r2 = {'direction': 'BEARISH'}
        cache.put("KCN5", "PriceSentinel", base_state, r1)
        cache.put("KCN5", "WeatherSentinel", base_state, r2)

        cache.invalidate_by_ticker("KC")

        assert cache.get("KCN5", "PriceSentinel", base_state) is None
        assert cache.get("KCN5", "WeatherSentinel", base_state) is None

    def test_invalidate_nonexistent_ticker_no_crash(self, cache):
        """Invalidating a ticker with no entries should not crash."""
        cache.invalidate_by_ticker("ZZ")  # no entries

    def test_invalidate_contract_prefix_match(self, cache, base_state):
        """invalidate(contract='KCN5') removes all partitions starting with KCN5."""
        result = {'direction': 'BULLISH'}
        cache.put("KCN5", "PriceSentinel", base_state, result)
        cache.invalidate(contract="KCN5")

        assert cache.get("KCN5", "PriceSentinel", base_state) is None


# --- TTL Tests ---

class TestTTL:
    """Cache entries should expire after ttl_minutes."""

    def test_expired_entry_returns_none(self, cache, base_state):
        """An entry older than TTL should not be returned."""
        result = {'direction': 'BULLISH', 'confidence': 'HIGH'}
        cache.put("KCN5", "PriceSentinel", base_state, result)

        # Manually age the entry
        key = cache._partition_key("KCN5", "PriceSentinel", base_state)
        old_time = datetime.now(timezone.utc) - timedelta(minutes=61)
        entries = cache._cache[key]
        cache._cache[key] = [(v, r, old_time) for v, r, _ in entries]

        got = cache.get("KCN5", "PriceSentinel", base_state)
        assert got is None

    def test_fresh_entry_returns_result(self, cache, base_state):
        """An entry within TTL should be returned."""
        result = {'direction': 'BULLISH', 'confidence': 'HIGH'}
        cache.put("KCN5", "PriceSentinel", base_state, result)

        got = cache.get("KCN5", "PriceSentinel", base_state)
        assert got == result


# --- Disabled Cache Tests ---

class TestDisabledCache:

    def test_disabled_get_returns_none(self, disabled_config, base_state):
        """When disabled, get() always returns None."""
        cache = SemanticCache(disabled_config)
        cache._cache["KCN5|PriceSentinel|TRENDING_UP"] = [
            ([0.1, 0.2, 0.3, 0.1], {'direction': 'BULLISH'}, datetime.now(timezone.utc))
        ]
        assert cache.get("KCN5", "PriceSentinel", base_state) is None

    def test_disabled_put_is_noop(self, disabled_config, base_state):
        """When disabled, put() does not store anything."""
        cache = SemanticCache(disabled_config)
        cache.put("KCN5", "PriceSentinel", base_state, {'direction': 'BULLISH'})
        assert len(cache._cache) == 0


# --- Singleton Tests ---

class TestSingleton:

    def test_two_calls_return_same_instance(self, config):
        """get_semantic_cache() should return the same object."""
        c1 = get_semantic_cache(config)
        c2 = get_semantic_cache()
        assert c1 is c2

    def test_first_call_without_config_raises(self):
        """First call without config should raise ValueError."""
        with pytest.raises(ValueError, match="must provide config"):
            get_semantic_cache()

    def test_singleton_is_proper_type(self, config):
        """Singleton should be a SemanticCache instance."""
        c = get_semantic_cache(config)
        assert isinstance(c, SemanticCache)


# --- Stats Tests ---

class TestStats:

    def test_stats_accuracy(self, cache, base_state):
        """Stats should accurately count hits, misses, invalidations."""
        result = {'direction': 'BULLISH'}
        cache.put("KCN5", "PriceSentinel", base_state, result)

        # Miss (different trigger)
        cache.get("KCN5", "WeatherSentinel", base_state)
        # Hit
        cache.get("KCN5", "PriceSentinel", base_state)
        # Another hit
        cache.get("KCN5", "PriceSentinel", base_state)
        # Invalidate
        cache.invalidate(contract="KCN5")

        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['invalidations'] == 1
        assert stats['hit_rate'] == pytest.approx(2 / 3)
        assert stats['enabled'] is True

    def test_stats_partitions_count(self, cache, base_state):
        """Stats should report number of distinct partitions."""
        cache.put("KCN5", "PriceSentinel", base_state, {'d': 'BULL'})
        cache.put("KCN5", "WeatherSentinel", base_state, {'d': 'BEAR'})

        stats = cache.get_stats()
        assert stats['partitions'] == 2

    def test_stats_entries_count(self, cache, base_state):
        """Stats entries count should reflect total cached items."""
        cache.put("KCN5", "PriceSentinel", base_state, {'d': 'BULL'})

        # Same partition, different vector (different price)
        state2 = dict(base_state)
        state2['price_vs_sma'] = 0.10
        cache.put("KCN5", "PriceSentinel", state2, {'d': 'BEAR'})

        stats = cache.get_stats()
        assert stats['entries'] == 2

    def test_empty_stats(self, cache):
        """Empty cache should have zero stats."""
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        assert stats['entries'] == 0
        assert stats['partitions'] == 0


# --- Invalidate All Tests ---

class TestInvalidateAll:

    def test_invalidate_all_clears_everything(self, cache, base_state):
        """invalidate() with no args clears all entries."""
        cache.put("KCN5", "PriceSentinel", base_state, {'d': 'BULL'})
        cache.put("CCN5", "WeatherSentinel", base_state, {'d': 'BEAR'})

        cache.invalidate()

        assert cache.get("KCN5", "PriceSentinel", base_state) is None
        assert cache.get("CCN5", "WeatherSentinel", base_state) is None
        assert cache.get_stats()['entries'] == 0


# --- Max Entries Eviction ---

class TestEviction:

    def test_max_entries_eviction(self, base_state):
        """Cache should evict oldest entries when max_entries is exceeded."""
        config = {
            'semantic_cache': {
                'enabled': True,
                'similarity_threshold': 0.92,
                'ttl_minutes': 60,
                'max_entries': 2,
            }
        }
        cache = SemanticCache(config)

        # Put 3 entries in the same partition (different states)
        for i in range(3):
            state = dict(base_state)
            state['price_vs_sma'] = 0.01 * (i + 1)
            cache.put("KCN5", "PriceSentinel", state, {'idx': i})

        # Should have at most 2 entries in the partition
        key = cache._partition_key("KCN5", "PriceSentinel", base_state)
        assert len(cache._cache[key]) <= 2
