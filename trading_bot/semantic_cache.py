"""
Semantic Cache — Avoids redundant council runs when market state is unchanged.

Roadmap Item C.1. Expected savings: $0.60–$0.90/day.

Uses ONLY keys that ACTUALLY EXIST in market_data_provider.py.
If new keys are added to build_market_context(), update _vectorize_market_state().

Partition key: contract|trigger_source|regime (exact match)
Vector dimensions: 4 (all scaled to ~[0, 1])
  [0]   price_vs_sma   — normalized momentum, scaled by 0.15
  [1]   volatility_5d   — realized volatility, scaled by 0.06
  [2]   sentiment_score — from agent aggregation (if available)
  [3]   alert_density   — normalized sentinel alert count
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class SemanticCache:
    """Cache council decisions based on market state similarity.

    Uses a two-layer lookup:
    1. Partition key (contract|trigger_source|regime) — exact match
    2. Cosine similarity on scaled numeric vector — fuzzy match
    """

    def __init__(self, config: dict):
        cache_config = config.get('semantic_cache', {})
        self.enabled = cache_config.get('enabled', False)
        self.similarity_threshold = cache_config.get('similarity_threshold', 0.92)
        self.ttl_minutes = cache_config.get('ttl_minutes', 60)
        self.max_entries = cache_config.get('max_entries', 100)

        # In-memory cache: {partition_key: [(vector, result, timestamp), ...]}
        self._cache: Dict[str, list] = {}
        self._stats = {'hits': 0, 'misses': 0, 'invalidations': 0}

    def _partition_key(self, contract: str, trigger_source: str, state: dict) -> str:
        """Build partition key from contract, trigger source, and regime.

        Different regimes, trigger sources, or contracts never cross-hit.
        """
        regime = state.get('regime', 'UNKNOWN')
        return f"{contract}|{trigger_source}|{regime}"

    def get(self, contract: str, trigger_source: str, market_state: dict) -> Optional[dict]:
        """Check cache for a similar market state.

        Returns cached council result if found, None otherwise.
        """
        if not self.enabled:
            return None

        key = self._partition_key(contract, trigger_source, market_state)
        vector = self._vectorize_market_state(market_state)
        entries = self._cache.get(key, [])
        now = datetime.now(timezone.utc)

        for cached_vector, cached_result, cached_time in entries:
            # TTL check
            age = (now - cached_time).total_seconds() / 60
            if age > self.ttl_minutes:
                continue

            # Similarity check
            similarity = self._cosine_similarity(vector, cached_vector)
            if similarity >= self.similarity_threshold:
                self._stats['hits'] += 1
                logger.info(
                    f"CACHE HIT ({key}): similarity={similarity:.4f}, "
                    f"age={age:.0f}min"
                )
                return cached_result

        self._stats['misses'] += 1
        return None

    def put(self, contract: str, trigger_source: str, market_state: dict, result: dict):
        """Store a council result for this market state."""
        if not self.enabled:
            return

        key = self._partition_key(contract, trigger_source, market_state)
        vector = self._vectorize_market_state(market_state)
        now = datetime.now(timezone.utc)

        if key not in self._cache:
            self._cache[key] = []

        self._cache[key].append((vector, result, now))

        # Evict old entries
        self._cache[key] = [
            (v, r, t) for v, r, t in self._cache[key]
            if (now - t).total_seconds() / 60 <= self.ttl_minutes
        ][-self.max_entries:]

    def invalidate(self, contract: str = None):
        """Invalidate cache entries.

        If contract is given, removes all partitions whose key starts with that
        contract name. If None, clears everything.
        """
        if contract:
            keys_to_remove = [k for k in self._cache if k.startswith(contract)]
            removed = sum(len(self._cache.pop(k, [])) for k in keys_to_remove)
        else:
            removed = sum(len(v) for v in self._cache.values())
            self._cache.clear()

        self._stats['invalidations'] += 1
        logger.info(f"CACHE INVALIDATED: {removed} entries removed (contract={contract or 'ALL'})")

    def invalidate_by_ticker(self, ticker: str):
        """Invalidate all cache entries for a given ticker.

        Clears all partitions where the contract starts with the ticker.
        E.g. invalidate_by_ticker("KC") clears "KC...|...|..." but not "CC...|...|...".
        """
        keys_to_remove = [k for k in self._cache if k.split('|')[0].startswith(ticker)]
        removed = sum(len(self._cache.pop(k, [])) for k in keys_to_remove)
        if removed > 0:
            self._stats['invalidations'] += 1
            logger.info(f"CACHE INVALIDATED BY TICKER ({ticker}): {removed} entries removed")

    def _vectorize_market_state(self, state: dict) -> List[float]:
        """Convert market state to numerical vector for similarity matching.

        4 dimensions, all scaled to ~[0, 1] range. Regime is NOT in the vector —
        it's a hard partition key (exact match) to prevent the cosine dominance
        bug where regime one-hot (4 binary dims) overwhelmed small numeric changes.

        IMPORTANT: Uses ONLY keys from market_data_provider.py::build_market_context().
        If you add keys to build_market_context(), update this method.
        """
        # Price momentum: price_vs_sma typically in [-0.15, +0.15]
        price_vs_sma = state.get('price_vs_sma') or 0.0
        price_vs_sma_scaled = price_vs_sma / 0.15  # 15% move = 1.0

        # Volatility: typically 0.01–0.06
        vol_5d = state.get('volatility_5d') or 0.02
        vol_5d_scaled = vol_5d / 0.06  # 6% daily vol = 1.0

        # Sentiment: already in [-1, 1]
        sentiment = state.get('sentiment_score') or 0.0

        # Alert density: count normalized to [0, 1]
        alert_count = state.get('recent_alert_count') or 0
        alerts_scaled = min(alert_count / 10.0, 1.0)

        return [price_vs_sma_scaled, vol_5d_scaled, sentiment, alerts_scaled]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def get_stats(self) -> dict:
        """Cache performance statistics for dashboard."""
        total = self._stats['hits'] + self._stats['misses']
        return {
            'enabled': self.enabled,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': self._stats['hits'] / total if total > 0 else 0.0,
            'invalidations': self._stats['invalidations'],
            'entries': sum(len(v) for v in self._cache.values()),
            'partitions': len(self._cache),
        }


# --- Singleton factory ---

_cache_instance: Optional['SemanticCache'] = None


def get_semantic_cache(config: dict = None) -> 'SemanticCache':
    """Get or create the singleton SemanticCache instance.

    First call must provide config. Subsequent calls can omit it.
    """
    global _cache_instance
    if _cache_instance is None:
        if config is None:
            raise ValueError("First call to get_semantic_cache() must provide config")
        _cache_instance = SemanticCache(config)
    return _cache_instance
