"""
Semantic Cache — Avoids redundant LLM calls when market state is unchanged.

Roadmap Item C.1. Expected savings: $0.60–$0.90/day.

Uses ONLY keys that ACTUALLY EXIST in market_data_provider.py.
If new keys are added to build_market_context(), update _vectorize_market_state().

Vector dimensions: 8
  [0]   price_vs_sma        — normalized momentum
  [1]   volatility_5d       — realized volatility
  [2:6] regime one-hot      — RANGE_BOUND, TRENDING_UP, TRENDING_DOWN, HIGH_VOLATILITY
  [6]   sentiment_score     — from agent aggregation (if available)
  [7]   alert_density       — normalized sentinel alert count (if available)
"""

import logging
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class SemanticCache:
    """Cache council decisions based on market state similarity."""

    def __init__(self, config: dict):
        cache_config = config.get('semantic_cache', {})
        self.enabled = cache_config.get('enabled', False)
        self.similarity_threshold = cache_config.get('similarity_threshold', 0.92)
        self.ttl_minutes = cache_config.get('ttl_minutes', 60)
        self.max_entries = cache_config.get('max_entries', 100)

        # In-memory cache: {contract: [(vector, result, timestamp), ...]}
        self._cache: Dict[str, list] = {}
        self._stats = {'hits': 0, 'misses': 0, 'invalidations': 0}

    def get(self, contract: str, market_state: dict) -> Optional[dict]:
        """
        Check cache for a similar market state.

        Returns cached council result if found, None otherwise.
        """
        if not self.enabled:
            return None

        vector = self._vectorize_market_state(market_state)
        entries = self._cache.get(contract, [])
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
                    f"CACHE HIT ({contract}): similarity={similarity:.4f}, "
                    f"age={age:.0f}min"
                )
                return cached_result

        self._stats['misses'] += 1
        return None

    def put(self, contract: str, market_state: dict, result: dict):
        """Store a council result for this market state."""
        if not self.enabled:
            return

        vector = self._vectorize_market_state(market_state)
        now = datetime.now(timezone.utc)

        if contract not in self._cache:
            self._cache[contract] = []

        self._cache[contract].append((vector, result, now))

        # Evict old entries
        self._cache[contract] = [
            (v, r, t) for v, r, t in self._cache[contract]
            if (now - t).total_seconds() / 60 <= self.ttl_minutes
        ][-self.max_entries:]

    def invalidate(self, contract: str = None):
        """
        Invalidate cache entries.

        Called by sentinel alerts when market conditions change rapidly.
        """
        if contract:
            removed = len(self._cache.pop(contract, []))
        else:
            removed = sum(len(v) for v in self._cache.values())
            self._cache.clear()

        self._stats['invalidations'] += 1
        logger.info(f"CACHE INVALIDATED: {removed} entries removed (contract={contract or 'ALL'})")

    def _vectorize_market_state(self, state: dict) -> List[float]:
        """
        Convert market state to numerical vector for similarity matching.

        IMPORTANT: Uses ONLY keys from market_data_provider.py::build_market_context().
        If you add keys to build_market_context(), update this method.

        Current schema (verified 2026-02-04):
            price, sma_200, regime, volatility_5d, price_vs_sma
        """
        features = []

        # [0] Price momentum (normalized: positive = above SMA)
        # price_vs_sma = (price - sma_200) / sma_200, already normalized
        price_vs_sma = state.get('price_vs_sma', 0.0)
        if price_vs_sma is None:
            price_vs_sma = 0.0
        features.append(price_vs_sma)

        # [1] Volatility (5-day realized)
        vol_5d = state.get('volatility_5d', 0.02)  # Default 2% if missing
        if vol_5d is None:
            vol_5d = 0.02
        features.append(vol_5d)

        # [2:6] Regime (one-hot encoding, 4 dimensions)
        regime = state.get('regime', 'UNKNOWN')
        regime_map = {
            'RANGE_BOUND':      [1, 0, 0, 0],
            'TRENDING_UP':      [0, 1, 0, 0],
            'TRENDING':         [0, 1, 0, 0],  # Alias
            'TRENDING_DOWN':    [0, 0, 1, 0],
            'HIGH_VOLATILITY':  [0, 0, 0, 1],
            'HIGH_VOL':         [0, 0, 0, 1],  # Alias from _detect_market_regime
            'UNKNOWN':          [0, 0, 0, 0],
        }
        features.extend(regime_map.get(regime, [0, 0, 0, 0]))

        # [6] Sentiment score (from agent aggregation, if available)
        sentiment_score = state.get('sentiment_score', 0.0)
        if sentiment_score is None:
            sentiment_score = 0.0
        features.append(sentiment_score)

        # [7] Alert density (from sentinel, if available)
        alert_count = state.get('recent_alert_count', 0)
        if alert_count is None:
            alert_count = 0
        features.append(min(alert_count / 10.0, 1.0))  # Normalize to [0, 1]

        return features  # 8 dimensions total

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
        }
