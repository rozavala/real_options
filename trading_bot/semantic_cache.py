"""
Semantic Cache — Reduce redundant LLM calls via embedding similarity.

Uses ChromaDB (already deployed for TMS) with a SEPARATE persistent directory.
Cache TTLs are role-dependent: sentinels (15min), analysts (2hr), master (NEVER cached).

IMPORTANT: Uses ./data/semantic_cache — NOT the TMS path (./data/tms).
"""

import logging
import time
import hashlib
import os
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# Role-based TTLs in seconds (0 = never cache)
CACHE_TTLS = {
    # Sentinels — short TTL
    "weather_sentinel": 900, "logistics_sentinel": 900,
    "news_sentinel": 900, "price_sentinel": 300,
    "microstructure_sentinel": 300,
    # Tier 2 Analysts — medium TTL
    "agronomist": 7200, "macro": 7200, "geopolitical": 7200,
    "sentiment": 3600, "technical": 3600, "volatility": 3600,
    "inventory": 7200, "supply_chain": 7200,
    # Tier 3 Decision Makers — NEVER cache
    "permabear": 0, "permabull": 0, "master": 0, "compliance": 0,
}

# Approximate cost per LLM call by role (for savings tracking)
COST_PER_CALL = {
    "agronomist": 0.08, "macro": 0.08, "geopolitical": 0.08,
    "supply_chain": 0.08, "inventory": 0.08,
    "sentiment": 0.05, "technical": 0.05, "volatility": 0.05,
    "weather_sentinel": 0.02, "news_sentinel": 0.02,
    "logistics_sentinel": 0.02, "price_sentinel": 0.01,
}

# CRITICAL: Separate from TMS path (./data/tms)
CACHE_PERSIST_PATH = "./data/semantic_cache"


class SemanticCache:
    """Embedding-based LLM response cache using ChromaDB."""

    def __init__(self, cache_dir: str = CACHE_PERSIST_PATH):
        self._hash_cache: Dict[str, Tuple[str, float, str]] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._savings_usd = 0.0
        self._use_embeddings = False

        try:
            import chromadb
            os.makedirs(cache_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=cache_dir)
            self._collection = self._client.get_or_create_collection(
                name="semantic_cache",
                metadata={"hnsw:space": "cosine"}
            )
            self._use_embeddings = True
            logger.info(f"SemanticCache initialized with ChromaDB at {cache_dir}")
        except Exception as e:
            logger.warning(f"ChromaDB unavailable for semantic cache, using hash-only: {e}")

    def get(self, prompt: str, role: str) -> Optional[str]:
        """Check cache. Returns cached response or None."""
        ttl = CACHE_TTLS.get(role, 0)
        if ttl == 0:
            return None

        now = time.time()
        prompt_hash = self._hash_prompt(prompt, role)

        # Fast path: exact hash match
        if prompt_hash in self._hash_cache:
            response, timestamp, cached_role = self._hash_cache[prompt_hash]
            if now - timestamp < ttl:
                self._hit_count += 1
                self._savings_usd += COST_PER_CALL.get(role, 0.05)
                logger.debug(f"Cache HIT (exact) for {role}")
                return response
            else:
                del self._hash_cache[prompt_hash]

        # Slow path: embedding similarity
        if self._use_embeddings:
            try:
                results = self._collection.query(
                    query_texts=[prompt], n_results=1, where={"role": role}
                )
                if results and results['distances'] and results['distances'][0]:
                    distance = results['distances'][0][0]
                    similarity = 1.0 - distance
                    if similarity > 0.92:
                        metadata = results['metadatas'][0][0]
                        cached_time = metadata.get('timestamp', 0)
                        if now - cached_time < ttl:
                            cached_response = metadata.get('response', '')
                            if cached_response:
                                self._hit_count += 1
                                self._savings_usd += COST_PER_CALL.get(role, 0.05)
                                logger.debug(f"Cache HIT (semantic, sim={similarity:.3f}) for {role}")
                                return cached_response
            except Exception as e:
                logger.debug(f"Semantic cache query failed: {e}")

        self._miss_count += 1
        return None

    def set(self, prompt: str, role: str, response: str):
        """Store a prompt→response pair."""
        ttl = CACHE_TTLS.get(role, 0)
        if ttl == 0:
            return

        now = time.time()
        prompt_hash = self._hash_prompt(prompt, role)
        self._hash_cache[prompt_hash] = (response, now, role)

        if self._use_embeddings:
            try:
                self._collection.upsert(
                    ids=[prompt_hash],
                    documents=[prompt],
                    metadatas=[{
                        "role": role,
                        "timestamp": now,
                        "response": response[:5000],
                    }]
                )
            except Exception as e:
                logger.debug(f"Failed to store in embedding cache: {e}")

    def invalidate_role(self, role: str):
        """Invalidate all entries for a specific role."""
        to_remove = [k for k, (_, _, r) in self._hash_cache.items() if r == role]
        for k in to_remove:
            del self._hash_cache[k]
        logger.info(f"Invalidated {len(to_remove)} cache entries for {role}")

    def invalidate_all_analysts(self):
        """Invalidate all analyst caches (e.g., after >2% price move)."""
        analyst_roles = [r for r, ttl in CACHE_TTLS.items()
                        if ttl > 0 and "sentinel" not in r]
        count = 0
        for role in analyst_roles:
            to_remove = [k for k, (_, _, r) in self._hash_cache.items() if r == role]
            for k in to_remove:
                del self._hash_cache[k]
            count += len(to_remove)
        logger.info(f"Invalidated {count} analyst cache entries")

    def get_stats(self) -> dict:
        total = self._hit_count + self._miss_count
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": round(self._hit_count / total, 3) if total > 0 else 0.0,
            "estimated_savings_usd": round(self._savings_usd, 2),
            "entries": len(self._hash_cache),
        }

    @staticmethod
    def _hash_prompt(prompt: str, role: str) -> str:
        return hashlib.sha256(f"{role}:{prompt}".encode()).hexdigest()[:16]
