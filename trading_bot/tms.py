"""
Enhanced Transactive Memory System with Temporal Filtering.

CHANGELOG:
- Added valid_from metadata to all documents
- Added simulation_time parameter to retrieve()
- Added temporal filtering to prevent look-ahead bias
- Preserved all existing CROSS_CUE_RULES and functionality
"""

import chromadb
from datetime import datetime, timezone, timedelta
import logging
import os
import json
from typing import List, Optional
import math

logger = logging.getLogger(__name__)

# EXISTING: Preserved exactly as-is
CROSS_CUE_RULES = {
    'meteorologist': {
        'drought': ['fundamentalist', 'volatility'],
        'frost': ['fundamentalist', 'volatility', 'technical'],
        'rain': ['fundamentalist'],
    },
    'macro': {
        'inflation': ['sentiment', 'technical'],
        'brl': ['fundamentalist', 'geopolitical'],
        'fed': ['volatility', 'sentiment'],
    },
    'logistics': {
        'strike': ['fundamentalist', 'geopolitical'],
        'port': ['fundamentalist'],
        'suez': ['volatility'],
    },
    'technical': {
        'breakout': ['sentiment', 'volatility'],
        'oversold': ['fundamentalist'],
    }
}


# Default TMS path — overridden by set_data_dir() for multi-commodity
_default_tms_path = os.path.join("./data", os.environ.get("COMMODITY_TICKER", "KC"), "tms")


def set_data_dir(data_dir: str):
    """Configure default TMS persist path for a commodity-specific data directory."""
    global _default_tms_path
    _default_tms_path = os.path.join(data_dir, "tms")
    logger.info(f"TMS default path set to: {_default_tms_path}")


def _get_default_tms_path() -> str:
    """Resolve TMS path via ContextVar (multi-engine) or module global (legacy)."""
    try:
        from trading_bot.data_dir_context import get_engine_data_dir
        return os.path.join(get_engine_data_dir(), "tms")
    except LookupError:
        return _default_tms_path


class TransactiveMemory:
    """
    Shared memory system for cross-agent knowledge retrieval using Vector DB.

    ENHANCED: Now supports temporal filtering for backtest integrity.
    """

    def __init__(self, persist_path: str = None):
        persist_path = persist_path or _get_default_tms_path()
        """Initialize TMS with ChromaDB backend."""
        os.makedirs(os.path.dirname(persist_path) if os.path.dirname(persist_path) else '.', exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection(
                name="agent_insights",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"TMS initialized at {persist_path}")
        except Exception as e:
            logger.error(f"Failed to initialize TMS: {e}")
            self.collection = None

        # NEW: Simulation clock (None = live mode, datetime = backtest mode)
        self._simulation_time: Optional[datetime] = None

    # =========================================================================
    # NEW: Simulation Clock Protocol
    # =========================================================================

    def set_simulation_time(self, sim_time: Optional[datetime]) -> None:
        """
        Set the simulation clock for backtesting.

        Args:
            sim_time: datetime for backtest mode, None for live mode
        """
        self._simulation_time = sim_time
        if sim_time:
            logger.info(f"TMS: Simulation time set to {sim_time.isoformat()}")
        else:
            logger.info("TMS: Simulation time cleared (live mode)")

    def get_current_time(self) -> datetime:
        """
        Get the current time respecting simulation clock.

        Returns:
            Simulation time if set, otherwise current UTC time
        """
        if self._simulation_time:
            return self._simulation_time
        return datetime.now(timezone.utc)

    def is_backtest_mode(self) -> bool:
        """Check if TMS is in backtest mode."""
        return self._simulation_time is not None

    def _compute_decay_factor(
        self,
        document_metadata: dict,
        query_time: datetime,
        decay_rates: dict = None
    ) -> float:
        """
        Compute temporal relevance decay factor for a document.

        Returns a multiplier in (0.0, 1.0] where 1.0 = brand new,
        approaching 0.0 = very stale.

        Formula: decay = exp(-lambda × age_days)

        Args:
            document_metadata: ChromaDB document metadata dict
            query_time: The reference time to compute age against
            decay_rates: Dict mapping document types to lambda values

        Returns:
            Float decay factor, floored at 0.01 (never fully zero)
        """
        if decay_rates is None:
            decay_rates = {'default': 0.05}

        # Get document timestamp (prefer valid_from, fall back to timestamp)
        valid_from_str = document_metadata.get('valid_from')
        if valid_from_str:
            try:
                if isinstance(valid_from_str, str):
                    doc_time = datetime.fromisoformat(
                        valid_from_str.replace('Z', '+00:00')
                    )
                else:
                    doc_time = valid_from_str
            except (ValueError, TypeError):
                ts_str = document_metadata.get('timestamp')
                if ts_str:
                    try:
                        doc_time = datetime.fromisoformat(
                            ts_str.replace('Z', '+00:00')
                        )
                    except (ValueError, TypeError):
                        return 1.0  # Can't compute age, assume fresh
                else:
                    return 1.0
        else:
            ts_str = document_metadata.get('timestamp')
            if ts_str:
                try:
                    doc_time = datetime.fromisoformat(
                        ts_str.replace('Z', '+00:00')
                    )
                except (ValueError, TypeError):
                    return 1.0
            else:
                return 1.0

        # Ensure timezone-aware comparison
        if doc_time.tzinfo is None:
            doc_time = doc_time.replace(tzinfo=timezone.utc)
        if query_time.tzinfo is None:
            query_time = query_time.replace(tzinfo=timezone.utc)

        # Calculate age in days
        age_days = max(0, (query_time - doc_time).total_seconds() / 86400.0)

        # Determine document type for decay rate lookup
        # Priority: explicit 'type' metadata → 'agent' name → substring match → default
        doc_type = document_metadata.get('type', '').lower()
        agent = document_metadata.get('agent', '').lower()

        lambda_val = decay_rates.get(doc_type, None)
        if lambda_val is None:
            lambda_val = decay_rates.get(agent, None)
        if lambda_val is None:
            # Check if agent name contains a known type keyword
            for key in decay_rates:
                if key in agent or key in doc_type:
                    lambda_val = decay_rates[key]
                    break
        if lambda_val is None:
            lambda_val = decay_rates.get('default', 0.05)

        # Exponential decay
        decay = math.exp(-lambda_val * age_days)

        return max(0.01, decay)  # Floor at 1% — never fully zero


    # =========================================================================
    # ENHANCED: Encode with valid_from timestamp
    # =========================================================================

    def encode(
        self,
        agent: str,
        insight: str,
        metadata: dict = None,
        valid_from: Optional[datetime] = None
    ) -> None:
        """
        Store an agent's insight with metadata.

        ENHANCED: Now includes valid_from timestamp for temporal filtering.

        Args:
            agent: Agent name (e.g., 'agronomist', 'macro')
            insight: The insight text to store
            metadata: Additional metadata dict
            valid_from: When this insight became valid (default: now)
        """
        if not self.collection:
            return

        try:
            current_time = self.get_current_time()

            # Use provided valid_from or current time
            effective_valid_from = valid_from or current_time

            # Generate unique document ID
            doc_id = f"{agent}_{current_time.isoformat()}_{hash(insight) % 10000}"

            # Build metadata with temporal info
            doc_metadata = {
                "agent": agent,
                "timestamp": current_time.isoformat(),
                # NEW: Temporal validity fields
                "valid_from": effective_valid_from.isoformat(),
                "valid_from_ts": effective_valid_from.timestamp(),  # Numeric for filtering
                **(metadata or {})
            }

            self.collection.add(
                documents=[insight],
                metadatas=[doc_metadata],
                ids=[doc_id]
            )
            logger.debug(f"TMS: Stored insight from {agent} (valid_from: {effective_valid_from.isoformat()})")

            # EXISTING: Check for Cross-Cues (preserved)
            cues = self.get_cross_cue_agents(agent, insight)
            if cues:
                logger.info(f"TMS: {agent} insight cues -> {cues}")

        except Exception as e:
            logger.error(f"TMS Encode failed: {e}")

    # =========================================================================
    # ENHANCED: Retrieve with temporal filtering
    # =========================================================================

    def retrieve(
        self,
        query: str,
        agent_filter: str = None,
        n_results: int = 5,
        max_age_days: int = 30,
        simulation_time: Optional[datetime] = None,
        decay_rates: dict = None
    ) -> list:
        """
        Retrieve relevant insights with temporal filtering.

        ENHANCED: Now respects simulation_time to prevent look-ahead bias.

        Args:
            query: Search query for semantic similarity
            agent_filter: Optional agent name to filter results
            n_results: Maximum number of results to return
            max_age_days: Only return insights from the last N days
            simulation_time: Override simulation clock for this query
            decay_rates: Dict of decay rates per doc type

        Returns:
            List of insight strings matching the query
        """
        if not self.collection:
            return []

        try:
            # Determine the "now" for this query
            effective_time = simulation_time or self._simulation_time or datetime.now(timezone.utc)

            # Calculate the cutoff time for max_age
            if max_age_days and max_age_days > 0:
                cutoff_time = effective_time - timedelta(days=max_age_days)
            else:
                cutoff_time = None

            # Build where filter
            # NOTE: ChromaDB filtering has limitations, so we do temporal
            # filtering in Python after retrieval for robustness
            where_filter = {}
            if agent_filter:
                where_filter["agent"] = agent_filter

            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 3,  # Over-fetch for post-filtering
                where=where_filter if where_filter else None
            )

            if not results or not results['documents'] or not results['documents'][0]:
                return []

            # Post-filter for temporal validity + inline decay scoring
            use_decay = decay_rates is not None
            scored_insights = []  # List of (decay_score, doc) tuples

            for rank, (doc, meta) in enumerate(
                zip(results['documents'][0], results['metadatas'][0])
            ):
                # Handle None metadata from ChromaDB
                if meta is None:
                    meta = {}

                # Parse valid_from timestamp
                valid_from_str = meta.get('valid_from')
                if valid_from_str:
                    try:
                        valid_from = datetime.fromisoformat(
                            valid_from_str.replace('Z', '+00:00')
                        )
                    except (ValueError, TypeError):
                        valid_from_ts = meta.get('valid_from_ts')
                        if valid_from_ts:
                            valid_from = datetime.fromtimestamp(
                                valid_from_ts, tz=timezone.utc
                            )
                        else:
                            if self.is_backtest_mode():
                                logger.warning(
                                    "TMS: Skipping legacy document without "
                                    "valid_from in backtest mode"
                                )
                                continue
                            valid_from = None
                else:
                    if self.is_backtest_mode():
                        continue
                    valid_from = None

                # CRITICAL: Temporal filtering (future documents)
                if valid_from is not None and valid_from > effective_time:
                    logger.debug(
                        f"TMS: Filtered out future document "
                        f"(valid_from: {valid_from}, sim_time: {effective_time})"
                    )
                    continue

                # Age filtering (hard cutoff — preserved from original)
                if (valid_from is not None and cutoff_time
                        and valid_from < cutoff_time):
                    continue

                # === DECAY SCORING (computed inline where metadata is accessible) ===
                if use_decay:
                    # ChromaDB returns results by semantic similarity rank.
                    # Base score: gentle rank penalty (rank 0 = 1.0, rank 5 = 0.67)
                    base_similarity = 1.0 / (1.0 + rank * 0.1)
                    decay_factor = self._compute_decay_factor(
                        meta, effective_time, decay_rates
                    )
                    combined_score = base_similarity * decay_factor
                    scored_insights.append((combined_score, doc))
                else:
                    # No decay — preserve ChromaDB's original ranking
                    scored_insights.append((1.0, doc))

                # Over-fetch limit: collect up to 3x n_results for re-ranking
                if len(scored_insights) >= n_results * 3:
                    break

            # === RE-RANK BY DECAY-ADJUSTED SCORE ===
            if use_decay and len(scored_insights) > 1:
                scored_insights.sort(key=lambda x: x[0], reverse=True)
                logger.debug(
                    f"TMS: Decay re-ranked {len(scored_insights)} results. "
                    f"Top score: {scored_insights[0][0]:.3f}, "
                    f"Bottom: {scored_insights[-1][0]:.3f}"
                )

            # Extract doc strings, trim to n_results
            filtered_insights = [doc for _, doc in scored_insights[:n_results]]

            return filtered_insights

        except Exception as e:
            logger.error(f"TMS Retrieve failed: {e}")
            return []

    # =========================================================================
    # NEW: Migration Support
    # =========================================================================

    def backfill_valid_from(self) -> int:
        """
        Backfill valid_from on documents that lack it.

        IMPORTANT: Run this ONCE before enabling backtest mode.
        Uses existing 'timestamp' field as valid_from.

        Returns:
            Number of documents migrated
        """
        if not self.collection:
            return 0

        all_docs = self.collection.get(include=["metadatas"])
        if not all_docs or not all_docs['ids']:
            return 0

        migrated = 0
        for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']):
            # FIX (Final Review): Handle None metadata from ChromaDB
            if metadata is None:
                metadata = {}

            if metadata.get('valid_from'):
                continue  # Already has valid_from

            # Use timestamp as valid_from
            ts_str = metadata.get('timestamp')
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
            else:
                ts = datetime(2020, 1, 1, tzinfo=timezone.utc)

            # Update metadata
            new_metadata = {
                **metadata,
                'valid_from': ts.isoformat(),
                'valid_from_ts': ts.timestamp()
            }
            self.collection.update(ids=[doc_id], metadatas=[new_metadata])
            migrated += 1

        logger.info(f"TMS: Backfilled valid_from on {migrated} documents")
        return migrated

    # =========================================================================
    # EXISTING: Preserved methods
    # =========================================================================

    def get_cross_cue_agents(self, source_agent: str, insight: str) -> list:
        """
        Determine which agents should be notified based on insight content.

        PRESERVED: Existing functionality unchanged.
        """
        insight_lower = insight.lower()
        cued_agents = set()

        # Handle role names if passed as enum
        simple_role = str(source_agent).split('.')[-1].lower().replace('_sentinel', '').replace('_analyst', '')

        rules = CROSS_CUE_RULES.get(simple_role, {})
        for keyword, target_agents in rules.items():
            if keyword in insight_lower:
                cued_agents.update(target_agents)

        return list(cued_agents)

    def get_collection_stats(self) -> dict:
        """Get statistics about the TMS collection."""
        if not self.collection:
            return {"status": "unavailable"}

        try:
            count = self.collection.count()
            return {
                "status": "active",
                "document_count": count,
                "backtest_mode": self.is_backtest_mode(),
                "simulation_time": self._simulation_time.isoformat() if self._simulation_time else None
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # Preserved methods from original file (record_trade_thesis, etc) need to be kept?
    # The instructions say "Modify the existing file". I just overwrote it with the code from the guide.
    # The guide's code in 4.3.1 says "PRESERVED: Existing functionality unchanged." and lists `get_cross_cue_agents`.
    # But checking the original file, there were `record_trade_thesis`, `retrieve_thesis`, `get_active_theses_by_guardian`, `invalidate_thesis`.
    # The guide's code block ends with `__all__ = ['TransactiveMemory', 'CROSS_CUE_RULES']`.
    # I should verify if I dropped those thesis methods.
    # Re-reading the guide: "Note: We are MODIFYING the existing file, not replacing it."
    # The guide shows `EXISTING: Preserved methods` section but only explicitly lists `get_cross_cue_agents` and `get_collection_stats` (which wasn't in original).
    # I should probably restore the thesis methods to be safe, as they seem important for the system.

    def record_trade_thesis(self, trade_id: str, thesis_data: dict):
        """
        Records the entry thesis for a trade, enabling continuous re-evaluation.
        """
        if not self.collection:
            return None

        try:
            doc_id = f"thesis_{trade_id}"

            self.collection.add(
                documents=[json.dumps(thesis_data)],
                metadatas=[{
                    "type": "entry_thesis",
                    "trade_id": trade_id,
                    "strategy_type": thesis_data.get('strategy_type', 'UNKNOWN'),
                    "guardian_agent": thesis_data.get('guardian_agent', 'UNKNOWN'),
                    "entry_timestamp": thesis_data.get('entry_timestamp', datetime.now(timezone.utc).isoformat()),
                    "active": "true"
                }],
                ids=[doc_id]
            )
            logger.info(f"TMS: Recorded entry thesis for trade {trade_id}")
            return doc_id
        except Exception as e:
            logger.error(f"TMS record_trade_thesis failed: {e}")
            return None

    def retrieve_thesis(self, trade_id: str) -> dict | None:
        """Retrieves the entry thesis for a specific trade."""
        if not self.collection:
            return None

        try:
            results = self.collection.get(
                ids=[f"thesis_{trade_id}"],
                include=['documents', 'metadatas']
            )
            if results and results['documents']:
                data = json.loads(results['documents'][0])
                return data if isinstance(data, dict) else None
            return None
        except Exception as e:
            logger.error(f"TMS retrieve_thesis failed: {e}")
            return None

    def get_active_theses_by_guardian(self, guardian_agent: str) -> list:
        """
        Retrieves all active trade theses owned by a specific agent.
        """
        if not self.collection:
            return []

        try:
            results = self.collection.get(
                where={"$and": [
                    {"guardian_agent": guardian_agent},
                    {"active": "true"}
                ]},
                include=['documents', 'metadatas']
            )
            docs = [json.loads(doc) for doc in results.get('documents', [])]
            return [d for d in docs if isinstance(d, dict)]
        except Exception as e:
            logger.error(f"TMS get_active_theses failed: {e}")
            return []

    def invalidate_thesis(self, trade_id: str, reason: str):
        """Marks a thesis as invalidated (position closed)."""
        if not self.collection:
            return

        try:
            results = self.collection.get(ids=[f"thesis_{trade_id}"])
            if results and results['metadatas']:
                current_meta = results['metadatas'][0]
                current_meta['active'] = "false"
                current_meta['invalidation_reason'] = reason

                self.collection.update(
                    ids=[f"thesis_{trade_id}"],
                    metadatas=[current_meta]
                )
                logger.info(f"TMS: Invalidated thesis for {trade_id}: {reason}")
        except Exception as e:
            logger.error(f"TMS invalidate_thesis failed: {e}")

    def get_all_theses(self) -> list:
        """Returns all theses (active + invalidated) from the TMS."""
        if not self.collection:
            return []

        try:
            # Query all documents with type="entry_thesis"
            results = self.collection.get(
                where={"type": "entry_thesis"},
                include=['metadatas', 'documents']
            )

            theses = []
            if results and results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    try:
                        doc_content = results['documents'][i]
                        if not doc_content:
                            continue

                        thesis_data = json.loads(doc_content)
                        if not isinstance(thesis_data, dict):
                            continue
                        # Ensure trade_id is present
                        metadata = results['metadatas'][i]
                        if 'trade_id' not in thesis_data and metadata:
                            thesis_data['trade_id'] = metadata.get('trade_id')

                        # Add metadata flags to the returned object if they exist (like migrated_v6_5_1)
                        # because they might be in metadata but not in the document yet (if written by legacy code)
                        # However, we prefer the document as source of truth.
                        # Migration script reads from thesis dict.

                        theses.append(thesis_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse thesis {doc_id}: {e}")
            return theses
        except Exception as e:
            logger.error(f"TMS get_all_theses failed: {e}")
            return []

    def update_thesis_supporting_data(self, thesis_id: str, new_supporting_data: dict):
        """
        Updates the supporting_data field for a specific thesis.

        Args:
            thesis_id: The trade_id of the thesis (e.g., UUID)
            new_supporting_data: The dictionary to replace or update supporting_data
        """
        if not self.collection:
            return

        doc_id = f"thesis_{thesis_id}"

        try:
            # 1. Get current doc
            results = self.collection.get(ids=[doc_id], include=['documents', 'metadatas'])
            if not results or not results['documents']:
                logger.error(f"TMS update failed: Thesis {thesis_id} not found")
                return

            # 2. Parse
            current_doc_str = results['documents'][0]
            current_doc = json.loads(current_doc_str)
            if not isinstance(current_doc, dict):
                logger.error(f"TMS update failed: Thesis {thesis_id} doc is not a dict")
                return

            # 3. Update field
            current_doc['supporting_data'] = new_supporting_data

            # 4. Save back - preserve existing metadata
            current_metadata = results['metadatas'][0]

            self.collection.update(
                ids=[doc_id],
                documents=[json.dumps(current_doc)],
                metadatas=[current_metadata]
            )
            logger.info(f"TMS: Updated supporting_data for thesis {thesis_id}")

        except Exception as e:
            logger.error(f"TMS update_thesis_supporting_data failed: {e}")


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Ensure existing code that imports TransactiveMemory continues to work
__all__ = ['TransactiveMemory', 'CROSS_CUE_RULES']
