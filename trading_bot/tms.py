import chromadb
from datetime import datetime, timezone
import logging
import os
import json

logger = logging.getLogger(__name__)

# MAPPING: If Agent A sees [Keyword], Trigger Agent B
CROSS_CUE_RULES = {
    'meteorologist': {
        'drought': ['fundamentalist', 'volatility'],
        'frost': ['fundamentalist', 'volatility', 'technical'],
        'rain': ['fundamentalist'],
    },
    'macro': {
        'inflation': ['sentiment', 'technical'],
        'brl': ['fundamentalist', 'geopolitical'], # Brazilian Real
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

class TransactiveMemory:
    """Shared memory system for cross-agent knowledge retrieval using Vector DB."""

    def __init__(self, persist_path: str = "./data/tms"):
        # Ensure data directory exists
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)

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

    def encode(self, agent: str, insight: str, metadata: dict = None):
        """Store an agent's insight with metadata."""
        if not self.collection: return

        try:
            # 1. Standard storage
            doc_id = f"{agent}_{datetime.now(timezone.utc).isoformat()}"
            self.collection.add(
                documents=[insight],
                metadatas=[{
                    "agent": agent,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **(metadata or {})
                }],
                ids=[doc_id]
            )
            logger.info(f"TMS: Stored insight from {agent}")

            # 2. Check for Cross-Cues
            cues = self.get_cross_cue_agents(agent, insight)
            if cues:
                logger.info(f"TMS: {agent} insight on '{insight[:20]}...' cues -> {cues}")

        except Exception as e:
            logger.error(f"TMS Encode failed: {e}")

    def retrieve(self, query: str, agent_filter: str = None, n_results: int = 5) -> list:
        """Retrieve relevant insights, optionally filtered by agent."""
        if not self.collection: return []

        try:
            where_filter = {"agent": agent_filter} if agent_filter else None
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logger.error(f"TMS Retrieve failed: {e}")
            return []

    def get_cross_cue_agents(self, source_role: str, insight_text: str) -> list:
        """
        Analyzes an insight to see if it should trigger other agents.
        Returns a list of AgentRoles that should be alerted.
        """
        triggered_roles = set()
        insight_lower = insight_text.lower()

        # Look up the source agent's rules
        # Handle role names (e.g. 'AgentRole.METEOROLOGIST' -> 'meteorologist')
        simple_role = str(source_role).split('.')[-1].lower().replace('_sentinel', '').replace('_analyst', '')

        rules = CROSS_CUE_RULES.get(simple_role, {})

        for keyword, targets in rules.items():
            if keyword in insight_lower:
                for target in targets:
                    triggered_roles.add(target)

        return list(triggered_roles)

    def record_trade_thesis(self, trade_id: str, thesis_data: dict):
        """
        Records the entry thesis for a trade, enabling continuous re-evaluation.

        Args:
            trade_id: Unique identifier for the trade (position_id from trade ledger)
            thesis_data: Dictionary containing:
                - strategy_type: 'BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD', 'IRON_CONDOR', 'LONG_STRADDLE'
                - guardian_agent: Which specialist "owns" this thesis (e.g., 'Agronomist', 'VolatilityAnalyst')
                - primary_rationale: The core reason for the trade (e.g., 'frost_risk_minas_gerais')
                - invalidation_triggers: List of conditions that would kill the thesis
                - supporting_data: Original market context at entry
                - entry_timestamp: When the position was opened
                - entry_regime: Market regime at entry ('HIGH_VOLATILITY', 'RANGE_BOUND', 'TRENDING')
        """
        if not self.collection:
            return None

        try:
            doc_id = f"thesis_{trade_id}"
            # Ensure complex objects in metadata are stringified if needed, but ChromaDB supports basic types.
            # thesis_data contains nested dicts/lists which ChromaDB metadata DOES NOT support well.
            # So we store the full json in 'documents' and key fields in 'metadatas'.

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
                return json.loads(results['documents'][0])
            return None
        except Exception as e:
            logger.error(f"TMS retrieve_thesis failed: {e}")
            return None

    def get_active_theses_by_guardian(self, guardian_agent: str) -> list:
        """
        Retrieves all active trade theses owned by a specific agent.
        Used when a Sentinel fires to check if any existing positions are affected.
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
            return [json.loads(doc) for doc in results.get('documents', [])]
        except Exception as e:
            logger.error(f"TMS get_active_theses failed: {e}")
            return []

    def invalidate_thesis(self, trade_id: str, reason: str):
        """Marks a thesis as invalidated (position closed)."""
        if not self.collection:
            return

        try:
            # Check if it exists first
            existing = self.retrieve_thesis(trade_id)
            if not existing:
                return

            # Update metadata. Note: ChromaDB update overwrites. We need to preserve other metadata or just update specific fields?
            # ChromaDB's update method requires re-supplying the document or metadata.
            # We fetch existing metadata first.
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
