import chromadb
from datetime import datetime, timezone
import logging
import os

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
