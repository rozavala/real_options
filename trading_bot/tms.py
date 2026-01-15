import chromadb
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

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
            doc_id = f"{agent}_{datetime.now().isoformat()}"
            self.collection.add(
                documents=[insight],
                metadatas=[{
                    "agent": agent,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }],
                ids=[doc_id]
            )
            logger.info(f"TMS: Stored insight from {agent}")
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
