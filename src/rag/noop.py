"""
No-op RAG retriever implementation.

Used when RAG is bypassed (RAG_MODE=none).
Returns empty results for all queries.
"""

from typing import List
from rag.base import BaseRetriever, Memory, RetrievalResult


class NoOpRetriever(BaseRetriever):
    """
    No-operation retriever that always returns empty results.
    
    Used when RAG is disabled or bypassed. This allows the system
    to function without any memory retrieval while maintaining
    the same interface.
    """
    
    @property
    def retriever_type(self) -> str:
        """Return the retriever type."""
        return "none"
    
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> RetrievalResult:
        """
        Return empty retrieval results.
        
        Args:
            query: The user's query (ignored)
            limit: Maximum memories (ignored)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Empty RetrievalResult
        """
        return RetrievalResult(
            memories=[],
            retriever_type=self.retriever_type,
            query=query,
            metadata={"bypassed": True}
        )
    
    def add_memory(self, memory: Memory) -> None:
        """
        No-op memory addition.
        
        Args:
            memory: Memory to add (ignored)
        """
        pass
