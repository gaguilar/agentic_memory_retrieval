"""
Graph-based RAG retriever implementation.

Extends semantic retrieval with 1-hop graph traversal. Uses semantic
search to find seed memories, then follows edges in the memory graph
to surface situationally relevant memories that semantic search alone
would miss.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from rag.base import BaseRetriever, Memory as RAGMemory, RetrievalResult
from rag.semantic import SemanticRetriever
from database.repository import Repository
from database.models import Memory as DBMemory
from memory.tag_embedder import TagEmbedder

logger = logging.getLogger(__name__)


class GraphRetriever(BaseRetriever):
    """
    Retrieves memories using semantic search + 1-hop graph traversal.
    
    Workflow:
    1. Accept multiple entry queries (each with seed_types and hop_types)
    2. For each query, find seed memories via semantic similarity
    3. From seed memories, follow edges (1-hop) in the memory graph
    4. Prioritize edges whose labels align with the requested hop_types
    5. Merge, deduplicate, and return the combined result set
    """
    
    def __init__(
        self,
        repository: Repository,
        tag_embedder: TagEmbedder,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize the graph retriever.
        
        Args:
            repository: Database repository for memory and edge access
            tag_embedder: TagEmbedder instance (shares sentence-transformer model)
            similarity_threshold: Minimum cosine similarity for semantic seed retrieval
        """
        self.repository = repository
        self.tag_embedder = tag_embedder
        self.similarity_threshold = similarity_threshold
        
        # Internal semantic retriever for seed retrieval
        self.semantic_retriever = SemanticRetriever(
            repository=repository,
            tag_embedder=tag_embedder,
            similarity_threshold=similarity_threshold
        )
    
    @property
    def retriever_type(self) -> str:
        """Return the type of this retriever."""
        return "graph"
    
    def _prioritize_by_type(self, connected: List[tuple], hop_types: List[str]) -> List[tuple]:
        """
        Prioritize connected memories whose memory_type matches the
        requested hop_types. Non-matching memories are kept but ranked lower.
        
        Args:
            connected: List of (Memory, edge_label) tuples
            hop_types: List of preferred memory types (e.g., ["procedural", "episodic"])
            
        Returns:
            Reordered list of (Memory, edge_label) tuples
        """
        if not hop_types:
            return connected
        
        hop_set = set(hop_types)
        prioritized = []
        others = []
        
        for memory, label in connected:
            if memory.memory_type in hop_set:
                prioritized.append((memory, label))
            else:
                others.append((memory, label))
        
        return prioritized + others
    
    def _db_memory_to_rag_memory(
        self,
        db_mem: DBMemory,
        score: float,
        source: str = "graph_hop",
        edge_label: Optional[str] = None
    ) -> RAGMemory:
        """
        Convert a database Memory to a RAG Memory dataclass.
        
        Args:
            db_mem: Database Memory instance
            score: Relevance score
            source: How this memory was found ("semantic_seed" or "graph_hop")
            edge_label: Edge label if found via graph traversal
            
        Returns:
            RAG Memory dataclass instance
        """
        tags = db_mem.get_tags()
        metadata = {
            "db_id": db_mem.id,
            "tags": tags,
            "source": source,
            "score": score
        }
        if edge_label:
            metadata["edge_label"] = edge_label
        
        return RAGMemory(
            memory_id=str(db_mem.id),
            text=db_mem.statement,
            memory_type=db_mem.memory_type,
            timestamp=db_mem.created_at,
            polarity=0.0,
            confidence=score,
            entities=[],
            topics=tags,
            metadata=metadata
        )
    
    def retrieve(self, query: str, limit: int = 10, **kwargs) -> RetrievalResult:
        """
        Retrieve memories using semantic search + 1-hop graph traversal.
        
        Args:
            query: Fallback single query (used if 'queries' not provided)
            limit: Maximum total memories to retrieve
            **kwargs:
                user_id (int): Required. User ID for memory access.
                queries (list): Optional. List of query dicts, each with:
                    - "query": search query text
                    - "seed_types": list of preferred seed memory types
                    - "hop_types": list of preferred hop memory types
            
        Returns:
            RetrievalResult with combined semantic + graph memories
        """
        user_id = kwargs.get("user_id")
        if user_id is None:
            logger.warning("GraphRetriever.retrieve() called without user_id")
            return RetrievalResult(
                memories=[], retriever_type=self.retriever_type, query=query
            )
        
        queries = kwargs.get("queries", [{"query": query, "seed_types": [], "hop_types": []}])
        
        # Track all memories by ID to avoid duplicates
        # id -> (RAGMemory, score) where higher score wins
        all_results: Dict[int, tuple] = {}
        
        # Allocate seed slots per query
        seeds_per_query = max(2, limit // max(len(queries), 1))
        
        total_seeds = 0
        total_hops = 0
        
        for q in queries:
            q_text = q.get("query", query)
            seed_types = q.get("seed_types", [])
            hop_types = q.get("hop_types", [])
            
            # Step 1: Semantic search for seed memories
            seed_memories = self.semantic_retriever.retrieve_db_memories(
                query=q_text,
                limit=seeds_per_query,
                user_id=user_id
            )
            
            # Filter seeds by preferred type if specified
            if seed_types:
                type_set = set(seed_types)
                preferred = [m for m in seed_memories if m.memory_type in type_set]
                others = [m for m in seed_memories if m.memory_type not in type_set]
                seed_memories = preferred + others
                seed_memories = seed_memories[:seeds_per_query]
            
            # Add seeds to results (score 1.0 for direct semantic matches)
            for i, mem in enumerate(seed_memories):
                # Score decreases with rank position
                score = 1.0 - (i * 0.05)
                db_id = mem.id
                if db_id not in all_results or all_results[db_id][1] < score:
                    rag_mem = self._db_memory_to_rag_memory(
                        mem, score, source="semantic_seed"
                    )
                    all_results[db_id] = (rag_mem, score)
            
            total_seeds += len(seed_memories)
            
            # Step 2: 1-hop graph traversal from seeds
            if seed_memories:
                seed_ids = [m.id for m in seed_memories]
                connected_with_labels = self.repository.get_connected_memories_with_edges(seed_ids)
                
                # Prioritize by requested hop types
                connected_with_labels = self._prioritize_by_type(
                    connected_with_labels, hop_types
                )
                
                for i, (mem, edge_label) in enumerate(connected_with_labels):
                    # Graph-hop memories get lower base score than seeds
                    score = 0.7 - (i * 0.03)
                    db_id = mem.id
                    if db_id not in all_results or all_results[db_id][1] < score:
                        rag_mem = self._db_memory_to_rag_memory(
                            mem, score, source="graph_hop", edge_label=edge_label
                        )
                        all_results[db_id] = (rag_mem, score)
                
                total_hops += len(connected_with_labels)
        
        # Sort by score and cap at limit
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        final_memories = [rag_mem for rag_mem, _ in sorted_results[:limit]]
        
        # Build combined query string for logging
        combined_query = " | ".join(q.get("query", query) for q in queries)
        
        return RetrievalResult(
            memories=final_memories,
            retriever_type=self.retriever_type,
            query=combined_query,
            metadata={
                "num_queries": len(queries),
                "total_seeds": total_seeds,
                "total_hops": total_hops,
                "unique_memories": len(all_results),
                "returned": len(final_memories)
            }
        )
    
    def add_memory(self, memory: RAGMemory) -> None:
        """
        No-op: memories and edges are managed via the repository
        and MemoryLinker respectively.
        """
        pass
