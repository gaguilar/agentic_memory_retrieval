"""
Abstract base class for RAG retrievers.

Defines the interface that all RAG implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class Memory:
    """
    A memory item that can be retrieved by RAG systems.
    
    This structure is shared across all RAG implementations
    (semantic, graph-based, etc.).
    """
    memory_id: str
    text: str
    memory_type: str  # episodic, semantic, procedural, affective
    timestamp: datetime
    polarity: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    entities: List[str]
    topics: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat(),
            "polarity": self.polarity,
            "confidence": self.confidence,
            "entities": self.entities,
            "topics": self.topics,
            "metadata": self.metadata
        }
    
    def format_for_prompt(self) -> str:
        """Format this memory for inclusion in a prompt."""
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp_str}] ({self.memory_type}) {self.text}"


@dataclass
class RetrievalResult:
    """Result from a RAG retrieval operation."""
    memories: List[Memory]
    retriever_type: str
    query: str
    metadata: Optional[Dict[str, Any]] = None
    
    def format_for_prompt(self) -> str:
        """Format all retrieved memories for prompt injection."""
        if not self.memories:
            return ""
        
        lines = ["Relevant memories from past conversations:"]
        for memory in self.memories:
            lines.append(f"- {memory.format_for_prompt()}")
        
        return "\n".join(lines)
    
    @property
    def is_empty(self) -> bool:
        """Check if no memories were retrieved."""
        return len(self.memories) == 0


class BaseRetriever(ABC):
    """
    Abstract base class for RAG retrievers.
    
    All RAG implementations (semantic, graph-based, etc.) must inherit
    from this class and implement the retrieve method.
    """
    
    @property
    @abstractmethod
    def retriever_type(self) -> str:
        """Return the type of this retriever (e.g., 'semantic', 'graph', 'none')."""
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: The user's query or message
            limit: Maximum number of memories to retrieve
            **kwargs: Additional retriever-specific parameters
            
        Returns:
            RetrievalResult containing the retrieved memories
        """
        pass
    
    def add_memory(self, memory: Memory) -> None:
        """
        Add a memory to the retriever's store.
        
        This is optional and may not be implemented by all retrievers.
        
        Args:
            memory: Memory to add
        """
        pass
