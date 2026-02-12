"""
Statement embedding utilities for semantic memory retrieval.

Reuses the same sentence-transformers model as TagEmbedder for consistency.
Provides serialization/deserialization of numpy arrays to bytes for DB storage,
and batch cosine similarity computation.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from memory.tag_embedder import TagEmbedder

logger = logging.getLogger(__name__)


class MemoryEmbedder:
    """
    Embeds full memory statements for semantic retrieval.
    
    Shares the underlying sentence-transformer model with TagEmbedder
    to avoid loading the model twice. Provides serialization utilities
    for storing embeddings in the database.
    """
    
    def __init__(self, tag_embedder: TagEmbedder):
        """
        Initialize the memory embedder.
        
        Args:
            tag_embedder: TagEmbedder instance (model will be shared)
        """
        self.tag_embedder = tag_embedder
    
    def embed_statement(self, statement: str) -> np.ndarray:
        """
        Embed a memory statement into a vector.
        
        Args:
            statement: Memory statement text
            
        Returns:
            Normalized embedding vector as numpy array
        """
        embedding = self.tag_embedder.model.encode(
            statement,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding
    
    def embed_statements_batch(self, statements: List[str]) -> List[np.ndarray]:
        """
        Embed multiple statements in a single batch call.
        
        Args:
            statements: List of statement texts
            
        Returns:
            List of normalized embedding vectors
        """
        if not statements:
            return []
        embeddings = self.tag_embedder.model.encode(
            statements,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return list(embeddings)
    
    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """
        Serialize a numpy embedding to bytes for DB storage.
        
        Args:
            embedding: Numpy array embedding vector
            
        Returns:
            Bytes representation
        """
        return embedding.astype(np.float32).tobytes()
    
    @staticmethod
    def deserialize_embedding(blob: bytes) -> np.ndarray:
        """
        Deserialize bytes from DB back into a numpy embedding.
        
        Args:
            blob: Bytes from the database embedding column
            
        Returns:
            Numpy array embedding vector
        """
        return np.frombuffer(blob, dtype=np.float32)
    
    def embed_and_serialize(self, statement: str) -> bytes:
        """
        Embed a statement and serialize to bytes in one step.
        
        Args:
            statement: Memory statement text
            
        Returns:
            Serialized embedding bytes ready for DB storage
        """
        embedding = self.embed_statement(statement)
        return self.serialize_embedding(embedding)
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized vectors.
        
        Since sentence-transformers normalizes embeddings, the dot product
        equals cosine similarity.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Cosine similarity score between -1.0 and 1.0
        """
        return float(np.dot(a, b))
    
    @staticmethod
    def cosine_similarity_batch(
        query_embedding: np.ndarray,
        memory_embeddings: List[np.ndarray]
    ) -> List[float]:
        """
        Compute cosine similarity between a query and a batch of memory embeddings.
        
        Args:
            query_embedding: Query embedding vector (normalized)
            memory_embeddings: List of memory embedding vectors (normalized)
            
        Returns:
            List of cosine similarity scores
        """
        if not memory_embeddings:
            return []
        
        # Stack into matrix for vectorized dot product
        matrix = np.stack(memory_embeddings)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            similarities = matrix @ query_embedding
        return similarities.tolist()
