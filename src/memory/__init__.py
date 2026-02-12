"""
Memory extraction, consolidation, embedding, and linking module.

Provides functionality for extracting memories from completed conversations,
consolidating them against existing memories, tag-based similarity matching,
statement embedding for semantic retrieval, and memory-to-memory graph linking.
"""

from memory.extractor import MemoryExtractor, ExtractedMemoryData
from memory.consolidator import MemoryConsolidator, ConsolidationAction, ConsolidationDecision
from memory.tag_embedder import TagEmbedder
from memory.embedder import MemoryEmbedder
from memory.linker import MemoryLinker

__all__ = [
    "MemoryExtractor",
    "ExtractedMemoryData",
    "MemoryConsolidator",
    "ConsolidationAction",
    "ConsolidationDecision",
    "TagEmbedder",
    "MemoryEmbedder",
    "MemoryLinker",
]
