"""
Prompt templates used across the conversational agent system.

Each file in this package contains a single prompt constant,
named by its purpose for easy discovery and readability.
"""

from prompts.system import SYSTEM_PROMPT
from prompts.memory_extraction import EXTRACTION_PROMPT
from prompts.memory_consolidation import CONSOLIDATION_PROMPT
from prompts.semantic_query import SEMANTIC_QUERY_PROMPT
from prompts.graph_query import GRAPH_QUERY_PROMPT
from prompts.memory_linking import MEMORY_LINKING_PROMPT

__all__ = [
    "SYSTEM_PROMPT",
    "EXTRACTION_PROMPT",
    "CONSOLIDATION_PROMPT",
    "SEMANTIC_QUERY_PROMPT",
    "GRAPH_QUERY_PROMPT",
    "MEMORY_LINKING_PROMPT",
]
