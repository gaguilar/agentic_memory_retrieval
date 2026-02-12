"""
Memory extractor for extracting memories from completed conversations.

Uses an LLM to analyze conversation turns and extract structured memories
with tags and temporal grounding.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from agents.base import BaseAgent, Message
from prompts import EXTRACTION_PROMPT


MEMORY_TYPES = {
    "semantic", 
    "episodic", 
    "procedural", 
    "affective",
}


@dataclass
class ExtractedMemoryData:
    """Data class for extracted memory before database storage."""
    statement: str
    type: str
    tags: List[str] = field(default_factory=list)


class MemoryExtractor:
    """
    Extracts memories from completed conversations using an LLM.
    
    This is an independent module that takes conversation turns
    and uses an LLM to extract structured memories with tags
    and temporal grounding.
    """
    
    def __init__(self, agent: BaseAgent):
        """
        Initialize the memory extractor.
        
        Args:
            agent: LLM agent to use for extraction
        """
        self.agent = agent
    
    def _format_turns_for_prompt(self, turns: List[Dict[str, Any]]) -> str:
        """
        Format conversation turns for the extraction prompt.
        
        Args:
            turns: List of turn dictionaries with role, content, and optional timestamp
            
        Returns:
            Formatted string of the conversation
        """
        lines = []
        for turn in turns:
            role = turn.get("role", "unknown").upper()
            content = turn.get("content", "")
            timestamp = turn.get("timestamp")
            
            if timestamp:
                # Format: [2024-01-15 14:30] [USER]: message
                lines.append(f"[{timestamp}] [{role}]: {content}")
            else:
                lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)
    
    def _parse_response(self, response: str) -> List[ExtractedMemoryData]:
        """
        Parse the LLM response into extracted memories.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            List of ExtractedMemoryData objects
        """
        try:
            # Try to find JSON array in response
            response = response.strip()
            
            # Handle cases where response might have markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                # Remove first and last lines (code block markers)
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```") and not in_json:
                        in_json = True
                        continue
                    elif line.startswith("```") and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response = "\n".join(json_lines)
            
            # Parse JSON
            data = json.loads(response)
            
            if not isinstance(data, list):
                return []
            
            memories = []
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                statement = item.get("statement", "").strip()
                memory_type = item.get("type", "").strip().lower()
                raw_tags = item.get("tags", [])
                
                # Validate required fields
                if not statement:
                    continue
                
                # Validate memory type
                if memory_type not in MEMORY_TYPES:
                    memory_type = "semantic"  # Default to semantic if invalid
                
                # Normalize tags
                tags = []
                if isinstance(raw_tags, list):
                    tags = [
                        t.lower().strip().replace(" ", "_")
                        for t in raw_tags
                        if isinstance(t, str) and t.strip()
                    ]
                
                memories.append(ExtractedMemoryData(
                    statement=statement,
                    type=memory_type,
                    tags=tags
                ))
            
            return memories
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list
            return []
    
    def extract(
        self,
        turns: List[Dict[str, Any]],
        max_tokens: int = 2000,
        repository: Optional[Any] = None,
        conversation_id: Optional[int] = None
    ) -> List[ExtractedMemoryData]:
        """
        Extract memories from conversation turns.
        
        Args:
            turns: List of turn dictionaries with role and content keys
            max_tokens: Maximum tokens for LLM response
            repository: Optional database repository for logging the inference
            conversation_id: Optional conversation ID for inference logging
            
        Returns:
            List of ExtractedMemoryData objects
        """
        if not turns:
            return []
        
        # Format conversation for prompt
        conversation_text = self._format_turns_for_prompt(turns)
        full_prompt = EXTRACTION_PROMPT.format(conversation_text=conversation_text)
        
        # Create messages for LLM
        messages = [Message(role="user", content=full_prompt)]
        
        # Call LLM (Lower temperature for more consistent extraction)
        response = self.agent.generate(messages=messages, max_tokens=max_tokens, temperature=0.3)
        
        # Log the inference if repository is available
        if repository is not None:
            try:
                repository.add_inference(
                    prompt=full_prompt,
                    response=response.content,
                    model=response.model,
                    provider=response.provider,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    conversation_id=conversation_id,
                    task="memory_extraction"
                )
            except Exception:
                pass  # Don't fail extraction if logging fails
        
        # Parse and return memories
        return self._parse_response(response.content)
    
    def extract_to_dicts(
        self,
        turns: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> List[Dict[str, Any]]:
        """
        Extract memories and return as dictionaries (for repository batch insert).
        
        Args:
            turns: List of turn dictionaries with role and content keys
            max_tokens: Maximum tokens for LLM response
            
        Returns:
            List of dictionaries with statement, type, tags keys
        """
        memories = self.extract(turns, max_tokens)
        return [
            {
                "statement": m.statement,
                "type": m.type,
                "tags": m.tags
            }
            for m in memories
        ]
    
    def extract_and_consolidate(
        self,
        turns: List[Dict[str, Any]],
        user_id: int,
        conversation_id: int,
        repository: Any,
        tag_embedder: Optional[Any] = None,
        similarity_threshold: float = 0.8,
        max_tokens: int = 4000
    ) -> List[Any]:
        """
        Extract memories with consolidation against existing user memories.
        
        This is the full pipeline:
        1. Extract candidate memories with tags from conversation (with temporal grounding)
        2. For each candidate, find relevant existing memories by tag matching
        3. If no exact tag matches, fall back to embedding similarity on tags
        4. Use LLM to decide: SKIP, UPDATE, MERGE, or STORE
        5. Execute the appropriate database operation
        
        Args:
            turns: List of turn dictionaries with role and content keys
            user_id: User ID for fetching existing memories
            conversation_id: Conversation ID for new memory records
            repository: Database repository instance
            tag_embedder: Optional TagEmbedder for similarity-based fallback
            similarity_threshold: Cosine similarity threshold for tag matching
            max_tokens: Maximum tokens for extraction LLM response
            
        Returns:
            List of Memory instances (None entries for SKIPped candidates)
        """
        from memory.consolidator import MemoryConsolidator
        
        # Step 1: Extract candidates (pass repository + conversation_id for inference logging)
        candidates = self.extract(
            turns, max_tokens,
            repository=repository,
            conversation_id=conversation_id
        )
        
        if not candidates:
            return []
        
        # Step 2: Consolidate against existing memories
        consolidator = MemoryConsolidator(
            agent=self.agent,
            repository=repository,
            tag_embedder=tag_embedder,
            similarity_threshold=similarity_threshold
        )
        
        return consolidator.consolidate(
            candidates=candidates,
            user_id=user_id,
            conversation_id=conversation_id
        )
