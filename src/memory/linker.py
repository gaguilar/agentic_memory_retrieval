"""
Memory linking agent for constructing the memory-to-memory graph.

Uses an LLM to identify meaningful relationships between memories
and creates edges in the database. Handles both initial linking of
new memories and re-evaluation of edges when memories change.
"""

import json
import logging
from typing import List, Optional, Dict, Any

from agents.base import BaseAgent, Message
from database.repository import Repository
from database.models import Memory
from prompts.memory_linking import MEMORY_LINKING_PROMPT

logger = logging.getLogger(__name__)


class MemoryLinker:
    """
    Creates and manages edges in the memory-to-memory graph.
    
    After memory extraction and consolidation, the linker analyzes
    new memories against existing ones to discover meaningful
    relationships that enable graph-based retrieval.
    """
    
    def __init__(self, agent: BaseAgent, repository: Repository):
        """
        Initialize the memory linker.
        
        Args:
            agent: LLM agent for generating edge proposals
            repository: Database repository for edge CRUD
        """
        self.agent = agent
        self.repository = repository
    
    def _format_memories_for_prompt(self, memories: List[Memory], label: str = "memory") -> str:
        """
        Format a list of memories for inclusion in the linking prompt.
        
        Args:
            memories: List of Memory instances
            label: Label prefix for display
            
        Returns:
            Formatted string
        """
        if not memories:
            return "(none)"
        
        lines = []
        for mem in memories:
            tags = mem.get_tags()
            tags_str = ", ".join(tags) if tags else "none"
            lines.append(
                f"- [ID: {mem.id}] ({mem.memory_type}) \"{mem.statement}\" "
                f"[tags: {tags_str}]"
            )
        return "\n".join(lines)
    
    def _parse_edge_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response into a list of edge proposals.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            List of dicts with source_id, target_id, label keys
        """
        try:
            response = response.strip()
            
            # Handle markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
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
            
            data = json.loads(response)
            
            if not isinstance(data, list):
                return []
            
            edges = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                source_id = item.get("source_id")
                target_id = item.get("target_id")
                label = item.get("label", "").strip()
                
                if source_id is None or target_id is None or not label:
                    continue
                
                # Don't allow self-loops
                if source_id == target_id:
                    continue
                
                edges.append({
                    "source_id": int(source_id),
                    "target_id": int(target_id),
                    "label": label.upper().replace(" ", "_"),
                    "reasoning": item.get("reasoning", "")
                })
            
            return edges
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse edge response: {e}")
            return []
    
    def link_memories(
        self, new_memories: List[Memory], user_id: int, conversation_id: int
    ) -> List[Dict[str, Any]]:
        """
        Analyze new memories and create edges to existing memories.
        
        Args:
            new_memories: Newly extracted/consolidated memories
            user_id: User ID for fetching existing memories
            conversation_id: Conversation ID for inference logging
            
        Returns:
            List of dicts with source_id, target_id, label, reasoning for display
        """
        if not new_memories:
            return []
        
        # Get all active memories for the user (including the new ones)
        all_active = self.repository.get_user_active_memories(user_id, limit=None)
        
        # Separate new from existing for the prompt
        new_ids = {m.id for m in new_memories}
        existing_memories = [m for m in all_active if m.id not in new_ids]
        
        # Build the prompt
        new_text = self._format_memories_for_prompt(new_memories)
        existing_text = self._format_memories_for_prompt(existing_memories)
        
        prompt = MEMORY_LINKING_PROMPT.format(
            new_memories=new_text,
            existing_memories=existing_text
        )
        
        # Call LLM
        messages = [Message(role="user", content=prompt)]
        response = self.agent.generate(
            messages=messages,
            max_tokens=2000,
            temperature=0.2
        )
        
        # Log the inference
        try:
            self.repository.add_inference(
                prompt=prompt,
                response=response.content,
                model=response.model,
                provider=response.provider,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                conversation_id=conversation_id,
                task="memory_linking"
            )
        except Exception:
            pass
        
        # Parse edge proposals
        edge_proposals = self._parse_edge_response(response.content)

        # Build id -> statement and id -> type maps for display
        id_to_statement = {m.id: m.statement for m in all_active}
        id_to_type = {m.id: m.memory_type for m in all_active}

        # Validate that referenced memory IDs exist in active memories
        active_ids = {m.id for m in all_active}

        created_edges = []
        for proposal in edge_proposals:
            source_id = proposal["source_id"]
            target_id = proposal["target_id"]
            label = proposal["label"]
            reasoning = proposal.get("reasoning", "")

            if source_id not in active_ids or target_id not in active_ids:
                logger.warning(
                    f"Skipping edge {source_id}--[{label}]-->{target_id}: "
                    f"memory ID not found in active memories"
                )
                continue

            try:
                self.repository.add_memory_edge(source_id, target_id, label)
                created_edges.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "label": label,
                    "reasoning": reasoning,
                    "source_statement": id_to_statement.get(source_id, "(unknown)"),
                    "target_statement": id_to_statement.get(target_id, "(unknown)"),
                    "source_type": id_to_type.get(source_id, "unknown"),
                    "target_type": id_to_type.get(target_id, "unknown"),
                })
            except Exception as e:
                logger.warning(f"Failed to create edge: {e}")

        return created_edges
    
    def reevaluate_edges_for_memory(
        self, memory_id: int, user_id: int, conversation_id: int
    ) -> List[Dict[str, Any]]:
        """
        Re-evaluate edges for a memory that has been updated or merged.
        
        Deactivates old edges and re-runs linking for the updated memory.
        
        Args:
            memory_id: ID of the updated memory
            user_id: User ID
            conversation_id: Conversation ID for inference logging
            
        Returns:
            List of dicts with source_id, target_id, label, reasoning for display
        """
        # Deactivate existing edges for this memory
        deactivated = self.repository.deactivate_edges_for_memory(memory_id)
        if deactivated:
            logger.info(f"Deactivated {deactivated} edges for memory {memory_id}")
        
        # Get the updated memory
        with self.repository.get_session() as session:
            memory = session.get(Memory, memory_id)
            if not memory or not memory.is_active:
                return []
            session.expunge(memory)
        
        # Re-run linking for just this memory
        return self.link_memories([memory], user_id, conversation_id)
