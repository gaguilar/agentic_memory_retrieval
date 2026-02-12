"""
Memory consolidation logic for deduplication and intelligent merging.

Compares new memory candidates against existing user memories to determine
the appropriate CRUD action: SKIP, UPDATE, MERGE, or STORE.

All candidates are evaluated in a single LLM inference pass against a
deduplicated pool of relevant existing memories.
"""

import json
import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from agents.base import BaseAgent, Message
from database.models import Memory
from database.repository import Repository
from memory.extractor import ExtractedMemoryData
from memory.tag_embedder import TagEmbedder
from prompts import CONSOLIDATION_PROMPT


logger = logging.getLogger(__name__)


class ConsolidationAction(str, Enum):
    """Possible consolidation actions for a memory candidate."""
    SKIP = "skip"       # Candidate is a duplicate; do nothing
    UPDATE = "update"   # Candidate contradicts existing; supersede old, store new
    MERGE = "merge"     # Candidate complements existing; enrich existing memory
    STORE = "store"     # Candidate is novel; store as new active memory


@dataclass
class ConsolidationDecision:
    """Result of consolidating a single memory candidate."""
    action: ConsolidationAction
    candidate: ExtractedMemoryData
    existing_memory_id: Optional[int] = None
    merged_statement: Optional[str] = None
    merged_tags: Optional[List[str]] = None
    reasoning: Optional[str] = None


class MemoryConsolidator:
    """
    Consolidates new memory candidates against existing user memories.
    
    Uses tag-based filtering (exact match first, then embedding similarity)
    to build a deduplicated pool of relevant existing memories, then uses
    a single LLM call to decide the consolidation action for all candidates.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        repository: Repository,
        tag_embedder: Optional[TagEmbedder] = None,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize the memory consolidator.
        
        Args:
            agent: LLM agent for consolidation decisions
            repository: Database repository for memory CRUD operations
            tag_embedder: Optional tag embedder for similarity-based fallback
            similarity_threshold: Cosine similarity threshold for tag matching
        """
        self.agent = agent
        self.repository = repository
        self.tag_embedder = tag_embedder
        self.similarity_threshold = similarity_threshold
    
    # ── Formatting helpers ──────────────────────────────────────────────
    
    def _format_existing_memories(self, memories: List[Memory]) -> str:
        """
        Format existing memories for the consolidation prompt.
        
        Args:
            memories: List of existing Memory instances
            
        Returns:
            Formatted string listing memories with their IDs
        """
        if not memories:
            return "(no existing memories)"
        
        lines = []
        for mem in memories:
            tags = mem.get_tags()
            tags_str = ", ".join(tags) if tags else "none"
            lines.append(
                f"- [ID: {mem.id}] ({mem.memory_type}) \"{mem.statement}\" "
                f"[tags: {tags_str}]"
            )
        return "\n".join(lines)
    
    def _format_candidates(self, candidates: List[ExtractedMemoryData]) -> str:
        """
        Format candidate memories for the consolidation prompt.
        
        Args:
            candidates: List of candidate memory data
            
        Returns:
            Formatted string listing candidates with 1-based indices
        """
        lines = []
        for i, candidate in enumerate(candidates, start=1):
            tags_str = ", ".join(candidate.tags) if candidate.tags else "none"
            lines.append(
                f"- [Candidate {i}] ({candidate.type}) \"{candidate.statement}\" "
                f"[tags: {tags_str}]"
            )
        return "\n".join(lines)
    
    # ── Semantic similarity retrieval (no LLM) ─────────────────────────
    
    def _find_relevant_memories_for_candidate(
        self,
        candidate: ExtractedMemoryData,
        all_active_memories: List[Memory]
    ) -> List[Memory]:
        """
        Find existing memories relevant to a candidate via tag-based
        semantic similarity.
        
        Builds a tag→memories index, then collects all existing tags that
        match the candidate's tags — both exact overlaps and semantically
        similar tags (via TagEmbedder). Returns the union of memories
        associated with any matching tag.
        
        Args:
            candidate: The memory candidate to find matches for
            all_active_memories: Pre-fetched list of all active user memories
            
        Returns:
            List of relevant existing memories
        """
        if not candidate.tags or not all_active_memories:
            return all_active_memories  # Return all if no tags to filter by
        
        candidate_tags = {t.lower().strip() for t in candidate.tags if t.strip()}
        
        # Build tag → memories index
        tag_to_memories: Dict[str, List[Memory]] = {}
        for mem in all_active_memories:
            for tag in mem.get_tags():
                tag_to_memories.setdefault(tag, []).append(mem)
        
        all_existing_tags = set(tag_to_memories.keys())
        if not all_existing_tags:
            return []
        
        # Collect matching tags: exact overlap + semantic similarity
        matching_tags: set = candidate_tags & all_existing_tags
        
        if self.tag_embedder is not None:
            similar = self.tag_embedder.get_similar_existing_tags(
                candidate_tags=list(candidate_tags),
                existing_tags=list(all_existing_tags),
                threshold=self.similarity_threshold
            )
            matching_tags.update(similar)
        
        if not matching_tags:
            return []
        
        # Collect memories that have any matching tag
        matched_ids: set = set()
        for tag in matching_tags:
            for mem in tag_to_memories.get(tag, []):
                matched_ids.add(mem.id)
        
        return [m for m in all_active_memories if m.id in matched_ids]
    
    def _build_relevant_memory_pool(
        self,
        candidates: List[ExtractedMemoryData],
        all_active_memories: List[Memory]
    ) -> List[Memory]:
        """
        Build a deduplicated pool of existing memories relevant to any candidate.
        
        Iterates over all candidates, finds relevant existing memories for each
        via tag matching / embedding similarity (no LLM call), and merges them
        into a single deduplicated list.
        
        Args:
            candidates: All memory candidates to consolidate
            all_active_memories: Pre-fetched list of all active user memories
            
        Returns:
            Deduplicated list of relevant existing memories
        """
        if not all_active_memories:
            return []
        
        seen_ids: set = set()
        pool: List[Memory] = []
        
        for candidate in candidates:
            relevant = self._find_relevant_memories_for_candidate(
                candidate, all_active_memories
            )
            for mem in relevant:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    pool.append(mem)
        
        return pool
    
    # ── LLM response parsing ───────────────────────────────────────────
    
    def _parse_batch_response(
        self,
        response: str,
        candidates: List[ExtractedMemoryData]
    ) -> List[ConsolidationDecision]:
        """
        Parse the LLM batch consolidation response into a list of decisions.
        
        Expects a JSON array with one object per candidate, each containing
        candidate_index, action, memory_id, merged_statement, merged_tags,
        and reasoning.
        
        Args:
            response: Raw LLM response text
            candidates: The candidate list (for mapping indices back)
            
        Returns:
            List of ConsolidationDecision instances, one per candidate
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
            
            # Handle case where LLM returns a single object instead of array
            if isinstance(data, dict):
                data = [data]
            
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array, got {type(data)}")
            
            # Build decisions indexed by candidate
            decisions_by_index: Dict[int, ConsolidationDecision] = {}
            
            for i, item in enumerate(data):
                # candidate_index is 1-based in the prompt; convert to 0-based
                candidate_idx = item.get("candidate_index", i + 1) - 1
                if candidate_idx < 0 or candidate_idx >= len(candidates):
                    candidate_idx = i  # fallback to positional order
                
                candidate = candidates[min(candidate_idx, len(candidates) - 1)]
                
                action_str = item.get("action", "store").lower()
                try:
                    action = ConsolidationAction(action_str)
                except ValueError:
                    action = ConsolidationAction.STORE
                
                decisions_by_index[candidate_idx] = ConsolidationDecision(
                    action=action,
                    candidate=candidate,
                    existing_memory_id=item.get("memory_id"),
                    merged_statement=item.get("merged_statement"),
                    merged_tags=item.get("merged_tags"),
                    reasoning=item.get("reasoning")
                )
            
            # Ensure every candidate has a decision (STORE as fallback)
            decisions = []
            for i, candidate in enumerate(candidates):
                if i in decisions_by_index:
                    decisions.append(decisions_by_index[i])
                else:
                    decisions.append(ConsolidationDecision(
                        action=ConsolidationAction.STORE,
                        candidate=candidate,
                        reasoning="Not addressed in LLM response; defaulting to STORE"
                    ))
            
            return decisions
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse batch consolidation response: {e}")
            # Default: STORE all candidates
            return [
                ConsolidationDecision(
                    action=ConsolidationAction.STORE,
                    candidate=candidate,
                    reasoning="Failed to parse LLM consolidation response; defaulting to STORE"
                )
                for candidate in candidates
            ]
    
    # ── Decision execution ──────────────────────────────────────────────
    
    def execute_decision(
        self,
        decision: ConsolidationDecision,
        conversation_id: int
    ) -> Optional[Memory]:
        """
        Execute a consolidation decision by performing the appropriate
        database operation.
        
        When a memory is superseded (UPDATE) or merged (MERGE), any
        graph edges involving the affected memory are deactivated so
        the MemoryLinker can re-evaluate them for the updated memory.
        
        Args:
            decision: The consolidation decision to execute
            conversation_id: Conversation ID for new memory records
            
        Returns:
            The resulting Memory instance (new or updated), or None for SKIP
        """
        if decision.action == ConsolidationAction.SKIP:
            logger.info(
                f"SKIP: '{decision.candidate.statement[:60]}...' "
                f"Reason: {decision.reasoning}"
            )
            return None
        
        elif decision.action == ConsolidationAction.UPDATE:
            # Store new memory first
            new_memory = self.repository.add_memory(
                conversation_id=conversation_id,
                statement=decision.candidate.statement,
                memory_type=decision.candidate.type,
                tags=decision.candidate.tags
            )
            
            # Mark old memory as superseded and deactivate its graph edges
            if decision.existing_memory_id:
                self.repository.mark_memory_superseded(
                    memory_id=decision.existing_memory_id,
                    superseded_by_id=new_memory.id
                )
                # Deactivate graph edges for the superseded memory
                deactivated = self.repository.deactivate_edges_for_memory(
                    decision.existing_memory_id
                )
                if deactivated:
                    logger.info(
                        f"UPDATE: Deactivated {deactivated} graph edges "
                        f"for superseded memory {decision.existing_memory_id}"
                    )
                logger.info(
                    f"UPDATE: Superseded memory {decision.existing_memory_id} "
                    f"with new memory {new_memory.id}. "
                    f"Reason: {decision.reasoning}"
                )
            
            return new_memory
        
        elif decision.action == ConsolidationAction.MERGE:
            if decision.existing_memory_id and decision.merged_statement:
                # Deactivate graph edges before updating (will be re-linked)
                deactivated = self.repository.deactivate_edges_for_memory(
                    decision.existing_memory_id
                )
                if deactivated:
                    logger.info(
                        f"MERGE: Deactivated {deactivated} graph edges "
                        f"for merged memory {decision.existing_memory_id} "
                        f"(will be re-linked)"
                    )
                
                updated = self.repository.update_memory_content(
                    memory_id=decision.existing_memory_id,
                    statement=decision.merged_statement,
                    tags=decision.merged_tags or decision.candidate.tags
                )
                logger.info(
                    f"MERGE: Enriched memory {decision.existing_memory_id}. "
                    f"Reason: {decision.reasoning}"
                )
                return updated
            else:
                # Fallback: if merge data is incomplete, store as new
                logger.warning(
                    f"MERGE: Missing memory_id or merged_statement, "
                    f"falling back to STORE"
                )
                return self.repository.add_memory(
                    conversation_id=conversation_id,
                    statement=decision.candidate.statement,
                    memory_type=decision.candidate.type,
                    tags=decision.candidate.tags
                )
        
        elif decision.action == ConsolidationAction.STORE:
            new_memory = self.repository.add_memory(
                conversation_id=conversation_id,
                statement=decision.candidate.statement,
                memory_type=decision.candidate.type,
                tags=decision.candidate.tags
            )
            logger.info(
                f"STORE: New memory {new_memory.id}: "
                f"'{decision.candidate.statement[:60]}...'. "
                f"Reason: {decision.reasoning}"
            )
            return new_memory
        
        return None
    
    # ── Main entry point ────────────────────────────────────────────────
    
    def consolidate(
        self,
        candidates: List[ExtractedMemoryData],
        user_id: int,
        conversation_id: int
    ) -> List[Optional[Memory]]:
        """
        Consolidate a list of memory candidates against existing user memories
        in a single LLM inference pass.
        
        Steps:
        1. Pre-fetch all active memories for the user.
        2. For each candidate, find relevant existing memories via tag-based
           semantic similarity (no LLM call).
        3. Pool and deduplicate the relevant existing memories.
        4. Build a single prompt with the existing pool + all candidates.
        5. Make one LLM call to get consolidation decisions for all candidates.
        6. Parse and execute the decisions.
        
        If no relevant existing memories are found for any candidate, all
        candidates are STOREd directly without an LLM call.
        
        Args:
            candidates: List of extracted memory candidates
            user_id: The user ID for fetching existing memories
            conversation_id: Conversation ID for new memory records
            
        Returns:
            List of Memory instances (None entries for SKIPped candidates)
        """
        if not candidates:
            return []
        
        # Step 1: Pre-fetch all active memories for this user once (no limit)
        all_active = self.repository.get_user_active_memories(user_id, limit=None)
        
        # Step 2-3: Build deduplicated pool of relevant existing memories
        relevant_pool = self._build_relevant_memory_pool(candidates, all_active)
        
        # Shortcut: if no relevant existing memories, STORE everything directly
        if not relevant_pool:
            logger.info(
                f"No relevant existing memories found for {len(candidates)} "
                f"candidate(s); storing all as new"
            )
            results = []
            for candidate in candidates:
                decision = ConsolidationDecision(
                    action=ConsolidationAction.STORE,
                    candidate=candidate,
                    reasoning="No relevant existing memories found"
                )
                result = self.execute_decision(decision, conversation_id)
                results.append(result)
            return results
        
        # Step 4: Build the single consolidation prompt
        existing_text = self._format_existing_memories(relevant_pool)
        candidates_text = self._format_candidates(candidates)
        prompt = CONSOLIDATION_PROMPT.format(
            existing_memories=existing_text,
            candidate_memories=candidates_text
        )
        
        # Step 5: Single LLM inference call
        messages = [Message(role="user", content=prompt)]
        response = self.agent.generate(
            messages=messages,
            max_tokens=2000,
            temperature=0.2  # Low temperature for consistent decisions
        )
        
        # Log the consolidation inference
        try:
            self.repository.add_inference(
                prompt=prompt,
                response=response.content,
                model=response.model,
                provider=response.provider,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                conversation_id=conversation_id,
                task="memory_consolidation"
            )
        except Exception:
            pass  # Don't fail consolidation if logging fails
        
        # Step 6: Parse batch response and execute decisions
        decisions = self._parse_batch_response(response.content, candidates)
        
        results = []
        for decision in decisions:
            result = self.execute_decision(decision, conversation_id)
            results.append(result)
        
        return results
