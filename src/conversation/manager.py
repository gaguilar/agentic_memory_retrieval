"""
Conversation manager that orchestrates the agent and database.

This is the core component that ties everything together:
- Manages conversation state and history
- Builds prompts within token limits using XML-templated system prompt
- Retrieves active memories from the database
- Logs inference calls
- Extracts memories when conversations end
"""

import json
import logging
from typing import List, Optional, Tuple

from config import Settings, LLMProvider, RAGMode
from agents.base import BaseAgent, Message
from agents.openai_agent import OpenAIAgent
from agents.anthropic_agent import AnthropicAgent
from database.repository import Repository
from database.models import Turn, User, Conversation, Memory
from utils.tokens import TokenCounter
from memory.extractor import MemoryExtractor
from memory.tag_embedder import TagEmbedder
from memory.embedder import MemoryEmbedder
from memory.linker import MemoryLinker
from rag.semantic import SemanticRetriever
from rag.graph import GraphRetriever
from prompts.semantic_query import SEMANTIC_QUERY_PROMPT

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversations between users and the AI agent.
    
    Responsibilities:
    - Maintain conversation state
    - Build prompts with history and memories
    - Enforce token limits
    - Persist all interactions to database
    - Log inference calls
    - Extract memories when conversations end
    """
    
    def __init__(self, settings: Optional[Settings] = None, agent: Optional[BaseAgent] = None, repository: Optional[Repository] = None):
        """
        Initialize the conversation manager.
        
        Args:
            settings: Application settings (uses global if not provided)
            agent: LLM agent (created from settings if not provided)
            repository: Database repository (created from settings if not provided)
        """
        from config import settings as default_settings
        self.settings = settings or default_settings
        
        # Initialize components
        self.agent = agent or self._create_agent()
        self.repository = repository or Repository(self.settings.database_url)
        
        # Initialize memory extractor (uses same agent)
        self.memory_extractor = MemoryExtractor(self.agent)
        
        # Initialize retriever and linker based on RAG mode
        self.retriever = None
        self.memory_linker = None
        self._tag_embedder = None
        self._memory_embedder = None
        
        if self.settings.rag_mode in (RAGMode.SEMANTIC, RAGMode.GRAPH):
            self._tag_embedder = TagEmbedder(
                model_name=self.settings.embedding_model,
                similarity_threshold=self.settings.tag_similarity_threshold
            )
            self._memory_embedder = MemoryEmbedder(self._tag_embedder)
            
            if self.settings.rag_mode == RAGMode.SEMANTIC:
                self.retriever = SemanticRetriever(
                    repository=self.repository,
                    tag_embedder=self._tag_embedder,
                    similarity_threshold=self.settings.semantic_similarity_threshold
                )
            elif self.settings.rag_mode == RAGMode.GRAPH:
                self.retriever = GraphRetriever(
                    repository=self.repository,
                    tag_embedder=self._tag_embedder,
                    similarity_threshold=self.settings.semantic_similarity_threshold
                )
                self.memory_linker = MemoryLinker(
                    agent=self.agent,
                    repository=self.repository,
                )
        
        # Current conversation state
        self.current_user: Optional[User] = None
        self.current_conversation: Optional[Conversation] = None
        
        # Memories used in the last send_message() call (for UI display)
        self.last_used_memories: List[Memory] = []
        # RAG queries from last inference (for UI display)
        self.last_semantic_query: Optional[str] = None
        self.last_graph_queries: Optional[List[dict]] = None
        # Memory graph edges created in last end_conversation() (for UI display)
        self.last_created_edges: List[dict] = []
    
    def _create_agent(self) -> BaseAgent:
        """Create an agent based on settings."""
        if self.settings.llm_provider == LLMProvider.OPENAI:
            return OpenAIAgent(
                api_key=self.settings.get_api_key(),
                model=self.settings.openai_model
            )
        elif self.settings.llm_provider == LLMProvider.ANTHROPIC:
            return AnthropicAgent(
                api_key=self.settings.get_api_key(),
                model=self.settings.anthropic_model
            )
        else:
            raise ValueError(f"Unknown provider: {self.settings.llm_provider}")
    
    def start_conversation(self, username: str) -> User:
        """
        Start a new conversation for a user.
        
        If there's an active conversation, it will be ended first
        (with memory extraction).
        
        Args:
            username: Username for the conversation
            
        Returns:
            User instance
        """
        # Get or create user
        self.current_user = self.repository.get_or_create_user(username)
        
        # End any existing active conversation for this user
        existing_conversation = self.repository.get_active_conversation(self.current_user.id)
        if existing_conversation:
            self._end_conversation_internal(existing_conversation.id)
        
        # Create new conversation
        self.current_conversation = self.repository.create_conversation(self.current_user.id)
        
        return self.current_user
    
    def _end_conversation_internal(self, conversation_id: int) -> List[Memory]:
        """
        Internal method to end a conversation and extract memories.
        
        When consolidation is enabled, uses extract_and_consolidate to
        deduplicate, supersede contradictions, and merge complementary
        memories. Otherwise extracts and batch-inserts only.
        
        Args:
            conversation_id: ID of the conversation to end
            
        Returns:
            List of extracted/consolidated memories (active Memory instances)
        """
        self.last_created_edges = []

        # Need user_id for consolidation; load conversation
        conversation = self.repository.get_conversation(conversation_id)
        if not conversation:
            self.repository.end_conversation(conversation_id)
            return []
        
        user_id = conversation.user_id
        
        # Get all turns for this conversation
        turns = self.repository.get_conversation_turns(conversation_id)
        
        extracted_memories: List[Memory] = []
        
        if turns:
            # Format turns for extraction (include timestamps)
            turns_data = [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
                for turn in turns
            ]
            
            tag_embedder = TagEmbedder(
                model_name=self.settings.embedding_model, 
                similarity_threshold=self.settings.tag_similarity_threshold
            )

            # Compare against existing memories to SKIP / UPDATE / MERGE / STORE as appropriate
            results = self.memory_extractor.extract_and_consolidate(
                turns=turns_data,
                user_id=user_id,
                conversation_id=conversation_id,
                repository=self.repository,
                tag_embedder=tag_embedder,
                similarity_threshold=self.settings.tag_similarity_threshold,
            )

            extracted_memories = [m for m in results if m is not None]
            
            # Generate embeddings for new/updated memories (semantic + graph modes)
            if extracted_memories and self.settings.rag_mode in (RAGMode.SEMANTIC, RAGMode.GRAPH):
                self._generate_embeddings_for_memories(extracted_memories)
            
            # Run graph linking for new memories (graph mode only)
            if extracted_memories and self.settings.rag_mode == RAGMode.GRAPH and self.memory_linker:
                try:
                    self.last_created_edges = self.memory_linker.link_memories(
                        new_memories=extracted_memories,
                        user_id=user_id,
                        conversation_id=conversation_id
                    )
                except Exception as e:
                    logger.warning(f"Memory linking failed: {e}")
                    self.last_created_edges = []
            
        # Mark conversation as ended
        self.repository.end_conversation(conversation_id)
        
        return extracted_memories
    
    def end_conversation(self) -> List[Memory]:
        """
        End the current conversation and extract memories.
        
        This method:
        1. Gets all turns from the current conversation
        2. Uses the MemoryExtractor to extract memories
        3. Saves memories to database
        4. Marks conversation as ended
        
        Returns:
            List of Memory instances that were extracted
            
        Raises:
            ValueError: If no conversation is active
        """
        if not self.current_conversation:
            raise ValueError("No active conversation to end.")
        
        conversation_id = self.current_conversation.id
        extracted_memories = self._end_conversation_internal(conversation_id)
        
        # Clear current conversation state
        self.current_conversation = None
        
        return extracted_memories
    
    @staticmethod
    def _format_memories_for_prompt(memories: List[Memory]) -> str:
        """
        Format database Memory objects into text for the system prompt.
        
        Args:
            memories: List of Memory instances from the database
            
        Returns:
            Formatted string with one line per memory
        """
        if not memories:
            return "No memories available yet."
        
        lines = []
        for m in memories:
            tags = m.get_tags()
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"- ({m.memory_type}) {m.statement}{tag_str}")
        return "\n".join(lines)
    
    @staticmethod
    def _format_history_for_prompt(history: List[Turn]) -> str:
        """
        Format conversation turns into text for the system prompt.
        
        Args:
            history: List of Turn instances
            
        Returns:
            Formatted string with one line per turn
        """
        if not history:
            return "No previous conversation history."
        
        lines = []
        for turn in history:
            timestamp_str = turn.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[{timestamp_str}] {turn.role}: {turn.content}")
        return "\n".join(lines)
    
    def _build_messages(
        self,
        user_message: str,
        historical_turns: List[Turn],
        current_turns: List[Turn],
        memories: List[Memory]
    ) -> Tuple[List[Message], str]:
        """
        Build the message list for the LLM.
        
        Replaces the {memories}, {conversation_history}, and
        {current_conversation} placeholders in the system prompt
        template with formatted content.
        
        Args:
            user_message: Current user message
            historical_turns: Turns from previous sessions (truncated)
            current_turns: Turns from the active session (unlimited)
            memories: User memories from the database
            
        Returns:
            Tuple of (messages list, formatted prompt string for logging)
        """
        messages = []
        
        # Build system content by replacing placeholders in the template
        memory_text = self._format_memories_for_prompt(memories)
        history_text = self._format_history_for_prompt(historical_turns)
        current_text = self._format_history_for_prompt(current_turns)
        
        system_content = (
            self.settings.system_prompt
            .replace("{memories}", memory_text)
            .replace("{conversation_history}", history_text)
            .replace("{current_conversation}", current_text)
        )
        
        messages.append(Message(role="system", content=system_content))
        
        # Add historical turns with timestamps
        for turn in historical_turns:
            timestamp_str = turn.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            content_with_timestamp = f"[{timestamp_str}] {turn.content}"
            messages.append(Message(role=turn.role, content=content_with_timestamp))
        
        # Add current conversation turns with timestamps
        for turn in current_turns:
            timestamp_str = turn.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            content_with_timestamp = f"[{timestamp_str}] {turn.content}"
            messages.append(Message(role=turn.role, content=content_with_timestamp))
        
        # Add current user message (no timestamp prefix needed, it's the current message)
        messages.append(Message(role="user", content=user_message))
        
        # Build prompt string for logging
        prompt_parts = []
        for msg in messages:
            prompt_parts.append(f"[{msg.role.upper()}]\n{msg.content}")
        prompt_string = "\n\n".join(prompt_parts)
        
        return messages, prompt_string
    
    def _truncate_history_to_fit(
        self,
        historical_turns: List[Turn],
        current_turns: List[Turn],
        user_message: str,
        memories: List[Memory],
        max_tokens: int
    ) -> List[Turn]:
        """
        Truncate historical conversation turns to fit within token limit.
        
        Current conversation turns are treated as fixed overhead (never
        truncated). Only historical turns from previous sessions are
        trimmed to fit the remaining token budget.
        
        Args:
            historical_turns: Turns from previous sessions (subject to truncation)
            current_turns: Turns from the active session (always included in full)
            user_message: Current user message
            memories: User memories from the database
            max_tokens: Maximum tokens for prompt
            
        Returns:
            Truncated list of historical turns that fits within limit
        """
        provider = self.settings.llm_provider.value
        model = self.settings.get_model()
        
        # Calculate base tokens (system prompt with memories + current message)
        # Use the template with memories filled in but history/current as empty
        # to estimate the overhead before we add turns
        memory_text = self._format_memories_for_prompt(memories)
        system_content = (
            self.settings.system_prompt
            .replace("{memories}", memory_text)
            .replace("{conversation_history}", "")
            .replace("{current_conversation}", "")
        )
        
        base_tokens = TokenCounter.estimate_message_tokens(
            "system", system_content, model, provider
        )
        base_tokens += TokenCounter.estimate_message_tokens(
            "user", user_message, model, provider
        )
        
        # Account for current conversation turns as fixed overhead
        current_turns_tokens = 0
        for turn in current_turns:
            current_turns_tokens += TokenCounter.estimate_message_tokens(
                turn.role, turn.content, model, provider
            )
        
        # Reserve tokens for response and current conversation
        available_tokens = (
            max_tokens - base_tokens - self.settings.max_response_tokens - current_turns_tokens
        )
        
        if available_tokens <= 0:
            return []
        
        # Add historical turns from most recent, stopping when we hit the limit
        truncated = []
        tokens_used = 0
        
        for turn in reversed(historical_turns):
            turn_tokens = TokenCounter.estimate_message_tokens(
                turn.role, turn.content, model, provider
            )
            
            if tokens_used + turn_tokens > available_tokens:
                break
            
            truncated.insert(0, turn)
            tokens_used += turn_tokens
        
        return truncated
    
    def _generate_embeddings_for_memories(self, memories: List[Memory]) -> None:
        """
        Generate and store embeddings for a list of memories.
        
        Args:
            memories: List of Memory instances to embed
        """
        if not self._memory_embedder:
            return
        
        for memory in memories:
            try:
                blob = self._memory_embedder.embed_and_serialize(memory.statement)
                self.repository.update_memory_embedding(memory.id, blob)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for memory {memory.id}: {e}")
    
    def _generate_semantic_query(self, user_message: str, current_turns: List[Turn], conversation_id: int, turn_id: Optional[int] = None) -> str:
        """
        Use the LLM to generate an optimized semantic search query.
        
        Args:
            user_message: The user's latest message
            current_turns: Current conversation turns for context
            conversation_id: For inference logging
            turn_id: User turn ID for inference logging
            
        Returns:
            Generated search query string
        """
        current_text = self._format_history_for_prompt(current_turns)
        prompt = SEMANTIC_QUERY_PROMPT.format(
            current_conversation=current_text,
            user_message=user_message
        )
        
        messages = [Message(role="user", content=prompt)]
        response = self.agent.generate(messages=messages, max_tokens=200, temperature=0.3)
        
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
                turn_id=turn_id,
                task="semantic_query"
            )
        except Exception:
            pass
        
        return response.content.strip()
    
    def _generate_graph_queries(self, user_message: str, current_turns: List[Turn], conversation_id: int, turn_id: Optional[int] = None) -> List[dict]:
        """
        Use the LLM to generate multiple entry queries for graph traversal.
        
        Args:
            user_message: The user's latest message
            current_turns: Current conversation turns for context
            conversation_id: For inference logging
            turn_id: User turn ID for inference logging
            
        Returns:
            List of query dicts with 'query', 'seed_types', 'hop_types' keys
        """
        from prompts.graph_query import GRAPH_QUERY_PROMPT
        
        current_text = self._format_history_for_prompt(current_turns)
        prompt = GRAPH_QUERY_PROMPT.format(
            current_conversation=current_text,
            user_message=user_message
        )
        
        messages = [Message(role="user", content=prompt)]
        response = self.agent.generate(
            messages=messages,
            max_tokens=500,
            temperature=0.3
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
                turn_id=turn_id,
                task="graph_query"
            )
        except Exception:
            pass
        
        # Parse the JSON response
        try:
            text = response.content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
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
                text = "\n".join(json_lines)
            
            queries = json.loads(text)
            if isinstance(queries, list):
                return queries
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse graph queries response, falling back to single query")
        
        # Fallback: use the raw response as a single query
        return [{"query": response.content.strip(), "seed_types": [], "hop_types": []}]
    
    def _retrieve_memories(self, user_message: str, user_id: int, conversation_id: int, current_conv_turns: List[Turn], turn_id: Optional[int] = None) -> List[Memory]:
        """
        Retrieve memories based on the configured RAG mode.
        
        Args:
            user_message: The user's latest message
            user_id: User ID
            conversation_id: Current conversation ID
            current_conv_turns: Current conversation turns (for query context)
            
        Returns:
            List of Memory instances to include in the prompt
        """
        self.last_semantic_query = None
        self.last_graph_queries = None

        if self.settings.rag_mode == RAGMode.NONE:
            # Baseline: most recent N memories
            return self.repository.get_user_active_memories(
                user_id, self.settings.max_memories
            )
        
        if self.settings.rag_mode == RAGMode.SEMANTIC:
            # Generate optimized query, then semantic retrieval
            query = self._generate_semantic_query(
                user_message, current_conv_turns, conversation_id, turn_id
            )
            self.last_semantic_query = query
            
            from rag.semantic import SemanticRetriever
            if isinstance(self.retriever, SemanticRetriever):
                return self.retriever.retrieve_db_memories(
                    query=query,
                    limit=self.settings.max_memories,
                    user_id=user_id
                )
            return []
        
        if self.settings.rag_mode == RAGMode.GRAPH:
            # Generate multiple queries, then graph retrieval
            queries = self._generate_graph_queries(user_message, current_conv_turns, conversation_id, turn_id)
            self.last_graph_queries = queries
            
            if self.retriever:
                result = self.retriever.retrieve(
                    query=user_message, 
                    limit=self.settings.max_memories, 
                    user_id=user_id, 
                    queries=queries
                )

                # Convert RAG memories back to DB Memory objects
                db_memories = []
                for rag_mem in result.memories:
                    db_id = rag_mem.metadata.get("db_id") if rag_mem.metadata else None
                    if db_id:
                        with self.repository.get_session() as session:
                            mem = session.get(Memory, db_id)
                            if mem:
                                session.expunge(mem)
                                db_memories.append(mem)
                return db_memories
            return []
        
        return []
    
    def send_message(self, user_message: str) -> str:
        """
        Send a message and get a response.
        
        This is the main method that:
        1. Retrieves active memories from the database
        2. Gets conversation history (limited, excluding current conversation)
        3. Gets current conversation turns (unlimited)
        4. Builds the prompt using the XML template
        5. Calls the LLM
        6. Persists everything to database
        
        Args:
            user_message: The user's message
            
        Returns:
            The assistant's response
            
        Raises:
            ValueError: If no conversation is active
        """
        if not self.current_user:
            raise ValueError("No active user. Call start_conversation first.")
        
        if not self.current_conversation:
            raise ValueError("No active conversation. Call start_conversation first.")
        
        user_id = self.current_user.id
        conversation_id = self.current_conversation.id
        
        # Get next turn number
        last_turn_number = self.repository.get_last_turn_number(user_id)
        user_turn_number = last_turn_number + 1
        assistant_turn_number = user_turn_number + 1
        
        # Add user turn to database (with conversation_id)
        user_turn = self.repository.add_turn(
            user_id=user_id,
            role="user",
            content=user_message,
            turn_number=user_turn_number,
            conversation_id=conversation_id
        )
        
        # Get current conversation turns (no limit for current conversation)
        current_conv_turns = self.repository.get_conversation_turns(conversation_id)
        
        # Remove the user turn we just added
        current_conv_turns = [t for t in current_conv_turns if t.id != user_turn.id]
        
        # Retrieve memories using the configured RAG mode
        memories = self._retrieve_memories(
            user_message=user_message,
            user_id=user_id,
            conversation_id=conversation_id,
            current_conv_turns=current_conv_turns,
            turn_id=user_turn.id
        )
        self.last_used_memories = memories
        
        # Get historical turns from past conversations (with limit)
        historical_turns = self.repository.get_user_turns(
            user_id=user_id,
            limit=self.settings.max_conversation_turns,
            exclude_conversation_id=conversation_id
        )
        
        # Truncate only historical turns to fit token limit
        # (current conversation turns are always included in full)
        historical_turns = self._truncate_history_to_fit(
            historical_turns=historical_turns,
            current_turns=current_conv_turns,
            user_message=user_message,
            memories=memories,
            max_tokens=self.settings.max_prompt_tokens
        )
        
        # Build messages
        messages, prompt_string = self._build_messages(
            user_message=user_message,
            historical_turns=historical_turns,
            current_turns=current_conv_turns,
            memories=memories
        )
        
        # Call the agent
        response = self.agent.generate(
            messages=messages,
            max_tokens=self.settings.max_response_tokens,
            temperature=0.7
        )
        
        # Add assistant turn to database (with conversation_id)
        assistant_turn = self.repository.add_turn(
            user_id=user_id,
            role="assistant",
            content=response.content,
            turn_number=assistant_turn_number,
            conversation_id=conversation_id
        )
        
        # Log the inference
        self.repository.add_inference(
            prompt=prompt_string,
            response=response.content,
            model=response.model,
            provider=response.provider,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            turn_id=assistant_turn.id,
            conversation_id=conversation_id,
            task="conversation"
        )
        
        return response.content
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Turn]:
        """
        Get the current conversation history.
        
        Args:
            limit: Maximum turns to return
            
        Returns:
            List of Turn instances
        """
        if not self.current_user:
            return []
        
        return self.repository.get_user_turns(user_id=self.current_user.id, limit=limit)
    
    def get_current_conversation_turns(self) -> List[Turn]:
        """
        Get turns from the current active conversation only.
        
        Returns:
            List of Turn instances for the current conversation
        """
        if not self.current_conversation:
            return []
        
        return self.repository.get_conversation_turns(self.current_conversation.id)
    
    def get_memories(self, conversation_id: Optional[int] = None) -> List[Memory]:
        """
        Get memories for a conversation.
        
        Args:
            conversation_id: Specific conversation ID, or None for all user memories
            
        Returns:
            List of Memory instances
        """
        if conversation_id:
            return self.repository.get_memories(conversation_id)
        
        if not self.current_user:
            return []
        
        return self.repository.get_user_memories(self.current_user.id)