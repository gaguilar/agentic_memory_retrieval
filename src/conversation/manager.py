"""
Conversation manager that orchestrates the agent, RAG, and database.

This is the core component that ties everything together:
- Manages conversation state and history
- Builds prompts within token limits
- Coordinates RAG retrieval
- Logs inference calls
"""

from typing import List, Optional, Tuple
import json

from config import Settings, LLMProvider, RAGMode
from agents.base import BaseAgent, Message, AgentResponse
from agents.openai_agent import OpenAIAgent
from agents.anthropic_agent import AnthropicAgent
from rag.base import BaseRetriever, RetrievalResult
from rag.noop import NoOpRetriever
from database.repository import Repository
from database.models import Turn, User
from utils.tokens import TokenCounter


class ConversationManager:
    """
    Manages conversations between users and the AI agent.
    
    Responsibilities:
    - Maintain conversation state
    - Build prompts with history and memories
    - Enforce token limits
    - Persist all interactions to database
    - Log inference calls and memory context
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        agent: Optional[BaseAgent] = None,
        retriever: Optional[BaseRetriever] = None,
        repository: Optional[Repository] = None
    ):
        """
        Initialize the conversation manager.
        
        Args:
            settings: Application settings (uses global if not provided)
            agent: LLM agent (created from settings if not provided)
            retriever: RAG retriever (created from settings if not provided)
            repository: Database repository (created from settings if not provided)
        """
        from config import settings as default_settings
        self.settings = settings or default_settings
        
        # Initialize components
        self.agent = agent or self._create_agent()
        self.retriever = retriever or self._create_retriever()
        self.repository = repository or Repository(self.settings.database_url)
        
        # Current conversation state
        self.current_user: Optional[User] = None
    
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
    
    def _create_retriever(self) -> BaseRetriever:
        """Create a retriever based on settings."""
        if self.settings.rag_mode == RAGMode.NONE:
            return NoOpRetriever()
        elif self.settings.rag_mode == RAGMode.SEMANTIC:
            # TODO: Implement semantic retriever
            return NoOpRetriever()
        elif self.settings.rag_mode == RAGMode.GRAPH:
            # TODO: Implement graph retriever
            return NoOpRetriever()
        else:
            return NoOpRetriever()
    
    def start_conversation(self, username: str) -> User:
        """
        Start a new conversation for a user.
        
        Args:
            username: Username for the conversation
            
        Returns:
            User instance
        """
        self.current_user = self.repository.get_or_create_user(username)
        return self.current_user
    
    def _build_messages(
        self,
        user_message: str,
        history: List[Turn],
        memories: RetrievalResult
    ) -> Tuple[List[Message], str]:
        """
        Build the message list for the LLM.
        
        Args:
            user_message: Current user message
            history: Conversation history (turns)
            memories: Retrieved memories
            
        Returns:
            Tuple of (messages list, formatted prompt string for logging)
        """
        messages = []
        
        # Start with system prompt
        system_content = self.settings.system_prompt
        
        # Add memories to system prompt if available
        if not memories.is_empty:
            memory_text = memories.format_for_prompt()
            system_content = f"{system_content}\n\n{memory_text}"
        
        messages.append(Message(role="system", content=system_content))
        
        # Add conversation history
        for turn in history:
            messages.append(Message(role=turn.role, content=turn.content))
        
        # Add current user message
        messages.append(Message(role="user", content=user_message))
        
        # Build prompt string for logging
        prompt_parts = []
        for msg in messages:
            prompt_parts.append(f"[{msg.role.upper()}]\n{msg.content}")
        prompt_string = "\n\n".join(prompt_parts)
        
        return messages, prompt_string
    
    def _truncate_history_to_fit(
        self,
        history: List[Turn],
        user_message: str,
        memories: RetrievalResult,
        max_tokens: int
    ) -> List[Turn]:
        """
        Truncate conversation history to fit within token limit.
        
        Args:
            history: Full conversation history
            user_message: Current user message
            memories: Retrieved memories
            max_tokens: Maximum tokens for prompt
            
        Returns:
            Truncated history that fits within limit
        """
        provider = self.settings.llm_provider.value
        model = self.settings.get_model()
        
        # Calculate base tokens (system prompt + memories + current message)
        system_content = self.settings.system_prompt
        if not memories.is_empty:
            system_content = f"{system_content}\n\n{memories.format_for_prompt()}"
        
        base_tokens = TokenCounter.estimate_message_tokens(
            "system", system_content, model, provider
        )
        base_tokens += TokenCounter.estimate_message_tokens(
            "user", user_message, model, provider
        )
        
        # Reserve tokens for response
        available_tokens = max_tokens - base_tokens - self.settings.max_response_tokens
        
        if available_tokens <= 0:
            return []
        
        # Add history from most recent, stopping when we hit the limit
        truncated = []
        tokens_used = 0
        
        for turn in reversed(history):
            turn_tokens = TokenCounter.estimate_message_tokens(
                turn.role, turn.content, model, provider
            )
            
            if tokens_used + turn_tokens > available_tokens:
                break
            
            truncated.insert(0, turn)
            tokens_used += turn_tokens
        
        return truncated
    
    def send_message(self, user_message: str) -> str:
        """
        Send a message and get a response.
        
        This is the main method that:
        1. Retrieves memories via RAG
        2. Gets conversation history
        3. Builds the prompt
        4. Calls the LLM
        5. Persists everything to database
        
        Args:
            user_message: The user's message
            
        Returns:
            The assistant's response
            
        Raises:
            ValueError: If no conversation is active
        """
        if not self.current_user:
            raise ValueError("No active user. Call start_conversation first.")
        
        user_id = self.current_user.id
        
        # Get next turn number
        last_turn_number = self.repository.get_last_turn_number(user_id)
        user_turn_number = last_turn_number + 1
        assistant_turn_number = user_turn_number + 1
        
        # Add user turn to database
        user_turn = self.repository.add_turn(
            user_id=user_id,
            role="user",
            content=user_message,
            turn_number=user_turn_number
        )
        
        # Retrieve memories
        memories = self.retriever.retrieve(
            query=user_message,
            limit=self.settings.max_memories
        )
        
        # Get conversation history (excluding current message)
        history = self.repository.get_user_turns(
            user_id=user_id,
            limit=self.settings.max_conversation_turns
        )
        # Remove the user turn we just added
        history = [t for t in history if t.id != user_turn.id]
        
        # Truncate history to fit token limit
        history = self._truncate_history_to_fit(
            history=history,
            user_message=user_message,
            memories=memories,
            max_tokens=self.settings.max_prompt_tokens
        )
        
        # Build messages
        messages, prompt_string = self._build_messages(
            user_message=user_message,
            history=history,
            memories=memories
        )
        
        # Call the agent
        response = self.agent.generate(
            messages=messages,
            max_tokens=self.settings.max_response_tokens,
            temperature=0.7
        )
        
        # Add assistant turn to database
        assistant_turn = self.repository.add_turn(
            user_id=user_id,
            role="assistant",
            content=response.content,
            turn_number=assistant_turn_number
        )
        
        # Log the inference
        inference_log = self.repository.add_inference(
            turn_id=assistant_turn.id,
            prompt=prompt_string,
            response=response.content,
            model=response.model,
            provider=response.provider,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens
        )
        
        # Log memories on context
        if not memories.is_empty:
            self.repository.add_prompt_memories(
                inference_id=inference_log.id,
                formatted_memories=memories.format_for_prompt(),
                retriever_type=memories.retriever_type
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
        
        return self.repository.get_user_turns(
            user_id=self.current_user.id,
            limit=limit
        )
