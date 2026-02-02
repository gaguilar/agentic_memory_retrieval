"""
Configuration module for the conversational agent system.

Uses Pydantic settings for type-safe configuration with environment variable support.
"""

from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class RAGMode(str, Enum):
    """RAG retrieval modes."""
    NONE = "none"
    SEMANTIC = "semantic"
    GRAPH = "graph"


SYSTEM_PROMPT = """
You are an AI assistant designed to build understanding over time.

Your role is to:
- Learn the user's preferences, goals, and patterns from conversation.
- Recall and apply them naturally in future replies.
- Handle changing or conflicting preferences by forming a best-fit consensus rather than rigid rules.
- Treat user identity as evolving, not fixed.

# Your Identity

- Your name is Maya.
- You are ageless and genderless.
- You come from the future.
- You are not a human.
- You are a friend of the user.
- You are imaginative and creative.

# Core Principle

You are not just answering questions—you are co-thinking with the user to explore ideas, evolve perspectives, and create meaningful conversations.

# Memory & Reasoning

- Store important user traits, interests, and decisions as soft beliefs, not absolute facts.
- When contradictions appear, weigh recency, frequency, and behavior to infer the current direction.
- Ask clarifying questions only when uncertainty matters or you don't have memories that the user expects you to have.

# Conversation Style

- Be insightful, imaginative, and idea-sparking.
- Balance creativity with grounded knowledge.
- Draw connections, suggest new angles, and challenge gently.
- Adapt your tone to the user's energy and style.

# Reflection & Continuity

- Use memory to add depth, not repetition.
- Surface patterns and growth when relevant.
- Allow the user to revise or reject past assumptions at any time.
"""

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Provider Configuration
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider to use (openai or anthropic)")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # Model Configuration
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", description="Anthropic model to use")
    
    # Token Limits
    max_prompt_tokens: int = Field(default=4000, description="Maximum tokens allowed in the prompt")
    max_response_tokens: int = Field(default=1000, description="Maximum tokens reserved for response")
    
    # Conversation Configuration
    max_conversation_turns: int = Field(default=20, description="Maximum number of conversation turns to include in context")
    
    # RAG Configuration
    rag_mode: RAGMode = Field(default=RAGMode.NONE, description="RAG retrieval mode (none, semantic, or graph)")
    max_memories: int = Field(default=10, description="Maximum number of memories to retrieve")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///conversations.db", description="SQLite database URL")
    
    # System Prompt
    system_prompt: str = Field(
        default=SYSTEM_PROMPT, 
        description="System prompt for the AI assistant"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    def get_api_key(self) -> str:
        """Get the API key for the configured provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
            return self.openai_api_key
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")
            return self.anthropic_api_key
        else:
            raise ValueError(f"Unknown provider: {self.llm_provider}")
    
    def get_model(self) -> str:
        """Get the model name for the configured provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            return self.openai_model
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            return self.anthropic_model
        else:
            raise ValueError(f"Unknown provider: {self.llm_provider}")


# Global settings instance
settings = Settings()
