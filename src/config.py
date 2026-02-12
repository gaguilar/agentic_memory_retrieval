"""
Configuration module for the conversational agent system.

Uses Pydantic settings for type-safe configuration with environment variable support.
Loads from .env in the project root (parent of src/).
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

from prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Resolve .env path relative to project root (parent of src/)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class RAGMode(str, Enum):
    """RAG retrieval modes."""
    NONE = "none"
    SEMANTIC = "semantic"
    GRAPH = "graph"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Provider Configuration
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider to use (openai or anthropic)")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # Model Configuration
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", description="Anthropic model to use")
    
    # Token Limits
    max_prompt_tokens: int = Field(default=4000, description="Maximum tokens allowed in the prompt")
    max_response_tokens: int = Field(default=1000, description="Maximum tokens reserved for response")
    
    # Conversation Configuration
    max_conversation_turns: int = Field(default=20, description="Maximum number of conversation turns to include in context")
    
    # RAG Configuration
    rag_mode: RAGMode = Field(default=RAGMode.NONE, description="RAG retrieval mode (none, semantic, or graph)")
    max_memories: int = Field(default=10, description="Maximum number of memories to retrieve")
    
    # Semantic Retrieval Configuration
    semantic_similarity_threshold: float = Field(
        default=0.5,
        description="Cosine similarity threshold for semantic memory retrieval (0.0 to 1.0)"
    )
    
    # Memory Consolidation Configuration
    tag_similarity_threshold: float = Field(
        default=0.3,
        description="Cosine similarity threshold for tag embedding matching (0.0 to 1.0)"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name for tag embeddings"
    )
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///conversations.db", description="SQLite database URL")
    
    # System Prompt
    system_prompt: str = Field(
        default=SYSTEM_PROMPT, 
        description="System prompt for the AI assistant"
    )
    
    model_config = {
        "env_file": _ENV_FILE,
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


def log_settings(s: Settings) -> None:
    """Log the current settings values at startup (excludes API keys)."""
    logger.info(
        "Settings loaded: llm_provider=%s model=%s rag_mode=%s max_memories=%s "
        "max_prompt_tokens=%s max_response_tokens=%s max_conversation_turns=%s "
        "database_url=%s",
        s.llm_provider.value,
        s.get_model(),
        s.rag_mode.value,
        s.max_memories,
        s.max_prompt_tokens,
        s.max_response_tokens,
        s.max_conversation_turns,
        s.database_url,
    )


# Global settings instance
settings = Settings()
