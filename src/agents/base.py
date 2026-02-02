"""
Abstract base class for LLM agents.

Defines the interface that all LLM provider implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # "system", "user", or "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}


@dataclass
class AgentResponse:
    """Response from an LLM agent."""
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: Optional[Any] = None
    
    @property
    def usage(self) -> Dict[str, int]:
        """Get token usage as a dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


class BaseAgent(ABC):
    """
    Abstract base class for LLM agents.
    
    All LLM provider implementations (OpenAI, Anthropic, etc.) must inherit
    from this class and implement the generate method.
    """
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize the agent.
        
        Args:
            api_key: API key for the provider
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider (e.g., 'openai', 'anthropic')."""
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AgentResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            AgentResponse with the generated content and metadata
        """
        pass
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert messages to list of dictionaries."""
        return [msg.to_dict() for msg in messages]
