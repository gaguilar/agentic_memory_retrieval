"""
Anthropic Claude agent implementation.

Uses the Anthropic SDK to interact with Claude models.
"""

from typing import List
from anthropic import Anthropic

from agents.base import BaseAgent, Message, AgentResponse


class AnthropicAgent(BaseAgent):
    """
    Anthropic Claude agent implementation.
    
    Supports Claude 3 and later models.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the Anthropic agent.
        
        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-3-5-sonnet-20241022)
        """
        super().__init__(api_key, model)
        self.client = Anthropic(api_key=api_key)
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"
    
    def generate(
        self,
        messages: List[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AgentResponse:
        """
        Generate a response using Anthropic's API.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            **kwargs: Additional parameters passed to the API
            
        Returns:
            AgentResponse with the generated content and metadata
        """
        # Separate system message from conversation messages
        system_content = ""
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                conversation_messages.append(msg.to_dict())
        
        # Ensure we have at least one message
        if not conversation_messages:
            conversation_messages = [{"role": "user", "content": "Hello"}]
        
        # Make the API call
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_content if system_content else None,
            messages=conversation_messages,
            **kwargs
        )
        
        # Extract the response content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        
        # Extract token usage
        usage = response.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        
        return AgentResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=response
        )
