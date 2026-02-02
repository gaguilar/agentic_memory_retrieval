"""
OpenAI LLM agent implementation.

Uses the OpenAI SDK to interact with GPT models.
"""

from typing import List
from openai import OpenAI

from agents.base import BaseAgent, Message, AgentResponse


class OpenAIAgent(BaseAgent):
    """
    OpenAI GPT agent implementation.
    
    Supports GPT-4, GPT-4o, GPT-3.5-turbo and other OpenAI chat models.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI agent.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
        """
        super().__init__(api_key, model)
        self.client = OpenAI(api_key=api_key)
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"
    
    def generate(
        self,
        messages: List[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AgentResponse:
        """
        Generate a response using OpenAI's API.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            **kwargs: Additional parameters passed to the API
            
        Returns:
            AgentResponse with the generated content and metadata
        """
        # Convert messages to OpenAI format
        formatted_messages = self.format_messages(messages)
        
        # Make the API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Extract the response content
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # Extract token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        
        return AgentResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=response
        )
