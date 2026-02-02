"""
Token counting utilities.

Provides token counting for different LLM providers.
Uses tiktoken for OpenAI models and approximation for others.
"""

from typing import List, Dict, Union
import tiktoken


class TokenCounter:
    """
    Utility class for counting tokens in text.
    
    Uses tiktoken for accurate OpenAI token counts and
    character-based approximation for other providers.
    """
    
    # Cache for tokenizer encodings
    _encoders: Dict[str, tiktoken.Encoding] = {}
    
    # Approximate characters per token for non-OpenAI models
    CHARS_PER_TOKEN_APPROX = 4
    
    @classmethod
    def get_encoder(cls, model: str = "gpt-4o-mini") -> tiktoken.Encoding:
        """
        Get or create a tokenizer encoder for a model.
        
        Args:
            model: Model name to get encoder for
            
        Returns:
            tiktoken Encoding instance
        """
        if model not in cls._encoders:
            try:
                cls._encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                cls._encoders[model] = tiktoken.get_encoding("cl100k_base")
        return cls._encoders[model]
    
    @classmethod
    def count_tokens(
        cls,
        text: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai"
    ) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens in
            model: Model name for encoding
            provider: LLM provider ('openai' uses tiktoken, others use approximation)
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if provider == "openai":
            encoder = cls.get_encoder(model)
            return len(encoder.encode(text))
        else:
            # Approximate for non-OpenAI models
            return len(text) // cls.CHARS_PER_TOKEN_APPROX + 1
    
    @classmethod
    def count_messages_tokens(
        cls,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        provider: str = "openai"
    ) -> int:
        """
        Count tokens in a list of messages.
        
        Accounts for message formatting overhead in chat models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name for encoding
            provider: LLM provider
            
        Returns:
            Total token count
        """
        if not messages:
            return 0
        
        total = 0
        
        # Token overhead per message (role, formatting, etc.)
        # This is an approximation based on OpenAI's documentation
        tokens_per_message = 4  # <|im_start|>role\n...content...<|im_end|>\n
        
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                total += cls.count_tokens(str(value), model, provider)
        
        # Add overhead for the reply priming
        total += 3  # Every reply is primed with <|im_start|>assistant<|im_sep|>
        
        return total
    
    @classmethod
    def truncate_to_token_limit(
        cls,
        text: str,
        max_tokens: int,
        model: str = "gpt-4o-mini",
        provider: str = "openai"
    ) -> str:
        """
        Truncate text to fit within a token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            model: Model name for encoding
            provider: LLM provider
            
        Returns:
            Truncated text
        """
        if not text:
            return text
        
        current_tokens = cls.count_tokens(text, model, provider)
        
        if current_tokens <= max_tokens:
            return text
        
        if provider == "openai":
            encoder = cls.get_encoder(model)
            tokens = encoder.encode(text)
            truncated_tokens = tokens[:max_tokens]
            return encoder.decode(truncated_tokens)
        else:
            # Approximate truncation for non-OpenAI
            char_limit = max_tokens * cls.CHARS_PER_TOKEN_APPROX
            return text[:char_limit]
    
    @classmethod
    def estimate_message_tokens(
        cls,
        role: str,
        content: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai"
    ) -> int:
        """
        Estimate tokens for a single message.
        
        Args:
            role: Message role
            content: Message content
            model: Model name
            provider: LLM provider
            
        Returns:
            Estimated token count
        """
        # Overhead for role and message structure
        overhead = 4
        return overhead + cls.count_tokens(content, model, provider)
