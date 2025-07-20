"""Abstract interface for LLM providers used to generate SQL."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers that generate SQL from natural language."""
    
    @abstractmethod
    def generate_sql(
        self, 
        question: str, 
        available_tables: Optional[list[str]] = None,
        **kwargs: Any
    ) -> str:
        """Generate SQL from a natural language question.
        
        Args:
            question: The user's natural language question
            available_tables: List of available table names for context
            **kwargs: Provider-specific additional parameters
            
        Returns:
            str: A SQL SELECT query
            
        Raises:
            UserExperienceError: If the question is invalid or unsafe
            ProviderError: If the LLM provider encounters an error
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider.
        
        Returns:
            str: Provider name (e.g., "openai", "anthropic", "google")
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the specific model being used.
        
        Returns:
            str: Model name (e.g., "gpt-3.5-turbo", "claude-3-sonnet")
        """
        pass
    
    def validate_configuration(self) -> bool:
        """Validate that the provider is properly configured.
        
        Returns:
            bool: True if configuration is valid
        """
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and limitations.
        
        Returns:
            Dict containing provider capabilities like max_tokens, 
            supported_features, etc.
        """
        return {}


class ProviderError(Exception):
    """Base exception for LLM provider errors."""
    
    def __init__(self, message: str, provider: str, error_code: Optional[str] = None):
        self.provider = provider
        self.error_code = error_code
        super().__init__(message)


class ProviderTimeoutError(ProviderError):
    """Exception raised when LLM provider times out."""
    pass


class ProviderRateLimitError(ProviderError):
    """Exception raised when LLM provider rate limit is exceeded."""
    pass


class ProviderAuthenticationError(ProviderError):
    """Exception raised when LLM provider authentication fails."""
    pass