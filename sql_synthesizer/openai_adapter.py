"""OpenAI implementation of the LLM provider interface."""

from __future__ import annotations

import time
from typing import Dict, Any, Optional

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional
    openai = None

from . import metrics
from .llm_interface import LLMProvider, ProviderError, ProviderTimeoutError, ProviderAuthenticationError
from .circuit_breaker import CircuitBreaker
from .user_experience import (
    create_openai_package_missing_error,
    create_empty_question_error,
)


class OpenAIAdapter(LLMProvider):
    """OpenAI implementation of the LLM provider interface."""

    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-3.5-turbo", 
        timeout: float | None = None,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: float = 60.0
    ) -> None:
        if not openai:
            raise create_openai_package_missing_error()
        openai.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        
        # Initialize circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            provider_name="openai",
            failure_threshold=circuit_breaker_failure_threshold,
            recovery_timeout=circuit_breaker_recovery_timeout
        )

    def generate_sql(
        self, 
        question: str, 
        available_tables: Optional[list[str]] = None,
        **kwargs: Any
    ) -> str:
        """Generate SQL with proper schema context and safety constraints."""
        # Sanitize input question
        question = question.strip()
        if not question:
            raise create_empty_question_error()
        
        # Check circuit breaker before making request
        if not self.circuit_breaker.is_request_allowed():
            metrics.record_openai_request(0, "circuit_breaker_open")
            raise ProviderError(
                f"OpenAI service temporarily unavailable (circuit breaker open). "
                f"Failures: {self.circuit_breaker.failure_count}/{self.circuit_breaker.failure_threshold}",
                "openai"
            )
        
        # Build secure prompt with table context
        table_context = ""
        if available_tables:
            table_list = ", ".join(available_tables)
            table_context = f"\nAvailable tables: {table_list}\n"
        
        prompt = (
            "You are a SQL assistant. Generate ONLY safe SELECT queries based on the following constraints:\n"
            "- Use only SELECT statements\n"
            "- Use only the provided table names\n"
            "- Do not use multiple statements or semicolons\n"
            "- Do not use DROP, DELETE, UPDATE, INSERT, or other destructive operations\n"
            f"{table_context}"
            f"Translate this natural language request into a single SQL SELECT query:\n"
            f"{question}\n\n"
            "Return only the SQL query without any explanation or additional text:\n"
        )
        
        start_time = time.time()
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=self.timeout,
            )
            duration = time.time() - start_time
            
            # Record success in circuit breaker and metrics
            self.circuit_breaker.record_success()
            metrics.record_openai_request(duration, "success")
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failure in circuit breaker
            self.circuit_breaker.record_failure()
            
            # Check exception type by name to handle import/mocking issues
            exception_name = type(e).__name__
            
            if "Timeout" in exception_name:
                metrics.record_openai_request(duration, "timeout")
                raise ProviderTimeoutError(f"OpenAI request timed out: {e}", "openai")
            elif "Authentication" in exception_name:
                metrics.record_openai_request(duration, "auth_error")
                raise ProviderAuthenticationError(f"OpenAI authentication failed: {e}", "openai")
            else:
                metrics.record_openai_request(duration, "error")
                raise ProviderError(f"OpenAI request failed: {e}", "openai")
    
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return "openai"
    
    def get_model_name(self) -> str:
        """Get the specific model being used."""
        return self.model
    
    def validate_configuration(self) -> bool:
        """Validate that the OpenAI provider is properly configured."""
        if not openai:
            return False
        if not self.api_key:
            return False
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get OpenAI provider capabilities and limitations."""
        return {
            "max_tokens": 4096 if "gpt-3.5" in self.model else 8192,
            "supports_json_mode": True,
            "supports_function_calling": True,
            "temperature_range": (0.0, 2.0),
            "model": self.model,
            "timeout": self.timeout,
            "circuit_breaker": self.circuit_breaker.get_status()
        }
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring and debugging."""
        return self.circuit_breaker.get_status()
