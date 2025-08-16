"""Async SQL generation service using LLM providers."""

import logging
from typing import Any, List, Optional

from .. import metrics
from ..llm_interface import LLMProvider, ProviderError

logger = logging.getLogger(__name__)


class AsyncSQLGeneratorService:
    """Async service for generating SQL from natural language questions."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize the async SQL generator service.

        Args:
            llm_provider: Optional async LLM provider for SQL generation
        """
        self.llm_provider = llm_provider

    async def generate_sql(
        self, question: str, available_tables: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        """Generate SQL from natural language question asynchronously.

        Args:
            question: The user's natural language question
            available_tables: List of available table names
            **kwargs: Additional parameters for the LLM provider

        Returns:
            str: Generated SQL query
        """
        if not self.llm_provider:
            # Fallback to naive SQL generation without LLM
            return self._generate_naive_sql(question, available_tables)

        try:
            # Use async LLM provider to generate SQL
            sql = await self.llm_provider.generate_sql(
                question, available_tables=available_tables, **kwargs
            )

            # Clean up the generated SQL
            sql = sql.strip()
            if sql.endswith(";"):
                sql = sql[:-1]

            logger.info(
                f"Generated SQL via {self.llm_provider.get_provider_name()}: {sql[:100]}..."
            )
            return sql

        except ProviderError as e:
            logger.warning(f"LLM provider failed, falling back to naive SQL: {e}")
            metrics.record_sql_generation_fallback(
                self.llm_provider.get_provider_name()
            )
            return self._generate_naive_sql(question, available_tables)
        except Exception as e:
            logger.error(f"Unexpected error in SQL generation: {e}")
            metrics.record_sql_generation_error("unexpected_error")
            return self._generate_naive_sql(question, available_tables)

    def _generate_naive_sql(
        self, question: str, available_tables: Optional[List[str]] = None
    ) -> str:
        """Generate naive SQL when LLM is unavailable.

        Args:
            question: The user's natural language question
            available_tables: List of available table names

        Returns:
            str: Naive SQL query or comment
        """
        # Simple pattern matching for basic queries
        question_lower = question.lower().strip()

        # Try to identify table from question or use first available table
        target_table = None
        if available_tables:
            for table in available_tables:
                if table.lower() in question_lower:
                    target_table = table
                    break

            # Use first table if no specific table mentioned
            if not target_table:
                target_table = available_tables[0]

        # Generate simple queries based on keywords
        if target_table:
            if any(
                word in question_lower for word in ["count", "how many", "number of"]
            ):
                return f"SELECT COUNT(*) FROM {target_table}"
            elif any(word in question_lower for word in ["all", "show", "list", "get"]):
                return f"SELECT * FROM {target_table}"
            elif "distinct" in question_lower:
                # Try to identify column name
                words = question_lower.split()
                for i, word in enumerate(words):
                    if word == "distinct" and i + 1 < len(words):
                        column = words[i + 1].rstrip(".,?!")
                        return f"SELECT DISTINCT {column} FROM {target_table}"
                return f"SELECT DISTINCT * FROM {target_table}"
            else:
                return f"SELECT * FROM {target_table}"
        else:
            # No tables available or identified
            return f"-- Unable to generate SQL: {question[:50]}..."

    def get_provider_info(self) -> dict:
        """Get information about the current LLM provider.

        Returns:
            dict: Provider information
        """
        if not self.llm_provider:
            return {"provider": "none", "model": "naive_fallback", "status": "active"}

        return {
            "provider": self.llm_provider.get_provider_name(),
            "model": self.llm_provider.get_model_name(),
            "status": "active",
        }

    def is_llm_available(self) -> bool:
        """Check if LLM provider is available.

        Returns:
            bool: True if LLM provider is available and functional
        """
        if not self.llm_provider:
            return False

        # Check if provider is available (could add health check here)
        try:
            # For OpenAI adapter, check circuit breaker status
            if hasattr(self.llm_provider, "circuit_breaker"):
                return self.llm_provider.circuit_breaker.is_request_allowed()
            return True
        except AttributeError:
            # Circuit breaker not available or not properly configured
            return False
        except Exception:
            # Any other unexpected error in circuit breaker check
            return False
