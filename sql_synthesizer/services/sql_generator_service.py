"""SQL generation service for converting natural language to SQL."""

import logging
from typing import List, Optional

from ..generator import naive_generate_sql
from ..llm_interface import LLMProvider, ProviderError, ProviderTimeoutError, ProviderAuthenticationError
from .. import metrics

logger = logging.getLogger(__name__)


class SQLGeneratorService:
    """Service for generating SQL from natural language questions."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize the SQL generator service.
        
        Args:
            llm_provider: Optional LLM provider for AI-based generation
        """
        self.llm_provider = llm_provider

    def generate_sql(self, question: str, available_tables: List[str]) -> str:
        """Generate SQL from a natural language question.
        
        Args:
            question: The user's natural language question
            available_tables: List of available table names
            
        Returns:
            str: The generated SQL statement
        """
        logger.info(f"Generating SQL for question: {question[:50]}...")
        
        # Try LLM generation first if available
        if self.llm_provider:
            try:
                sql = self._generate_with_llm(question, available_tables)
                provider_name = self.llm_provider.get_provider_name()
                logger.info(f"Successfully generated SQL using {provider_name}")
                return sql
            except (ProviderError, ProviderTimeoutError, ProviderAuthenticationError, ValueError) as e:
                provider_name = self.llm_provider.get_provider_name()
                logger.warning(f"{provider_name} generation failed, falling back to naive: {e}")
                metrics.record_query_error("llm_generation_failed")
        
        # Fall back to naive generation
        return self._generate_naive(question, available_tables)

    def _generate_with_llm(self, question: str, available_tables: List[str]) -> str:
        """Generate SQL using LLM provider.
        
        Args:
            question: The user's natural language question
            available_tables: List of available table names
            
        Returns:
            str: The generated SQL statement
            
        Raises:
            Exception: If LLM generation fails
        """
        if not self.llm_provider:
            raise RuntimeError("LLM provider not configured")
        
        try:
            sql = self.llm_provider.generate_sql(question, available_tables)
            # Note: Duration tracking would be handled by caller
            return sql
        except (ProviderError, ProviderTimeoutError, ProviderAuthenticationError) as e:
            metrics.record_query_error("llm_api_failed")
            raise

    def _generate_naive(self, question: str, available_tables: List[str]) -> str:
        """Generate SQL using naive keyword matching.
        
        Args:
            question: The user's natural language question
            available_tables: List of available table names
            
        Returns:
            str: The generated SQL statement
        """
        logger.info("Using naive SQL generation")
        
        try:
            sql = naive_generate_sql(question, available_tables)
            # Note: Duration tracking would be handled by caller
            return sql
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Naive generation failed: {e}")
            metrics.record_query_error("naive_generation_failed")
            
            # Return a helpful error message as a SQL comment
            if not available_tables:
                return "-- No tables available in the database schema"
            else:
                tables_list = ", ".join(available_tables[:5])
                return f"-- Unable to generate SQL. Available tables: {tables_list}"