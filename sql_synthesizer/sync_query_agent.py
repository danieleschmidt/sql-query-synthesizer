"""Synchronous query agent for natural language to SQL translation with schema discovery."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine

from .cache import CacheError, create_cache_backend
from .config import config
from .database import DatabaseConnectionManager
from .openai_adapter import OpenAIAdapter
from .services.query_service import QueryService
from .services.query_validator_service import QueryValidatorService
from .services.sql_generator_service import SQLGeneratorService
from .types import QueryResult

logger = logging.getLogger(__name__)


class QueryAgent:
    """Synchronous query agent for natural language to SQL with enhanced database operations."""

    def __init__(
        self,
        database_url: str,
        *,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        openai_timeout: Optional[float] = None,
        schema_cache_ttl: int = 3600,
        query_cache_ttl: int = 0,
        max_rows: int = 5,
        enable_structured_logging: bool = False,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: float = 60.0,
        max_page_size: int = 1000,
    ):
        """Initialize the query agent.

        Args:
            database_url: Database connection URL
            openai_api_key: OpenAI API key for LLM-based SQL generation
            openai_model: OpenAI model to use
            openai_timeout: Timeout for OpenAI requests
            schema_cache_ttl: Time to live for schema cache in seconds
            query_cache_ttl: Time to live for query cache in seconds (0 = disabled)
            max_rows: Maximum rows to return for queries
            enable_structured_logging: Enable structured logging with trace IDs
            circuit_breaker_failure_threshold: Circuit breaker failure threshold
            circuit_breaker_recovery_timeout: Circuit breaker recovery timeout
            max_page_size: Maximum allowed page size for pagination
        """
        self.database_url = database_url

        # Create sync engine
        engine_kwargs = {"echo": False}

        # Only add pool parameters for databases that support them
        if not self.database_url.startswith("sqlite://"):
            engine_kwargs.update(
                {
                    "pool_size": config.db_pool_size,
                    "max_overflow": config.db_max_overflow,
                    "pool_recycle": config.db_pool_recycle,
                    "pool_pre_ping": config.db_pool_pre_ping,
                }
            )

        self.engine = create_engine(self.database_url, **engine_kwargs)

        # Initialize caches using configured backend
        try:
            cache_backend_config = {
                "redis_host": config.redis_host,
                "redis_port": config.redis_port,
                "redis_db": config.redis_db,
                "redis_password": config.redis_password,
                "memcached_servers": config.memcached_servers,
                "max_size": config.cache_max_size,
            }

            # Create schema cache (longer TTL)
            self.schema_cache = create_cache_backend(
                backend_type=config.cache_backend,
                ttl=schema_cache_ttl,
                **cache_backend_config,
            )

            # Create query cache (shorter TTL)
            self.query_cache = create_cache_backend(
                backend_type=config.cache_backend,
                ttl=query_cache_ttl,
                **cache_backend_config,
            )

            logger.info(
                f"Initialized {config.cache_backend} cache backend for sync agent"
            )

        except (CacheError, ValueError) as e:
            logger.warning(
                f"Failed to initialize {config.cache_backend} cache backend: {e}"
            )
            logger.info("Falling back to memory cache backend")

            # Fallback to in-memory cache
            self.schema_cache = create_cache_backend(
                "memory", ttl=schema_cache_ttl, max_size=100
            )
            self.query_cache = create_cache_backend(
                "memory", ttl=query_cache_ttl, max_size=1000
            )

        # Initialize services
        self.validator = QueryValidatorService()

        # Initialize LLM provider if API key provided
        llm_provider = None
        if openai_api_key:
            llm_provider = OpenAIAdapter(
                api_key=openai_api_key,
                model=openai_model,
                timeout=openai_timeout,
                circuit_breaker_failure_threshold=circuit_breaker_failure_threshold,
                circuit_breaker_recovery_timeout=circuit_breaker_recovery_timeout,
            )

        self.generator = SQLGeneratorService(llm_provider=llm_provider)

        # Initialize query service
        self.query_service = QueryService(
            engine=self.engine,
            validator=self.validator,
            generator=self.generator,
            schema_cache=self.schema_cache,
            query_cache=self.query_cache,
            max_rows=max_rows,
            enable_structured_logging=enable_structured_logging,
            max_page_size=max_page_size,
        )

        # Database connection manager for health checks
        self.connection_manager = DatabaseConnectionManager(database_url)

        # Store configuration for tests
        self._structured_logging = enable_structured_logging
        self._max_rows = max_rows

    def query(
        self, question: str, *, explain: bool = False, trace_id: Optional[str] = None
    ) -> QueryResult:
        """Process a natural language question and return results.

        Args:
            question: The user's natural language question
            explain: Whether to return query execution plan
            trace_id: Optional trace ID for request correlation

        Returns:
            QueryResult: The query result containing SQL, data, and explanation
        """
        return self.query_service.query(question, explain=explain, trace_id=trace_id)

    def execute_sql(
        self, sql: str, *, explain: bool = False, trace_id: Optional[str] = None
    ) -> QueryResult:
        """Execute raw SQL and return results.

        Args:
            sql: The SQL statement to execute
            explain: Whether to return query execution plan
            trace_id: Optional trace ID for request correlation

        Returns:
            QueryResult: The query result containing SQL, data, and explanation
        """
        return self.query_service.execute_sql(sql, explain=explain, trace_id=trace_id)

    def query_paginated(
        self,
        question: str,
        *,
        page: int = 1,
        page_size: int = None,
        trace_id: str = None,
    ) -> QueryResult:
        """Process a natural language question and return paginated results.

        Args:
            question: The user's natural language question
            page: Page number (1-based)
            page_size: Number of items per page (defaults to max_rows)
            trace_id: Optional trace ID for request correlation

        Returns:
            QueryResult with pagination information
        """
        if page_size is None:
            page_size = self.query_service.max_rows

        # First get the SQL from natural language
        available_tables = self.query_service.discover_schema()
        sql = self.generator.generate_sql(question, available_tables)
        validated_sql = self.validator.validate_sql(sql)

        # Execute with pagination
        return self.query_service.query_paginated(
            validated_sql, page, page_size, trace_id=trace_id
        )

    def execute_sql_paginated(
        self, sql: str, *, page: int = 1, page_size: int = None, trace_id: str = None
    ) -> QueryResult:
        """Execute raw SQL and return paginated results.

        Args:
            sql: The SQL statement to execute
            page: Page number (1-based)
            page_size: Number of items per page (defaults to max_rows)
            trace_id: Optional trace ID for request correlation

        Returns:
            QueryResult with pagination information
        """
        if page_size is None:
            page_size = self.query_service.max_rows

        return self.query_service.query_paginated(
            sql, page, page_size, trace_id=trace_id
        )

    def get_table_names(self) -> List[str]:
        """Get list of available table names."""
        return self.query_service.discover_schema()

    def discover_schema(self) -> List[str]:
        """Get list of available table names (alias for get_table_names)."""
        return self.get_table_names()

    def describe_table(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific table."""
        return self.query_service.describe_table(table_name)

    def row_count(self, table_name: str) -> int:
        """Get row count for a single table."""
        # Validate table name to prevent SQL injection
        available_tables = self.get_table_names()
        if table_name not in available_tables:
            raise ValueError(
                f"Table '{table_name}' not found in database. Available tables: {available_tables}"
            )

        result = self.execute_sql(f"SELECT COUNT(*) as count FROM {table_name}")
        return result.data[0]["count"] if result.data else 0

    def batch_row_counts(self, table_names: List[str]) -> Dict[str, int]:
        """Get row counts for multiple tables."""
        counts = {}
        for table_name in table_names:
            try:
                counts[table_name] = self.row_count(table_name)
            except ValueError:
                counts[table_name] = 0
        return counts

    def list_table_counts(self) -> List[tuple[str, int]]:
        """Get list of tables with their row counts."""
        tables = self.get_table_names()
        counts = self.batch_row_counts(tables)
        return [(table, counts[table]) for table in tables]

    def generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question."""
        available_tables = self.discover_schema()
        return self.generator.generate_sql(question, available_tables)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.schema_cache.clear()
        self.query_cache.clear()
        logger.info("All caches cleared")

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        return self.connection_manager.health_check()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return self.connection_manager.get_connection_stats()

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            if hasattr(self, "engine"):
                self.engine.dispose()
            if hasattr(self, "connection_manager"):
                self.connection_manager.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
