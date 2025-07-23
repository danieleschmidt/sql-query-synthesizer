"""Async query agent for natural language to SQL translation with schema discovery."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import inspect, text

from .services.async_query_service import AsyncQueryService
from .services.query_validator_service import QueryValidatorService  
from .services.async_sql_generator_service import AsyncSQLGeneratorService
from .async_openai_adapter import AsyncOpenAIAdapter
from .cache import TTLCache
from .types import QueryResult
from .database import DatabaseConnectionManager
from .config import config
from .logging_utils import configure_logging

logger = logging.getLogger(__name__)


class AsyncQueryAgent:
    """Async query agent for natural language to SQL with enhanced database operations."""

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
        """Initialize the async query agent.
        
        Args:
            database_url: Database connection URL (must be async-compatible)
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
        # Convert sync database URL to async if needed
        if database_url.startswith("sqlite://"):
            self.database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
        elif database_url.startswith("postgresql://"):
            self.database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        elif database_url.startswith("mysql://"):
            self.database_url = database_url.replace("mysql://", "mysql+aiomysql://")
        else:
            self.database_url = database_url

        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            pool_size=config.db_pool_size,
            max_overflow=config.db_max_overflow,
            pool_recycle=config.db_pool_recycle,
            pool_pre_ping=config.db_pool_pre_ping,
            echo=False
        )

        # Initialize caches
        self.schema_cache = TTLCache(maxsize=100, ttl=schema_cache_ttl)
        self.query_cache = TTLCache(maxsize=1000, ttl=query_cache_ttl)

        # Initialize services
        self.validator = QueryValidatorService()

        # Initialize async LLM provider if API key provided
        llm_provider = None
        if openai_api_key:
            llm_provider = AsyncOpenAIAdapter(
                api_key=openai_api_key,
                model=openai_model,
                timeout=openai_timeout,
                circuit_breaker_failure_threshold=circuit_breaker_failure_threshold,
                circuit_breaker_recovery_timeout=circuit_breaker_recovery_timeout,
            )

        self.generator = AsyncSQLGeneratorService(llm_provider=llm_provider)

        # Initialize async query service
        self.query_service = AsyncQueryService(
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
        self.connection_manager = DatabaseConnectionManager(
            database_url=database_url,  # Use sync URL for connection manager
            pool_size=config.db_pool_size,
            max_overflow=config.db_max_overflow,
            pool_recycle=config.db_pool_recycle,
            pool_pre_ping=config.db_pool_pre_ping,
            connect_retries=config.db_connect_retries,
            retry_delay=config.db_retry_delay,
        )

    async def query(self, question: str, *, explain: bool = False, trace_id: Optional[str] = None) -> QueryResult:
        """Process a natural language question and return results asynchronously.
        
        Args:
            question: The user's natural language question
            explain: Whether to return query execution plan
            trace_id: Optional trace ID for request correlation
            
        Returns:
            QueryResult: The query result containing SQL, data, and explanation
        """
        return await self.query_service.query(question, explain=explain, trace_id=trace_id)

    async def execute_sql(self, sql: str, *, explain: bool = False, trace_id: Optional[str] = None) -> QueryResult:
        """Execute raw SQL and return results asynchronously.
        
        Args:
            sql: The SQL statement to execute
            explain: Whether to return query execution plan
            trace_id: Optional trace ID for request correlation
            
        Returns:
            QueryResult: The query result containing SQL, data, and explanation
        """
        return await self.query_service.execute_sql(sql, explain=explain, trace_id=trace_id)

    async def query_paginated(self, question: str, *, page: int = 1, page_size: int = None, trace_id: str = None) -> QueryResult:
        """Process a natural language question and return paginated results asynchronously.
        
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
        available_tables = await self.query_service.discover_schema()
        sql = await self.generator.generate_sql(question, available_tables)
        validated_sql = self.validator.validate_sql(sql)

        # Execute with pagination
        return await self.query_service.query_paginated(validated_sql, page, page_size, trace_id=trace_id)

    async def execute_sql_paginated(self, sql: str, *, page: int = 1, page_size: int = None, trace_id: str = None) -> QueryResult:
        """Execute raw SQL and return paginated results asynchronously.
        
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

        return await self.query_service.query_paginated(sql, page, page_size, trace_id=trace_id)

    async def list_tables(self) -> List[str]:
        """List all available tables asynchronously.
        
        Returns:
            List[str]: List of table names
        """
        return await self.query_service.discover_schema()

    async def describe_table(self, table_name: str) -> List[Dict[str, Any]]:
        """Describe the columns of a specific table asynchronously.
        
        Args:
            table_name: Name of the table to describe
            
        Returns:
            List[Dict[str, Any]]: List of column information
        """
        async with self.engine.connect() as connection:
            # Get column information
            result = await connection.execute(text(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """))
            
            columns = []
            for row in result:
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "default": row[3]
                })
            
            return columns

    async def get_table_row_counts(self, tables: Optional[List[str]] = None) -> Dict[str, int]:
        """Get row counts for tables asynchronously.
        
        Args:
            tables: Optional list of table names (defaults to all tables)
            
        Returns:
            Dict[str, int]: Mapping of table names to row counts
        """
        if tables is None:
            tables = await self.list_tables()

        row_counts = {}
        
        # Execute all count queries concurrently
        async def get_count(table: str) -> tuple[str, int]:
            try:
                async with self.engine.connect() as connection:
                    result = await connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    return table, count
            except Exception as e:
                logger.warning(f"Failed to get row count for {table}: {e}")
                return table, 0

        # Run all count queries concurrently
        tasks = [get_count(table) for table in tables]
        results = await asyncio.gather(*tasks)
        
        for table, count in results:
            row_counts[table] = count

        return row_counts

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        return self.query_service.get_cache_stats()

    async def clear_cache(self) -> None:
        """Clear all caches asynchronously."""
        await self.query_service.clear_cache()

    async def cleanup_expired_cache_entries(self) -> Dict[str, int]:
        """Clean up expired cache entries asynchronously."""
        return await self.query_service.cleanup_expired_cache_entries()

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        # Use sync connection manager for health check
        return self.connection_manager.health_check()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get database connection statistics.
        
        Returns:
            Dict[str, Any]: Connection pool statistics
        """
        # Use sync connection manager for connection stats
        return self.connection_manager.get_connection_stats()

    async def close(self) -> None:
        """Close async database connections and cleanup resources."""
        try:
            await self.engine.dispose()
            logger.info("Async database connections closed")
        except Exception as e:
            logger.error(f"Error closing async database connections: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()