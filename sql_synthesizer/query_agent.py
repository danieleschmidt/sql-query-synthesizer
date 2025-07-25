"""Refactored query agent using service layer architecture."""

import os
import time
import threading
import atexit
import logging
from typing import List, Any, Tuple, Optional, Dict
from dataclasses import dataclass, field

from sqlalchemy import (
    inspect,
    text,
    Table,
    MetaData,
    select,
    func,
    union_all,
    literal,
)
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError

from .cache import TTLCache, create_cache_backend, CacheError
from .openai_adapter import OpenAIAdapter
from .config import config
from .database import DatabaseConnectionManager
from .services import QueryValidatorService, SQLGeneratorService, QueryService

logger = logging.getLogger(__name__)


from .types import QueryResult


class QueryAgent:
    """Natural language to SQL agent with service layer architecture."""

    def __init__(
        self,
        database_url: str,
        schema_cache_ttl: int = 0,
        max_rows: int = 5,
        query_cache_ttl: int = 0,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-3.5-turbo",
        openai_timeout: float | None = None,
        enable_structured_logging: bool = False,
    ):
        """Create a new agent for the given database_url.

        Parameters
        ----------
        database_url:
            SQLAlchemy connection string.
        schema_cache_ttl:
            Time-to-live for cached table names in seconds. 0 disables caching.
        max_rows:
            Maximum number of rows returned for SELECT * queries.
        query_cache_ttl:
            Time-to-live for cached query results in seconds. 0 disables caching.
        openai_api_key:
            API key for the OpenAI service. If None (default), the agent
            falls back to naive keyword matching.
        openai_model:
            Name of the OpenAI model used to generate SQL when an API key is
            provided.
        openai_timeout:
            Request timeout in seconds for OpenAI API calls.
        enable_structured_logging:
            Enable structured logging with trace IDs and JSON formatting.
        """
        # Initialize database connection manager with pooling and error handling
        self.db_manager = DatabaseConnectionManager(database_url)
        self.engine = self.db_manager.engine  # Backward compatibility
        self._structured_logging = enable_structured_logging
        self.max_rows = max_rows

        # Configure structured logging if enabled
        if enable_structured_logging:
            from .logging_utils import configure_logging
            configure_logging(enable_json=True)

        # Initialize caches using configured backend
        try:
            cache_backend_config = {
                "redis_host": config.redis_host,
                "redis_port": config.redis_port,
                "redis_db": config.redis_db,
                "redis_password": config.redis_password,
                "memcached_servers": config.memcached_servers,
                "max_size": config.cache_max_size
            }
            
            # Create schema cache (longer TTL)
            self.schema_cache = create_cache_backend(
                backend_type=config.cache_backend,
                ttl=schema_cache_ttl,
                **cache_backend_config
            )
            
            # Create query cache (shorter TTL)
            self.query_cache = create_cache_backend(
                backend_type=config.cache_backend,
                ttl=query_cache_ttl,
                **cache_backend_config
            )
            
            logger.info(f"Initialized {config.cache_backend} cache backend")
            
        except (CacheError, ValueError) as e:
            logger.warning(f"Failed to initialize {config.cache_backend} cache backend: {e}")
            logger.info("Falling back to memory cache backend")
            
            # Fallback to in-memory cache
            self.schema_cache = create_cache_backend("memory", ttl=schema_cache_ttl, max_size=config.cache_max_size)
            self.query_cache = create_cache_backend("memory", ttl=query_cache_ttl, max_size=config.cache_max_size)

        # Set up automatic cache cleanup if TTL is enabled
        self._cleanup_timer = None
        self._setup_cache_cleanup()

        # Initialize LLM provider if API key provided
        llm_provider = None
        key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            try:
                llm_provider = OpenAIAdapter(
                    api_key=key, 
                    model=openai_model, 
                    timeout=openai_timeout,
                    circuit_breaker_failure_threshold=config.circuit_breaker_failure_threshold,
                    circuit_breaker_recovery_timeout=config.circuit_breaker_recovery_timeout
                )
            except RuntimeError:
                llm_provider = None

        # Initialize validator service (enhanced or legacy)
        if config.use_enhanced_sql_validation:
            from .services.enhanced_query_validator import EnhancedQueryValidatorService
            self.validator = EnhancedQueryValidatorService(
                max_question_length=config.max_question_length,
                allowed_tables=None,  # Will be set dynamically based on discovered schema
                allowed_columns=None   # Will be set dynamically based on discovered schema
            )
        else:
            self.validator = QueryValidatorService(max_question_length=config.max_question_length)
        self.generator = SQLGeneratorService(llm_provider=llm_provider)
        self.query_service = QueryService(
            engine=self.engine,
            validator=self.validator,
            generator=self.generator,
            schema_cache=self.schema_cache,
            query_cache=self.query_cache,
            max_rows=max_rows,
            enable_structured_logging=enable_structured_logging,
            max_page_size=config.max_page_size,
        )

        # Maintain backward compatibility attributes
        self.inspector = self.query_service.inspector
        self.openai_adapter = llm_provider  # For backward compatibility

    def clear_cache(self) -> None:
        """Empty in-memory caches for schema and query results."""
        self.query_service.clear_cache()

    def get_cache_stats(self) -> dict[str, Any]:
        """Return comprehensive cache statistics for monitoring."""
        return self.query_service.get_cache_stats()

    def cleanup_expired_cache_entries(self) -> dict[str, int]:
        """Clean up expired entries from all caches and return cleanup stats."""
        return self.query_service.cleanup_expired_cache_entries()

    def _setup_cache_cleanup(self) -> None:
        """Set up automatic cache cleanup if TTL is enabled."""
        if self.schema_cache.ttl > 0 or self.query_cache.ttl > 0:
            self._start_cleanup_timer()
            # Register cleanup on exit
            atexit.register(self._stop_cleanup_timer)

    def _start_cleanup_timer(self) -> None:
        """Start the periodic cache cleanup timer."""
        if self._cleanup_timer is None:
            interval = config.cache_cleanup_interval
            self._cleanup_timer = threading.Timer(interval, self._periodic_cleanup)
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()

    def _stop_cleanup_timer(self) -> None:
        """Stop the periodic cache cleanup timer."""
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

    def _periodic_cleanup(self) -> None:
        """Perform periodic cache cleanup and reschedule."""
        try:
            stats = self.cleanup_expired_cache_entries()
            if stats["total_cleaned"] > 0:
                logger.debug(f"Cleaned up {stats['total_cleaned']} expired cache entries")
        except AttributeError as e:
            logger.error(f"Cache cleanup failed - cache not properly initialized: {e}")
        except RuntimeError as e:
            logger.error(f"Cache cleanup failed - runtime error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during cache cleanup: {e}")
        
        # Reschedule the next cleanup
        self._cleanup_timer = None
        self._start_cleanup_timer()

    def discover_schema(self) -> List[str]:
        """Discover and return available table names."""
        return self.query_service.discover_schema()

    def row_count(self, table: str) -> int:
        """Return the row count for a given table."""
        # Validate table name
        table = self.validator.validate_table_name(table)
        
        # Execute count query
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
                return result.scalar()
        except OperationalError as e:
            # Handle database operational errors (table doesn't exist, connection issues)
            from .user_experience import create_invalid_table_error
            available_tables = self.discover_schema()
            raise create_invalid_table_error(table, available_tables)
        except DatabaseError as e:
            logger.error(f"Database error getting row count for {table}: {e}")
            raise ValueError(f"Unable to access table '{table}': database error")
        except Exception as e:
            logger.error(f"Unexpected error getting row count for {table}: {e}")
            raise RuntimeError(f"Failed to get row count for table '{table}'")

    def batch_row_counts(self, tables: List[str]) -> dict[str, int]:
        """Return row counts for multiple tables efficiently."""
        # Validate all table names first
        validated_tables = [self.validator.validate_table_name(table) for table in tables]
        
        counts = {}
        with self.engine.connect() as connection:
            for table in validated_tables:
                try:
                    result = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    counts[table] = result.scalar()
                except OperationalError as e:
                    logger.warning(f"Table access error for {table}: {e}")
                    counts[table] = 0
                except DatabaseError as e:
                    logger.warning(f"Database error for {table}: {e}")
                    counts[table] = 0
                except Exception as e:
                    logger.warning(f"Unexpected error for {table}: {e}")
                    counts[table] = 0
        return counts

    def list_table_counts(self) -> List[tuple[str, int]]:
        """Return a list of (table_name, row_count) tuples."""
        tables = self.discover_schema()
        counts = self.batch_row_counts(tables)
        return [(table, counts.get(table, 0)) for table in tables]

    def explain_sql(self, sql: str) -> List[Any]:
        """Return the execution plan for a SQL query."""
        result = self.query_service.execute_sql(sql, explain=True)
        return result.data

    def table_columns(self, table: str) -> List[Tuple[str, str]]:
        """Return column information for a table."""
        table = self.validator.validate_table_name(table)
        columns = self.inspector.get_columns(table)
        return [(col["name"], str(col["type"])) for col in columns]

    def generate_sql_llm(self, question: str) -> str:
        """Generate SQL using OpenAI LLM if available."""
        if not self.openai_adapter:
            from .user_experience import create_openai_not_configured_error
            raise create_openai_not_configured_error()
        
        tables = self.discover_schema()
        return self.generator._generate_with_openai(question, tables)

    def generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question."""
        sanitized_question = self.validator.validate_question(question)
        tables = self.discover_schema()
        return self.generator.generate_sql(sanitized_question, tables)

    def query(self, question: str, *, explain: bool = False, trace_id: str = None) -> QueryResult:
        """Process a natural language question and return a QueryResult."""
        return self.query_service.query(question, explain=explain, trace_id=trace_id)

    def execute_sql(self, sql: str, *, explain: bool = False, trace_id: str = None) -> QueryResult:
        """Execute raw SQL and return a QueryResult."""
        return self.query_service.execute_sql(sql, explain=explain, trace_id=trace_id)
    
    def query_paginated(self, question: str, *, page: int = 1, page_size: int = None, trace_id: str = None) -> QueryResult:
        """Process a natural language question and return paginated results.
        
        Args:
            question: Natural language question
            page: Page number (1-based)
            page_size: Number of items per page (uses default if None)
            trace_id: Optional trace ID for request correlation
            
        Returns:
            QueryResult with pagination information
        """
        if page_size is None:
            page_size = self.max_rows
        
        # Generate SQL from question
        sanitized_question = self.validator.validate_question(question)
        tables = self.discover_schema()
        sql = self.generator.generate_sql(sanitized_question, tables)
        
        # Execute with pagination
        return self.query_service.query_paginated(sql, page, page_size, trace_id=trace_id)
    
    def execute_sql_paginated(self, sql: str, *, page: int = 1, page_size: int = None, trace_id: str = None) -> QueryResult:
        """Execute raw SQL and return paginated results.
        
        Args:
            sql: SQL query to execute
            page: Page number (1-based)
            page_size: Number of items per page (uses default if None)
            trace_id: Optional trace ID for request correlation
            
        Returns:
            QueryResult with pagination information
        """
        if page_size is None:
            page_size = self.max_rows
        
        return self.query_service.query_paginated(sql, page, page_size, trace_id=trace_id)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the query agent.
        
        Returns:
            Dict containing health status of all components
        """
        # Check OpenAI API availability
        openai_health = self._check_openai_api_health()
        
        health_status = {
            "database": self.db_manager.health_check(),
            "caches": {
                "schema_cache": {
                    "size": len(self.schema_cache._cache),
                    "ttl": self.schema_cache.ttl,
                    "healthy": True
                },
                "query_cache": {
                    "size": len(self.query_cache._cache),
                    "ttl": self.query_cache.ttl,
                    "healthy": True
                }
            },
            "services": {
                "validator": {"healthy": True},
                "generator": {
                    "healthy": True,
                    "llm_provider_available": self.generator.llm_provider is not None
                },
                "openai_api": openai_health
            },
            "timestamp": time.time()
        }
        
        # Overall health status
        health_status["overall_healthy"] = (
            health_status["database"]["healthy"] and
            health_status["caches"]["schema_cache"]["healthy"] and
            health_status["caches"]["query_cache"]["healthy"] and
            health_status["services"]["validator"]["healthy"] and
            health_status["services"]["generator"]["healthy"] and
            health_status["services"]["openai_api"]["healthy"]
        )
        
        return health_status
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get database connection statistics for monitoring.
        
        Returns:
            Dict containing connection pool statistics
        """
        return self.db_manager.get_connection_stats()
    
    def _check_openai_api_health(self) -> Dict[str, Any]:
        """
        Check OpenAI API availability and health.
        
        Returns:
            Dict containing OpenAI API health status
        """
        health_info = {
            "healthy": False,
            "available": False,
            "error": None,
            "response_time_ms": None
        }
        
        # Check if adapter exists and has API key
        if not self.generator or not self.generator.llm_provider:
            health_info["error"] = "OpenAI adapter not initialized"
            return health_info
        
        # Get API key from environment or adapter
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            health_info["error"] = "OpenAI API key not configured"
            return health_info
        
        try:
            # Perform a simple API call to check availability
            start_time = time.time()
            
            # Try to get models list as a lightweight check
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Make a minimal API call with timeout
            models = client.models.list()
            
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            health_info.update({
                "healthy": True,
                "available": True,
                "response_time_ms": response_time_ms
            })
            
        except Exception as e:
            # Log the error but don't expose sensitive details
            logger.warning(f"OpenAI API health check failed: {str(e)}")
            health_info["error"] = f"API unavailable: {type(e).__name__}"
        
        return health_info