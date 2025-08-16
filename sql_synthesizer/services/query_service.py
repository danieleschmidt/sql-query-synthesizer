"""Core query service for orchestrating SQL generation and execution."""

import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import Engine, inspect, text
from sqlalchemy import exc as sqlalchemy_exc

from .. import metrics
from ..cache import TTLCache
from ..config import config
from ..security_audit import (
    get_security_audit_logger,
)
from ..types import PaginationInfo, QueryResult
from .query_validator_service import QueryValidatorService
from .sql_generator_service import SQLGeneratorService

logger = logging.getLogger(__name__)


class QueryService:
    """Service for orchestrating query processing, caching, and execution."""

    def __init__(
        self,
        engine: Engine,
        validator: QueryValidatorService,
        generator: SQLGeneratorService,
        schema_cache: TTLCache,
        query_cache: TTLCache,
        max_rows: int = 5,
        enable_structured_logging: bool = False,
        inspector=None,
        max_page_size: int = 1000,
    ):
        """Initialize the query service.

        Args:
            engine: SQLAlchemy database engine
            validator: Query validation service
            generator: SQL generation service
            schema_cache: Cache for schema information
            query_cache: Cache for query results
            max_rows: Maximum rows to return for queries
            enable_structured_logging: Enable structured logging with trace IDs
            inspector: Optional database inspector (for testing)
            max_page_size: Maximum allowed page size for pagination
        """
        self.engine = engine
        self.validator = validator
        self.generator = generator
        self.schema_cache = schema_cache
        self.query_cache = query_cache
        self.max_rows = max_rows
        self.max_page_size = max_page_size
        self._structured_logging = enable_structured_logging

        # Create database inspector
        self.inspector = inspector or inspect(engine)

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
        # Generate trace ID for this request if needed
        if trace_id is None and self._structured_logging:
            from ..logging_utils import get_trace_id

            trace_id = get_trace_id()

        # Validate and sanitize the question
        sanitized_question = self.validator.validate_question(question)

        # Check cache first if enabled
        if self.query_cache.ttl > 0:
            try:
                cached_result = self.query_cache.get(sanitized_question)
                metrics.record_cache_hit("query")
                logger.info(f"Cache hit for question: {sanitized_question[:50]}...")
                return cached_result
            except KeyError:
                metrics.record_cache_miss("query")

        # Get available tables for SQL generation
        available_tables = self.discover_schema()

        # Generate SQL
        start_time = time.time()
        sql = self.generator.generate_sql(sanitized_question, available_tables)
        generation_duration = time.time() - start_time

        logger.info(
            f"SQL generated in {generation_duration:.3f}s",
            extra={"trace_id": trace_id} if trace_id else {},
        )

        # Handle placeholder responses (comments)
        if sql.startswith("--"):
            result = QueryResult(
                sql=sql, explanation="SQL generation placeholder", data=[]
            )
        else:
            # Validate the generated SQL
            validated_sql = self.validator.validate_sql(sql)

            # Execute the SQL
            result = self._execute_sql_internal(
                validated_sql, explain=explain, trace_id=trace_id
            )

        # Cache the result if caching is enabled
        if self.query_cache.ttl > 0:
            self.query_cache.set(sanitized_question, result)

        return result

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
        # Generate trace ID for this request if needed
        if trace_id is None and self._structured_logging:
            from ..logging_utils import get_trace_id

            trace_id = get_trace_id()

        # Validate the SQL
        validated_sql = self.validator.validate_sql(sql)

        # Check cache first if enabled
        if self.query_cache.ttl > 0:
            try:
                cached_result = self.query_cache.get(validated_sql)
                metrics.record_cache_hit("sql_execute")
                return cached_result
            except KeyError:
                metrics.record_cache_miss("sql_execute")

        # Execute the SQL
        result = self._execute_sql_internal(
            validated_sql, explain=explain, trace_id=trace_id
        )
        if not explain:
            result.explanation = "Executed raw SQL"

        # Cache the result if caching is enabled
        if self.query_cache.ttl > 0:
            self.query_cache.set(validated_sql, result)

        return result

    def discover_schema(self) -> List[str]:
        """Discover and return available table names.

        Returns:
            List[str]: List of available table names
        """
        # Check cache first if enabled
        if self.schema_cache.ttl > 0:
            try:
                cached_tables = self.schema_cache.get("tables")
                metrics.record_cache_hit("schema")
                return cached_tables
            except KeyError:
                metrics.record_cache_miss("schema")

        # Discover tables from database
        try:
            tables = self.inspector.get_table_names()
            logger.info(f"Discovered {len(tables)} tables")

            # Cache the result if caching is enabled
            if self.schema_cache.ttl > 0:
                self.schema_cache.set("tables", tables)

            return tables
        except (sqlalchemy_exc.SQLAlchemyError, ConnectionError, AttributeError) as e:
            logger.error(f"Failed to discover schema: {e}")
            metrics.record_query_error("schema_discovery_failed")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        schema_stats = self.schema_cache.get_stats()
        query_stats = self.query_cache.get_stats()

        # Update Prometheus metrics
        metrics.update_cache_metrics("schema", schema_stats)
        metrics.update_cache_metrics("query", query_stats)

        total_operations = schema_stats.get("total_operations", 0) + query_stats.get(
            "total_operations", 0
        )
        total_hits = schema_stats.get("hit_count", 0) + query_stats.get("hit_count", 0)

        return {
            "schema_cache": schema_stats,
            "query_cache": query_stats,
            "total_cache_size": schema_stats.get("size", 0)
            + query_stats.get("size", 0),
            "overall_hit_rate": total_hits / max(1, total_operations),
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.schema_cache.clear()
        self.query_cache.clear()
        logger.info("All caches cleared")

    def cleanup_expired_cache_entries(self) -> Dict[str, int]:
        """Clean up expired cache entries.

        Returns:
            Dict[str, int]: Cleanup statistics
        """
        schema_cleaned = self.schema_cache.cleanup_expired()
        query_cleaned = self.query_cache.cleanup_expired()

        return {
            "schema_cache_cleaned": schema_cleaned,
            "query_cache_cleaned": query_cleaned,
            "total_cleaned": schema_cleaned + query_cleaned,
        }

    def _execute_sql_internal(
        self, sql: str, *, explain: bool = False, trace_id: Optional[str] = None
    ) -> QueryResult:
        """Execute SQL against the database.

        Args:
            sql: The validated SQL to execute
            explain: Whether to return execution plan
            trace_id: Optional trace ID for request correlation

        Returns:
            QueryResult: The execution result
        """
        start_time = time.time()

        log_extra = {"sql": sql}
        if trace_id:
            log_extra["trace_id"] = trace_id
        logger.info("Executing SQL", extra=log_extra)

        try:
            with self.engine.connect() as connection:
                if explain:
                    # Execute EXPLAIN query
                    result = connection.execute(text(f"EXPLAIN {sql}"))
                    data = [dict(row._mapping) for row in result]
                    explanation = "Execution plan via EXPLAIN"
                    operation_type = "explain"
                else:
                    # Execute normal query with row limit
                    if (
                        sql.upper().strip().startswith("SELECT")
                        and "LIMIT" not in sql.upper()
                    ):
                        limited_sql = f"{sql} LIMIT {self.max_rows}"
                    else:
                        limited_sql = sql

                    result = connection.execute(text(limited_sql))
                    data = [dict(row._mapping) for row in result]
                    explanation = "Generated and executed SQL"
                    operation_type = "query"

            duration = time.time() - start_time

            # Log successful execution
            log_extra = {
                "sql": sql,
                "duration_ms": int(duration * 1000),
                "row_count": len(data),
            }
            if trace_id:
                log_extra["trace_id"] = trace_id
            logger.info("Query executed successfully", extra=log_extra)

            # Record metrics
            metrics.record_query(duration, operation_type)

            # Log security audit event for query execution
            get_security_audit_logger(config).log_query_execution(
                sql_query=sql,
                execution_time_ms=duration * 1000,
                row_count=len(data),
                trace_id=trace_id,
                operation_type=operation_type,
            )

            return QueryResult(sql=sql, explanation=explanation, data=data)

        except (sqlalchemy_exc.SQLAlchemyError, ConnectionError, ValueError) as e:
            duration = time.time() - start_time
            logger.error(
                f"Query execution failed: {e}",
                extra={"sql": sql, "trace_id": trace_id} if trace_id else {"sql": sql},
            )
            metrics.record_query_error("query_execution_failed")
            raise

    def query_paginated(
        self, sql: str, page: int, page_size: int, trace_id: Optional[str] = None
    ) -> QueryResult:
        """Execute a SQL query with pagination support.

        Args:
            sql: The SQL statement to execute
            page: Page number (1-based)
            page_size: Number of items per page
            trace_id: Optional trace ID for request correlation

        Returns:
            QueryResult with data and pagination information

        Raises:
            ValueError: If pagination parameters are invalid
        """
        # Validate pagination parameters
        self._validate_pagination_params(page, page_size)

        # Create cache key including pagination parameters
        cache_key = f"{sql}|page={page}|page_size={page_size}"

        # Check cache first
        if self.query_cache:
            cached_result = self.query_cache.get(cache_key)
            if cached_result:
                logger.info(
                    "Returning cached paginated result",
                    extra={"trace_id": trace_id} if trace_id else {},
                )
                return cached_result

        start_time = time.time()
        try:
            with self.engine.begin() as connection:
                # First, get the total count
                total_count = self._get_total_count(connection, sql)

                # Calculate offset
                offset = (page - 1) * page_size

                # Create paginated SQL
                paginated_sql = self._add_pagination_to_sql(sql, page_size, offset)

                # Execute paginated query
                result = connection.execute(text(paginated_sql))
                data = [list(row) for row in result.fetchall()]
                column_names = list(result.keys())

                # Create pagination info
                pagination = PaginationInfo.create(page, page_size, total_count)

                # Create result with pagination
                query_result = QueryResult(
                    sql=paginated_sql,
                    explanation=f"Paginated query: page {page} of {pagination.total_pages}",
                    data=data,
                    pagination=pagination,
                )

                # Cache the result
                if self.query_cache:
                    self.query_cache.set(cache_key, query_result)

                duration = time.time() - start_time

                # Log successful execution
                log_extra = {
                    "sql": paginated_sql,
                    "duration_ms": int(duration * 1000),
                    "row_count": len(data),
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                }
                if trace_id:
                    log_extra["trace_id"] = trace_id
                logger.info("Paginated query executed successfully", extra=log_extra)

                # Record metrics
                metrics.record_query(duration, "paginated_query")

                # Log security audit event for paginated query execution
                get_security_audit_logger(config).log_query_execution(
                    sql_query=paginated_sql,
                    execution_time_ms=duration * 1000,
                    row_count=len(data),
                    trace_id=trace_id,
                    operation_type="paginated_query",
                    page=page,
                    page_size=page_size,
                    total_count=total_count,
                )

                return query_result

        except (sqlalchemy_exc.SQLAlchemyError, ConnectionError, ValueError) as e:
            duration = time.time() - start_time
            logger.error(
                f"Paginated query execution failed: {e}",
                extra={"sql": sql, "trace_id": trace_id} if trace_id else {"sql": sql},
            )
            metrics.record_query_error("paginated_query_execution_failed")
            raise

    def _validate_pagination_params(self, page: int, page_size: int) -> None:
        """Validate pagination parameters.

        Args:
            page: Page number to validate
            page_size: Page size to validate

        Raises:
            ValueError: If parameters are invalid
        """
        try:
            page = int(page)
            page_size = int(page_size)
        except (ValueError, TypeError):
            raise ValueError("Page and page_size must be integers")

        if page < 1:
            raise ValueError("Page number must be positive")

        if page_size < 1:
            raise ValueError("Page size must be positive")

        if page_size > self.max_page_size:
            raise ValueError(
                f"Page size exceeds maximum allowed ({self.max_page_size})"
            )

    def _get_total_count(self, connection, sql: str) -> int:
        """Get total count of rows for pagination.

        Args:
            connection: Database connection
            sql: Original SQL query

        Returns:
            Total number of rows
        """
        # Create a count query by wrapping the original SQL
        count_sql = f"SELECT COUNT(*) FROM ({sql}) AS count_query"

        try:
            result = connection.execute(text(count_sql))
            return result.scalar() or 0
        except (sqlalchemy_exc.SQLAlchemyError, ConnectionError) as e:
            logger.warning(f"Failed to get count for pagination, using fallback: {e}")
            # Fallback: execute original query and count results (less efficient)
            result = connection.execute(text(sql))
            return len(result.fetchall())

    def _add_pagination_to_sql(self, sql: str, page_size: int, offset: int) -> str:
        """Add LIMIT and OFFSET to SQL query.

        Args:
            sql: Original SQL query
            page_size: Number of items per page
            offset: Number of items to skip

        Returns:
            SQL with pagination clauses
        """
        # Remove trailing semicolon if present
        sql = sql.rstrip().rstrip(";")

        # Add LIMIT and OFFSET
        return f"{sql} LIMIT {page_size} OFFSET {offset}"

    def get_pagination_config(self) -> Dict[str, Any]:
        """Get pagination configuration information.

        Returns:
            Dict with pagination configuration
        """
        return {"default_page_size": self.max_rows, "max_page_size": self.max_page_size}
