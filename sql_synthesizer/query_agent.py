"""Core query agent implementation."""

from dataclasses import dataclass, field
from typing import List, Any, Tuple

import time
import os
import threading
import atexit

import re
import logging
from sqlalchemy import (
    create_engine,
    inspect,
    text,
    Table,
    MetaData,
    select,
    func,
    union_all,
    literal,
)

from .cache import TTLCache
from .openai_adapter import OpenAIAdapter
from .generator import naive_generate_sql
from . import metrics
from .user_experience import (
    create_empty_question_error,
    create_invalid_table_error,
    create_unsafe_input_error,
    create_question_too_long_error,
    create_invalid_sql_error,
    create_multiple_statements_error,
    create_openai_not_configured_error,
    create_invalid_question_type_error,
)

logger = logging.getLogger(__name__)

VALID_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

@dataclass
class QueryResult:
    """Container for the generated SQL, optional explanation and data."""

    sql: str
    explanation: str = ""
    data: List[Any] = field(default_factory=list)


class QueryAgent:
    """Simple natural language to SQL agent."""

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
        """Create a new agent for the given ``database_url``.

        Parameters
        ----------
        database_url:
            SQLAlchemy connection string.
        schema_cache_ttl:
            Time-to-live for cached table names in seconds. ``0`` disables
            caching.
        max_rows:
            Maximum number of rows returned for ``SELECT *`` queries.
        query_cache_ttl:
            Time-to-live for cached query results in seconds. ``0`` disables
            caching.
        openai_api_key:
            API key for the OpenAI service. If ``None`` (default), the agent
            falls back to naive keyword matching.
        openai_model:
            Name of the OpenAI model used to generate SQL when an API key is
            provided.
        openai_timeout:
            Request timeout in seconds for OpenAI API calls.
        enable_structured_logging:
            Enable structured logging with trace IDs and JSON formatting.
        """

        self.engine = create_engine(database_url)
        self._structured_logging = enable_structured_logging
        
        # Configure structured logging if enabled
        if enable_structured_logging:
            from .logging_utils import configure_logging
            configure_logging(enable_json=True)
        self.inspector = inspect(self.engine)
        self.max_rows = max_rows
        self.schema_cache = TTLCache(schema_cache_ttl)
        self.query_cache = TTLCache(query_cache_ttl)
        self.openai_adapter: OpenAIAdapter | None = None
        
        # Set up automatic cache cleanup if TTL is enabled
        self._cleanup_timer = None
        self._setup_cache_cleanup()
        
        key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            try:
                self.openai_adapter = OpenAIAdapter(
                    key, model=openai_model, timeout=openai_timeout
                )
            except RuntimeError:
                self.openai_adapter = None

    def clear_cache(self) -> None:
        """Empty in-memory caches for schema and query results."""
        self.schema_cache.clear()
        self.query_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Return comprehensive cache statistics for monitoring."""
        schema_stats = self.schema_cache.get_stats()
        query_stats = self.query_cache.get_stats()
        
        # Update Prometheus metrics
        metrics.update_cache_metrics("schema", schema_stats)
        metrics.update_cache_metrics("query", query_stats)
        
        return {
            "schema_cache": schema_stats,
            "query_cache": query_stats,
            "total_cache_size": schema_stats["size"] + query_stats["size"],
            "overall_hit_rate": (
                (schema_stats["hit_count"] + query_stats["hit_count"]) /
                max(1, schema_stats["total_operations"] + query_stats["total_operations"])
            ),
        }

    def cleanup_expired_cache_entries(self) -> dict[str, int]:
        """Clean up expired entries from all caches and return cleanup stats."""
        schema_cleaned = self.schema_cache.cleanup_expired()
        query_cleaned = self.query_cache.cleanup_expired()
        
        return {
            "schema_cache_cleaned": schema_cleaned,
            "query_cache_cleaned": query_cleaned,
            "total_cleaned": schema_cleaned + query_cleaned,
        }

    def _setup_cache_cleanup(self) -> None:
        """Set up automatic cache cleanup if TTL is enabled."""
        if self.schema_cache.ttl > 0 or self.query_cache.ttl > 0:
            self._start_cleanup_timer()
            # Register cleanup on exit
            atexit.register(self._stop_cleanup_timer)

    def _start_cleanup_timer(self) -> None:
        """Start the periodic cache cleanup timer."""
        if self._cleanup_timer is not None:
            return
        
        # Run cleanup every 5 minutes
        cleanup_interval = 300  # 5 minutes in seconds
        self._cleanup_timer = threading.Timer(cleanup_interval, self._periodic_cleanup)
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
            # Clean expired entries
            cleanup_stats = self.cleanup_expired_cache_entries()
            
            # Log cleanup if significant activity
            if cleanup_stats["total_cleaned"] > 0:
                logger.debug(
                    "Cache cleanup completed",
                    extra={
                        "schema_cleaned": cleanup_stats["schema_cache_cleaned"],
                        "query_cleaned": cleanup_stats["query_cache_cleaned"],
                        "total_cleaned": cleanup_stats["total_cleaned"],
                    }
                )
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
        finally:
            # Reschedule next cleanup
            self._cleanup_timer = None
            self._start_cleanup_timer()

    def _execute_with_metrics(self, sql_text: str, operation_type: str = "query"):
        """Execute SQL with database connection and query metrics tracking."""
        start_time = time.time()
        try:
            metrics.record_database_connection("success")
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_text))
                query_duration = time.time() - start_time
                metrics.record_database_query(query_duration)
                return result
        except Exception as e:
            query_duration = time.time() - start_time
            metrics.record_database_connection("error")
            metrics.record_database_query(query_duration)
            metrics.record_query_error(f"database_{operation_type}_error")
            raise

    def _validate_table(self, table: str) -> str:
        """Return *table* if valid and known, else raise user-friendly error."""
        if not VALID_TABLE_RE.match(table):
            available_tables = self.discover_schema()
            raise create_invalid_table_error(table, available_tables)
        if table not in self.discover_schema():
            available_tables = self.discover_schema()
            raise create_invalid_table_error(table, available_tables)
        return table

    def _validate_sql(self, sql: str) -> str:
        """Validate that *sql* is a safe single ``SELECT`` statement."""
        try:
            cleaned = sql.strip().rstrip(";")
            if ";" in cleaned:
                metrics.record_query_error("multiple_statements")
                raise create_multiple_statements_error()
            if not re.match(r"^(select|with|explain)\b", cleaned, re.IGNORECASE):
                # Try to determine what operation they attempted
                attempted = cleaned.split()[0] if cleaned.split() else "unknown"
                metrics.record_query_error(f"invalid_sql_{attempted.lower()}")
                raise create_invalid_sql_error(attempted.upper())
            return cleaned + ";"
        except Exception as e:
            # Record SQL validation error if not already recorded
            if not any(attr in str(e) for attr in ['multiple_statements', 'invalid_sql']):
                metrics.record_query_error("unknown_sql_validation_error")
            raise

    def discover_schema(self) -> List[str]:
        """Return a list of available tables in the database."""
        try:
            tables = self.schema_cache.get("tables")
            metrics.record_cache_hit("schema")
            return tables
        except KeyError:
            metrics.record_cache_miss("schema")

        tables = self.inspector.get_table_names()
        self.schema_cache.set("tables", tables)
        return tables

    def row_count(self, table: str) -> int:
        """Return the number of rows for *table* using SQLAlchemy constructs."""
        table = self._validate_table(table)
        tbl = Table(table, MetaData(), autoload_with=self.engine)
        stmt = select(func.count()).select_from(tbl)
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            return result.scalar() or 0

    def batch_row_counts(self, tables: List[str]) -> dict[str, int]:
        """Return row counts for multiple *tables* with a single query."""
        valid = [self._validate_table(t) for t in tables]
        if not valid:
            return {}
        meta = MetaData()
        selects = [
            select(literal(t).label("tbl"), func.count().label("cnt")).select_from(
                Table(t, meta, autoload_with=self.engine)
            )
            for t in valid
        ]
        union = union_all(*selects)
        with self.engine.connect() as conn:
            result = conn.execute(union)
            return {row._mapping["tbl"]: row._mapping["cnt"] for row in result}

    def list_table_counts(self) -> List[tuple[str, int]]:
        """Return table names with corresponding row counts."""
        tables = self.discover_schema()
        counts = self.batch_row_counts(tables)
        return [(t, counts.get(t, -1)) for t in tables]

    def explain_sql(self, sql: str) -> List[Any]:
        """Return the database's execution plan for *sql* using ``EXPLAIN``."""
        sql = self._validate_sql(sql)
        result = self._execute_with_metrics(f"EXPLAIN {sql}", "explain")
        return [dict(row._mapping) for row in result]

    def table_columns(self, table: str) -> List[Tuple[str, str]]:
        """Return column names and types for *table*."""
        table = self._validate_table(table)
        columns = self.inspector.get_columns(table)
        return [(col["name"], str(col["type"])) for col in columns]

    def generate_sql_llm(self, question: str) -> str:
        """Generate SQL using an OpenAI adapter if configured."""
        if not self.openai_adapter:
            raise create_openai_not_configured_error()

        available_tables = self.discover_schema()
        return self.openai_adapter.generate_sql(question, available_tables)

    def generate_sql(self, question: str) -> str:
        """Generate SQL from *question* using OpenAI if available."""
        if self.openai_adapter:
            try:
                return self.generate_sql_llm(question)
            except Exception:
                pass  # Fall back to naive pattern

        tables = self.discover_schema()
        return naive_generate_sql(question, tables, self.max_rows)

    def _sanitize_question(self, question: str) -> str:
        """Sanitize and validate user question input."""
        try:
            if not isinstance(question, str):
                metrics.record_input_validation_error("invalid_type")
                raise create_invalid_question_type_error()
            
            question = question.strip()
            if not question:
                metrics.record_input_validation_error("empty_question")
                raise create_empty_question_error()
            
            # Check for suspicious patterns that might indicate SQL injection attempts
            suspicious_patterns = [
                r';\s*drop\s+',
                r';\s*delete\s+',
                r';\s*update\s+',
                r';\s*insert\s+',
                r';\s*truncate\s+',
                r';\s*alter\s+',
                r';\s*create\s+',
                r'union\s+select',
                r'exec\s*\(',
                r'xp_cmdshell',
            ]
            
            question_lower = question.lower()
            for pattern in suspicious_patterns:
                if re.search(pattern, question_lower):
                    metrics.record_input_validation_error("sql_injection_attempt")
                    raise create_unsafe_input_error()
            
            # Limit question length to prevent abuse
            if len(question) > 1000:
                metrics.record_input_validation_error("question_too_long")
                raise create_question_too_long_error(1000)
                
            return question
        except Exception as e:
            # Record validation error if not already recorded
            if not any(hasattr(e, attr) for attr in ['__cause__', '__context__']):
                metrics.record_input_validation_error("unknown_validation_error")
            raise

    def query(self, question: str, *, explain: bool = False, trace_id: str = None) -> QueryResult:
        """Generate and optionally execute SQL for *question*.

        Parameters
        ----------
        question:
            Natural language prompt.
        explain:
            If ``True``, return the ``EXPLAIN`` plan instead of query results.
        trace_id:
            Optional trace ID for request correlation. Auto-generated if not provided.
        """
        # Generate trace ID for this request
        if trace_id is None and self._structured_logging:
            from .logging_utils import get_trace_id
            trace_id = get_trace_id()
        
        # Sanitize input
        question = self._sanitize_question(question)

        # Return cached result if available and valid
        if self.query_cache.ttl:
            try:
                result = self.query_cache.get(question)
                metrics.record_cache_hit("query")
                return result
            except KeyError:
                metrics.record_cache_miss("query")

        sql = self.generate_sql(question)
        data: List[Any] = []
        explanation = ""
        if sql.startswith("--"):
            explanation = "SQL generation placeholder"
        else:
            sql = self._validate_sql(sql)
            start = time.time()
            log_extra = {"sql": sql}
            if trace_id:
                log_extra["trace_id"] = trace_id
            logger.info("Executing SQL", extra=log_extra)
            
            try:
                if explain:
                    result = self._execute_with_metrics(f"EXPLAIN {sql}", "explain")
                    data = [dict(row._mapping) for row in result]
                    explanation = "Execution plan via EXPLAIN"
                else:
                    result = self._execute_with_metrics(sql, "query")
                    data = [dict(row._mapping) for row in result]
                    explanation = "Generated and executed SQL using naive keyword match"
                    
                duration = time.time() - start
                log_extra = {"sql": sql, "duration_ms": int(duration * 1000)}
                if trace_id:
                    log_extra["trace_id"] = trace_id
                logger.info(
                    "Query executed",
                    extra=log_extra,
                )
                metrics.record_query(duration, "query")
            except Exception as e:
                duration = time.time() - start
                metrics.record_query_error("query_execution_failed")
                raise
        result = QueryResult(sql=sql, explanation=explanation, data=data)
        if self.query_cache.ttl:
            self.query_cache.set(question, result)
        return result

    def execute_sql(self, sql: str, *, explain: bool = False, trace_id: str = None) -> QueryResult:
        """Execute raw SQL and return a :class:`QueryResult`."""
        
        # Generate trace ID for this request
        if trace_id is None and self._structured_logging:
            from .logging_utils import get_trace_id
            trace_id = get_trace_id()

        sql = self._validate_sql(sql)
        if self.query_cache.ttl:
            try:
                result = self.query_cache.get(sql)
                metrics.record_cache_hit("query")
                return result
            except KeyError:
                metrics.record_cache_miss("query")

        data: list[Any] = []
        explanation = ""
        start = time.time()
        log_extra = {"sql": sql}
        if trace_id:
            log_extra["trace_id"] = trace_id
        logger.info("Executing SQL", extra=log_extra)
        
        try:
            if explain:
                result = self._execute_with_metrics(f"EXPLAIN {sql}", "explain")
                data = [dict(row._mapping) for row in result]
                explanation = "Execution plan via EXPLAIN"
            else:
                result = self._execute_with_metrics(sql, "execute")
                data = [dict(row._mapping) for row in result]
                explanation = "Executed raw SQL"
            duration = time.time() - start
        except Exception as e:
            duration = time.time() - start
            metrics.record_query_error("sql_execution_failed")
            raise
        log_extra = {"sql": sql, "duration_ms": int(duration * 1000)}
        if trace_id:
            log_extra["trace_id"] = trace_id
        logger.info(
            "Query executed",
            extra=log_extra,
        )
        metrics.record_query(duration, "execute")

        res = QueryResult(sql=sql, explanation=explanation, data=data)
        if self.query_cache.ttl:
            self.query_cache.set(sql, res)
        return res
