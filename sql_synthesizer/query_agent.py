"""Core query agent implementation."""

from dataclasses import dataclass, field
from typing import List, Any, Tuple

import time
import os

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
        """

        self.engine = create_engine(database_url)
        self.inspector = inspect(self.engine)
        self.max_rows = max_rows
        self.schema_cache = TTLCache(schema_cache_ttl)
        self.query_cache = TTLCache(query_cache_ttl)
        self.openai_adapter: OpenAIAdapter | None = None
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

    def _validate_table(self, table: str) -> str:
        """Return *table* if valid and known, else raise ``ValueError``."""
        if not VALID_TABLE_RE.match(table):
            raise ValueError(f"Invalid table name: {table!r}")
        if table not in self.discover_schema():
            raise ValueError(f"Unknown table: {table}")
        return table

    def _validate_sql(self, sql: str) -> str:
        """Validate that *sql* is a safe single ``SELECT`` statement."""
        cleaned = sql.strip().rstrip(";")
        if ";" in cleaned:
            raise ValueError("Only single SQL statements are allowed")
        if not re.match(r"^(select|with|explain)\b", cleaned, re.IGNORECASE):
            raise ValueError("Only SELECT queries are permitted")
        return cleaned + ";"

    def discover_schema(self) -> List[str]:
        """Return a list of available tables in the database."""
        try:
            return self.schema_cache.get("tables")
        except KeyError:
            pass

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
        with self.engine.connect() as conn:
            result = conn.execute(text(f"EXPLAIN {sql}"))
            return [dict(row._mapping) for row in result]

    def table_columns(self, table: str) -> List[Tuple[str, str]]:
        """Return column names and types for *table*."""
        table = self._validate_table(table)
        columns = self.inspector.get_columns(table)
        return [(col["name"], str(col["type"])) for col in columns]

    def generate_sql_llm(self, question: str) -> str:
        """Generate SQL using an OpenAI adapter if configured."""
        if not self.openai_adapter:
            raise RuntimeError("OpenAI API key not configured")

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
        if not isinstance(question, str):
            raise ValueError("Question must be a string")
        
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        
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
                raise ValueError("Potentially unsafe input detected: contains suspicious SQL patterns")
        
        # Limit question length to prevent abuse
        if len(question) > 1000:
            raise ValueError("Question too long (max 1000 characters)")
            
        return question

    def query(self, question: str, *, explain: bool = False) -> QueryResult:
        """Generate and optionally execute SQL for *question*.

        Parameters
        ----------
        question:
            Natural language prompt.
        explain:
            If ``True``, return the ``EXPLAIN`` plan instead of query results.
        """
        # Sanitize input
        question = self._sanitize_question(question)

        # Return cached result if available and valid
        if self.query_cache.ttl:
            try:
                return self.query_cache.get(question)
            except KeyError:
                pass

        sql = self.generate_sql(question)
        data: List[Any] = []
        explanation = ""
        if sql.startswith("--"):
            explanation = "SQL generation placeholder"
        else:
            sql = self._validate_sql(sql)
            start = time.time()
            logger.info("Executing SQL", extra={"sql": sql})
            with self.engine.connect() as conn:
                if explain:
                    result = conn.execute(text(f"EXPLAIN {sql}"))
                    data = [dict(row._mapping) for row in result]
                    explanation = "Execution plan via EXPLAIN"
                else:
                    result = conn.execute(text(sql))
                    data = [dict(row._mapping) for row in result]
                    explanation = "Generated and executed SQL using naive keyword match"
            duration = time.time() - start
            logger.info(
                "Query executed",
                extra={"sql": sql, "duration_ms": int(duration * 1000)},
            )
            metrics.record_query(duration, "query")
        result = QueryResult(sql=sql, explanation=explanation, data=data)
        if self.query_cache.ttl:
            self.query_cache.set(question, result)
        return result

    def execute_sql(self, sql: str, *, explain: bool = False) -> QueryResult:
        """Execute raw SQL and return a :class:`QueryResult`."""

        sql = self._validate_sql(sql)
        if self.query_cache.ttl:
            try:
                return self.query_cache.get(sql)
            except KeyError:
                pass

        data: list[Any] = []
        explanation = ""
        start = time.time()
        logger.info("Executing SQL", extra={"sql": sql})
        with self.engine.connect() as conn:
            if explain:
                result = conn.execute(text(f"EXPLAIN {sql}"))
                data = [dict(row._mapping) for row in result]
                explanation = "Execution plan via EXPLAIN"
            else:
                result = conn.execute(text(sql))
                data = [dict(row._mapping) for row in result]
                explanation = "Executed raw SQL"
        duration = time.time() - start
        logger.info(
            "Query executed",
            extra={"sql": sql, "duration_ms": int(duration * 1000)},
        )
        metrics.record_query(duration, "execute")

        res = QueryResult(sql=sql, explanation=explanation, data=data)
        if self.query_cache.ttl:
            self.query_cache.set(sql, res)
        return res
