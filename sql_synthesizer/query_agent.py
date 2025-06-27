"""Core query agent implementation."""

from dataclasses import dataclass, field
from typing import List, Any, Tuple, Dict

try:  # Optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional
    openai = None

import time
import os

import re
from sqlalchemy import create_engine, inspect, text

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
        """

        self.engine = create_engine(database_url)
        self.inspector = inspect(self.engine)
        self.schema_cache_ttl = schema_cache_ttl
        self._schema_cache: List[str] = []
        self._schema_ts = 0.0
        self.max_rows = max_rows
        self.query_cache_ttl = query_cache_ttl
        self._query_cache: Dict[str, tuple[float, QueryResult]] = {}
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_model = openai_model
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key

    def clear_cache(self) -> None:
        """Empty in-memory caches for schema and query results."""
        self._schema_cache = []
        self._schema_ts = 0.0
        self._query_cache.clear()

    def discover_schema(self) -> List[str]:
        """Return a list of available tables in the database."""
        now = time.time()
        if (
            self.schema_cache_ttl
            and now - self._schema_ts <= self.schema_cache_ttl
            and self._schema_cache
        ):
            return self._schema_cache

        tables = self.inspector.get_table_names()
        self._schema_cache = tables
        self._schema_ts = now
        return tables

    def row_count(self, table: str) -> int:
        """Return the number of rows for *table*."""
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            return result.scalar() or 0

    def list_table_counts(self) -> List[tuple[str, int]]:
        """Return table names with corresponding row counts."""
        info = []
        for table in self.discover_schema():
            try:
                count = self.row_count(table)
            except Exception:
                count = -1
            info.append((table, count))
        return info

    def explain_sql(self, sql: str) -> List[Any]:
        """Return the database's execution plan for *sql* using ``EXPLAIN``."""
        with self.engine.connect() as conn:
            result = conn.execute(text(f"EXPLAIN {sql}"))
            return [dict(row._mapping) for row in result]

    def table_columns(self, table: str) -> List[Tuple[str, str]]:
        """Return column names and types for *table*."""
        columns = self.inspector.get_columns(table)
        return [(col["name"], str(col["type"])) for col in columns]

    def generate_sql_llm(self, question: str) -> str:
        """Generate SQL using an OpenAI model if configured."""
        if not (openai and self.openai_api_key):
            raise RuntimeError("OpenAI API key not configured")

        prompt = (
            "Translate the following natural language request into an SQL "
            f"query:\n{question}\nSQL:"
        )
        response = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message["content"].strip()

    def generate_sql(self, question: str) -> str:
        """Generate SQL from *question* using OpenAI if available."""
        if openai and self.openai_api_key:
            try:
                return self.generate_sql_llm(question)
            except Exception:
                pass  # Fall back to naive pattern

        q = question.lower()
        tables = self.discover_schema()
        for table in tables:
            if re.search(rf"\b{re.escape(table.lower())}\b", q):
                if any(word in q for word in ["count", "how many", "number"]):
                    return f"SELECT COUNT(*) FROM {table};"
                return f"SELECT * FROM {table} LIMIT {self.max_rows};"
        return f"-- SQL for: {question}"

    def query(self, question: str, *, explain: bool = False) -> QueryResult:
        """Generate and optionally execute SQL for *question*.

        Parameters
        ----------
        question:
            Natural language prompt.
        explain:
            If ``True``, return the ``EXPLAIN`` plan instead of query results.
        """

        # Return cached result if available and valid
        now = time.time()
        if (
            self.query_cache_ttl
            and question in self._query_cache
            and now - self._query_cache[question][0] <= self.query_cache_ttl
        ):
            return self._query_cache[question][1]

        sql = self.generate_sql(question)
        data: List[Any] = []
        explanation = ""
        if sql.startswith("--"):
            explanation = "SQL generation placeholder"
        else:
            with self.engine.connect() as conn:
                if explain:
                    result = conn.execute(text(f"EXPLAIN {sql}"))
                    data = [dict(row._mapping) for row in result]
                    explanation = "Execution plan via EXPLAIN"
                else:
                    result = conn.execute(text(sql))
                    data = [dict(row._mapping) for row in result]
                    explanation = "Generated and executed SQL using naive keyword match"
        result = QueryResult(sql=sql, explanation=explanation, data=data)
        if self.query_cache_ttl:
            self._query_cache[question] = (now, result)
        return result

    def execute_sql(self, sql: str, *, explain: bool = False) -> QueryResult:
        """Execute raw SQL and return a :class:`QueryResult`."""

        now = time.time()
        if (
            self.query_cache_ttl
            and sql in self._query_cache
            and now - self._query_cache[sql][0] <= self.query_cache_ttl
        ):
            return self._query_cache[sql][1]

        data: list[Any] = []
        explanation = ""
        with self.engine.connect() as conn:
            if explain:
                result = conn.execute(text(f"EXPLAIN {sql}"))
                data = [dict(row._mapping) for row in result]
                explanation = "Execution plan via EXPLAIN"
            else:
                result = conn.execute(text(sql))
                data = [dict(row._mapping) for row in result]
                explanation = "Executed raw SQL"

        res = QueryResult(sql=sql, explanation=explanation, data=data)
        if self.query_cache_ttl:
            self._query_cache[sql] = (now, res)
        return res
