"""QueryExecutor: runs SQL against an in-memory SQLite database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import Schema


@dataclass
class QueryResult:
    success: bool
    rows: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None

    def __str__(self) -> str:
        if not self.success:
            return f"Error: {self.error}"
        if not self.rows:
            return f"(0 rows)"
        col_widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in self.rows)) for c in self.columns}
        header = " | ".join(c.ljust(col_widths[c]) for c in self.columns)
        sep = "-+-".join("-" * col_widths[c] for c in self.columns)
        lines = [header, sep]
        for row in self.rows:
            lines.append(" | ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in self.columns))
        lines.append(f"({self.row_count} row{'s' if self.row_count != 1 else ''})")
        return "\n".join(lines)


class QueryExecutor:
    """
    Executes SQL queries against an in-memory SQLite database.

    Usage:
      1. Create with a Schema
      2. Call setup_schema() to create tables
      3. Call insert() or insert_many() to add test data
      4. Call execute() to run a query
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._initialized = False

    def setup_schema(self) -> None:
        """Create all tables from the schema in the SQLite database."""
        ddl = self.schema.ddl()
        # SQLite doesn't support FOREIGN KEY enforcement by default — that's fine for testing
        for stmt in ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._conn.commit()
        self._initialized = True

    def insert(self, table_name: str, row: Dict[str, Any]) -> None:
        """Insert a single row into a table."""
        table = self.schema.table(table_name)
        if not table:
            raise ValueError(f"Unknown table: {table_name}")
        # Only insert columns that exist in the schema
        valid_cols = [k for k in row if table.has_column(k)]
        cols = ", ".join(valid_cols)
        placeholders = ", ".join("?" for _ in valid_cols)
        values = [row[k] for k in valid_cols]
        self._conn.execute(f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})", values)
        self._conn.commit()

    def insert_many(self, table_name: str, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self.insert(table_name, row)

    def execute(self, sql: str) -> QueryResult:
        """Execute a SQL query and return a QueryResult."""
        try:
            cursor = self._conn.execute(sql)
            if cursor.description is None:
                # Non-SELECT statement
                return QueryResult(success=True, row_count=cursor.rowcount)
            columns = [desc[0] for desc in cursor.description]
            raw_rows = cursor.fetchall()
            rows = [dict(zip(columns, row)) for row in raw_rows]
            return QueryResult(
                success=True,
                rows=rows,
                columns=columns,
                row_count=len(rows),
            )
        except sqlite3.Error as e:
            return QueryResult(success=False, error=str(e))

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
