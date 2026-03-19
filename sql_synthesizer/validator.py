"""QueryValidator: validates generated SQL against the schema."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import List

from .schema import Schema


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def __str__(self) -> str:
        if self.valid:
            return "Valid" + (f" (warnings: {'; '.join(self.warnings)})" if self.warnings else "")
        return "Invalid: " + "; ".join(self.errors)


class QueryValidator:
    """
    Validates generated SQL queries against a Schema.

    Checks:
      - Referenced tables exist in the schema
      - Referenced columns exist in their tables
      - JOIN conditions reference valid columns in both tables
      - SQL is parseable by SQLite (syntax check via EXPLAIN)
    """

    # Patterns to extract table/column references from SQL
    _FROM_RE = re.compile(r"\bFROM\s+(\w+)", re.IGNORECASE)
    _JOIN_RE = re.compile(r"\bJOIN\s+(\w+)\s+ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", re.IGNORECASE)
    _SELECT_COLS_RE = re.compile(r"\bSELECT\s+(.*?)\s+FROM\b", re.IGNORECASE | re.DOTALL)
    _WHERE_COLS_RE = re.compile(r"\bWHERE\b(.+?)(?=\bGROUP\b|\bORDER\b|\bLIMIT\b|\Z)", re.IGNORECASE | re.DOTALL)
    _COL_REF_RE = re.compile(r"(\w+)\.(\w+)")
    _BARE_COL_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
    _KEYWORDS = frozenset({
        "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "JOIN", "ON", "LEFT", "RIGHT",
        "INNER", "OUTER", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET",
        "AS", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX", "NULL", "IS",
        "IN", "LIKE", "BETWEEN", "CASE", "WHEN", "THEN", "ELSE", "END", "ASC", "DESC",
        "ALL", "ANY", "EXISTS", "UNION", "INTERSECT", "EXCEPT", "WITH",
    })

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def validate(self, sql: str) -> ValidationResult:
        result = ValidationResult(valid=True)

        if sql.strip().startswith("--"):
            result.valid = False
            result.errors.append("SQL is a comment/error placeholder")
            return result

        # 1. Table existence check
        self._check_tables(sql, result)

        # 2. JOIN column validity
        self._check_join_columns(sql, result)

        # 3. Qualified column references (table.column)
        self._check_qualified_columns(sql, result)

        # 4. SQLite syntax check
        if result.valid:
            self._check_syntax(sql, result)

        return result

    # ------------------------------------------------------------------ #
    # Checks                                                               #
    # ------------------------------------------------------------------ #

    def _check_tables(self, sql: str, result: ValidationResult) -> None:
        for m in self._FROM_RE.finditer(sql):
            tname = m.group(1)
            if not self.schema.has_table(tname):
                result.valid = False
                result.errors.append(f"Table '{tname}' not found in schema")

        for m in self._JOIN_RE.finditer(sql):
            tname = m.group(1)
            if not self.schema.has_table(tname):
                result.valid = False
                result.errors.append(f"Joined table '{tname}' not found in schema")

    def _check_join_columns(self, sql: str, result: ValidationResult) -> None:
        for m in self._JOIN_RE.finditer(sql):
            t1, c1 = m.group(2), m.group(3)
            t2, c2 = m.group(4), m.group(5)
            for tname, cname in [(t1, c1), (t2, c2)]:
                table = self.schema.table(tname)
                if table and not table.has_column(cname):
                    result.valid = False
                    result.errors.append(f"Column '{cname}' not found in table '{tname}'")

    def _check_qualified_columns(self, sql: str, result: ValidationResult) -> None:
        """Check table.column references are valid."""
        for m in self._COL_REF_RE.finditer(sql):
            tname, cname = m.group(1), m.group(2)
            if tname.upper() in self._KEYWORDS:
                continue
            table = self.schema.table(tname)
            if table is None:
                # May be an alias — warn but don't fail
                result.warnings.append(f"Unknown table reference '{tname}' (may be alias)")
                continue
            if not table.has_column(cname):
                result.valid = False
                result.errors.append(
                    f"Column '{cname}' not found in table '{tname}'"
                )

    def _check_syntax(self, sql: str, result: ValidationResult) -> None:
        """Use SQLite to verify syntax (EXPLAIN doesn't execute the query)."""
        try:
            conn = sqlite3.connect(":memory:")
            # Create stub tables so SQLite can parse column/table references
            for table in self.schema._tables.values():
                cols = ", ".join(
                    f"{c.name} {c.type}" for c in table.columns
                ) or "id INTEGER"
                try:
                    conn.execute(f"CREATE TABLE {table.name} ({cols})")
                except sqlite3.Error:
                    pass
            conn.execute(f"EXPLAIN {sql}")
            conn.close()
        except sqlite3.Error as e:
            result.valid = False
            result.errors.append(f"SQL syntax error: {e}")
