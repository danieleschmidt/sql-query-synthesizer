"""Utility functions for SQL generation."""

import re


def naive_generate_sql(question: str, tables: list[str], max_rows: int = 5) -> str:
    """Return a simple SQL statement for *question* using keyword matching."""
    q = question.lower()
    for table in tables:
        if re.search(rf"\b{re.escape(table.lower())}\b", q):
            if any(word in q for word in ["count", "how many", "number"]):
                return f"SELECT COUNT(*) FROM {table};"
            return f"SELECT * FROM {table} LIMIT {max_rows};"
    return f"-- SQL for: {question}"

