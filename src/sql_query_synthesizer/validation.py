"""Utilities for validating SQL queries."""

from dataclasses import dataclass, field
from typing import List

import sqlparse


@dataclass
class ValidationResult:
    """Outcome of SQL validation."""

    is_valid: bool
    suggestions: List[str] = field(default_factory=list)


def validate_sql(query: str) -> ValidationResult:
    """Return ValidationResult for the given SQL string."""

    if not query or not query.strip():
        return ValidationResult(False, ["Query is empty"])

    try:
        statements = sqlparse.parse(query)
    except Exception as exc:  # pragma: no cover - defensive
        return ValidationResult(False, [f"SQL parse error: {exc}"])

    if not statements:
        return ValidationResult(False, ["Invalid SQL syntax"])

    # sqlparse does not raise errors for malformed keywords; attempt a basic check
    from sqlparse import tokens as T

    first_statement = statements[0]
    first = first_statement.token_first(skip_ws=True)
    if first is None or first.ttype not in (T.DML, T.DDL):
        return ValidationResult(False, ["Invalid SQL syntax"])

    suggestions: List[str] = []

    stmt_type = first_statement.get_type()
    if stmt_type == "SELECT":
        for token in first_statement.tokens:
            if token.ttype is T.Wildcard:
                suggestions.append("Avoid SELECT *; specify columns explicitly")
                break
    elif stmt_type in {"DELETE", "UPDATE"}:
        has_where = any(
            token.ttype is T.Keyword and token.value.upper() == "WHERE"
            for token in first_statement.tokens
        )
        if not has_where:
            suggestions.append("DELETE/UPDATE without WHERE may affect all rows")

    return ValidationResult(True, suggestions)
