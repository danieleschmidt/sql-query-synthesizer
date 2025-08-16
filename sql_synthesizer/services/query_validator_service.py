"""Query validation service for input sanitization and SQL validation."""

import logging
import re

from .. import metrics
from ..user_experience import (
    create_empty_question_error,
    create_invalid_sql_error,
    create_invalid_table_error,
    create_multiple_statements_error,
    create_question_too_long_error,
    create_unsafe_input_error,
)

logger = logging.getLogger(__name__)

# Regular expression for valid table names (alphanumeric, underscores, no leading numbers)
VALID_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# SQL injection patterns to detect
SQL_INJECTION_PATTERNS = [
    r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|GRANT|REVOKE|TRUNCATE)",
    r"--\s*[^'\s]",  # SQL comments that aren't in quotes
    r"/\*.*?\*/",  # Multi-line comments
    r"UNION\s+SELECT",
    r"OR\s+1\s*=\s*1",
    r"AND\s+1\s*=\s*1",
    r"'\s*(OR|AND)\s*'",
    r";\s*EXEC",
    r";\s*xp_",
]

# Compile patterns for better performance
COMPILED_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in SQL_INJECTION_PATTERNS
]


class QueryValidatorService:
    """Service for validating user input and SQL statements."""

    def __init__(self, max_question_length: int = 1000):
        """Initialize the validator service.

        Args:
            max_question_length: Maximum allowed length for user questions
        """
        self.max_question_length = max_question_length

    def validate_question(self, question: str) -> str:
        """Validate and sanitize a user question.

        Args:
            question: The user's natural language question

        Returns:
            str: The sanitized question

        Raises:
            UserExperienceError: If the question is invalid or unsafe
        """
        # Check for non-string input
        if not isinstance(question, str):
            metrics.record_query_error("invalid_question_type")
            raise ValueError("Questions must be provided as text")

        # Check for empty question
        if not question or not question.strip():
            metrics.record_query_error("empty_question")
            raise create_empty_question_error()

        # Check question length
        if len(question) > self.max_question_length:
            metrics.record_query_error("question_too_long")
            raise create_question_too_long_error(self.max_question_length)

        # Sanitize the question
        sanitized = self.sanitize_question(question)

        # Check for SQL injection attempts
        if self._contains_sql_injection(sanitized):
            metrics.record_query_error("sql_injection_attempt")
            raise create_unsafe_input_error()

        return sanitized

    def validate_sql(self, sql: str) -> str:
        """Validate a SQL statement for safety and syntax.

        Args:
            sql: The SQL statement to validate

        Returns:
            str: The validated SQL statement

        Raises:
            UserExperienceError: If the SQL is invalid or unsafe
        """
        # Check for empty SQL
        if not sql or not sql.strip():
            metrics.record_query_error("empty_sql")
            raise create_invalid_sql_error("SQL statement cannot be empty")

        sql = sql.strip()

        # Check for multiple statements (basic semicolon detection)
        if self._contains_multiple_statements(sql):
            metrics.record_query_error("multiple_statements")
            raise create_multiple_statements_error()

        # Basic SQL syntax validation
        if not self._is_valid_sql_syntax(sql):
            metrics.record_query_error("invalid_sql_syntax")
            raise create_invalid_sql_error("Invalid SQL syntax")

        return sql

    def validate_table_name(self, table_name: str) -> str:
        """Validate a table name for safety.

        Args:
            table_name: The table name to validate

        Returns:
            str: The validated table name

        Raises:
            UserExperienceError: If the table name is invalid
        """
        if not table_name or not table_name.strip():
            metrics.record_query_error("invalid_table_name")
            raise create_invalid_table_error("", [])

        table_name = table_name.strip()

        if not VALID_TABLE_RE.match(table_name):
            metrics.record_query_error("invalid_table_name")
            raise create_invalid_table_error(table_name, [])

        return table_name

    def sanitize_question(self, question: str) -> str:
        """Sanitize a user question by removing dangerous content.

        Args:
            question: The raw user question

        Returns:
            str: The sanitized question
        """
        if not question:
            return ""

        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", question.strip())

        # Remove control characters but preserve basic punctuation
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", sanitized)

        return sanitized

    def _contains_sql_injection(self, text: str) -> bool:
        """Check if text contains potential SQL injection patterns.

        Args:
            text: The text to check

        Returns:
            bool: True if injection patterns are detected
        """
        for pattern in COMPILED_INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning(f"SQL injection pattern detected: {pattern.pattern}")
                return True
        return False

    def _contains_multiple_statements(self, sql: str) -> bool:
        """Check if SQL contains multiple statements.

        Args:
            sql: The SQL to check

        Returns:
            bool: True if multiple statements are detected
        """
        # Simple check for semicolons not in quotes
        # This is a basic implementation - more sophisticated parsing would be better
        in_single_quote = False
        in_double_quote = False
        semicolon_count = 0

        i = 0
        while i < len(sql):
            char = sql[i]

            if char == "'" and not in_double_quote:
                # Check for escaped quote
                if i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 1  # Skip escaped quote
                else:
                    in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                if i + 1 < len(sql) and sql[i + 1] == '"':
                    i += 1  # Skip escaped quote
                else:
                    in_double_quote = not in_double_quote
            elif char == ";" and not in_single_quote and not in_double_quote:
                semicolon_count += 1
                # If we find more than one meaningful semicolon, it's multiple statements
                if semicolon_count > 1 or i < len(sql) - 1:  # Not at end
                    return True

            i += 1

        return False

    def _is_valid_sql_syntax(self, sql: str) -> bool:
        """Basic SQL syntax validation.

        Args:
            sql: The SQL to validate

        Returns:
            bool: True if basic syntax appears valid
        """
        sql_upper = sql.upper().strip()

        # Must start with a valid SQL command
        valid_starts = ["SELECT", "WITH", "EXPLAIN", "DESCRIBE", "DESC", "SHOW"]

        if not any(sql_upper.startswith(start) for start in valid_starts):
            return False

        # Basic structure checks for SELECT statements
        if sql_upper.startswith("SELECT"):
            # Must contain FROM for most SELECT statements (except SELECT 1, etc.)
            if "FROM" not in sql_upper and not re.match(
                r'SELECT\s+[\d\'"]+', sql_upper
            ):
                return False

            # Check for basic syntax errors
            if sql_upper.endswith("WHERE") or sql_upper.endswith("FROM"):
                return False

        return True
