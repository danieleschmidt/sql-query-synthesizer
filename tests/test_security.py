"""Tests for security features and input validation."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from sql_synthesizer import QueryAgent


@pytest.fixture()
def agent(tmp_path: Path) -> QueryAgent:
    db = tmp_path / "test.db"
    url = f"sqlite:///{db}"
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"))
        conn.execute(text("INSERT INTO users (name) VALUES ('Alice'), ('Bob');"))
    return QueryAgent(url)


def test_sanitize_question_empty(agent: QueryAgent):
    """Test that empty questions are rejected."""
    with pytest.raises(ValueError, match="Please provide a question"):
        agent.query("")

    with pytest.raises(ValueError, match="Please provide a question"):
        agent.query("   ")


def test_sanitize_question_non_string(agent: QueryAgent):
    """Test that non-string inputs are rejected."""
    with pytest.raises(ValueError, match="Questions must be provided as text"):
        agent.query(123)

    with pytest.raises(ValueError, match="Questions must be provided as text"):
        agent.query(None)


def test_sanitize_question_too_long(agent: QueryAgent):
    """Test that overly long questions are rejected."""
    long_question = "a" * 1001
    with pytest.raises(ValueError, match="too long"):
        agent.query(long_question)


def test_sanitize_question_sql_injection_attempts(agent: QueryAgent):
    """Test that potential SQL injection patterns are blocked."""
    injection_attempts = [
        "Show users; DROP TABLE users;",
        "List users; DELETE FROM users;",
        "Count users; UPDATE users SET name='hacked';",
        "Get data; INSERT INTO users VALUES (999, 'evil');",
        "Select data; TRUNCATE TABLE users;",
        "Show tables; ALTER TABLE users DROP COLUMN name;",
        "List all; CREATE TABLE evil (id INT);",
        "Count all UNION SELECT password FROM secret_table",
        "Show data; EXEC xp_cmdshell('dir');",
    ]

    for injection in injection_attempts:
        with pytest.raises(ValueError, match="unsafe.*security"):
            agent.query(injection)


def test_valid_questions_pass_sanitization(agent: QueryAgent):
    """Test that normal, safe questions pass validation."""
    safe_questions = [
        "How many users are there?",
        "List all users",
        "Show me the user table",
        "Count the number of records",
        "What's in the users table?",
    ]

    for question in safe_questions:
        # Should not raise an exception
        result = agent.query(question)
        assert result is not None


def test_sql_validation_rejects_multiple_statements(agent: QueryAgent):
    """Test that SQL validation rejects multiple statements."""
    with pytest.raises(ValueError, match="Multiple.*statements.*not allowed"):
        agent.execute_sql("SELECT * FROM users; DROP TABLE users;")


def test_sql_validation_rejects_non_select(agent: QueryAgent):
    """Test that SQL validation only allows SELECT statements."""
    non_select_statements = [
        "DROP TABLE users;",
        "DELETE FROM users;",
        "UPDATE users SET name='evil';",
        "INSERT INTO users VALUES (999, 'bad');",
        "CREATE TABLE evil (id INT);",
    ]

    for sql in non_select_statements:
        with pytest.raises(ValueError, match="Only SELECT.*allowed.*security"):
            agent.execute_sql(sql)


def test_table_validation_rejects_invalid_names(agent: QueryAgent):
    """Test that table name validation works correctly."""
    invalid_tables = [
        "users; DROP TABLE users;--",
        "users/*comment*/",
        "users'",
        "users--",
        "users union select",
    ]

    for table in invalid_tables:
        with pytest.raises(ValueError, match="not found.*database"):
            agent.row_count(table)


def test_table_validation_rejects_unknown_tables(agent: QueryAgent):
    """Test that unknown table names are rejected."""
    with pytest.raises(ValueError, match="not found.*database"):
        agent.row_count("nonexistent_table")
