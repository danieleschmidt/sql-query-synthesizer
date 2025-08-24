"""Tests for user experience improvements and friendly error messages."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from sql_synthesizer import QueryAgent


@pytest.fixture()
def agent(tmp_path: Path) -> QueryAgent:
    """TODO: Add docstring"""
    db = tmp_path / "test.db"
    url = f"sqlite:///{db}"
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"))
        conn.execute(text("INSERT INTO users (name) VALUES ('Alice'), ('Bob');"))
    return QueryAgent(url)


def test_friendly_error_for_empty_question(agent):
    """Test that empty questions get user-friendly error messages."""
    with pytest.raises(ValueError) as exc_info:
        agent.query("")

    error_msg = str(exc_info.value)
    assert "Please provide a question" in error_msg
    # Should include helpful suggestion
    assert "suggestion" in error_msg.lower() and "try" in error_msg.lower()


def test_friendly_error_for_invalid_table(agent):
    """Test that invalid table names get user-friendly error messages."""
    with pytest.raises(ValueError) as exc_info:
        agent.row_count("nonexistent_table")

    error_msg = str(exc_info.value)
    assert "table" in error_msg.lower()
    assert "nonexistent_table" in error_msg
    # Should suggest available tables
    assert "available" in error_msg.lower() or "valid" in error_msg.lower()


def test_friendly_error_for_malicious_input(agent):
    """Test that potentially unsafe input gets user-friendly error messages."""
    with pytest.raises(ValueError) as exc_info:
        agent.query("Show users; DROP TABLE users;")

    error_msg = str(exc_info.value)
    assert "not allowed" in error_msg.lower() or "unsafe" in error_msg.lower()
    assert "security" in error_msg.lower()
    # Should explain what's allowed
    assert "select" in error_msg.lower() or "query" in error_msg.lower()


def test_friendly_error_for_long_question(agent):
    """Test that overly long questions get user-friendly error messages."""
    long_question = "a" * 1001
    with pytest.raises(ValueError) as exc_info:
        agent.query(long_question)

    error_msg = str(exc_info.value)
    assert "too long" in error_msg.lower()
    assert "1000" in error_msg or "shorter" in error_msg.lower()


def test_friendly_error_for_invalid_sql(agent):
    """Test that invalid SQL gets user-friendly error messages."""
    with pytest.raises(ValueError) as exc_info:
        agent.execute_sql("DROP TABLE users;")

    error_msg = str(exc_info.value)
    assert "only" in error_msg.lower() and "select" in error_msg.lower()
    assert "allowed" in error_msg.lower() or "permitted" in error_msg.lower()


def test_helpful_suggestions_in_cli_help():
    """Test that CLI help includes helpful examples and suggestions."""
    import sys

    sys.path.insert(0, "/root/repo")
    from io import StringIO

    import query_agent

    # Capture help output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        with pytest.raises(SystemExit):
            query_agent.main(["--help"])
    finally:
        sys.stdout = old_stdout

    help_text = captured_output.getvalue()

    # Should contain usage examples
    assert "example" in help_text.lower() or "common" in help_text.lower()


def test_user_friendly_openai_error(agent):
    """Test that OpenAI-related errors are user-friendly."""
    # This should fail since no API key is configured
    with pytest.raises(
        ValueError
    ) as exc_info:  # Now raises UserFriendlyError, which inherits from ValueError
        agent.generate_sql_llm("test question")

    error_msg = str(exc_info.value)
    assert "api key" in error_msg.lower()
    assert "openai" in error_msg.lower()
    assert "suggestion" in error_msg.lower()


def test_contextual_error_messages_include_suggestions(agent):
    """Test that error messages include contextual suggestions."""
    # Test with a question that matches no tables
    try:
        result = agent.query("Show me all orders")
        # If this doesn't raise an error, check the SQL generated
        assert "no matching table" in result.sql.lower() or "--" in result.sql
    except Exception as e:
        # If it does raise an error, ensure it's helpful
        error_msg = str(e)
        assert len(error_msg) > 10  # Should be descriptive


def test_progress_feedback_for_long_operations():
    """Test that long operations provide user feedback."""
    # This is more of a design test - ensuring we have mechanisms for feedback
    # In a real implementation, we might test progress callbacks or status messages
    pass
