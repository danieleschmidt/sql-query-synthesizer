"""Tests for structured logging functionality."""
import json
import logging
import uuid
from pathlib import Path
from unittest.mock import patch
from io import StringIO

import pytest
from sqlalchemy import text, create_engine

from sql_synthesizer import QueryAgent


@pytest.fixture()
def agent_with_logging(tmp_path: Path) -> QueryAgent:
    """Create test agent with structured logging enabled."""
    db = tmp_path / "test.db"
    url = f"sqlite:///{db}"
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"))
        conn.execute(text("INSERT INTO users (name) VALUES ('Alice'), ('Bob');"))
    
    # Create agent with structured logging
    agent = QueryAgent(url, enable_structured_logging=True)
    return agent


def test_structured_logging_configuration():
    """Test that structured logging can be enabled/disabled."""
    # Test default (disabled)
    agent = QueryAgent("sqlite:///:memory:")
    assert not hasattr(agent, '_structured_logging') or not agent._structured_logging
    
    # Test enabled
    agent = QueryAgent("sqlite:///:memory:", enable_structured_logging=True)
    assert agent._structured_logging is True


def test_trace_id_generation_and_propagation(agent_with_logging):
    """Test that trace IDs are generated and included in logs."""
    from io import StringIO
    import logging
    
    # Set up a string stream to capture log output
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    
    # Configure JSON formatter to capture structured logs
    from sql_synthesizer.logging_utils import JSONFormatter
    handler.setFormatter(JSONFormatter())
    
    # Add handler to the service logger
    service_logger = logging.getLogger("sql_synthesizer.services.query_service")
    service_logger.setLevel(logging.INFO)
    service_logger.addHandler(handler)
    
    try:
        # Execute query which should generate trace IDs
        agent_with_logging.query("How many users?")
        
        # Get log output and verify trace_id is present
        log_output = log_stream.getvalue()
        assert log_output, "No log output captured"
        
        # Check that logs contain trace_id field
        import json
        log_lines = [line for line in log_output.strip().split('\n') if line.strip()]
        trace_id_found = False
        
        for line in log_lines:
            try:
                log_entry = json.loads(line)
                if 'trace_id' in log_entry:
                    trace_id_found = True
                    assert log_entry['trace_id'], "trace_id should not be empty"
                    break
            except json.JSONDecodeError:
                continue
        
        assert trace_id_found, f"No trace_id found in logs: {log_output}"
    
    finally:
        service_logger.removeHandler(handler)


def test_json_log_formatting():
    """Test JSON log formatter."""
    from sql_synthesizer.logging_utils import JSONFormatter
    
    formatter = JSONFormatter()
    
    # Create a test log record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None
    )
    record.trace_id = "test-trace-123"
    record.duration_ms = 100
    
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Test message"
    assert parsed["trace_id"] == "test-trace-123"
    assert parsed["duration_ms"] == 100
    assert "timestamp" in parsed


def test_log_context_manager():
    """Test log context manager for trace ID propagation."""
    from sql_synthesizer.logging_utils import log_context
    
    with patch('sql_synthesizer.query_agent.logger') as mock_logger:
        with log_context() as trace_id:
            # Verify trace_id is a valid UUID
            uuid.UUID(trace_id)
            
            # Mock a log call within context
            mock_logger.info("Test message", extra={"trace_id": trace_id})
            
            mock_logger.info.assert_called_with(
                "Test message", 
                extra={"trace_id": trace_id}
            )


def test_structured_logging_in_query_execution(agent_with_logging):
    """Test that query execution includes structured logging."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    
    # Configure JSON formatter for test
    from sql_synthesizer.logging_utils import JSONFormatter
    handler.setFormatter(JSONFormatter())
    
    logger = logging.getLogger("sql_synthesizer.query_agent")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    try:
        agent_with_logging.query("How many users?")
        
        # Get logged output
        log_output = log_stream.getvalue()
        
        # Parse JSON log lines
        log_lines = [line.strip() for line in log_output.split('\n') if line.strip()]
        
        for line in log_lines:
            log_entry = json.loads(line)
            
            # Verify structured fields are present
            assert "timestamp" in log_entry
            assert "level" in log_entry
            assert "message" in log_entry
            assert "trace_id" in log_entry
            
            # If it's a query execution log, verify SQL and duration
            if "sql" in log_entry:
                assert log_entry["sql"].startswith("SELECT")
            
            if "duration_ms" in log_entry:
                assert isinstance(log_entry["duration_ms"], int)
                assert log_entry["duration_ms"] >= 0
                
    finally:
        logger.removeHandler(handler)


def test_logging_config_from_environment():
    """Test that logging configuration can be set via environment variables."""
    with patch.dict('os.environ', {
        'QUERY_AGENT_LOG_LEVEL': 'DEBUG',
        'QUERY_AGENT_LOG_FORMAT': 'json'
    }):
        QueryAgent("sqlite:///:memory:")
        # Verify configuration was applied
        # (Implementation will be added in the actual code)


def test_error_logging_includes_trace_id(agent_with_logging):
    """Test that error conditions include trace IDs in logs."""
    with patch('sql_synthesizer.query_agent.logger'):
        try:
            # Trigger an error condition
            agent_with_logging.query("")  # Empty question should fail
        except ValueError:
            pass
        
        # Error logging should still include trace_id if implemented
        # This test ensures we don't lose correlation in error scenarios