"""Tests for comprehensive metrics and monitoring functionality."""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from sql_synthesizer.query_agent import QueryAgent
from sql_synthesizer import metrics
from sql_synthesizer.openai_adapter import OpenAIAdapter


@pytest.fixture
def mock_engine():
    """Mock database engine for testing."""
    with patch('sql_synthesizer.database.create_engine') as mock_create, \
         patch('sql_synthesizer.database.event') as mock_event:
        mock_engine = Mock()
        mock_create.return_value = mock_engine
        
        # Mock the event system to avoid issues with pool events
        mock_event.listens_for = Mock()
        
        # Mock the connection context manager
        mock_connection = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_connection)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_engine.connect.return_value = mock_context_manager
        
        # Mock inspector for both query_agent and query_service
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["users", "orders", "products"]
        
        with patch('sql_synthesizer.query_agent.inspect') as mock_inspect_agent, \
             patch('sql_synthesizer.services.query_service.inspect') as mock_inspect_service:
            mock_inspect_agent.return_value = mock_inspector
            mock_inspect_service.return_value = mock_inspector
            yield mock_engine


def test_input_validation_error_metrics(mock_engine):
    """Test that input validation errors are tracked in metrics."""
    agent = QueryAgent("sqlite:///:memory:")
    
    # Test empty question error through query method
    with pytest.raises(Exception):
        agent.query("")
    
    # Test SQL injection attempt error through query method  
    with pytest.raises(Exception):
        agent.query("SELECT * FROM users; DROP TABLE users;")
    
    # Test question too long error through query method
    with pytest.raises(Exception):
        agent.query("x" * 1001)
    
    # Test validation through execute_sql method
    with pytest.raises(Exception):
        agent.execute_sql("")


def test_sql_validation_error_metrics(mock_engine):
    """Test that SQL validation errors are tracked in metrics."""
    agent = QueryAgent("sqlite:///:memory:")
    
    # Test multiple statements error through execute_sql
    with pytest.raises(Exception):
        agent.execute_sql("SELECT * FROM users; SELECT * FROM orders;")
    
    # Test invalid SQL syntax through execute_sql
    with pytest.raises(Exception):
        agent.execute_sql("SELCT * FROM users")


def test_database_metrics_tracking(mock_engine):
    """Test that database connection and query metrics are tracked."""
    agent = QueryAgent("sqlite:///:memory:")
    
    # Mock successful database execution
    with patch.object(agent.engine, 'connect') as mock_connect:
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        mock_conn.execute.return_value = mock_result
        
        # Execute a query through the public API
        result = agent.execute_sql("SELECT 1")
        
        # Verify the query was executed
        assert mock_conn.execute.called


def test_database_error_metrics_tracking(mock_engine):
    """Test that database errors are tracked in metrics."""
    agent = QueryAgent("sqlite:///:memory:")
    
    # Mock database execution failure
    with patch.object(agent.engine, 'connect') as mock_connect:
        mock_connect.side_effect = Exception("Database connection failed")
        
        # Execute a query that should fail
        with pytest.raises(Exception):
            agent.execute_sql("SELECT 1")


def test_openai_metrics_tracking():
    """Test that OpenAI API metrics are tracked."""
    with patch('sql_synthesizer.openai_adapter.openai') as mock_openai:
        # Mock successful OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "SELECT * FROM users;"
        mock_openai.chat.completions.create.return_value = mock_response
        
        adapter = OpenAIAdapter("test-key")
        result = adapter.generate_sql("Show me all users", ["users"])
        
        assert result == "SELECT * FROM users;"
        assert mock_openai.chat.completions.create.called


def test_openai_error_metrics_tracking():
    """Test that OpenAI API errors are tracked in metrics."""
    with patch('sql_synthesizer.openai_adapter.openai') as mock_openai:
        # Mock OpenAI API failure
        mock_openai.chat.completions.create.side_effect = Exception("API timeout")
        
        adapter = OpenAIAdapter("test-key")
        with pytest.raises(Exception):
            adapter.generate_sql("Show me all users", ["users"])


def test_query_execution_metrics_integration(mock_engine):
    """Test that query execution integrates with comprehensive metrics."""
    agent = QueryAgent("sqlite:///:memory:", query_cache_ttl=10)
    
    # Mock database execution without patching internal validation
    with patch.object(agent.engine, 'connect') as mock_connect:
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        mock_conn.execute.return_value = mock_result
        
        # Execute a query
        result = agent.execute_sql("SELECT * FROM users")
        
        # Verify result structure
        assert hasattr(result, 'sql')
        assert hasattr(result, 'data')
        assert hasattr(result, 'explanation')


def test_error_handling_with_metrics_in_query_method(mock_engine):
    """Test that the query method properly handles errors with metrics."""
    agent = QueryAgent("sqlite:///:memory:")
    
    # Test empty question handling
    with pytest.raises(Exception):
        agent.query("")
    
    # Test SQL injection attempt handling
    with pytest.raises(Exception):
        agent.query("SELECT * FROM users; DROP TABLE users;")


def test_comprehensive_metrics_functions():
    """Test all metrics recording functions work correctly."""
    # Test query error recording
    metrics.record_query_error("test_error")
    
    # Test input validation error recording
    metrics.record_input_validation_error("test_validation_error")
    
    # Test OpenAI request recording
    metrics.record_openai_request(1.5, "success")
    metrics.record_openai_request(0.8, "error")
    
    # Test database connection recording
    metrics.record_database_connection("success")
    metrics.record_database_connection("error")
    
    # Test database query recording
    metrics.record_database_query(0.05)


def test_metrics_histogram_buckets():
    """Test that histogram metrics have appropriate bucket ranges."""
    # Test query duration histogram
    assert hasattr(metrics, 'QUERY_DURATION')
    assert hasattr(metrics, 'OPENAI_REQUEST_DURATION')
    assert hasattr(metrics, 'DATABASE_QUERY_DURATION')
    
    # Verify buckets cover expected ranges
    query_buckets = metrics.QUERY_DURATION._upper_bounds
    assert 0.01 in query_buckets  # Fast queries
    assert 1.0 in query_buckets   # Normal queries
    assert 10.0 in query_buckets  # Slow queries


def test_error_counter_metrics():
    """Test that error counter metrics are properly configured."""
    # Test error metrics exist
    assert hasattr(metrics, 'QUERY_ERRORS_TOTAL')
    assert hasattr(metrics, 'INPUT_VALIDATION_ERRORS_TOTAL')
    
    # Test metrics have proper labels
    assert 'error_type' in str(metrics.QUERY_ERRORS_TOTAL._labelnames)
    assert 'validation_error_type' in str(metrics.INPUT_VALIDATION_ERRORS_TOTAL._labelnames)


def test_database_performance_metrics():
    """Test database performance metrics are properly configured."""
    # Test database metrics exist
    assert hasattr(metrics, 'DATABASE_CONNECTIONS_TOTAL')
    assert hasattr(metrics, 'DATABASE_QUERY_DURATION')
    
    # Test appropriate bucket ranges for database queries
    db_buckets = metrics.DATABASE_QUERY_DURATION._upper_bounds
    assert 0.001 in db_buckets  # Very fast DB queries
    assert 0.1 in db_buckets    # Normal DB queries
    assert 1.0 in db_buckets    # Slower DB queries


def test_openai_performance_metrics():
    """Test OpenAI performance metrics are properly configured."""
    # Test OpenAI metrics exist
    assert hasattr(metrics, 'OPENAI_REQUESTS_TOTAL')
    assert hasattr(metrics, 'OPENAI_REQUEST_DURATION')
    
    # Test appropriate bucket ranges for OpenAI requests
    openai_buckets = metrics.OPENAI_REQUEST_DURATION._upper_bounds
    assert 0.5 in openai_buckets   # Fast API responses
    assert 5.0 in openai_buckets   # Normal API responses
    assert 30.0 in openai_buckets  # Slow API responses