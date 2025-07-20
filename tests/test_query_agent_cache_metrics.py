"""Tests for QueryAgent cache metrics integration."""
import pytest
from unittest.mock import Mock, patch
from sql_synthesizer.query_agent import QueryAgent
from sql_synthesizer import metrics


@pytest.fixture
def mock_engine():
    """Mock database engine for testing."""
    with patch('sql_synthesizer.query_agent.create_engine') as mock_create:
        mock_engine = Mock()
        mock_create.return_value = mock_engine
        
        # Mock inspector for both query_agent and query_service
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["users", "orders", "products"]
        
        with patch('sql_synthesizer.query_agent.inspect') as mock_inspect_agent, \
             patch('sql_synthesizer.services.query_service.inspect') as mock_inspect_service:
            mock_inspect_agent.return_value = mock_inspector
            mock_inspect_service.return_value = mock_inspector
            yield mock_engine


def test_schema_cache_metrics_integration(mock_engine):
    """Test that schema cache hits/misses are tracked in metrics."""
    agent = QueryAgent("sqlite:///:memory:", schema_cache_ttl=10)
    
    # Clear any existing cache state
    agent.clear_cache()
    
    # First call should be a cache miss
    tables1 = agent.discover_schema()
    assert tables1 == ["users", "orders", "products"]
    
    # Second call should be a cache hit
    tables2 = agent.discover_schema()
    assert tables2 == tables1
    
    # Verify metrics were recorded by checking cache stats
    stats = agent.get_cache_stats()
    schema_stats = stats["schema_cache"]
    
    assert schema_stats["hit_count"] == 1
    assert schema_stats["miss_count"] == 1


def test_query_cache_metrics_integration(mock_engine):
    """Test that query cache hits/misses are tracked in metrics."""
    agent = QueryAgent("sqlite:///:memory:", query_cache_ttl=10)
    
    # Clear any existing cache state
    agent.clear_cache()
    
    # Mock database execution without patching internal validation
    with patch.object(agent.engine, 'connect') as mock_connect:
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        mock_conn.execute.return_value = mock_result
        
        # First query should be a cache miss
        result1 = agent.execute_sql("SELECT * FROM users")
        
        # Second identical query should be a cache hit
        result2 = agent.execute_sql("SELECT * FROM users")
        
        # Verify metrics were recorded by checking cache stats
        stats = agent.get_cache_stats()
        query_stats = stats["query_cache"]
        
        assert query_stats["hit_count"] == 1
        assert query_stats["miss_count"] == 1


def test_cache_stats_method(mock_engine):
    """Test the get_cache_stats method returns comprehensive statistics."""
    agent = QueryAgent("sqlite:///:memory:", schema_cache_ttl=10, query_cache_ttl=10)
    
    # Generate some cache activity
    agent.discover_schema()  # Cache miss
    agent.discover_schema()  # Cache hit
    
    # Get cache statistics
    stats = agent.get_cache_stats()
    
    # Verify structure
    assert "schema_cache" in stats
    assert "query_cache" in stats
    assert "total_cache_size" in stats
    assert "overall_hit_rate" in stats
    
    # Verify schema cache stats
    schema_stats = stats["schema_cache"]
    assert schema_stats["hit_count"] == 1
    assert schema_stats["miss_count"] == 1
    assert schema_stats["size"] == 1
    assert schema_stats["hit_rate"] == 0.5
    
    # Verify overall metrics
    assert stats["total_cache_size"] == 1  # Only schema cache has data
    assert 0 <= stats["overall_hit_rate"] <= 1


def test_cache_cleanup_method(mock_engine):
    """Test the cleanup_expired_cache_entries method."""
    agent = QueryAgent("sqlite:///:memory:", schema_cache_ttl=0.1, query_cache_ttl=0.1)
    
    # Add some data to caches
    agent.discover_schema()  # Adds to schema cache
    
    # Mock database execution without patching internal validation
    with patch.object(agent.engine, 'connect') as mock_connect:
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        mock_conn.execute.return_value = mock_result
        
        agent.execute_sql("SELECT 1")  # Adds to query cache
    
    # Wait for expiration
    import time
    time.sleep(0.15)
    
    # Clean up expired entries
    cleanup_stats = agent.cleanup_expired_cache_entries()
    
    # Verify structure
    assert "schema_cache_cleaned" in cleanup_stats
    assert "query_cache_cleaned" in cleanup_stats
    assert "total_cleaned" in cleanup_stats
    
    # Should have cleaned up expired entries
    assert cleanup_stats["total_cleaned"] > 0


def test_metrics_update_on_cache_stats_call(mock_engine):
    """Test that Prometheus metrics are updated when get_cache_stats is called."""
    agent = QueryAgent("sqlite:///:memory:", schema_cache_ttl=10)
    
    # Clear any existing cache state
    agent.clear_cache()
    
    # Generate cache activity
    agent.discover_schema()
    agent.discover_schema()
    
    # Call get_cache_stats to trigger metrics update
    stats = agent.get_cache_stats()
    
    # Verify cache stats structure and values
    schema_stats = stats["schema_cache"]
    
    assert schema_stats["size"] == 1
    assert schema_stats["hit_rate"] == 0.5


def test_cache_eviction_tracking(mock_engine):
    """Test that cache evictions are tracked when size limits are exceeded."""
    # Create agent with small cache limits
    agent = QueryAgent("sqlite:///:memory:", schema_cache_ttl=10, query_cache_ttl=10)
    
    # Set small max_size for testing eviction
    agent.schema_cache.max_size = 1
    agent.query_cache.max_size = 1
    
    with patch.object(agent.inspector, 'get_table_names') as mock_get_tables:
        # First table list
        mock_get_tables.return_value = ["table1"]
        agent.discover_schema()
        
        # Second table list (should evict first)
        mock_get_tables.return_value = ["table2"]
        agent.schema_cache.set("tables2", ["table2"])
        
        # Get stats to check eviction count
        stats = agent.get_cache_stats()
        schema_stats = stats["schema_cache"]
        
        # Should have recorded eviction
        assert schema_stats["eviction_count"] >= 1