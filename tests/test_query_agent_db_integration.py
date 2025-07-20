"""
Integration tests for QueryAgent with DatabaseConnectionManager.
"""

import pytest
from unittest.mock import Mock, patch
from sqlalchemy import create_engine

from sql_synthesizer.query_agent import QueryAgent
from sql_synthesizer.database import DatabaseConnectionManager, DatabaseConnectionError


@pytest.fixture
def sqlite_url():
    """SQLite database URL for testing."""
    return "sqlite:///:memory:"


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    with patch('sql_synthesizer.query_agent.config') as mock_cfg, \
         patch('sql_synthesizer.database.config') as mock_db_cfg:
        # Set up config values
        for cfg in [mock_cfg, mock_db_cfg]:
            cfg.max_question_length = 1000
            cfg.db_pool_size = 5
            cfg.db_max_overflow = 10
            cfg.db_pool_recycle = 3600
            cfg.db_pool_pre_ping = True
            cfg.db_connect_retries = 2
            cfg.db_retry_delay = 0.1
            cfg.database_timeout = 30
        yield mock_cfg


class TestQueryAgentDatabaseIntegration:
    """Test QueryAgent integration with DatabaseConnectionManager."""
    
    def test_query_agent_initializes_db_manager(self, mock_config, sqlite_url):
        """Test that QueryAgent properly initializes DatabaseConnectionManager."""
        agent = QueryAgent(sqlite_url)
        
        assert hasattr(agent, 'db_manager')
        assert isinstance(agent.db_manager, DatabaseConnectionManager)
        assert agent.db_manager.database_url == sqlite_url
        assert agent.engine is not None
        assert agent.engine == agent.db_manager.engine
    
    def test_query_agent_db_manager_error_handling(self, mock_config):
        """Test QueryAgent handles DatabaseConnectionManager initialization errors."""
        invalid_url = "invalid://database/url"
        
        with pytest.raises(DatabaseConnectionError):
            QueryAgent(invalid_url)
    
    def test_query_agent_health_check_includes_database(self, mock_config, sqlite_url):
        """Test that QueryAgent health check includes database status."""
        agent = QueryAgent(sqlite_url)
        health = agent.health_check()
        
        assert "database" in health
        assert "overall_healthy" in health
        assert health["database"]["healthy"] is True
        assert health["overall_healthy"] is True
        
        # Check that all expected components are included
        assert "caches" in health
        assert "services" in health
        assert "schema_cache" in health["caches"]
        assert "query_cache" in health["caches"]
        assert "validator" in health["services"]
        assert "generator" in health["services"]
    
    def test_query_agent_health_check_database_failure(self, mock_config, sqlite_url):
        """Test health check reports database failure correctly."""
        agent = QueryAgent(sqlite_url)
        
        # Mock database health check failure
        with patch.object(agent.db_manager, 'health_check', 
                         return_value={"healthy": False, "error": "Connection failed"}):
            health = agent.health_check()
            
            assert health["database"]["healthy"] is False
            assert health["overall_healthy"] is False
    
    def test_query_agent_connection_stats(self, mock_config, sqlite_url):
        """Test that QueryAgent provides connection statistics."""
        agent = QueryAgent(sqlite_url)
        stats = agent.get_connection_stats()
        
        assert "total_connections" in stats
        assert "failed_connections" in stats
        assert "retries_attempted" in stats
        assert isinstance(stats["total_connections"], int)
        assert isinstance(stats["failed_connections"], int)
    
    def test_query_agent_backward_compatibility(self, mock_config, sqlite_url):
        """Test that QueryAgent maintains backward compatibility."""
        agent = QueryAgent(sqlite_url)
        
        # Test that engine property still works
        assert hasattr(agent, 'engine')
        assert agent.engine is not None
        
        # Test that basic query operations still work
        tables = agent.list_tables()
        assert isinstance(tables, list)
    
    def test_query_agent_service_integration(self, mock_config, sqlite_url):
        """Test that QueryAgent services work with new database manager."""
        agent = QueryAgent(sqlite_url)
        
        # Test that services are properly initialized
        assert agent.query_service is not None
        assert agent.validator is not None
        assert agent.generator is not None
        
        # Test that services use the same engine
        assert agent.query_service.engine == agent.engine
    
    def test_query_agent_database_manager_disposal(self, mock_config, sqlite_url):
        """Test that DatabaseConnectionManager can be properly disposed."""
        agent = QueryAgent(sqlite_url)
        
        # Verify initial state
        assert agent.db_manager._engine is not None
        
        # Test disposal
        agent.db_manager.dispose()
        assert agent.db_manager._engine is None
    
    def test_query_agent_with_connection_pool_configuration(self, sqlite_url):
        """Test QueryAgent respects connection pool configuration."""
        # Test with custom environment variables
        env_vars = {
            "QUERY_AGENT_DB_POOL_SIZE": "8",
            "QUERY_AGENT_DB_MAX_OVERFLOW": "15",
            "QUERY_AGENT_DB_POOL_RECYCLE": "1800",
            "QUERY_AGENT_DB_POOL_PRE_PING": "false"
        }
        
        with patch.dict('os.environ', env_vars):
            with patch('sql_synthesizer.config.Config') as mock_config_class:
                # Create a mock config instance
                mock_config_instance = Mock()
                mock_config_instance.max_question_length = 1000
                mock_config_instance.db_pool_size = 8
                mock_config_instance.db_max_overflow = 15
                mock_config_instance.db_pool_recycle = 1800
                mock_config_instance.db_pool_pre_ping = False
                mock_config_instance.db_connect_retries = 2
                mock_config_instance.db_retry_delay = 0.1
                mock_config_instance.database_timeout = 30
                
                mock_config_class.return_value = mock_config_instance
                
                with patch('sql_synthesizer.query_agent.config', mock_config_instance), \
                     patch('sql_synthesizer.database.config', mock_config_instance):
                    
                    agent = QueryAgent(sqlite_url)
                    
                    # Verify that the database manager was created
                    assert agent.db_manager is not None
                    assert agent.engine is not None
    
    def test_query_agent_error_recovery(self, mock_config, sqlite_url):
        """Test QueryAgent handles database connection errors gracefully."""
        agent = QueryAgent(sqlite_url)
        
        # Mock a connection error in the database manager
        with patch.object(agent.db_manager, 'get_connection', 
                         side_effect=DatabaseConnectionError("Connection lost")):
            
            # Health check should report the error
            health = agent.health_check()
            assert health["database"]["healthy"] is False
            
            # Connection stats should still be accessible
            stats = agent.get_connection_stats()
            assert isinstance(stats, dict)


class TestQueryAgentConfigurationIntegration:
    """Test QueryAgent configuration integration with DatabaseConnectionManager."""
    
    def test_config_validation_during_initialization(self, sqlite_url):
        """Test that configuration validation works during QueryAgent initialization."""
        # Test with invalid configuration values
        invalid_env = {"QUERY_AGENT_DB_POOL_SIZE": "-1"}
        
        with patch.dict('os.environ', invalid_env):
            with pytest.raises(ValueError) as exc_info:
                from sql_synthesizer.config import Config
                Config(force_reload=True)
            
            assert "must be positive" in str(exc_info.value)
    
    def test_database_specific_configuration(self, mock_config):
        """Test database-specific configuration handling."""
        # Test PostgreSQL configuration
        pg_url = "postgresql://user:pass@localhost/db"
        
        with patch('sql_synthesizer.database.create_engine') as mock_create:
            mock_engine = Mock()
            mock_create.return_value = mock_engine
            
            agent = QueryAgent(pg_url)
            
            # Verify create_engine was called with PostgreSQL-specific args
            call_args = mock_create.call_args
            connect_args = call_args[1]["connect_args"]
            assert "application_name" in connect_args
            assert connect_args["application_name"] == "sql_synthesizer"