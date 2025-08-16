"""
Tests for database connection management.
"""

import os
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DisconnectionError, OperationalError

from sql_synthesizer.config import Config
from sql_synthesizer.database import DatabaseConnectionError, DatabaseConnectionManager


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    with patch("sql_synthesizer.database.config") as mock_cfg:
        mock_cfg.db_pool_size = 5
        mock_cfg.db_max_overflow = 10
        mock_cfg.db_pool_recycle = 3600
        mock_cfg.db_pool_pre_ping = True
        mock_cfg.db_connect_retries = 2
        mock_cfg.db_retry_delay = 0.1
        mock_cfg.database_timeout = 30
        yield mock_cfg


@pytest.fixture
def sqlite_url():
    """SQLite database URL for testing."""
    return "sqlite:///:memory:"


@pytest.fixture
def db_manager(mock_config, sqlite_url):
    """Database manager instance for testing."""
    return DatabaseConnectionManager(sqlite_url)


class TestDatabaseConnectionManager:
    """Test database connection manager functionality."""

    def test_initialization_success(self, mock_config, sqlite_url):
        """Test successful database manager initialization."""
        manager = DatabaseConnectionManager(sqlite_url)

        assert manager.database_url == sqlite_url
        assert manager._engine is not None
        assert manager._connection_stats["failed_connections"] == 0

    def test_initialization_with_invalid_url(self, mock_config):
        """Test initialization failure with invalid database URL."""
        invalid_url = "invalid://database/url"

        with pytest.raises(DatabaseConnectionError):
            DatabaseConnectionManager(invalid_url)

    def test_connection_retry_logic(self, mock_config):
        """Test connection retry logic with temporary failures."""
        mock_config.db_connect_retries = 2
        mock_config.db_retry_delay = 0.01  # Fast retries for testing

        # Mock create_engine to fail twice then succeed
        call_count = 0
        original_create_engine = create_engine

        def mock_create_engine(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise OperationalError("Connection failed", None, None)
            return original_create_engine("sqlite:///:memory:", **kwargs)

        with patch(
            "sql_synthesizer.database.create_engine", side_effect=mock_create_engine
        ):
            manager = DatabaseConnectionManager("postgresql://test")
            assert manager._connection_stats["retries_attempted"] == 2
            assert manager._engine is not None

    def test_connection_retry_exhausted(self, mock_config):
        """Test behavior when all connection retries are exhausted."""
        mock_config.db_connect_retries = 1
        mock_config.db_retry_delay = 0.01

        with patch(
            "sql_synthesizer.database.create_engine",
            side_effect=OperationalError("Connection failed", None, None),
        ):
            with pytest.raises(DatabaseConnectionError) as exc_info:
                DatabaseConnectionManager("postgresql://test")

            assert "Failed to connect to database after 2 attempts" in str(
                exc_info.value
            )

    def test_engine_property(self, db_manager):
        """Test engine property returns valid engine."""
        engine = db_manager.engine
        assert engine is not None
        assert hasattr(engine, "connect")

    def test_engine_property_when_not_initialized(self, mock_config):
        """Test engine property raises error when engine not initialized."""
        manager = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        manager._engine = None

        with pytest.raises(DatabaseConnectionError):
            _ = manager.engine

    def test_get_connection_context_manager(self, db_manager):
        """Test get_connection context manager functionality."""
        with db_manager.get_connection() as conn:
            assert conn is not None
            # Test that we can execute a simple query
            result = conn.execute(text("SELECT 1")).scalar()
            assert result == 1

    def test_get_connection_handles_disconnection_error(self, db_manager):
        """Test get_connection handles disconnection errors properly."""
        with patch.object(
            db_manager._engine,
            "connect",
            side_effect=DisconnectionError("Database disconnected", None, None),
        ):
            with patch.object(db_manager, "_initialize_engine") as mock_init:
                with pytest.raises(DatabaseConnectionError):
                    with db_manager.get_connection():
                        pass
                mock_init.assert_called_once()

    def test_get_connection_handles_operational_error(self, db_manager):
        """Test get_connection handles operational errors."""
        with patch.object(
            db_manager._engine,
            "connect",
            side_effect=OperationalError("Database error", None, None),
        ):
            with pytest.raises(DatabaseConnectionError) as exc_info:
                with db_manager.get_connection():
                    pass
            assert "Database operational error" in str(exc_info.value)

    def test_get_connection_stats(self, db_manager):
        """Test connection statistics reporting."""
        stats = db_manager.get_connection_stats()

        assert "total_connections" in stats
        assert "failed_connections" in stats
        assert "retries_attempted" in stats
        assert "last_error" in stats
        assert isinstance(stats["total_connections"], int)

    def test_health_check_success(self, db_manager):
        """Test successful health check."""
        health = db_manager.health_check()

        assert health["healthy"] is True
        assert health["error"] is None
        assert "timestamp" in health
        assert isinstance(health["timestamp"], float)

    def test_health_check_failure(self, db_manager):
        """Test health check failure handling."""
        with patch.object(
            db_manager,
            "get_connection",
            side_effect=DatabaseConnectionError("Connection failed"),
        ):
            health = db_manager.health_check()

            assert health["healthy"] is False
            assert "Connection failed" in health["error"]

    def test_dispose(self, db_manager):
        """Test engine disposal."""
        assert db_manager._engine is not None

        db_manager.dispose()

        assert db_manager._engine is None

    def test_database_specific_connection_args(self, mock_config):
        """Test database-specific connection arguments."""
        # Test PostgreSQL
        with patch("sql_synthesizer.database.create_engine") as mock_create:
            mock_engine = Mock()
            mock_create.return_value = mock_engine

            DatabaseConnectionManager("postgresql://test")

            call_args = mock_create.call_args
            connect_args = call_args[1]["connect_args"]
            assert "application_name" in connect_args
            assert connect_args["application_name"] == "sql_synthesizer"

        # Test MySQL
        with patch("sql_synthesizer.database.create_engine") as mock_create:
            mock_engine = Mock()
            mock_create.return_value = mock_engine

            DatabaseConnectionManager("mysql://test")

            call_args = mock_create.call_args
            connect_args = call_args[1]["connect_args"]
            assert "charset" in connect_args
            assert connect_args["charset"] == "utf8mb4"

    def test_pool_events_setup(self, mock_config, sqlite_url):
        """Test that pool events are properly set up."""
        with patch("sql_synthesizer.database.event.listens_for") as mock_listens:
            manager = DatabaseConnectionManager(sqlite_url)

            # Verify that event listeners were registered
            assert (
                mock_listens.call_count >= 4
            )  # connect, checkout, checkin, invalidate

    def test_connection_validation(self, mock_config):
        """Test connection validation during initialization."""
        # Mock an engine that fails validation
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.side_effect = OperationalError(
            "Validation failed", None, None
        )
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

        with patch("sql_synthesizer.database.create_engine", return_value=mock_engine):
            with pytest.raises(DatabaseConnectionError) as exc_info:
                DatabaseConnectionManager("sqlite:///:memory:")

            assert "Connection validation failed" in str(exc_info.value)


class TestConfigIntegration:
    """Test integration with configuration system."""

    def test_config_environment_variables(self):
        """Test that configuration properly reads environment variables."""
        env_vars = {
            "QUERY_AGENT_DB_POOL_SIZE": "15",
            "QUERY_AGENT_DB_MAX_OVERFLOW": "25",
            "QUERY_AGENT_DB_POOL_RECYCLE": "7200",
            "QUERY_AGENT_DB_POOL_PRE_PING": "false",
            "QUERY_AGENT_DB_CONNECT_RETRIES": "5",
            "QUERY_AGENT_DB_RETRY_DELAY": "2.5",
        }

        with patch.dict(os.environ, env_vars):
            config = Config(force_reload=True)

            assert config.db_pool_size == 15
            assert config.db_max_overflow == 25
            assert config.db_pool_recycle == 7200
            assert config.db_pool_pre_ping is False
            assert config.db_connect_retries == 5
            assert config.db_retry_delay == 2.5

    def test_config_validation_errors(self):
        """Test configuration validation for invalid values."""
        # Test invalid boolean
        with patch.dict(os.environ, {"QUERY_AGENT_DB_POOL_PRE_PING": "maybe"}):
            with pytest.raises(ValueError) as exc_info:
                Config(force_reload=True)
            assert "Invalid boolean value" in str(exc_info.value)

        # Test negative values
        with patch.dict(os.environ, {"QUERY_AGENT_DB_POOL_SIZE": "-1"}):
            with pytest.raises(ValueError) as exc_info:
                Config(force_reload=True)
            assert "must be positive" in str(exc_info.value)

        # Test invalid float
        with patch.dict(os.environ, {"QUERY_AGENT_DB_RETRY_DELAY": "not_a_number"}):
            with pytest.raises(ValueError) as exc_info:
                Config(force_reload=True)
            assert "is not a valid number" in str(exc_info.value)

    def test_config_as_dict_includes_db_settings(self):
        """Test that as_dict includes database configuration."""
        config = Config()
        config_dict = config.as_dict()

        db_keys = [
            "db_pool_size",
            "db_max_overflow",
            "db_pool_recycle",
            "db_pool_pre_ping",
            "db_connect_retries",
            "db_retry_delay",
        ]

        for key in db_keys:
            assert key in config_dict
