"""Tests for database layer functionality."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from sql_synthesizer.database.connection import (
    ConnectionConfig,
    ConnectionStats,
    DatabaseManager,
)
from sql_synthesizer.database.migrations import Migration, MigrationManager
from sql_synthesizer.database.repositories import (
    QueryHistoryRepository,
    SystemMetricsRepository,
)
from sql_synthesizer.database.schemas import QueryHistory, SystemMetrics


class TestConnectionConfig:
    """Test database connection configuration."""

    def test_connection_config_defaults(self):
        """Test default connection configuration values."""
        config = ConnectionConfig("sqlite:///test.db")

        assert config.url == "sqlite:///test.db"
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.pool_recycle == 3600
        assert config.pool_pre_ping is True
        assert config.connect_retries == 3
        assert config.retry_delay == 1.0
        assert config.timeout == 30

    def test_connection_config_custom(self):
        """Test custom connection configuration."""
        config = ConnectionConfig(
            url="postgresql://user:pass@host/db",
            pool_size=5,
            max_overflow=10,
            timeout=60,
        )

        assert config.url == "postgresql://user:pass@host/db"
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.timeout == 60


class TestConnectionStats:
    """Test connection statistics."""

    def test_connection_stats_creation(self):
        """Test connection statistics data structure."""
        stats = ConnectionStats(
            pool_size=10,
            checked_out=3,
            overflow=1,
            checked_in=6,
            total_connections=10,
            invalid_connections=0,
            health_score=0.85,
        )

        assert stats.pool_size == 10
        assert stats.checked_out == 3
        assert stats.total_connections == 10
        assert stats.health_score == 0.85


class TestMigration:
    """Test migration functionality."""

    def test_migration_creation(self):
        """Test migration object creation."""
        migration = Migration(
            version="20250803_001",
            name="Create Users Table",
            description="Initial user table",
            up_sql="CREATE TABLE users (id INTEGER PRIMARY KEY);",
            down_sql="DROP TABLE users;",
            created_at=datetime.now(),
        )

        assert migration.version == "20250803_001"
        assert migration.name == "Create Users Table"
        assert "CREATE TABLE users" in migration.up_sql
        assert "DROP TABLE users" in migration.down_sql

    def test_migration_filename(self):
        """Test migration filename generation."""
        migration = Migration(
            version="20250803_001",
            name="Create Users Table",
            description="",
            up_sql="",
            down_sql="",
            created_at=datetime.now(),
        )

        filename = migration.filename
        assert filename == "20250803_001_create_users_table.sql"

    def test_migration_to_dict(self):
        """Test migration serialization."""
        created_at = datetime.now()
        migration = Migration(
            version="20250803_001",
            name="Test Migration",
            description="Test description",
            up_sql="CREATE TABLE test;",
            down_sql="DROP TABLE test;",
            created_at=created_at,
        )

        result = migration.to_dict()

        assert result["version"] == "20250803_001"
        assert result["name"] == "Test Migration"
        assert result["description"] == "Test description"
        assert result["up_sql"] == "CREATE TABLE test;"
        assert result["down_sql"] == "DROP TABLE test;"
        assert result["created_at"] == created_at.isoformat()


class TestMigrationManager:
    """Test migration management."""

    def test_migration_manager_initialization(self, tmp_path):
        """Test migration manager initialization."""
        migration_path = tmp_path / "migrations"
        manager = MigrationManager(str(migration_path))

        assert manager.migrations_path.exists()
        assert not manager._loaded

    def test_create_migration(self, tmp_path):
        """Test migration file creation."""
        migration_path = tmp_path / "migrations"
        manager = MigrationManager(str(migration_path))

        file_path = manager.create_migration("Test Migration", "Test description")

        assert migration_path.exists()
        created_file = migration_path / file_path.split("/")[-1]
        assert created_file.exists()

        content = created_file.read_text()
        assert "-- Name: Test Migration" in content
        assert "-- Description: Test description" in content
        assert "-- UP" in content
        assert "-- DOWN" in content


class TestQueryHistory:
    """Test query history data model."""

    def test_query_history_creation(self):
        """Test query history object creation."""
        history = QueryHistory(
            query_id="test-123",
            user_question="Show me all users",
            generated_sql="SELECT * FROM users;",
            execution_time_ms=150.5,
            success=True,
            cache_hit=False,
        )

        assert history.query_id == "test-123"
        assert history.user_question == "Show me all users"
        assert history.generated_sql == "SELECT * FROM users;"
        assert history.execution_time_ms == 150.5
        assert history.success is True
        assert history.cache_hit is False

    def test_query_history_to_dict(self):
        """Test query history serialization."""
        created_at = datetime.now()
        history = QueryHistory(
            query_id="test-123",
            user_question="Show me all users",
            generated_sql="SELECT * FROM users;",
            execution_time_ms=150.5,
            success=True,
            created_at=created_at,
            id=1,
        )

        result = history.to_dict()

        assert result["id"] == 1
        assert result["query_id"] == "test-123"
        assert result["user_question"] == "Show me all users"
        assert result["execution_time_ms"] == 150.5
        assert result["success"] is True
        assert result["created_at"] == created_at.isoformat()


class TestSystemMetrics:
    """Test system metrics data model."""

    def test_system_metrics_creation(self):
        """Test system metrics object creation."""
        metrics = SystemMetrics(
            metric_name="cpu_usage",
            metric_value=75.5,
            metric_type="gauge",
            tags='{"host": "server1"}',
        )

        assert metrics.metric_name == "cpu_usage"
        assert metrics.metric_value == 75.5
        assert metrics.metric_type == "gauge"
        assert metrics.tags == '{"host": "server1"}'

    def test_system_metrics_defaults(self):
        """Test system metrics default values."""
        metrics = SystemMetrics(metric_name="requests_total", metric_value=1000)

        assert metrics.metric_type == "gauge"
        assert metrics.tags == "{}"

    def test_system_metrics_to_dict(self):
        """Test system metrics serialization."""
        recorded_at = datetime.now()
        metrics = SystemMetrics(
            metric_name="memory_usage",
            metric_value=512.0,
            metric_type="gauge",
            recorded_at=recorded_at,
            id=1,
        )

        result = metrics.to_dict()

        assert result["id"] == 1
        assert result["metric_name"] == "memory_usage"
        assert result["metric_value"] == 512.0
        assert result["metric_type"] == "gauge"
        assert result["recorded_at"] == recorded_at.isoformat()


# Mock repository tests that don't require actual database connections


class TestRepositoryMocks:
    """Test repository functionality with mocked database operations."""

    def test_query_history_repository_structure(self):
        """Test query history repository basic structure."""
        mock_db_manager = Mock()
        repository = QueryHistoryRepository(mock_db_manager)

        assert repository.db_manager == mock_db_manager
        assert hasattr(repository, "create")
        assert hasattr(repository, "get_by_id")
        assert hasattr(repository, "get_query_statistics")
        assert hasattr(repository, "get_recent_queries")

    def test_system_metrics_repository_structure(self):
        """Test system metrics repository basic structure."""
        mock_db_manager = Mock()
        repository = SystemMetricsRepository(mock_db_manager)

        assert repository.db_manager == mock_db_manager
        assert hasattr(repository, "create")
        assert hasattr(repository, "get_latest_metrics")
        assert hasattr(repository, "get_metrics_by_name")
