"""Pytest configuration and shared fixtures for SQL Query Synthesizer tests."""

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Test configuration
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_database_url():
    """Provide a sample database URL for testing."""
    return "sqlite:///test.db"


@pytest.fixture
def mock_database_manager():
    """Create a mock database manager for testing."""
    mock_manager = Mock()
    mock_manager.execute_query = AsyncMock()
    mock_manager.session = AsyncMock()
    mock_manager.health_check = AsyncMock(
        return_value={
            "healthy": True,
            "response_time_ms": 15.2,
            "connection_stats": Mock(),
        }
    )
    mock_manager.get_connection_stats = AsyncMock(
        return_value=Mock(
            pool_size=10,
            checked_out=2,
            overflow=0,
            checked_in=8,
            total_connections=10,
            invalid_connections=0,
            health_score=0.95,
        )
    )
    mock_manager.close = AsyncMock()
    return mock_manager


@pytest.fixture
def mock_query_agent():
    """Create a mock query agent for testing."""
    mock_agent = Mock()
    mock_agent.query = AsyncMock()
    mock_agent.query_paginated = AsyncMock()
    mock_agent.execute_sql = AsyncMock()
    mock_agent.execute_sql_paginated = AsyncMock()
    mock_agent.list_tables = AsyncMock()
    mock_agent.describe_table = AsyncMock()
    mock_agent.health_check = AsyncMock()
    mock_agent.get_connection_stats = AsyncMock()
    return mock_agent


@pytest.fixture
def sample_query_result():
    """Provide a sample query result for testing."""
    return {
        "sql": "SELECT * FROM users LIMIT 10;",
        "data": [
            {"id": 1, "name": "John Doe", "email": "john@example.com"},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
            {"id": 3, "name": "Bob Johnson", "email": "bob@example.com"},
        ],
        "explanation": "This query retrieves the first 10 users from the users table.",
        "columns": ["id", "name", "email"],
        "row_count": 3,
        "query_time_ms": 125.4,
    }


@pytest.fixture
def sample_query_history():
    """Provide sample query history data for testing."""
    return {
        "query_id": "query-test-123",
        "user_question": "Show me all users",
        "generated_sql": "SELECT * FROM users LIMIT 10;",
        "execution_time_ms": 125.4,
        "success": True,
        "error_message": None,
        "cache_hit": False,
        "user_agent": "TestAgent/1.0",
        "client_ip": "127.0.0.1",
        "created_at": datetime.utcnow(),
    }


@pytest.fixture
def sample_system_metrics():
    """Provide sample system metrics data for testing."""
    return [
        {
            "metric_name": "query_requests_total",
            "metric_value": 150.0,
            "metric_type": "counter",
            "tags": '{"status": "success"}',
            "recorded_at": datetime.utcnow(),
        },
        {
            "metric_name": "query_duration_seconds",
            "metric_value": 0.125,
            "metric_type": "histogram",
            "tags": '{"endpoint": "/api/query"}',
            "recorded_at": datetime.utcnow(),
        },
        {
            "metric_name": "cache_hit_rate",
            "metric_value": 85.5,
            "metric_type": "gauge",
            "tags": '{"cache_backend": "memory"}',
            "recorded_at": datetime.utcnow(),
        },
    ]


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""
    mock_provider = Mock()
    mock_provider.generate_sql = AsyncMock(
        return_value={
            "sql": "SELECT * FROM users LIMIT 10;",
            "explanation": "This query retrieves users from the database.",
        }
    )
    mock_provider.is_available = Mock(return_value=True)
    mock_provider.health_check = AsyncMock(return_value={"healthy": True})
    return mock_provider


@pytest.fixture
def mock_cache():
    """Create a mock cache for testing."""
    mock_cache = Mock()
    mock_cache.get = Mock(return_value=None)
    mock_cache.set = Mock()
    mock_cache.delete = Mock()
    mock_cache.clear = Mock()
    mock_cache.get_stats = Mock(
        return_value={"hits": 85, "misses": 15, "hit_rate": 85.0, "size": 150}
    )
    return mock_cache


@pytest.fixture
def mock_security_validator():
    """Create a mock security validator for testing."""
    mock_validator = Mock()
    mock_validator.validate_question = Mock(
        return_value={"is_valid": True, "reason": None}
    )
    mock_validator.validate_sql = Mock(return_value={"is_valid": True, "reason": None})
    mock_validator.sanitize_input = Mock(side_effect=lambda x: x)
    return mock_validator


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "database_url": "sqlite:///:memory:",
        "cache_backend": "memory",
        "cache_ttl": 300,
        "max_rows": 100,
        "openai_model": "gpt-3.5-turbo",
        "openai_timeout": 30,
        "rate_limit_per_minute": 60,
        "debug_mode": True,
        "testing": True,
    }


@pytest.fixture
def mock_migration_manager():
    """Create a mock migration manager for testing."""
    mock_manager = Mock()
    mock_manager.get_migrations = Mock(return_value=[])
    mock_manager.get_schema_version = AsyncMock(return_value="20250803_001")
    mock_manager.migrate_to_latest = AsyncMock(return_value=True)
    mock_manager.get_migration_status = AsyncMock(
        return_value={
            "current_version": "20250803_001",
            "total_migrations": 2,
            "applied_count": 2,
            "pending_count": 0,
            "applied_migrations": [],
            "pending_migrations": [],
        }
    )
    mock_manager.create_migration = Mock(return_value="/path/to/migration.sql")
    return mock_manager


@pytest.fixture
def mock_repositories():
    """Create mock repositories for testing."""
    query_repo = Mock()
    query_repo.create = AsyncMock()
    query_repo.get_by_id = AsyncMock()
    query_repo.get_query_statistics = AsyncMock(
        return_value={
            "total_queries": 100,
            "successful_queries": 85,
            "failed_queries": 15,
            "cache_hits": 30,
            "success_rate": 85.0,
            "cache_hit_rate": 30.0,
            "avg_execution_time_ms": 150.5,
        }
    )
    query_repo.get_recent_queries = AsyncMock(return_value=[])
    query_repo.cleanup_old_records = AsyncMock(return_value=10)

    metrics_repo = Mock()
    metrics_repo.create = AsyncMock()
    metrics_repo.get_latest_metrics = AsyncMock(return_value={})
    metrics_repo.get_metrics_by_name = AsyncMock(return_value=[])

    return {"query_history": query_repo, "system_metrics": metrics_repo}


@pytest.fixture
def sample_api_request():
    """Provide a sample API request for testing."""
    return {"question": "Show me the top 10 users by registration date", "max_rows": 10}


@pytest.fixture
def sample_api_response():
    """Provide a sample API response for testing."""
    return {
        "success": True,
        "sql": "SELECT * FROM users ORDER BY created_at DESC LIMIT 10;",
        "data": [
            {"id": 1, "name": "Alice", "created_at": "2025-08-01T10:00:00Z"},
            {"id": 2, "name": "Bob", "created_at": "2025-08-01T09:30:00Z"},
        ],
        "explanation": "This query retrieves the 10 most recently registered users.",
        "columns": ["id", "name", "created_at"],
        "row_count": 2,
        "query_time_ms": 89.7,
        "query_id": "query-abc123",
    }


@pytest.fixture
def error_scenarios():
    """Provide common error scenarios for testing."""
    return [
        {
            "type": "validation_error",
            "input": "",
            "expected_code": "INVALID_INPUT",
            "expected_message": "Question cannot be empty",
        },
        {
            "type": "sql_injection",
            "input": "'; DROP TABLE users; --",
            "expected_code": "SECURITY_VIOLATION",
            "expected_message": "Potential SQL injection detected",
        },
        {
            "type": "database_error",
            "input": "Show me users",
            "expected_code": "DATABASE_ERROR",
            "expected_message": "Database connection failed",
        },
        {
            "type": "llm_error",
            "input": "Show me users",
            "expected_code": "LLM_ERROR",
            "expected_message": "Query generation service unavailable",
        },
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env_vars = {
        "TESTING": "true",
        "DATABASE_URL": "sqlite:///:memory:",
        "OPENAI_API_KEY": "test-key-123",
        "QUERY_AGENT_CACHE_BACKEND": "memory",
        "QUERY_AGENT_CACHE_TTL": "300",
        "QUERY_AGENT_SECRET_KEY": "test-secret-key-for-testing",
        "QUERY_AGENT_RATE_LIMIT_PER_MINUTE": "1000",  # Higher limit for testing
        "QUERY_AGENT_DEBUG": "true",
    }

    # Set test environment variables
    original_env = {}
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def performance_benchmarks():
    """Provide performance benchmark expectations."""
    return {
        "simple_query_max_time": 2.0,  # seconds
        "complex_query_max_time": 5.0,  # seconds
        "health_check_max_time": 0.1,  # seconds
        "cache_operation_max_time": 0.01,  # seconds
        "database_connection_max_time": 1.0,  # seconds
        "max_memory_usage_mb": 512,
        "min_cache_hit_rate": 70.0,  # percentage
        "max_error_rate": 5.0,  # percentage
    }


# Test utilities
class TestHelpers:
    """Helper utilities for tests."""

    @staticmethod
    def assert_valid_uuid(uuid_string):
        """Assert that a string is a valid UUID."""
        import uuid

        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False

    @staticmethod
    def assert_valid_sql(sql_string):
        """Assert that a string looks like valid SQL."""
        sql_keywords = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
        ]
        return any(keyword in sql_string.upper() for keyword in sql_keywords)

    @staticmethod
    def assert_performance_within_limits(execution_time, max_time):
        """Assert that execution time is within performance limits."""
        return execution_time <= max_time

    @staticmethod
    def create_mock_request(question, max_rows=10, headers=None):
        """Create a mock HTTP request for testing."""
        mock_request = Mock()
        mock_request.json = {"question": question, "max_rows": max_rows}
        mock_request.headers = headers or {
            "Content-Type": "application/json",
            "User-Agent": "TestClient/1.0",
        }
        mock_request.remote_addr = "127.0.0.1"
        return mock_request


@pytest.fixture
def test_helpers():
    """Provide test helper utilities."""
    return TestHelpers()
