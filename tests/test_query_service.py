"""Tests for QueryService."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine

from sql_synthesizer.cache import TTLCache
from sql_synthesizer.services.query_service import QueryService
from sql_synthesizer.services.query_validator_service import QueryValidatorService
from sql_synthesizer.services.sql_generator_service import SQLGeneratorService
from sql_synthesizer.types import QueryResult


class TestQueryService:
    """Test QueryService functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_validator = Mock(spec=QueryValidatorService)
        self.mock_generator = Mock(spec=SQLGeneratorService)
        self.mock_schema_cache = Mock(spec=TTLCache)
        self.mock_schema_cache.ttl = 0  # Default to no caching
        self.mock_query_cache = Mock(spec=TTLCache)
        self.mock_query_cache.ttl = 0  # Default to no caching
        self.mock_inspector = Mock()

        self.service = QueryService(
            engine=self.mock_engine,
            validator=self.mock_validator,
            generator=self.mock_generator,
            schema_cache=self.mock_schema_cache,
            query_cache=self.mock_query_cache,
            max_rows=5,
            inspector=self.mock_inspector,
        )

    def test_init(self):
        """Test QueryService initialization."""
        assert self.service.engine is self.mock_engine
        assert self.service.validator is self.mock_validator
        assert self.service.generator is self.mock_generator
        assert self.service.schema_cache is self.mock_schema_cache
        assert self.service.query_cache is self.mock_query_cache
        assert self.service.max_rows == 5

    def test_query_with_cache_hit(self):
        """Test query execution with cache hit."""
        # Setup mocks
        cached_result = QueryResult(
            sql="SELECT * FROM users", data=[{"id": 1, "name": "John"}]
        )
        self.mock_query_cache.ttl = 300
        self.mock_query_cache.get.return_value = cached_result
        self.mock_validator.validate_question.return_value = "Show me users"

        result = self.service.query("Show me users")

        assert result is cached_result
        self.mock_validator.validate_question.assert_called_once_with("Show me users")
        self.mock_query_cache.get.assert_called_once_with("Show me users")
        # Should not call generator or execute SQL
        self.mock_generator.generate_sql.assert_not_called()

    def test_query_with_cache_miss(self):
        """Test query execution with cache miss."""
        # Setup mocks
        self.mock_query_cache.ttl = 300
        self.mock_query_cache.get.side_effect = KeyError("Not found")
        self.mock_validator.validate_question.return_value = "Show me users"
        self.mock_generator.generate_sql.return_value = "SELECT * FROM users LIMIT 5"
        self.mock_validator.validate_sql.return_value = "SELECT * FROM users LIMIT 5"

        # Mock database execution
        mock_connection = Mock()
        mock_result = Mock()
        mock_row = Mock()
        mock_row._mapping = {"id": 1, "name": "John"}
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_connection.execute.return_value = mock_result
        self.mock_engine.connect.return_value.__enter__ = Mock(
            return_value=mock_connection
        )
        self.mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

        with patch("sql_synthesizer.services.query_service.text") as mock_text:
            mock_text.return_value = "mocked_sql"
            result = self.service.query("Show me users")

        assert isinstance(result, QueryResult)
        assert result.sql == "SELECT * FROM users LIMIT 5"
        assert result.data == [{"id": 1, "name": "John"}]

        self.mock_validator.validate_question.assert_called_once_with("Show me users")
        self.mock_query_cache.get.assert_called_once_with("Show me users")
        self.mock_generator.generate_sql.assert_called_once()
        self.mock_validator.validate_sql.assert_called_once()

    def test_query_without_cache(self):
        """Test query execution without caching enabled."""
        # Setup mocks
        self.mock_query_cache.ttl = 0  # Caching disabled
        self.mock_validator.validate_question.return_value = "Show me users"
        self.mock_generator.generate_sql.return_value = "SELECT * FROM users LIMIT 5"
        self.mock_validator.validate_sql.return_value = "SELECT * FROM users LIMIT 5"

        # Mock database execution
        mock_connection = Mock()
        mock_result = Mock()
        mock_row = Mock()
        mock_row._mapping = {"id": 1, "name": "John"}
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_connection.execute.return_value = mock_result
        self.mock_engine.connect.return_value.__enter__ = Mock(
            return_value=mock_connection
        )
        self.mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

        with patch("sql_synthesizer.services.query_service.text") as mock_text:
            mock_text.return_value = "mocked_sql"
            result = self.service.query("Show me users")

        assert isinstance(result, QueryResult)
        # Should not call cache methods when caching disabled
        self.mock_query_cache.get.assert_not_called()
        self.mock_query_cache.set.assert_not_called()

    def test_execute_sql_direct(self):
        """Test direct SQL execution."""
        # Setup mocks
        self.mock_validator.validate_sql.return_value = "SELECT * FROM users"

        # Mock database execution
        mock_connection = Mock()
        mock_result = Mock()
        mock_row = Mock()
        mock_row._mapping = {"id": 1, "name": "John"}
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_connection.execute.return_value = mock_result
        self.mock_engine.connect.return_value.__enter__ = Mock(
            return_value=mock_connection
        )
        self.mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

        with patch("sql_synthesizer.services.query_service.text") as mock_text:
            mock_text.return_value = "mocked_sql"
            result = self.service.execute_sql("SELECT * FROM users")

        assert isinstance(result, QueryResult)
        assert result.sql == "SELECT * FROM users"
        assert result.data == [{"id": 1, "name": "John"}]
        assert result.explanation == "Executed raw SQL"

        self.mock_validator.validate_sql.assert_called_once_with("SELECT * FROM users")

    def test_execute_sql_with_explain(self):
        """Test SQL execution with EXPLAIN."""
        # Setup mocks
        self.mock_validator.validate_sql.return_value = "SELECT * FROM users"

        # Mock database execution
        mock_connection = Mock()
        mock_result = Mock()
        mock_row = Mock()
        mock_row._mapping = {"Plan": "Seq Scan on users"}
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_connection.execute.return_value = mock_result
        self.mock_engine.connect.return_value.__enter__ = Mock(
            return_value=mock_connection
        )
        self.mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

        with patch("sql_synthesizer.services.query_service.text") as mock_text:
            mock_text.return_value = "mocked_sql"
            result = self.service.execute_sql("SELECT * FROM users", explain=True)

        assert isinstance(result, QueryResult)
        assert result.explanation == "Execution plan via EXPLAIN"
        # Should execute EXPLAIN version
        mock_connection.execute.assert_called_once()

    def test_discover_schema_with_cache(self):
        """Test schema discovery with caching."""
        cached_tables = ["users", "orders", "products"]
        self.mock_schema_cache.ttl = 300
        self.mock_schema_cache.get.return_value = cached_tables

        result = self.service.discover_schema()

        assert result == cached_tables
        self.mock_schema_cache.get.assert_called_once_with("tables")

    def test_discover_schema_cache_miss(self):
        """Test schema discovery with cache miss."""
        self.mock_schema_cache.ttl = 300
        self.mock_schema_cache.get.side_effect = KeyError("Not found")

        # Mock inspector
        self.mock_inspector.get_table_names.return_value = ["users", "orders"]

        result = self.service.discover_schema()

        assert result == ["users", "orders"]
        self.mock_schema_cache.set.assert_called_once_with(
            "tables", ["users", "orders"]
        )

    def test_get_cache_stats(self):
        """Test cache statistics retrieval."""
        schema_stats = {
            "hit_count": 10,
            "miss_count": 2,
            "size": 5,
            "hit_rate": 0.8,
            "total_operations": 12,
        }
        query_stats = {
            "hit_count": 20,
            "miss_count": 3,
            "size": 8,
            "hit_rate": 0.87,
            "total_operations": 23,
        }

        self.mock_schema_cache.get_stats.return_value = schema_stats
        self.mock_query_cache.get_stats.return_value = query_stats

        result = self.service.get_cache_stats()

        assert result["schema_cache"] == schema_stats
        assert result["query_cache"] == query_stats
        assert "total_cache_size" in result
        assert "overall_hit_rate" in result

    def test_clear_cache(self):
        """Test cache clearing."""
        self.service.clear_cache()

        self.mock_schema_cache.clear.assert_called_once()
        self.mock_query_cache.clear.assert_called_once()

    def test_cleanup_expired_cache_entries(self):
        """Test cleanup of expired cache entries."""
        self.mock_schema_cache.cleanup_expired.return_value = 3
        self.mock_query_cache.cleanup_expired.return_value = 5

        result = self.service.cleanup_expired_cache_entries()

        assert result["schema_cache_cleaned"] == 3
        assert result["query_cache_cleaned"] == 5
        assert result["total_cleaned"] == 8
