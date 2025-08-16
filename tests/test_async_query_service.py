"""Tests for async query service functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from sql_synthesizer.cache import TTLCache
from sql_synthesizer.services.async_query_service import AsyncQueryService
from sql_synthesizer.services.async_sql_generator_service import (
    AsyncSQLGeneratorService,
)
from sql_synthesizer.services.query_validator_service import QueryValidatorService
from sql_synthesizer.types import PaginationInfo, QueryResult


class TestAsyncQueryService:
    """Test async query service capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock async engine
        self.mock_engine = Mock(spec=AsyncEngine)

        # Create mock async connection
        self.mock_connection = Mock(spec=AsyncConnection)
        self.mock_connection.execute = AsyncMock()
        self.mock_connection.begin = AsyncMock()

        # Configure async context manager for connection
        self.mock_context_manager = AsyncMock()
        self.mock_context_manager.__aenter__ = AsyncMock(
            return_value=self.mock_connection
        )
        self.mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        self.mock_engine.connect.return_value = self.mock_context_manager

        # Configure async context manager for transaction
        self.mock_transaction_manager = AsyncMock()
        self.mock_transaction_manager.__aenter__ = AsyncMock(
            return_value=self.mock_connection
        )
        self.mock_transaction_manager.__aexit__ = AsyncMock(return_value=None)
        self.mock_engine.begin.return_value = self.mock_transaction_manager

        # Create service dependencies
        self.validator = Mock(spec=QueryValidatorService)
        self.validator.validate_question.return_value = "test question"
        self.validator.validate_sql.return_value = "SELECT * FROM users"

        self.generator = Mock(spec=AsyncSQLGeneratorService)
        self.generator.generate_sql = AsyncMock(return_value="SELECT * FROM users")

        self.schema_cache = TTLCache(ttl=3600, max_size=100)
        self.query_cache = TTLCache(ttl=300, max_size=100)

        # Mock async inspector
        self.mock_inspector = AsyncMock()
        self.mock_inspector.get_table_names = AsyncMock(
            return_value=["users", "orders"]
        )

        # Create async query service
        self.async_query_service = AsyncQueryService(
            engine=self.mock_engine,
            validator=self.validator,
            generator=self.generator,
            schema_cache=self.schema_cache,
            query_cache=self.query_cache,
            max_rows=5,
            inspector=self.mock_inspector,
            max_page_size=1000,
        )

    @pytest.mark.asyncio
    async def test_async_query_basic(self):
        """Test basic async query execution."""
        # Setup mock result
        mock_result = Mock()
        mock_result._mapping = {"name": "John", "age": 30}
        self.mock_connection.execute.return_value = [mock_result]

        # Execute async query
        result = await self.async_query_service.query("Show me all users")

        # Verify result
        assert isinstance(result, QueryResult)
        assert result.sql == "SELECT * FROM users LIMIT 5"
        assert result.explanation == "Generated and executed SQL"
        assert len(result.data) == 1
        assert result.data[0] == {"name": "John", "age": 30}

        # Verify async methods were called
        self.mock_engine.connect.assert_called_once()
        self.mock_connection.execute.assert_called_once()
        self.generator.generate_sql.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_execute_sql(self):
        """Test async raw SQL execution."""
        # Setup mock result
        mock_result = Mock()
        mock_result._mapping = {"count": 42}
        self.mock_connection.execute.return_value = [mock_result]

        # Execute async SQL
        result = await self.async_query_service.execute_sql(
            "SELECT COUNT(*) FROM users"
        )

        # Verify result
        assert isinstance(result, QueryResult)
        assert result.sql == "SELECT COUNT(*) FROM users"
        assert result.explanation == "Executed raw SQL"
        assert result.data[0] == {"count": 42}

        # Verify async database call
        self.mock_connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_discover_schema(self):
        """Test async schema discovery."""
        # Execute async schema discovery
        tables = await self.async_query_service.discover_schema()

        # Verify result
        assert tables == ["users", "orders"]

        # Verify async inspector call
        self.mock_inspector.get_table_names.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_query_with_caching(self):
        """Test async query with caching enabled."""
        # Enable caching
        self.query_cache.ttl = 300

        # Setup mock result
        mock_result = Mock()
        mock_result._mapping = {"id": 1, "name": "John"}
        self.mock_connection.execute.return_value = [mock_result]

        # Execute first query (should cache)
        result1 = await self.async_query_service.query("Show users")

        # Execute same query again (should use cache)
        result2 = await self.async_query_service.query("Show users")

        # Verify both results are identical
        assert result1.data == result2.data

        # Verify database was only called once (second call used cache)
        self.mock_connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_query_paginated(self):
        """Test async paginated query execution."""
        # Setup mock results for pagination
        count_result = Mock()
        count_result.scalar.return_value = 25

        data_result = [
            Mock(_mapping={"id": 1, "name": "John"}),
            Mock(_mapping={"id": 2, "name": "Jane"}),
            Mock(_mapping={"id": 3, "name": "Bob"}),
        ]
        data_result_mock = Mock()
        data_result_mock.fetchall.return_value = [(1, "John"), (2, "Jane"), (3, "Bob")]
        data_result_mock.keys.return_value = ["id", "name"]

        # Configure execute to return count first, then data
        self.mock_connection.execute.side_effect = [count_result, data_result_mock]

        # Execute async paginated query
        result = await self.async_query_service.query_paginated(
            "SELECT * FROM users ORDER BY id", page=1, page_size=3
        )

        # Verify result
        assert isinstance(result, QueryResult)
        assert result.sql == "SELECT * FROM users ORDER BY id LIMIT 3 OFFSET 0"
        assert result.data == [(1, "John"), (2, "Jane"), (3, "Bob")]

        # Verify pagination info
        assert result.pagination is not None
        assert result.pagination.page == 1
        assert result.pagination.page_size == 3
        assert result.pagination.total_count == 25
        assert result.pagination.total_pages == 9
        assert result.pagination.has_next is True
        assert result.pagination.has_previous is False

        # Verify database calls (count + data)
        assert self.mock_connection.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_async_query_explain(self):
        """Test async query with explain plan."""
        # Setup mock explain result
        mock_result = Mock()
        mock_result._mapping = {"Query_Plan": "Seq Scan on users"}
        self.mock_connection.execute.return_value = [mock_result]

        # Execute async explain query
        result = await self.async_query_service.execute_sql(
            "SELECT * FROM users", explain=True
        )

        # Verify result
        assert result.explanation == "Execution plan via EXPLAIN"
        assert result.data[0] == {"Query_Plan": "Seq Scan on users"}

        # Verify EXPLAIN was prefixed to SQL
        call_args = self.mock_connection.execute.call_args[0]
        executed_sql = str(call_args[0])
        assert "EXPLAIN" in executed_sql

    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test async error handling."""
        # Configure mock to raise exception
        self.mock_connection.execute.side_effect = Exception("Database error")

        # Verify exception is properly raised
        with pytest.raises(Exception, match="Database error"):
            await self.async_query_service.execute_sql("SELECT * FROM users")

    @pytest.mark.asyncio
    async def test_async_concurrent_queries(self):
        """Test concurrent async query execution."""
        # Setup mock results for concurrent queries
        mock_result1 = Mock()
        mock_result1._mapping = {"count": 10}
        mock_result2 = Mock()
        mock_result2._mapping = {"count": 20}

        # Configure different results for each call
        self.mock_connection.execute.side_effect = [[mock_result1], [mock_result2]]

        # Execute concurrent queries
        tasks = [
            self.async_query_service.execute_sql("SELECT COUNT(*) FROM users"),
            self.async_query_service.execute_sql("SELECT COUNT(*) FROM orders"),
        ]

        results = await asyncio.gather(*tasks)

        # Verify both queries completed
        assert len(results) == 2
        assert results[0].data[0] == {"count": 10}
        assert results[1].data[0] == {"count": 20}

        # Verify both database calls were made
        assert self.mock_connection.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_async_pagination_validation(self):
        """Test async pagination parameter validation."""
        # Test invalid page number
        with pytest.raises(ValueError, match="Page number must be positive"):
            await self.async_query_service.query_paginated(
                "SELECT * FROM users", page=0, page_size=10
            )

        # Test invalid page size
        with pytest.raises(ValueError, match="Page size must be positive"):
            await self.async_query_service.query_paginated(
                "SELECT * FROM users", page=1, page_size=0
            )

        # Test page size exceeds maximum
        with pytest.raises(ValueError, match="Page size exceeds maximum"):
            await self.async_query_service.query_paginated(
                "SELECT * FROM users", page=1, page_size=1001
            )

    @pytest.mark.asyncio
    async def test_async_cache_cleanup(self):
        """Test async cache cleanup operations."""
        # Add some items to cache
        self.schema_cache.set("test_key", ["table1", "table2"])
        self.query_cache.set("test_query", QueryResult("SELECT 1", "test", []))

        # Clean up expired entries
        cleanup_stats = await self.async_query_service.cleanup_expired_cache_entries()

        # Verify cleanup stats structure
        assert "schema_cache_cleaned" in cleanup_stats
        assert "query_cache_cleaned" in cleanup_stats
        assert "total_cleaned" in cleanup_stats

        # Test clear all caches
        await self.async_query_service.clear_cache()

        # Verify caches are empty
        assert self.schema_cache.size() == 0
        assert self.query_cache.size() == 0
