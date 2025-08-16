"""Tests for query result pagination functionality."""

from unittest.mock import MagicMock, Mock

import pytest

from sql_synthesizer.services.query_service import QueryService
from sql_synthesizer.types import PaginationInfo, QueryResult


class TestQueryPagination:
    """Test query result pagination capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_validator = Mock()
        self.mock_generator = Mock()
        self.mock_schema_cache = Mock()
        self.mock_query_cache = Mock()

        self.mock_inspector = Mock()

        # Configure mock engine.begin() to return a context manager
        self.mock_connection = Mock()
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.__enter__.return_value = self.mock_connection
        self.mock_context_manager.__exit__.return_value = None
        self.mock_engine.begin.return_value = self.mock_context_manager

        # Configure cache to miss by default
        self.mock_query_cache.get.return_value = None

        self.query_service = QueryService(
            engine=self.mock_engine,
            validator=self.mock_validator,
            generator=self.mock_generator,
            schema_cache=self.mock_schema_cache,
            query_cache=self.mock_query_cache,
            max_rows=10,
            inspector=self.mock_inspector,
        )

    def test_pagination_info_creation(self):
        """Test creation of pagination info objects."""
        pagination = PaginationInfo(
            page=1,
            page_size=10,
            total_count=100,
            total_pages=10,
            has_next=True,
            has_previous=False,
        )

        assert pagination.page == 1
        assert pagination.page_size == 10
        assert pagination.total_count == 100
        assert pagination.total_pages == 10
        assert pagination.has_next is True
        assert pagination.has_previous is False

    def test_pagination_info_calculations(self):
        """Test pagination info calculations."""
        # Test middle page
        pagination = PaginationInfo.create(page=3, page_size=10, total_count=100)
        assert pagination.page == 3
        assert pagination.total_pages == 10
        assert pagination.has_next is True
        assert pagination.has_previous is True

        # Test last page
        pagination = PaginationInfo.create(page=10, page_size=10, total_count=100)
        assert pagination.has_next is False
        assert pagination.has_previous is True

        # Test single page
        pagination = PaginationInfo.create(page=1, page_size=10, total_count=5)
        assert pagination.total_pages == 1
        assert pagination.has_next is False
        assert pagination.has_previous is False

    def test_query_with_pagination_basic(self):
        """Test basic query with pagination."""
        # Mock database response
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("Alice", "alice@example.com"),
            ("Bob", "bob@example.com"),
            ("Charlie", "charlie@example.com"),
        ]
        mock_result.keys.return_value = ["name", "email"]

        # Mock count query
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 25
        self.mock_connection.execute.side_effect = [mock_count_result, mock_result]

        # Test pagination
        result = self.query_service.query_paginated(
            "SELECT name, email FROM users", page=1, page_size=3
        )

        assert isinstance(result, QueryResult)
        assert len(result.data) == 3
        assert result.pagination is not None
        assert result.pagination.page == 1
        assert result.pagination.page_size == 3
        assert result.pagination.total_count == 25
        assert result.pagination.total_pages == 9

    def test_query_with_pagination_validation(self):
        """Test pagination parameter validation."""
        # Test invalid page number
        with pytest.raises(ValueError, match="Page number must be positive"):
            self.query_service.query_paginated(
                "SELECT * FROM users", page=0, page_size=10
            )

        # Test invalid page size
        with pytest.raises(ValueError, match="Page size must be positive"):
            self.query_service.query_paginated(
                "SELECT * FROM users", page=1, page_size=0
            )

        # Test page size too large
        with pytest.raises(ValueError, match="Page size exceeds maximum"):
            self.query_service.query_paginated(
                "SELECT * FROM users", page=1, page_size=1001
            )

    def test_query_with_pagination_offset_calculation(self):
        """Test correct offset calculation for pagination."""
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = []

        # Mock count query
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 100
        self.mock_connection.execute.side_effect = [mock_count_result, mock_result]

        # Test page 3 with page size 10 should have offset 20
        self.query_service.query_paginated(
            "SELECT * FROM users ORDER BY id", page=3, page_size=10
        )

        # Verify the SQL includes correct LIMIT and OFFSET
        executed_sql = self.mock_connection.execute.call_args_list[1][0][0].text
        assert "LIMIT 10 OFFSET 20" in executed_sql

    def test_query_with_pagination_empty_results(self):
        """Test pagination with empty result sets."""
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = []

        # Mock count query returning 0
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 0
        self.mock_connection.execute.side_effect = [mock_count_result, mock_result]

        result = self.query_service.query_paginated(
            "SELECT * FROM empty_table", page=1, page_size=10
        )

        assert len(result.data) == 0
        assert result.pagination.total_count == 0
        assert result.pagination.total_pages == 0
        assert result.pagination.has_next is False
        assert result.pagination.has_previous is False

    def test_query_with_pagination_single_page(self):
        """Test pagination when all results fit on one page."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [("Alice",), ("Bob",)]
        mock_result.keys.return_value = ["name"]

        # Mock count query
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 2
        self.mock_connection.execute.side_effect = [mock_count_result, mock_result]

        result = self.query_service.query_paginated(
            "SELECT name FROM users", page=1, page_size=10
        )

        assert len(result.data) == 2
        assert result.pagination.total_pages == 1
        assert result.pagination.has_next is False
        assert result.pagination.has_previous is False

    def test_query_with_pagination_last_page(self):
        """Test pagination on the last page with partial results."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [("Last", "User")]
        mock_result.keys.return_value = ["first_name", "last_name"]

        # Mock count query - 21 total results, page size 10, so page 3 has 1 result
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 21
        self.mock_connection.execute.side_effect = [mock_count_result, mock_result]

        result = self.query_service.query_paginated(
            "SELECT first_name, last_name FROM users", page=3, page_size=10
        )

        assert len(result.data) == 1
        assert result.pagination.page == 3
        assert result.pagination.total_pages == 3
        assert result.pagination.has_next is False
        assert result.pagination.has_previous is True

    def test_query_with_pagination_configuration(self):
        """Test pagination with different configuration options."""
        # Test default page size
        result = self.query_service.get_pagination_config()
        assert result["default_page_size"] == 10
        assert result["max_page_size"] == 1000

        # Test custom configuration
        custom_service = QueryService(
            engine=self.mock_engine,
            validator=self.mock_validator,
            generator=self.mock_generator,
            schema_cache=self.mock_schema_cache,
            query_cache=self.mock_query_cache,
            max_rows=20,
            max_page_size=500,
            inspector=self.mock_inspector,
        )

        config = custom_service.get_pagination_config()
        assert config["default_page_size"] == 20
        assert config["max_page_size"] == 500

    def test_query_with_pagination_caching(self):
        """Test that paginated queries are properly cached."""
        # This test would verify that pagination parameters are included in cache keys
        # and that cached results include pagination metadata

        mock_result = Mock()
        mock_result.fetchall.return_value = [("Test",)]
        mock_result.keys.return_value = ["value"]

        # Mock count query
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 10
        self.mock_connection.execute.side_effect = [mock_count_result, mock_result]

        # Configure cache to return None (cache miss) then the result
        cache_key = None

        def mock_get(key):
            nonlocal cache_key
            cache_key = key
            return None

        self.mock_query_cache.get.side_effect = mock_get

        result = self.query_service.query_paginated(
            "SELECT value FROM test_table", page=2, page_size=5
        )

        # Verify cache key includes pagination parameters
        assert cache_key is not None
        assert "page=2" in cache_key
        assert "page_size=5" in cache_key

        # Verify cache.set was called with pagination metadata
        self.mock_query_cache.set.assert_called_once()
        cached_result = self.mock_query_cache.set.call_args[0][1]
        assert hasattr(cached_result, "pagination")

    def test_query_with_pagination_error_handling(self):
        """Test error handling in paginated queries."""
        # Test database error during count query
        self.mock_connection.execute.side_effect = Exception(
            "Database connection failed"
        )

        with pytest.raises(Exception, match="Database connection failed"):
            self.query_service.query_paginated(
                "SELECT * FROM users", page=1, page_size=10
            )

    def test_query_with_pagination_sql_injection_prevention(self):
        """Test that pagination parameters are properly sanitized."""
        # Verify that pagination parameters are treated as integers and cannot be injected

        # These should be safely converted to integers
        with pytest.raises(ValueError):
            self.query_service.query_paginated(
                "SELECT * FROM users", page="1; DROP TABLE users; --", page_size=10
            )

        with pytest.raises(ValueError):
            self.query_service.query_paginated(
                "SELECT * FROM users", page=1, page_size="10 OR 1=1"
            )

    def test_pagination_info_json_serialization(self):
        """Test that pagination info can be serialized to JSON."""
        pagination = PaginationInfo.create(page=2, page_size=10, total_count=50)

        # Should be serializable to dict
        data = pagination.to_dict()
        expected_keys = {
            "page",
            "page_size",
            "total_count",
            "total_pages",
            "has_next",
            "has_previous",
        }
        assert set(data.keys()) == expected_keys
        assert data["page"] == 2
        assert data["total_pages"] == 5
        assert data["has_next"] is True
        assert data["has_previous"] is True
