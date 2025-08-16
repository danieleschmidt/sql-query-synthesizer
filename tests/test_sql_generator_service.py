"""Tests for SQLGeneratorService."""

from unittest.mock import Mock, patch

import pytest

from sql_synthesizer.llm_interface import LLMProvider
from sql_synthesizer.services.sql_generator_service import SQLGeneratorService


class TestSQLGeneratorService:
    """Test SQLGeneratorService functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_provider = Mock(spec=LLMProvider)
        self.mock_llm_provider.get_provider_name.return_value = "test-provider"
        self.generator = SQLGeneratorService(llm_provider=self.mock_llm_provider)

    def test_init_without_openai(self):
        """Test initialization without LLM provider."""
        generator = SQLGeneratorService()
        assert generator.llm_provider is None

    def test_init_with_openai(self):
        """Test initialization with LLM provider."""
        assert self.generator.llm_provider is self.mock_llm_provider

    def test_generate_sql_with_openai_success(self):
        """Test SQL generation using LLM provider."""
        self.mock_llm_provider.generate_sql.return_value = "SELECT * FROM users"

        result = self.generator.generate_sql("Show me all users", ["users", "orders"])

        assert result == "SELECT * FROM users"
        self.mock_llm_provider.generate_sql.assert_called_once_with(
            "Show me all users", ["users", "orders"]
        )

    def test_generate_sql_with_openai_failure(self):
        """Test SQL generation when LLM provider fails."""
        self.mock_llm_provider.generate_sql.side_effect = Exception("API Error")

        # Should fall back to naive generation
        result = self.generator.generate_sql("Show me all users", ["users", "orders"])

        # Should return naive generation result
        assert "SELECT" in result
        assert "users" in result

    def test_generate_sql_naive_fallback(self):
        """Test fallback to naive generation when no LLM provider."""
        generator = SQLGeneratorService()  # No LLM provider

        result = generator.generate_sql("Show me all users", ["users", "orders"])

        # Should use naive generation
        assert "SELECT" in result
        assert "users" in result

    def test_generate_sql_naive_basic_patterns(self):
        """Test naive SQL generation with basic patterns."""
        generator = SQLGeneratorService()

        test_cases = [
            ("show users", ["users"], "SELECT * FROM users"),
            ("count orders", ["orders"], "SELECT COUNT(*) FROM orders"),
            ("list products", ["products"], "SELECT * FROM products"),
            ("describe customers", ["customers"], "SELECT * FROM customers"),
        ]

        for question, tables, expected_pattern in test_cases:
            result = generator.generate_sql(question, tables)
            # Check that result contains expected elements
            for word in expected_pattern.split():
                assert word.upper() in result.upper()

    def test_generate_sql_naive_no_matching_table(self):
        """Test naive generation when no table matches."""
        generator = SQLGeneratorService()

        result = generator.generate_sql("show inventory", ["users", "orders"])

        # Should return a comment explaining no match
        assert "-- No matching table found" in result
        assert "inventory" in result

    def test_generate_sql_naive_multiple_tables(self):
        """Test naive generation with multiple matching tables."""
        generator = SQLGeneratorService()

        result = generator.generate_sql(
            "show users and orders", ["users", "orders", "products"]
        )

        # Should handle multiple tables
        assert "SELECT" in result.upper()
        # Should include at least one of the tables
        assert any(table in result for table in ["users", "orders"])

    @patch("sql_synthesizer.services.sql_generator_service.naive_generate_sql")
    def test_generate_sql_uses_naive_function(self, mock_naive):
        """Test that naive generation uses the correct function."""
        mock_naive.return_value = "SELECT * FROM test"
        generator = SQLGeneratorService()

        result = generator.generate_sql("test question", ["test_table"])

        assert result == "SELECT * FROM test"
        mock_naive.assert_called_once_with("test question", ["test_table"])

    def test_generate_sql_empty_question(self):
        """Test generation with empty question."""
        generator = SQLGeneratorService()

        result = generator.generate_sql("", ["users"])

        # Should handle gracefully
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_sql_empty_tables(self):
        """Test generation with no available tables."""
        generator = SQLGeneratorService()

        result = generator.generate_sql("show data", [])

        # Should handle gracefully
        assert isinstance(result, str)
        assert "-- " in result  # Should be a comment explaining the issue
