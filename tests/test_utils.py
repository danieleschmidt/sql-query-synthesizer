"""Tests for utility functions."""

import pytest

from sql_synthesizer.utils.helpers import (
    calculate_cache_key,
    extract_client_info,
    format_duration,
    normalize_sql,
    parse_database_url,
    safe_json_dumps,
    sanitize_error_message,
    truncate_string,
    validate_pagination_params,
)
from sql_synthesizer.utils.validators import (
    contains_sql_injection_pattern,
    is_safe_sql_identifier,
    sanitize_input,
    validate_column_list,
    validate_limit_clause,
    validate_order_by_clause,
    validate_query_length,
    validate_table_name,
)


class TestValidators:
    """Test input validation functions."""

    def test_validate_query_length_valid(self):
        """Test query length validation with valid input."""
        assert validate_query_length("SELECT * FROM users") is True
        assert validate_query_length("A" * 1000) is True

    def test_validate_query_length_invalid(self):
        """Test query length validation with invalid input."""
        assert validate_query_length("A" * 20000) is False
        assert validate_query_length("A" * 10001) is False

    def test_validate_table_name_valid(self):
        """Test table name validation with valid names."""
        assert validate_table_name("users") is True
        assert validate_table_name("user_profiles") is True
        assert validate_table_name("_private") is True
        assert validate_table_name("table123") is True

    def test_validate_table_name_invalid(self):
        """Test table name validation with invalid names."""
        assert validate_table_name("") is False
        assert validate_table_name("123invalid") is False
        assert validate_table_name("table-name") is False
        assert validate_table_name("table.name") is False
        assert validate_table_name("select") is False  # Reserved word
        assert validate_table_name("A" * 200) is False  # Too long

    def test_sanitize_input_basic(self):
        """Test basic input sanitization."""
        assert sanitize_input("Hello World") == "Hello World"
        assert sanitize_input("  Multiple   spaces  ") == "Multiple spaces"
        assert sanitize_input("") == ""

    def test_sanitize_input_dangerous_chars(self):
        """Test sanitization of dangerous characters."""
        # Null bytes should be removed
        assert "\x00" not in sanitize_input("test\x00data")

        # Control characters should be removed (except tab, newline, carriage return)
        assert sanitize_input("test\x01\x02data") == "testdata"
        assert sanitize_input("test\t\n\rdata") == "test data"

    def test_is_safe_sql_identifier_valid(self):
        """Test safe SQL identifier validation with valid identifiers."""
        assert is_safe_sql_identifier("column_name") is True
        assert is_safe_sql_identifier("_private") is True
        assert is_safe_sql_identifier("id123") is True

    def test_is_safe_sql_identifier_invalid(self):
        """Test safe SQL identifier validation with invalid identifiers."""
        assert is_safe_sql_identifier("") is False
        assert is_safe_sql_identifier("123invalid") is False
        assert is_safe_sql_identifier("column-name") is False
        assert is_safe_sql_identifier("column.name") is False
        assert is_safe_sql_identifier("A" * 200) is False

    def test_contains_sql_injection_pattern_basic(self):
        """Test SQL injection pattern detection."""
        # Should detect dangerous patterns
        assert contains_sql_injection_pattern("'; DROP TABLE users; --") is not None
        assert contains_sql_injection_pattern("1' OR '1'='1") is not None
        assert (
            contains_sql_injection_pattern("UNION SELECT password FROM users")
            is not None
        )

        # Should not detect safe content
        assert contains_sql_injection_pattern("normal user input") is None
        assert contains_sql_injection_pattern("search for items") is None

    def test_validate_column_list_valid(self):
        """Test column list validation with valid lists."""
        assert validate_column_list(["id", "name", "email"]) is True
        assert validate_column_list(["_private", "user_id"]) is True

    def test_validate_column_list_invalid(self):
        """Test column list validation with invalid lists."""
        assert validate_column_list([]) is False
        assert validate_column_list(["id", "123invalid"]) is False
        assert validate_column_list(["col-name"]) is False
        assert validate_column_list(["col"] * 150) is False  # Too many columns

    def test_validate_order_by_clause_valid(self):
        """Test ORDER BY clause validation with valid clauses."""
        assert validate_order_by_clause("") is True  # Empty is valid
        assert validate_order_by_clause("name") is True
        assert validate_order_by_clause("name ASC") is True
        assert validate_order_by_clause("name DESC") is True
        assert validate_order_by_clause("name ASC, id DESC") is True

    def test_validate_order_by_clause_invalid(self):
        """Test ORDER BY clause validation with invalid clauses."""
        assert validate_order_by_clause("name; DROP TABLE") is False
        assert validate_order_by_clause("123invalid") is False
        assert validate_order_by_clause("name RANDOM") is False

    def test_validate_limit_clause_valid(self):
        """Test LIMIT clause validation with valid values."""
        assert validate_limit_clause("10") is True
        assert validate_limit_clause("100") is True
        assert validate_limit_clause("1000") is True

    def test_validate_limit_clause_invalid(self):
        """Test LIMIT clause validation with invalid values."""
        assert validate_limit_clause("0") is False
        assert validate_limit_clause("-1") is False
        assert validate_limit_clause("abc") is False
        assert validate_limit_clause("20000") is False  # Too large


class TestHelpers:
    """Test helper utility functions."""

    def test_format_duration_milliseconds(self):
        """Test duration formatting for milliseconds."""
        assert format_duration(500.5) == "500.5ms"
        assert format_duration(999.9) == "999.9ms"

    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        assert format_duration(1500) == "1.5s"
        assert format_duration(30000) == "30.0s"

    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        assert format_duration(60000) == "1m 0.0s"
        assert format_duration(90500) == "1m 30.5s"

    def test_truncate_string_no_truncation(self):
        """Test string truncation when no truncation needed."""
        text = "Short text"
        assert truncate_string(text, 20) == text
        assert truncate_string("", 10) == ""

    def test_truncate_string_with_truncation(self):
        """Test string truncation when truncation needed."""
        text = "This is a very long text that needs truncation"
        result = truncate_string(text, 20)
        assert len(result) == 20
        assert result.endswith("...")
        assert result == "This is a very lo..."

    def test_safe_json_dumps_basic(self):
        """Test safe JSON serialization."""
        data = {"key": "value", "number": 123}
        result = safe_json_dumps(data)
        assert '"key": "value"' in result
        assert '"number": 123' in result

    def test_safe_json_dumps_with_error(self):
        """Test safe JSON serialization with problematic objects."""

        # Create an object that can't be serialized normally
        class UnserializableClass:
            pass

        # Should handle gracefully and include error message
        result = safe_json_dumps(UnserializableClass())
        assert "error" in result.lower()

    def test_extract_client_info_basic(self):
        """Test client info extraction from headers."""
        headers = {
            "User-Agent": "Mozilla/5.0",
            "X-Forwarded-For": "192.168.1.1, 10.0.0.1",
            "Accept-Language": "en-US,en;q=0.9",
        }

        info = extract_client_info(headers)
        assert info["user_agent"] == "Mozilla/5.0"
        assert info["client_ip"] == "192.168.1.1"
        assert info["accept_language"] == "en-US,en;q=0.9"

    def test_extract_client_info_missing_headers(self):
        """Test client info extraction with missing headers."""
        headers = {}
        info = extract_client_info(headers)

        assert info["user_agent"] is None
        assert info["client_ip"] is None
        assert info["accept_language"] is None

    def test_sanitize_error_message_paths(self):
        """Test error message sanitization for file paths."""
        error = "Error in file /home/user/secret/file.py"
        sanitized = sanitize_error_message(error)
        assert "/home/user/secret/file.py" not in sanitized
        assert "<path>" in sanitized

    def test_sanitize_error_message_credentials(self):
        """Test error message sanitization for credentials."""
        error = "Connection failed: password=secret123 token=abc456"
        sanitized = sanitize_error_message(error)
        assert "secret123" not in sanitized
        assert "abc456" not in sanitized
        assert "password=<hidden>" in sanitized
        assert "token=<hidden>" in sanitized

    def test_normalize_sql_basic(self):
        """Test SQL normalization."""
        sql = "  SELECT   *   FROM   users  "
        normalized = normalize_sql(sql)
        assert normalized == "SELECT * FROM users;"

    def test_normalize_sql_with_semicolon(self):
        """Test SQL normalization with existing semicolon."""
        sql = "SELECT * FROM users;"
        normalized = normalize_sql(sql)
        assert normalized == "SELECT * FROM users;"
        assert normalized.count(";") == 1

    def test_parse_database_url_postgresql(self):
        """Test database URL parsing for PostgreSQL."""
        url = "postgresql://user:pass@localhost:5432/mydb?sslmode=require"
        parsed = parse_database_url(url)

        assert parsed["scheme"] == "postgresql"
        assert parsed["username"] == "user"
        assert parsed["password"] == "***"  # Should be hidden
        assert parsed["host"] == "localhost"
        assert parsed["port"] == "5432"
        assert parsed["database"] == "mydb"
        assert parsed["params"] == "sslmode=require"

    def test_parse_database_url_invalid(self):
        """Test database URL parsing with invalid URL."""
        parsed = parse_database_url("invalid-url")
        assert parsed == {}

    def test_calculate_cache_key_consistent(self):
        """Test cache key calculation consistency."""
        query1 = "SELECT * FROM users"
        query2 = "select * from users"  # Different case

        key1 = calculate_cache_key(query1)
        key2 = calculate_cache_key(query2)

        # Should be the same due to normalization
        assert key1 == key2
        assert len(key1) == 16  # 16 character hash

    def test_calculate_cache_key_with_params(self):
        """Test cache key calculation with parameters."""
        query = "SELECT * FROM users WHERE id = ?"
        params1 = {"id": 1}
        params2 = {"id": 2}

        key1 = calculate_cache_key(query, params1)
        key2 = calculate_cache_key(query, params2)

        # Should be different due to different parameters
        assert key1 != key2

    def test_validate_pagination_params_valid(self):
        """Test pagination parameter validation with valid input."""
        page, page_size = validate_pagination_params("2", "25")
        assert page == 2
        assert page_size == 25

        page, page_size = validate_pagination_params(1, 10)
        assert page == 1
        assert page_size == 10

    def test_validate_pagination_params_invalid(self):
        """Test pagination parameter validation with invalid input."""
        # Invalid input should return defaults
        page, page_size = validate_pagination_params("invalid", "also_invalid")
        assert page == 1
        assert page_size == 10

        # Zero/negative values should be corrected
        page, page_size = validate_pagination_params("0", "-5")
        assert page == 1
        assert page_size == 1

        # Too large page size should be capped
        page, page_size = validate_pagination_params("1", "5000")
        assert page == 1
        assert page_size == 1000
