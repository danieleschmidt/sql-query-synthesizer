"""Tests for QueryValidatorService."""

import pytest
from sql_synthesizer.services.query_validator_service import QueryValidatorService
from sql_synthesizer.user_experience import (
    UserFriendlyError,
    create_empty_question_error,
    create_question_too_long_error,
    create_unsafe_input_error,
    create_invalid_sql_error,
    create_multiple_statements_error,
    create_invalid_table_error,
)


class TestQueryValidatorService:
    """Test QueryValidatorService functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = QueryValidatorService(max_question_length=100)

    def test_validate_question_empty(self):
        """Test validation of empty questions."""
        with pytest.raises(UserFriendlyError):
            self.validator.validate_question("")
        
        with pytest.raises(UserFriendlyError):
            self.validator.validate_question("   ")

    def test_validate_question_too_long(self):
        """Test validation of overly long questions."""
        long_question = "a" * 101
        with pytest.raises(UserFriendlyError):
            self.validator.validate_question(long_question)

    def test_validate_question_unsafe_sql(self):
        """Test detection of SQL injection attempts."""
        unsafe_questions = [
            "SELECT * FROM users; DROP TABLE users;",
            "users'; DELETE FROM users; --",
            "'; UPDATE users SET password='hacked'; --",
            "Show users; INSERT INTO users VALUES('evil', 'hack');",
        ]
        
        for question in unsafe_questions:
            with pytest.raises(UserFriendlyError):
                self.validator.validate_question(question)

    def test_validate_question_valid(self):
        """Test validation of valid questions."""
        valid_questions = [
            "Show me all users",
            "What is the average order value?",
            "List customers from California",
            "Count products with high ratings",
        ]
        
        for question in valid_questions:
            # Should not raise exception
            sanitized = self.validator.validate_question(question)
            assert isinstance(sanitized, str)
            assert len(sanitized) > 0

    def test_validate_sql_empty(self):
        """Test validation of empty SQL."""
        with pytest.raises(UserFriendlyError):
            self.validator.validate_sql("")
        
        with pytest.raises(UserFriendlyError):
            self.validator.validate_sql("   ")

    def test_validate_sql_multiple_statements(self):
        """Test detection of multiple SQL statements."""
        multi_statements = [
            "SELECT * FROM users; DROP TABLE users;",
            "SELECT 1; SELECT 2;",
            "INSERT INTO users VALUES(1, 'test'); SELECT * FROM users;",
        ]
        
        for sql in multi_statements:
            with pytest.raises(UserFriendlyError):
                self.validator.validate_sql(sql)

    def test_validate_sql_invalid_syntax(self):
        """Test detection of invalid SQL syntax."""
        invalid_sql = [
            "SELCT * FROM users",  # Typo
            "SELECT * FRM users",   # Missing FROM
            "SELECT * FROM",        # Incomplete
            "SELECT * FROM users WHERE",  # Incomplete WHERE
        ]
        
        for sql in invalid_sql:
            with pytest.raises(UserFriendlyError):
                self.validator.validate_sql(sql)

    def test_validate_sql_valid(self):
        """Test validation of valid SQL."""
        valid_sql = [
            "SELECT * FROM users",
            "SELECT name, email FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM orders WHERE date > '2024-01-01'",
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
        ]
        
        for sql in valid_sql:
            # Should not raise exception
            validated = self.validator.validate_sql(sql)
            assert isinstance(validated, str)
            assert len(validated) > 0

    def test_validate_table_name_invalid(self):
        """Test validation of invalid table names."""
        invalid_tables = [
            "",
            "   ",
            "users; DROP TABLE users;",
            "123invalid",
            "user-name",
            "user name",
            "users'",
        ]
        
        for table in invalid_tables:
            with pytest.raises(UserFriendlyError):
                self.validator.validate_table_name(table)

    def test_validate_table_name_valid(self):
        """Test validation of valid table names."""
        valid_tables = [
            "users",
            "user_profiles",
            "Orders",
            "USER_DATA",
            "_internal_table",
            "table123",
        ]
        
        for table in valid_tables:
            # Should not raise exception
            validated = self.validator.validate_table_name(table)
            assert validated == table

    def test_sanitize_question(self):
        """Test question sanitization."""
        test_cases = [
            ("  Show me users  ", "Show me users"),
            ("Show\nme\tusers", "Show me users"),
            ("Show me users!", "Show me users!"),
            ("What's the average?", "What's the average?"),
        ]
        
        for input_q, expected in test_cases:
            result = self.validator.sanitize_question(input_q)
            assert result == expected