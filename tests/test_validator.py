"""Tests for QueryValidator."""

import pytest
from sql_synthesizer import Schema, QueryValidator


SCHEMA_DEF = {
    "tables": {
        "users": {
            "columns": [
                {"name": "id",   "type": "INTEGER", "primary_key": True},
                {"name": "name", "type": "TEXT"},
                {"name": "age",  "type": "INTEGER"},
            ]
        },
        "orders": {
            "columns": [
                {"name": "id",      "type": "INTEGER", "primary_key": True},
                {"name": "user_id", "type": "INTEGER"},
                {"name": "total",   "type": "REAL"},
                {"name": "status",  "type": "TEXT"},
            ]
        },
    },
    "foreign_keys": [
        {"from": "orders.user_id", "to": "users.id"}
    ],
}


@pytest.fixture
def validator():
    return QueryValidator(Schema.from_dict(SCHEMA_DEF))


class TestValidSQL:
    def test_simple_select(self, validator):
        result = validator.validate("SELECT * FROM users")
        assert result.valid, str(result)

    def test_select_with_where(self, validator):
        result = validator.validate("SELECT * FROM users WHERE age > 30")
        assert result.valid, str(result)

    def test_count(self, validator):
        result = validator.validate("SELECT COUNT(*) AS count FROM orders")
        assert result.valid, str(result)

    def test_join(self, validator):
        result = validator.validate(
            "SELECT * FROM orders JOIN users ON orders.user_id = users.id"
        )
        assert result.valid, str(result)

    def test_aggregate(self, validator):
        result = validator.validate("SELECT SUM(total) AS sum_total FROM orders")
        assert result.valid, str(result)

    def test_group_by(self, validator):
        result = validator.validate(
            "SELECT status, COUNT(*) AS count FROM orders GROUP BY status"
        )
        assert result.valid, str(result)

    def test_limit(self, validator):
        result = validator.validate("SELECT * FROM users LIMIT 10")
        assert result.valid, str(result)


class TestInvalidSQL:
    def test_nonexistent_table(self, validator):
        result = validator.validate("SELECT * FROM nonexistent_table")
        assert not result.valid
        assert any("nonexistent_table" in e for e in result.errors)

    def test_join_bad_column(self, validator):
        result = validator.validate(
            "SELECT * FROM orders JOIN users ON orders.bad_col = users.id"
        )
        assert not result.valid

    def test_comment_placeholder(self, validator):
        result = validator.validate("-- No matching table found")
        assert not result.valid

    def test_bad_syntax(self, validator):
        result = validator.validate("SELECT FROM WHERE *** GARBAGE")
        assert not result.valid
