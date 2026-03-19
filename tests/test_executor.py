"""Tests for QueryExecutor."""

import pytest
from sql_synthesizer import Schema, QueryExecutor


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

USERS = [
    {"id": 1, "name": "Alice", "age": 32},
    {"id": 2, "name": "Bob",   "age": 24},
    {"id": 3, "name": "Carol", "age": 41},
]

ORDERS = [
    {"id": 1, "user_id": 1, "total": 100.0, "status": "shipped"},
    {"id": 2, "user_id": 2, "total": 50.0,  "status": "pending"},
    {"id": 3, "user_id": 1, "total": 200.0, "status": "shipped"},
]


@pytest.fixture
def executor():
    schema = Schema.from_dict(SCHEMA_DEF)
    ex = QueryExecutor(schema)
    ex.setup_schema()
    ex.insert_many("users", USERS)
    ex.insert_many("orders", ORDERS)
    yield ex
    ex.close()


class TestExecutor:
    def test_select_all(self, executor):
        result = executor.execute("SELECT * FROM users")
        assert result.success
        assert result.row_count == 3

    def test_select_with_filter(self, executor):
        result = executor.execute("SELECT * FROM users WHERE age > 30")
        assert result.success
        assert result.row_count == 2
        for row in result.rows:
            assert row["age"] > 30

    def test_count(self, executor):
        result = executor.execute("SELECT COUNT(*) AS count FROM users")
        assert result.success
        assert result.rows[0]["count"] == 3

    def test_sum(self, executor):
        result = executor.execute("SELECT SUM(total) AS sum_total FROM orders")
        assert result.success
        assert result.rows[0]["sum_total"] == pytest.approx(350.0)

    def test_avg(self, executor):
        result = executor.execute("SELECT AVG(total) AS avg_total FROM orders")
        assert result.success
        assert result.rows[0]["avg_total"] == pytest.approx(350.0 / 3)

    def test_join(self, executor):
        result = executor.execute(
            "SELECT orders.id, users.name FROM orders "
            "JOIN users ON orders.user_id = users.id"
        )
        assert result.success
        assert result.row_count == 3

    def test_group_by(self, executor):
        result = executor.execute(
            "SELECT status, COUNT(*) AS count FROM orders GROUP BY status"
        )
        assert result.success
        totals = {r["status"]: r["count"] for r in result.rows}
        assert totals.get("shipped") == 2
        assert totals.get("pending") == 1

    def test_limit(self, executor):
        result = executor.execute("SELECT * FROM users LIMIT 2")
        assert result.success
        assert result.row_count == 2

    def test_order_by(self, executor):
        result = executor.execute("SELECT * FROM users ORDER BY age DESC")
        assert result.success
        ages = [r["age"] for r in result.rows]
        assert ages == sorted(ages, reverse=True)

    def test_error_handling(self, executor):
        result = executor.execute("SELECT * FROM nonexistent_table_xyz")
        assert not result.success
        assert result.error is not None

    def test_result_columns(self, executor):
        result = executor.execute("SELECT name, age FROM users")
        assert "name" in result.columns
        assert "age" in result.columns

    def test_str_representation(self, executor):
        result = executor.execute("SELECT * FROM users LIMIT 1")
        s = str(result)
        assert "Alice" in s or "name" in s.lower()
