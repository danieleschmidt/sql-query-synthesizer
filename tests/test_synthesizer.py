"""Tests for SQLSynthesizer."""

import pytest
from sql_synthesizer import Schema, NLParser, SQLSynthesizer


SCHEMA_DEF = {
    "tables": {
        "users": {
            "columns": [
                {"name": "id",      "type": "INTEGER", "primary_key": True},
                {"name": "name",    "type": "TEXT"},
                {"name": "age",     "type": "INTEGER"},
                {"name": "country", "type": "TEXT"},
            ]
        },
        "products": {
            "columns": [
                {"name": "id",       "type": "INTEGER", "primary_key": True},
                {"name": "name",     "type": "TEXT"},
                {"name": "price",    "type": "REAL"},
                {"name": "category", "type": "TEXT"},
            ]
        },
        "orders": {
            "columns": [
                {"name": "id",         "type": "INTEGER", "primary_key": True},
                {"name": "user_id",    "type": "INTEGER"},
                {"name": "product_id", "type": "INTEGER"},
                {"name": "total",      "type": "REAL"},
                {"name": "status",     "type": "TEXT"},
            ]
        },
    },
    "foreign_keys": [
        {"from": "orders.user_id",    "to": "users.id"},
        {"from": "orders.product_id", "to": "products.id"},
    ],
}


@pytest.fixture
def schema():
    return Schema.from_dict(SCHEMA_DEF)


@pytest.fixture
def pipeline(schema):
    return NLParser(schema), SQLSynthesizer(schema)


def synth(pipeline, nl):
    parser, synthesizer = pipeline
    intent = parser.parse(nl)
    return synthesizer.synthesize(intent).upper()


class TestSelectGeneration:
    def test_select_all(self, pipeline):
        sql = synth(pipeline, "Show all users")
        assert "SELECT" in sql
        assert "FROM USERS" in sql

    def test_select_with_limit(self, pipeline):
        sql = synth(pipeline, "Show top 5 products")
        assert "LIMIT 5" in sql
        assert "FROM PRODUCTS" in sql

    def test_select_with_where(self, pipeline):
        sql = synth(pipeline, "Show orders where status = shipped")
        assert "WHERE" in sql
        assert "STATUS" in sql
        assert "FROM ORDERS" in sql

    def test_select_with_order(self, pipeline):
        sql = synth(pipeline, "Show products ordered by price")
        assert "ORDER BY" in sql
        assert "PRICE" in sql


class TestCountGeneration:
    def test_count_all(self, pipeline):
        sql = synth(pipeline, "How many users are there?")
        assert "COUNT(*)" in sql
        assert "FROM USERS" in sql

    def test_count_with_filter(self, pipeline):
        sql = synth(pipeline, "How many orders have status = shipped")
        assert "COUNT(*)" in sql
        assert "WHERE" in sql

    def test_count_grouped(self, pipeline):
        sql = synth(pipeline, "Count orders grouped by status")
        assert "COUNT(*)" in sql
        assert "GROUP BY" in sql
        assert "STATUS" in sql


class TestAggregateGeneration:
    def test_sum(self, pipeline):
        sql = synth(pipeline, "What is the total revenue from orders?")
        assert "SUM(" in sql
        assert "FROM ORDERS" in sql

    def test_avg(self, pipeline):
        sql = synth(pipeline, "Average price of products")
        assert "AVG(" in sql
        assert "FROM PRODUCTS" in sql

    def test_max(self, pipeline):
        sql = synth(pipeline, "Most expensive product")
        assert "MAX(" in sql or "ORDER BY" in sql  # either approach is valid

    def test_min(self, pipeline):
        sql = synth(pipeline, "Cheapest product price")
        assert "MIN(" in sql


class TestJoinGeneration:
    def test_join_produced(self, pipeline):
        sql = synth(pipeline, "Show orders and users")
        assert "JOIN" in sql

    def test_join_on_clause(self, pipeline):
        sql = synth(pipeline, "List orders and users")
        assert "ON" in sql


class TestEdgeCases:
    def test_unknown_table_returns_comment(self, pipeline):
        sql = synth(pipeline, "Show all foobar")
        assert sql.startswith("--")
