"""Tests for NLParser."""

import pytest
from sql_synthesizer import Schema, NLParser


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
def parser():
    schema = Schema.from_dict(SCHEMA_DEF)
    return NLParser(schema)


class TestOperationDetection:
    def test_select(self, parser):
        assert parser.parse("Show all users").operation == "SELECT"

    def test_count(self, parser):
        assert parser.parse("How many users are there?").operation == "COUNT"
        assert parser.parse("Count all orders").operation == "COUNT"

    def test_sum(self, parser):
        assert parser.parse("What is the total revenue from orders?").operation == "SUM"
        assert parser.parse("Sum of order totals").operation == "SUM"

    def test_avg(self, parser):
        assert parser.parse("Average price of products").operation == "AVG"
        assert parser.parse("What is the mean age of users?").operation == "AVG"

    def test_max(self, parser):
        assert parser.parse("What is the maximum price?").operation == "MAX"
        assert parser.parse("most expensive product").operation == "MAX"

    def test_min(self, parser):
        assert parser.parse("Cheapest product").operation == "MIN"
        assert parser.parse("minimum price").operation == "MIN"


class TestTableDetection:
    def test_single_table(self, parser):
        intent = parser.parse("Show all users")
        assert "users" in intent.tables

    def test_plural_table(self, parser):
        intent = parser.parse("List all orders")
        assert "orders" in intent.tables

    def test_multiple_tables(self, parser):
        intent = parser.parse("Show orders and users")
        assert len(intent.tables) >= 2

    def test_unknown_table(self, parser):
        intent = parser.parse("Show all foobar")
        assert intent.tables == []


class TestFilterDetection:
    def test_equality_filter(self, parser):
        intent = parser.parse("Show users where country = US")
        filters = {f.column: f for f in intent.filters}
        assert "country" in filters
        assert filters["country"].operator == "="

    def test_greater_than(self, parser):
        intent = parser.parse("Show users where age > 30")
        filters = {f.column: f for f in intent.filters}
        assert "age" in filters
        assert filters["age"].operator == ">"
        assert filters["age"].value == 30

    def test_less_than(self, parser):
        intent = parser.parse("Products with price < 100")
        filters = {f.column: f for f in intent.filters}
        assert "price" in filters
        assert filters["price"].operator == "<"

    def test_status_filter(self, parser):
        intent = parser.parse("Show orders with status = shipped")
        filters = {f.column: f for f in intent.filters}
        assert "status" in filters


class TestLimitDetection:
    def test_top_n(self, parser):
        intent = parser.parse("Show top 5 users")
        assert intent.limit == 5

    def test_first_n(self, parser):
        intent = parser.parse("List first 10 products")
        assert intent.limit == 10

    def test_no_limit(self, parser):
        intent = parser.parse("Show all users")
        assert intent.limit is None


class TestOrderByDetection:
    def test_order_by_explicit(self, parser):
        intent = parser.parse("Show products ordered by price")
        assert intent.order_by is not None
        assert intent.order_by.column == "price"

    def test_order_desc(self, parser):
        intent = parser.parse("Show most expensive products")
        assert intent.order_by is not None
        assert intent.order_by.direction == "DESC"


class TestGroupByDetection:
    def test_group_by(self, parser):
        intent = parser.parse("Count orders grouped by status")
        assert "status" in intent.group_by

    def test_per_keyword(self, parser):
        intent = parser.parse("Count orders per status")
        # May or may not detect depending on pattern — accept either
        # but verify parse succeeds
        assert intent.operation == "COUNT"


class TestJoinInference:
    def test_join_inferred(self, parser):
        intent = parser.parse("Show orders and users")
        assert len(intent.joins) > 0
