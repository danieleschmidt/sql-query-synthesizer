"""Integration tests: NL query → parse → synthesize → validate → execute."""

import pytest
from sql_synthesizer import Schema, NLParser, SQLSynthesizer, QueryValidator, QueryExecutor


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
                {"name": "stock",    "type": "INTEGER"},
            ]
        },
        "orders": {
            "columns": [
                {"name": "id",         "type": "INTEGER", "primary_key": True},
                {"name": "user_id",    "type": "INTEGER"},
                {"name": "product_id", "type": "INTEGER"},
                {"name": "quantity",   "type": "INTEGER"},
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

USERS = [
    {"id": 1, "name": "Alice",  "age": 32, "country": "US"},
    {"id": 2, "name": "Bob",    "age": 24, "country": "UK"},
    {"id": 3, "name": "Carol",  "age": 41, "country": "US"},
    {"id": 4, "name": "Dave",   "age": 19, "country": "CA"},
    {"id": 5, "name": "Eve",    "age": 28, "country": "US"},
]

PRODUCTS = [
    {"id": 1, "name": "Laptop",     "price": 999.99,  "category": "Electronics", "stock": 15},
    {"id": 2, "name": "Headphones", "price": 79.99,   "category": "Electronics", "stock": 50},
    {"id": 3, "name": "Coffee Mug", "price": 12.99,   "category": "Kitchen",     "stock": 200},
    {"id": 4, "name": "Desk Chair", "price": 349.00,  "category": "Furniture",   "stock": 8},
    {"id": 5, "name": "Keyboard",   "price": 129.99,  "category": "Electronics", "stock": 30},
]

ORDERS = [
    {"id": 1, "user_id": 1, "product_id": 1, "quantity": 1, "total": 999.99,  "status": "shipped"},
    {"id": 2, "user_id": 2, "product_id": 2, "quantity": 2, "total": 159.98,  "status": "pending"},
    {"id": 3, "user_id": 3, "product_id": 3, "quantity": 5, "total": 64.95,   "status": "delivered"},
    {"id": 4, "user_id": 1, "product_id": 5, "quantity": 1, "total": 129.99,  "status": "shipped"},
    {"id": 5, "user_id": 4, "product_id": 4, "quantity": 1, "total": 349.00,  "status": "pending"},
    {"id": 6, "user_id": 5, "product_id": 2, "quantity": 1, "total": 79.99,   "status": "delivered"},
    {"id": 7, "user_id": 2, "product_id": 1, "quantity": 1, "total": 999.99,  "status": "delivered"},
    {"id": 8, "user_id": 3, "product_id": 5, "quantity": 2, "total": 259.98,  "status": "shipped"},
]


@pytest.fixture(scope="module")
def pipeline():
    schema = Schema.from_dict(SCHEMA_DEF)
    parser = NLParser(schema)
    synth = SQLSynthesizer(schema)
    validator = QueryValidator(schema)
    executor = QueryExecutor(schema)
    executor.setup_schema()
    executor.insert_many("users", USERS)
    executor.insert_many("products", PRODUCTS)
    executor.insert_many("orders", ORDERS)
    yield parser, synth, validator, executor
    executor.close()


def run(pipeline, nl):
    """Parse → synthesize → validate → execute. Returns (sql, result)."""
    parser, synth, validator, executor = pipeline
    intent = parser.parse(nl)
    sql = synth.synthesize(intent)
    validation = validator.validate(sql)
    assert validation.valid, f"SQL invalid for {nl!r}: {validation}  SQL={sql!r}"
    result = executor.execute(sql)
    assert result.success, f"Execution failed for {nl!r}: {result.error}  SQL={sql!r}"
    return sql, result


class TestEndToEnd:
    def test_select_all_users(self, pipeline):
        _, result = run(pipeline, "Show all users")
        assert result.row_count == 5

    def test_count_users(self, pipeline):
        _, result = run(pipeline, "How many users are there?")
        assert result.rows[0]["count"] == 5

    def test_filter_by_country(self, pipeline):
        _, result = run(pipeline, "Show users where country = US")
        assert result.row_count == 3
        for row in result.rows:
            assert row["country"] == "US"

    def test_products_below_price(self, pipeline):
        _, result = run(pipeline, "List all products with price < 200")
        for row in result.rows:
            assert row["price"] < 200

    def test_total_revenue(self, pipeline):
        _, result = run(pipeline, "What is the total revenue from orders?")
        expected = sum(o["total"] for o in ORDERS)
        assert result.rows[0][result.columns[0]] == pytest.approx(expected, rel=1e-4)

    def test_orders_by_status(self, pipeline):
        _, result = run(pipeline, "Show orders with status = shipped")
        assert result.row_count > 0
        for row in result.rows:
            assert row["status"] == "shipped"

    def test_count_grouped_by_status(self, pipeline):
        _, result = run(pipeline, "Count orders grouped by status")
        statuses = {r["status"] for r in result.rows}
        assert "shipped" in statuses
        assert "pending" in statuses
        assert "delivered" in statuses

    def test_top_3_expensive_products(self, pipeline):
        _, result = run(pipeline, "top 3 most expensive products")
        assert result.row_count <= 3

    def test_average_order_total(self, pipeline):
        _, result = run(pipeline, "What is the average order total?")
        expected = sum(o["total"] for o in ORDERS) / len(ORDERS)
        assert result.rows[0][result.columns[0]] == pytest.approx(expected, rel=1e-4)

    def test_join_orders_users(self, pipeline):
        _, result = run(pipeline, "List all orders and users")
        assert result.row_count > 0
        assert result.success
