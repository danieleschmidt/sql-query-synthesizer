#!/usr/bin/env python3
"""
Demo: schema with 3 tables (users, orders, products), 10 NL queries → SQL → executed results.
"""

from sql_synthesizer import Schema, NLParser, SQLSynthesizer, QueryValidator, QueryExecutor

# ── Schema definition ────────────────────────────────────────────────────────

SCHEMA_DEF = {
    "tables": {
        "users": {
            "columns": [
                {"name": "id",         "type": "INTEGER", "primary_key": True},
                {"name": "name",       "type": "TEXT",    "nullable": False},
                {"name": "email",      "type": "TEXT"},
                {"name": "age",        "type": "INTEGER"},
                {"name": "country",    "type": "TEXT"},
            ]
        },
        "products": {
            "columns": [
                {"name": "id",         "type": "INTEGER", "primary_key": True},
                {"name": "name",       "type": "TEXT",    "nullable": False},
                {"name": "price",      "type": "REAL"},
                {"name": "category",   "type": "TEXT"},
                {"name": "stock",      "type": "INTEGER"},
            ]
        },
        "orders": {
            "columns": [
                {"name": "id",         "type": "INTEGER", "primary_key": True},
                {"name": "user_id",    "type": "INTEGER", "nullable": False},
                {"name": "product_id", "type": "INTEGER", "nullable": False},
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

# ── Sample data ───────────────────────────────────────────────────────────────

USERS = [
    {"id": 1, "name": "Alice",   "email": "alice@example.com",   "age": 32, "country": "US"},
    {"id": 2, "name": "Bob",     "email": "bob@example.com",     "age": 24, "country": "UK"},
    {"id": 3, "name": "Carol",   "email": "carol@example.com",   "age": 41, "country": "US"},
    {"id": 4, "name": "Dave",    "email": "dave@example.com",    "age": 19, "country": "CA"},
    {"id": 5, "name": "Eve",     "email": "eve@example.com",     "age": 28, "country": "US"},
]

PRODUCTS = [
    {"id": 1, "name": "Laptop",      "price": 999.99,  "category": "Electronics", "stock": 15},
    {"id": 2, "name": "Headphones",  "price": 79.99,   "category": "Electronics", "stock": 50},
    {"id": 3, "name": "Coffee Mug",  "price": 12.99,   "category": "Kitchen",     "stock": 200},
    {"id": 4, "name": "Desk Chair",  "price": 349.00,  "category": "Furniture",   "stock": 8},
    {"id": 5, "name": "Keyboard",    "price": 129.99,  "category": "Electronics", "stock": 30},
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

# ── NL queries to demonstrate ─────────────────────────────────────────────────

QUERIES = [
    "Show all users",
    "How many users are there?",
    "List all products with price less than 200",
    "What is the total revenue from orders?",
    "Show orders with status shipped",
    "Count orders grouped by status",
    "List the top 3 most expensive products",
    "What is the average order total?",
    "Show all users from country US",
    "List all orders and users",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    schema = Schema.from_dict(SCHEMA_DEF)
    parser = NLParser(schema)
    synth = SQLSynthesizer(schema)
    validator = QueryValidator(schema)

    with QueryExecutor(schema) as executor:
        executor.setup_schema()
        executor.insert_many("users", USERS)
        executor.insert_many("products", PRODUCTS)
        executor.insert_many("orders", ORDERS)

        print("=" * 70)
        print("  sql-query-synthesizer — NL to SQL Demo")
        print("=" * 70)

        for i, nl_query in enumerate(QUERIES, 1):
            print(f"\n[{i:02d}] NL: {nl_query!r}")
            print("-" * 70)

            intent = parser.parse(nl_query)
            sql = synth.synthesize(intent)
            validation = validator.validate(sql)

            print(f"     SQL: {sql.replace(chr(10), ' ')}")
            print(f"     Valid: {validation}")

            if validation.valid:
                result = executor.execute(sql)
                print(f"     Result:\n{_indent(str(result), 10)}")
            else:
                print(f"     Skipped execution due to validation errors")

        print("\n" + "=" * 70)
        print("  Demo complete.")
        print("=" * 70)


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


if __name__ == "__main__":
    main()
