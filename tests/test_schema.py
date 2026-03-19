"""Tests for Schema."""

import pytest
from sql_synthesizer import Schema


SCHEMA_DEF = {
    "tables": {
        "users": {
            "columns": [
                {"name": "id",   "type": "INTEGER", "primary_key": True},
                {"name": "name", "type": "TEXT",    "nullable": False},
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


def make_schema():
    return Schema.from_dict(SCHEMA_DEF)


class TestSchemaFromDict:
    def test_tables_loaded(self):
        schema = make_schema()
        assert "users" in [t.lower() for t in schema.table_names]
        assert "orders" in [t.lower() for t in schema.table_names]

    def test_has_table(self):
        schema = make_schema()
        assert schema.has_table("users")
        assert schema.has_table("USERS")
        assert not schema.has_table("products")

    def test_columns_loaded(self):
        schema = make_schema()
        users = schema.table("users")
        assert users is not None
        assert users.has_column("id")
        assert users.has_column("name")
        assert users.has_column("age")
        assert not users.has_column("email")

    def test_primary_key(self):
        schema = make_schema()
        users = schema.table("users")
        id_col = users.column("id")
        assert id_col.primary_key is True

    def test_nullable(self):
        schema = make_schema()
        users = schema.table("users")
        name_col = users.column("name")
        assert name_col.nullable is False

    def test_type_normalization(self):
        schema = make_schema()
        orders = schema.table("orders")
        assert orders.column("total").type == "REAL"
        assert orders.column("status").type == "TEXT"

    def test_foreign_keys(self):
        schema = make_schema()
        fks = schema.foreign_keys
        assert len(fks) == 1
        assert fks[0].from_table == "orders"
        assert fks[0].from_column == "user_id"
        assert fks[0].to_table == "users"
        assert fks[0].to_column == "id"

    def test_join_path(self):
        schema = make_schema()
        fk = schema.join_path("orders", "users")
        assert fk is not None
        fk2 = schema.join_path("users", "orders")
        assert fk2 is not None

    def test_join_path_missing(self):
        schema = make_schema()
        assert schema.join_path("users", "nonexistent") is None

    def test_find_column(self):
        schema = make_schema()
        results = schema.find_column("id")
        table_names = [r[0] for r in results]
        assert "users" in table_names
        assert "orders" in table_names


class TestSchemaFromSQL:
    SQL = """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER
    );

    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        total REAL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """

    def test_from_sql_tables(self):
        schema = Schema.from_sql(self.SQL)
        assert schema.has_table("users")
        assert schema.has_table("orders")

    def test_from_sql_columns(self):
        schema = Schema.from_sql(self.SQL)
        users = schema.table("users")
        assert users.has_column("id")
        assert users.has_column("name")
        assert users.has_column("age")

    def test_from_sql_pk(self):
        schema = Schema.from_sql(self.SQL)
        assert schema.table("users").column("id").primary_key is True

    def test_from_sql_foreign_key(self):
        schema = Schema.from_sql(self.SQL)
        fk = schema.join_path("orders", "users")
        assert fk is not None

    def test_ddl_roundtrip(self):
        schema = Schema.from_dict(SCHEMA_DEF)
        ddl = schema.ddl()
        schema2 = Schema.from_sql(ddl)
        assert schema2.has_table("users")
        assert schema2.has_table("orders")
        assert schema2.table("users").has_column("name")
