from sql_synthesizer.generator import naive_generate_sql


def test_naive_generate_sql_count():
    sql = naive_generate_sql("How many users?", ["users"], 5)
    assert sql == 'SELECT COUNT(*) FROM "users"'


def test_naive_generate_sql_select():
    sql = naive_generate_sql("List users", ["users"], 3)
    assert sql == 'SELECT * FROM "users" LIMIT 3'


def test_naive_generate_sql_placeholder():
    sql = naive_generate_sql("unknown", ["users"], 5)
    assert sql.startswith("-- No matching table found for:")


def test_naive_generate_sql_invalid_max_rows():
    """Test that invalid max_rows values are handled safely."""
    # Test with negative max_rows
    sql = naive_generate_sql("List users", ["users"], -1)
    assert "LIMIT 5" in sql  # Should default to 5

    # Test with very large max_rows
    sql = naive_generate_sql("List users", ["users"], 50000)
    assert "LIMIT 5" in sql  # Should default to 5

    # Test with non-integer max_rows
    sql = naive_generate_sql("List users", ["users"], "invalid")
    assert "LIMIT 5" in sql  # Should default to 5


def test_naive_generate_sql_invalid_table_names():
    """Test that invalid table names are skipped safely."""
    # Test with SQL injection in table names
    invalid_tables = ["users; DROP TABLE users", "users--", "users/**/"]
    sql = naive_generate_sql("List users", invalid_tables, 5)
    assert sql.startswith("-- No matching table found for:")


def test_naive_generate_sql_empty_tables():
    """Test with empty table list."""
    sql = naive_generate_sql("List users", [], 5)
    assert sql.startswith("-- No matching table found for:")
