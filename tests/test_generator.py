from sql_synthesizer.generator import naive_generate_sql


def test_naive_generate_sql_count():
    sql = naive_generate_sql("How many users?", ["users"], 5)
    assert sql == 'SELECT COUNT(*) FROM "users";'


def test_naive_generate_sql_select():
    sql = naive_generate_sql("List users", ["users"], 3)
    assert sql == 'SELECT * FROM "users" LIMIT 3;'


def test_naive_generate_sql_placeholder():
    sql = naive_generate_sql("unknown", ["users"], 5)
    assert sql.startswith("-- No matching table found for:")

