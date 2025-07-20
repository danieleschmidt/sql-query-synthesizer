from pathlib import Path
import sys

import pytest
from sqlalchemy import text, create_engine

sys.path.append(str(Path(__file__).resolve().parents[1]))
import query_agent
from sql_synthesizer import QueryAgent


@pytest.fixture()
def agent(tmp_path: Path) -> QueryAgent:
    db = tmp_path / "test.db"
    url = f"sqlite:///{db}"
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"))
        conn.execute(text("INSERT INTO users (name) VALUES ('Alice'), ('Bob');"))
    return QueryAgent(url)


def test_discover_schema(agent: QueryAgent):
    tables = agent.discover_schema()
    assert tables == ["users"]
    # second call should hit cache
    assert agent.discover_schema() == ["users"]


def test_row_count(agent: QueryAgent):
    assert agent.row_count("users") == 2


def test_batch_row_counts(agent: QueryAgent):
    counts = agent.batch_row_counts(["users"])
    assert counts == {"users": 2}


def test_list_table_counts(agent: QueryAgent):
    pairs = agent.list_table_counts()
    assert pairs == [("users", 2)]


def test_generate_sql(agent: QueryAgent):
    sql = agent.generate_sql("How many users do we have?")
    assert sql == 'SELECT COUNT(*) FROM "users"'


def test_query_execute(agent: QueryAgent):
    res = agent.query("List users")
    assert res.sql.startswith('SELECT * FROM "users"')
    assert len(res.data) == 2


def test_cli_list_tables(agent: QueryAgent, capsys: pytest.CaptureFixture[str]):
    url = agent.engine.url.render_as_string(hide_password=False)
    query_agent.main(["--database-url", url, "--list-tables"])
    out = capsys.readouterr().out
    assert "users" in out


def test_cli_execute_sql(agent: QueryAgent, capsys: pytest.CaptureFixture[str]):
    url = agent.engine.url.render_as_string(hide_password=False)
    query_agent.main([
        "--database-url",
        url,
        "--execute-sql",
        "SELECT COUNT(*) AS c FROM users",
    ])
    out = capsys.readouterr().out
    assert "SELECT COUNT(*) AS c FROM users" in out


def test_row_count_injection(agent: QueryAgent):
    with pytest.raises(ValueError):
        agent.row_count("users; DROP TABLE users;")


def test_execute_sql_validation(agent: QueryAgent):
    with pytest.raises(ValueError):
        agent.execute_sql("SELECT 1; DROP TABLE users;")


def test_query_logs(agent: QueryAgent, caplog):
    with caplog.at_level("INFO"):
        res = agent.query("List users")
    assert any("Executing SQL" in rec.message for rec in caplog.records)
    assert any("Query executed" in rec.message for rec in caplog.records)
    assert res.data


def test_openai_timeout(monkeypatch, tmp_path: Path):
    db = tmp_path / "db.db"
    url = f"sqlite:///{db}"
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE t(id INTEGER PRIMARY KEY);"))

    captured = {}

    class DummyAdapter:
        def __init__(self, api_key: str, model: str = "gpt", timeout=None) -> None:
            captured["timeout"] = timeout

    monkeypatch.setattr("sql_synthesizer.query_agent.OpenAIAdapter", DummyAdapter)
    QueryAgent(url, openai_api_key="key", openai_timeout=7.5)  # pragma: allowlist secret
    assert captured["timeout"] == 7.5
