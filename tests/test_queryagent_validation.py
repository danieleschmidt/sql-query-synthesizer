import os
import tempfile

from sqlalchemy import create_engine, text

from sql_query_synthesizer import QueryAgent, ValidationResult

from sql_query_synthesizer.agent import QueryResult


def setup_sqlite_db():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "db.sqlite")
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"))
    engine.dispose()
    return url


def test_queryagent_returns_validation_result(monkeypatch):
    url = setup_sqlite_db()
    agent = QueryAgent(database_url=url)

    def fake_generate_sql(_prompt: str) -> QueryResult:
        return QueryResult(sql="SELECT id FROM users", explanation="")

    monkeypatch.setattr(agent, "generate_sql", fake_generate_sql)

    result = agent.query("show users")
    agent.dispose()
    assert isinstance(result.validation, ValidationResult)
    assert result.validation.suggestions == []
