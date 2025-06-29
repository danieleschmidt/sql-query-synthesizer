from pathlib import Path

from sqlalchemy import text, create_engine

from sql_synthesizer import QueryAgent, create_app


def make_agent(tmp_path: Path) -> QueryAgent:
    db = tmp_path / "web.db"
    url = f"sqlite:///{db}"
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE users(id INTEGER PRIMARY KEY);"))
    return QueryAgent(url)


def test_webapp_query(tmp_path):
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()
    resp = client.post("/api/query", json={"question": "How many users?"})
    assert resp.status_code == 200
    assert "sql" in resp.get_json()


def test_index_page_get(tmp_path):
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Ask a question" in resp.data


def test_index_page_post(tmp_path):
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()
    resp = client.post("/", data={"question": "hi"})
    assert resp.status_code == 200
    assert b"SQL" in resp.data


def test_metrics_endpoint(tmp_path):
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"queries_total" in resp.data
