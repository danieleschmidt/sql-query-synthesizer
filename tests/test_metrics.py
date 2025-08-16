from pathlib import Path

from prometheus_client import REGISTRY
from sqlalchemy import create_engine, text

from sql_synthesizer import QueryAgent


def make_agent(tmp_path: Path) -> QueryAgent:
    db = tmp_path / "m.db"
    url = f"sqlite:///{db}"
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE users(id INTEGER PRIMARY KEY);"))
    return QueryAgent(url)


def test_metrics_record(tmp_path: Path):
    agent = make_agent(tmp_path)
    before = REGISTRY.get_sample_value("queries_total", {"type": "query"}) or 0
    agent.query("How many users?")
    after = REGISTRY.get_sample_value("queries_total", {"type": "query"})
    assert after == before + 1
