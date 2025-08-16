from pathlib import Path
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, text

from sql_synthesizer import QueryAgent, create_app
from sql_synthesizer.webapp import main


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
    assert b"SQL Synthesizer" in resp.data
    assert b"Ask a question about your data" in resp.data


def test_index_page_post(tmp_path):
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()
    resp = client.post("/", data={"question": "How many users?"})
    assert resp.status_code == 200
    assert b"Generated SQL" in resp.data


def test_metrics_endpoint(tmp_path):
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"queries_total" in resp.data


@patch("sys.argv", ["webapp", "--database-url", "sqlite:///test.db", "--port", "5001"])
@patch("sql_synthesizer.webapp.QueryAgent")
@patch("sql_synthesizer.webapp.create_app")
def test_main_function(mock_create_app, mock_query_agent):
    """Test the main function for CLI argument parsing."""
    mock_agent = MagicMock()
    mock_query_agent.return_value = mock_agent

    mock_app = MagicMock()
    mock_create_app.return_value = mock_app

    main()

    # Verify QueryAgent was called with database URL
    mock_query_agent.assert_called_once_with("sqlite:///test.db")

    # Verify create_app was called with agent
    mock_create_app.assert_called_once_with(mock_agent)

    # Verify app.run was called with port
    mock_app.run.assert_called_once_with(port=5001)


def test_api_query_error_handling(tmp_path):
    """Test API error handling for invalid questions."""
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()

    # Test with empty question - should return 500 due to validation
    resp = client.post("/api/query", json={"question": ""})
    assert resp.status_code == 500

    # Test with valid question should work
    resp = client.post("/api/query", json={"question": "How many users?"})
    assert resp.status_code == 200
    assert "sql" in resp.get_json()


def test_index_post_empty_question(tmp_path):
    """Test index page with empty question."""
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()

    # Test with empty question - should show error gracefully
    resp = client.post("/", data={"question": ""})
    assert resp.status_code == 200
    assert b"Error" in resp.data


def test_template_security_headers(tmp_path):
    """Test that security headers are present in template."""
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Content-Security-Policy" in resp.data
    assert b"default-src 'self'" in resp.data


def test_template_input_escaping(tmp_path):
    """Test that user input is properly escaped in templates."""
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()

    # Test with potentially malicious input
    malicious_input = "<script>alert('xss')</script>"
    resp = client.post("/", data={"question": malicious_input})
    assert resp.status_code == 200
    # Should be escaped and not contain raw script tags
    assert b"<script>" not in resp.data
    assert b"&lt;script&gt;" in resp.data


def test_template_styling_and_ux(tmp_path):
    """Test that template includes proper styling and UX elements."""
    agent = make_agent(tmp_path)
    app = create_app(agent)
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200

    # Check for CSS styling
    assert b"<style>" in resp.data
    assert b"font-family" in resp.data

    # Check for UX improvements
    assert b"Try asking questions like:" in resp.data
    assert b"placeholder=" in resp.data
    assert b"required" in resp.data
