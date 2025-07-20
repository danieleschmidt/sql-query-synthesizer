"""Minimal Flask web app exposing :class:`QueryAgent`."""

from __future__ import annotations

from flask import Flask, request, jsonify, render_template, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .query_agent import QueryAgent
from .config import config

def create_app(agent: QueryAgent) -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        if request.method == "POST":
            q = request.form.get("question", "")
            try:
                res = agent.query(q)
                return render_template(
                    "index.html", 
                    sql=res.sql, 
                    data=res.data, 
                    input_size=config.webapp_input_size,
                    question=q
                )
            except Exception as e:
                return render_template(
                    "index.html", 
                    error=str(e), 
                    input_size=config.webapp_input_size,
                    question=q
                )
        return render_template("index.html", input_size=config.webapp_input_size)

    @app.post("/api/query")
    def api_query() -> tuple[str, int]:
        q = request.json.get("question", "")
        res = agent.query(q)
        return jsonify(sql=res.sql, data=res.data), 200

    @app.get("/metrics")
    def metrics() -> Response:
        """Expose Prometheus metrics."""
        data = generate_latest()
        return Response(data, mimetype=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run web UI for QueryAgent")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--port", type=int, default=config.webapp_port)
    args = parser.parse_args()

    agent = QueryAgent(args.database_url)
    app = create_app(agent)
    app.run(port=args.port)


if __name__ == "__main__":
    main()
