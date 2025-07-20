"""Secure Flask web app exposing :class:`QueryAgent` with comprehensive security features."""

from __future__ import annotations
import logging
import time

from flask import Flask, request, jsonify, render_template, Response, session
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .query_agent import QueryAgent
from .config import config
from .security import SecurityMiddleware, InputValidator

logger = logging.getLogger(__name__)

def create_app(agent: QueryAgent) -> Flask:
    app = Flask(__name__)
    
    # Initialize security middleware
    security = SecurityMiddleware(app)
    validator = InputValidator()

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        if request.method == "GET":
            # Generate CSRF token for the form
            csrf_token = None
            if config.webapp_csrf_enabled:
                csrf_token = security.csrf.generate_token()
                session['csrf_token'] = csrf_token
            
            return render_template(
                "index.html", 
                input_size=config.webapp_input_size,
                csrf_token=csrf_token
            )
        
        # POST request handling
        q = request.form.get("question", "")
        
        # Input validation
        if not validator.validate_question_length(q):
            error_msg = f"Question too long. Maximum {config.max_question_length} characters allowed."
            logger.warning(f"Question length validation failed: {len(q)} characters")
            return render_template(
                "index.html", 
                error=error_msg, 
                input_size=config.webapp_input_size,
                question=q[:100] + "..." if len(q) > 100 else q
            ), 400
        
        # Sanitize input
        sanitized_question = validator.sanitize_question(q)
        
        try:
            res = agent.query(sanitized_question)
            return render_template(
                "index.html", 
                sql=res.sql, 
                data=res.data, 
                input_size=config.webapp_input_size,
                question=sanitized_question
            )
        except Exception as e:
            # Log the full error but return sanitized message
            logger.error(f"Query execution failed: {str(e)}")
            
            # Provide user-friendly error without sensitive details
            if "invalid" in str(e).lower() or "syntax" in str(e).lower():
                error_msg = "Invalid query. Please check your question and try again."
            elif "timeout" in str(e).lower():
                error_msg = "Query timed out. Please try a simpler question."
            elif "connection" in str(e).lower():
                error_msg = "Database connection issue. Please try again later."
            else:
                error_msg = "An error occurred while processing your query. Please try again."
            
            return render_template(
                "index.html", 
                error=error_msg, 
                input_size=config.webapp_input_size,
                question=sanitized_question
            ), 500

    @app.post("/api/query")
    def api_query() -> tuple[str, int]:
        # Validate JSON structure
        if not request.is_json:
            logger.warning("API request without JSON content type")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        # Validate required fields
        is_valid, error_msg = validator.validate_json_structure(data, ['question'])
        if not is_valid:
            logger.warning(f"API validation failed: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        q = data.get("question", "")
        
        # Input validation
        if not validator.validate_question_length(q):
            error_msg = f"Question too long. Maximum {config.max_question_length} characters allowed."
            logger.warning(f"API question length validation failed: {len(q)} characters")
            return jsonify({'error': error_msg}), 400
        
        # Sanitize input
        sanitized_question = validator.sanitize_question(q)
        
        try:
            res = agent.query(sanitized_question)
            return jsonify({
                'sql': res.sql, 
                'data': res.data,
                'question': sanitized_question
            }), 200
            
        except Exception as e:
            # Log the full error but return sanitized message
            logger.error(f"API query execution failed: {str(e)}")
            
            # Provide user-friendly error without sensitive details
            if "invalid" in str(e).lower() or "syntax" in str(e).lower():
                error_msg = "Invalid query. Please check your question and try again."
                status_code = 400
            elif "timeout" in str(e).lower():
                error_msg = "Query timed out. Please try a simpler question."
                status_code = 408
            elif "connection" in str(e).lower():
                error_msg = "Database connection issue. Please try again later."
                status_code = 503
            else:
                error_msg = "An error occurred while processing your query."
                status_code = 500
            
            return jsonify({'error': error_msg}), status_code

    @app.get("/health")
    def health() -> tuple[dict, int]:
        """Health check endpoint."""
        try:
            health_status = agent.health_check()
            status_code = 200 if health_status.get('overall_healthy', False) else 503
            
            # Remove sensitive information from health response
            public_health = {
                'status': 'healthy' if health_status.get('overall_healthy', False) else 'unhealthy',
                'timestamp': health_status.get('timestamp', time.time()),
                'components': {
                    'database': health_status.get('database', {}).get('healthy', False),
                    'cache': True,  # Simplified cache status
                    'services': True  # Simplified services status
                }
            }
            
            return jsonify(public_health), status_code
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({
                'status': 'unhealthy', 
                'timestamp': time.time(),
                'error': 'Health check failed'
            }), 503

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
