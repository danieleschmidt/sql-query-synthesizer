"""Secure Flask web app exposing :class:`QueryAgent` with comprehensive security features."""

from __future__ import annotations
import logging
import time

from flask import Flask, request, jsonify, render_template, Response, session
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError, TimeoutError as SQLTimeoutError
import openai

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
        except SQLTimeoutError as e:
            logger.error(f"Query timeout: {str(e)}")
            error_msg = "Query timed out. Please try a simpler question."
        except OperationalError as e:
            logger.error(f"Database connection error: {str(e)}")
            error_msg = "Database connection issue. Please try again later."
        except DatabaseError as e:
            logger.error(f"Database error: {str(e)}")
            error_msg = "Database error occurred. Please try again."
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {str(e)}")
            error_msg = "AI service authentication failed. Please check configuration."
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {str(e)}")
            error_msg = "AI service temporarily unavailable. Please try again later."
        except openai.APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {str(e)}")
            error_msg = "AI service timed out. Please try again."
        except (OperationalError, DatabaseError, SQLTimeoutError) as e:
            logger.error(f"Database error during query execution: {str(e)}")
            error_msg = "Database connection or query execution error. Please try again."
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {str(e)}")
            error_msg = "Rate limit exceeded. Please wait a moment and try again."
        except (openai.APIError, openai.APIConnectionError) as e:
            logger.error(f"OpenAI API error: {str(e)}")
            error_msg = "AI service temporarily unavailable. Please try again later."
        except ValueError as e:
            logger.warning(f"Invalid input during query processing: {str(e)}")
            error_msg = "Invalid input provided. Please check your query and try again."
        except (OSError, IOError) as e:
            logger.error(f"I/O error during query processing: {str(e)}")
            error_msg = "System I/O error. Please try again."
        
        # Return error response if any exception occurred
        if 'error_msg' in locals():
            return render_template(
                "index.html", 
                error=error_msg, 
                input_size=config.webapp_input_size,
                question=sanitized_question if 'sanitized_question' in locals() else ""
            )

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
            
        except SQLTimeoutError as e:
            logger.error(f"API query timeout: {str(e)}")
            return jsonify({'error': 'Query timed out. Please try a simpler question.'}), 408
        except OperationalError as e:
            logger.error(f"API database connection error: {str(e)}")
            return jsonify({'error': 'Database connection issue. Please try again later.'}), 503
        except DatabaseError as e:
            logger.error(f"API database error: {str(e)}")
            return jsonify({'error': 'Database error occurred. Please try again.'}), 500
        except openai.AuthenticationError as e:
            logger.error(f"API OpenAI authentication error: {str(e)}")
            return jsonify({'error': 'AI service authentication failed.'}), 503
        except openai.RateLimitError as e:
            logger.error(f"API OpenAI rate limit exceeded: {str(e)}")
            return jsonify({'error': 'AI service temporarily unavailable. Please try again later.'}), 429
        except openai.APITimeoutError as e:
            logger.error(f"API OpenAI timeout: {str(e)}")
            return jsonify({'error': 'AI service timed out. Please try again.'}), 408
        except (OperationalError, DatabaseError, SQLTimeoutError) as e:
            logger.error(f"Database error in API query: {str(e)}")
            return jsonify({'error': 'Database connection or query execution error'}), 500
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded in API: {str(e)}")
            return jsonify({'error': 'Rate limit exceeded. Please wait and try again'}), 429
        except (openai.APIError, openai.APIConnectionError) as e:
            logger.error(f"OpenAI API error in API: {str(e)}")
            return jsonify({'error': 'AI service temporarily unavailable'}), 503
        except ValueError as e:
            logger.warning(f"Invalid input in API query: {str(e)}")
            return jsonify({'error': 'Invalid input provided'}), 400
        except (OSError, IOError) as e:
            logger.error(f"I/O error in API query: {str(e)}")
            return jsonify({'error': 'System I/O error occurred'}), 500

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
            
        except DatabaseError as e:
            logger.error(f"Health check database error: {str(e)}")
            return jsonify({
                'status': 'unhealthy', 
                'timestamp': time.time(),
                'error': 'Database health check failed'
            }), 503
        except AttributeError as e:
            logger.error(f"Health check configuration error: {str(e)}")
            return jsonify({
                'status': 'unhealthy', 
                'timestamp': time.time(),
                'error': 'Service not properly configured'
            }), 503
        except (OperationalError, DatabaseError, SQLTimeoutError) as e:
            logger.error(f"Database error in health check: {str(e)}")
            return jsonify({
                'status': 'unhealthy',
                'error': 'Database connection error'
            }), 503
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection error in health check: {str(e)}")
            return jsonify({
                'status': 'unhealthy',
                'error': 'Service connection error'
            }), 503
        except (OSError, IOError) as e:
            logger.error(f"I/O error in health check: {str(e)}")
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
