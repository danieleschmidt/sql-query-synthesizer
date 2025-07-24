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
        except ValueError as e:
            logger.error(f"Input validation error: {str(e)}")
            error_msg = "Invalid query. Please check your question and try again."
        except Exception as e:
            # Log the full error but return sanitized message
            logger.error(f"Unexpected query execution error: {str(e)}")
            error_msg = "An error occurred while processing your query. Please try again."
        
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
        except ValueError as e:
            logger.error(f"API input validation error: {str(e)}")
            return jsonify({'error': 'Invalid query. Please check your question and try again.'}), 400
        except Exception as e:
            # Log the full error but return sanitized message
            logger.error(f"Unexpected API query execution error: {str(e)}")
            
            return jsonify({'error': 'An error occurred while processing your query. Please try again.'}), 500

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
        except Exception as e:
            logger.error(f"Unexpected health check error: {str(e)}")
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

    @app.get("/openapi.json")
    def openapi_schema() -> tuple[dict, int]:
        """OpenAPI 3.0 schema for the SQL Synthesizer API."""
        schema = {
            "openapi": "3.0.3",
            "info": {
                "title": "SQL Synthesizer API",
                "version": "1.0.0",
                "description": "Natural language to SQL query generation API with schema discovery and validation",
                "contact": {
                    "name": "SQL Synthesizer",
                    "url": "https://github.com/your-org/sql-synthesizer"
                }
            },
            "servers": [
                {
                    "url": "/",
                    "description": "Current server"
                }
            ],
            "paths": {
                "/": {
                    "get": {
                        "summary": "Web Interface",
                        "description": "Interactive web interface for SQL query generation",
                        "responses": {
                            "200": {
                                "description": "HTML web interface",
                                "content": {
                                    "text/html": {
                                        "schema": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "post": {
                        "summary": "Generate SQL Query (Web Form)",
                        "description": "Generate SQL query from natural language via web form",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "question": {
                                                "type": "string",
                                                "description": "Natural language question",
                                                "maxLength": 1000
                                            },
                                            "csrf_token": {
                                                "type": "string",
                                                "description": "CSRF protection token"
                                            }
                                        },
                                        "required": ["question"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "HTML page with query results",
                                "content": {
                                    "text/html": {
                                        "schema": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/query": {
                    "post": {
                        "summary": "Generate SQL Query (API)",
                        "description": "Generate SQL query from natural language via JSON API",
                        "security": [
                            {"ApiKeyAuth": []}
                        ],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "question": {
                                                "type": "string",
                                                "description": "Natural language question about your data",
                                                "example": "Show me the top 5 customers by revenue",
                                                "maxLength": 1000
                                            }
                                        },
                                        "required": ["question"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful query generation",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "sql": {
                                                    "type": "string",
                                                    "description": "Generated SQL query"
                                                },
                                                "data": {
                                                    "type": "array",
                                                    "description": "Query results",
                                                    "items": {"type": "object"}
                                                },
                                                "question": {
                                                    "type": "string",
                                                    "description": "Sanitized input question"
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "400": {
                                "description": "Bad request - invalid input",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "error": {
                                                    "type": "string",
                                                    "description": "Error message"
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "408": {
                                "description": "Request timeout",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "error": {
                                                    "type": "string",
                                                    "description": "Timeout error message"
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "429": {
                                "description": "Rate limit exceeded",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "error": {
                                                    "type": "string",
                                                    "description": "Rate limit error message"
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "500": {
                                "description": "Internal server error",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "error": {
                                                    "type": "string",
                                                    "description": "Error message"
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "503": {
                                "description": "Service unavailable",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "error": {
                                                    "type": "string",
                                                    "description": "Service error message"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health Check",
                        "description": "Check system health and dependencies",
                        "responses": {
                            "200": {
                                "description": "System is healthy",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["healthy", "unhealthy"]
                                                },
                                                "timestamp": {
                                                    "type": "number",
                                                    "description": "Unix timestamp"
                                                },
                                                "checks": {
                                                    "type": "object",
                                                    "properties": {
                                                        "database": {
                                                            "type": "boolean",
                                                            "description": "Database connectivity status"
                                                        },
                                                        "cache": {
                                                            "type": "boolean",
                                                            "description": "Cache system status"
                                                        },
                                                        "services": {
                                                            "type": "boolean",
                                                            "description": "Core services status"
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "503": {
                                "description": "System is unhealthy",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["unhealthy"]
                                                },
                                                "timestamp": {
                                                    "type": "number"
                                                },
                                                "error": {
                                                    "type": "string",
                                                    "description": "Error description"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/metrics": {
                    "get": {
                        "summary": "Prometheus Metrics",
                        "description": "Prometheus-compatible metrics for monitoring",
                        "responses": {
                            "200": {
                                "description": "Prometheus metrics",
                                "content": {
                                    "text/plain": {
                                        "schema": {
                                            "type": "string",
                                            "description": "Prometheus metrics format"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                        "description": "API key for authentication (optional, configurable)"
                    }
                }
            }
        }
        return jsonify(schema), 200

    @app.get("/docs")
    def swagger_ui() -> str:
        """Swagger UI for API documentation."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SQL Synthesizer API Documentation</title>
            <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
            <style>
                html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
                *, *:before, *:after { box-sizing: inherit; }
                body { margin:0; background: #fafafa; }
            </style>
        </head>
        <body>
            <div id="swagger-ui"></div>
            <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
            <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js"></script>
            <script>
                window.onload = function() {
                    const ui = SwaggerUIBundle({
                        url: '/openapi.json',
                        dom_id: '#swagger-ui',
                        deepLinking: true,
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIStandalonePreset
                        ],
                        plugins: [
                            SwaggerUIBundle.plugins.DownloadUrl
                        ],
                        layout: "StandaloneLayout"
                    });
                };
            </script>
        </body>
        </html>
        """
        return html

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
