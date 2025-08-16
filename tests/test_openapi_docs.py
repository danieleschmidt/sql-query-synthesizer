"""Tests for OpenAPI documentation functionality."""

import json
from unittest.mock import patch

import pytest


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation endpoints and schema."""

    def test_openapi_endpoint_exists(self):
        """Test that /openapi.json endpoint exists."""
        from sql_synthesizer.query_agent import QueryAgent
        from sql_synthesizer.webapp import create_app

        # Mock QueryAgent to avoid database dependencies
        with patch("sql_synthesizer.query_agent.QueryAgent") as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.health_check.return_value = {"overall_healthy": True}

            app = create_app(mock_agent)

            with app.test_client() as client:
                response = client.get("/openapi.json")
                assert response.status_code == 200
                assert response.content_type == "application/json"

    def test_openapi_schema_structure(self):
        """Test that OpenAPI schema has required structure."""
        from sql_synthesizer.query_agent import QueryAgent
        from sql_synthesizer.webapp import create_app

        with patch("sql_synthesizer.query_agent.QueryAgent") as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.health_check.return_value = {"overall_healthy": True}

            app = create_app(mock_agent)

            with app.test_client() as client:
                response = client.get("/openapi.json")
                assert response.status_code == 200

                schema = json.loads(response.data)

                # Check required OpenAPI 3.0 fields
                assert "openapi" in schema
                assert schema["openapi"].startswith("3.0")
                assert "info" in schema
                assert "paths" in schema

                # Check info section
                info = schema["info"]
                assert "title" in info
                assert "version" in info
                assert "description" in info

    def test_openapi_includes_api_endpoints(self):
        """Test that OpenAPI schema includes all API endpoints."""
        from sql_synthesizer.query_agent import QueryAgent
        from sql_synthesizer.webapp import create_app

        with patch("sql_synthesizer.query_agent.QueryAgent") as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.health_check.return_value = {"overall_healthy": True}

            app = create_app(mock_agent)

            with app.test_client() as client:
                response = client.get("/openapi.json")
                schema = json.loads(response.data)

                paths = schema["paths"]

                # Check that main API endpoints are documented
                assert "/api/query" in paths
                assert "/health" in paths
                assert "/metrics" in paths

                # Check POST /api/query endpoint details
                query_endpoint = paths["/api/query"]
                assert "post" in query_endpoint

                post_spec = query_endpoint["post"]
                assert "summary" in post_spec
                assert "requestBody" in post_spec
                assert "responses" in post_spec

                # Check responses
                responses = post_spec["responses"]
                assert "200" in responses
                assert "400" in responses
                assert "500" in responses

    def test_swagger_ui_endpoint_exists(self):
        """Test that Swagger UI endpoint exists."""
        from sql_synthesizer.query_agent import QueryAgent
        from sql_synthesizer.webapp import create_app

        with patch("sql_synthesizer.query_agent.QueryAgent") as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.health_check.return_value = {"overall_healthy": True}

            app = create_app(mock_agent)

            with app.test_client() as client:
                response = client.get("/docs")
                assert response.status_code == 200
                assert b"swagger-ui" in response.data.lower()

    def test_openapi_schema_validation(self):
        """Test that the OpenAPI schema is valid."""
        from sql_synthesizer.query_agent import QueryAgent
        from sql_synthesizer.webapp import create_app

        with patch("sql_synthesizer.query_agent.QueryAgent") as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.health_check.return_value = {"overall_healthy": True}

            app = create_app(mock_agent)

            with app.test_client() as client:
                response = client.get("/openapi.json")
                schema = json.loads(response.data)

                # Basic schema validation
                required_fields = ["openapi", "info", "paths"]
                for field in required_fields:
                    assert field in schema, f"Missing required field: {field}"

                # Check that paths are properly structured
                for path, methods in schema["paths"].items():
                    assert isinstance(path, str) and path.startswith("/")
                    assert isinstance(methods, dict)

                    for method, spec in methods.items():
                        assert method.lower() in [
                            "get",
                            "post",
                            "put",
                            "delete",
                            "patch",
                        ]
                        assert "responses" in spec
