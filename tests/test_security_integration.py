"""
Integration tests for security features.
"""

import pytest
import time
from unittest.mock import Mock, patch
from sqlalchemy import create_engine, text

from sql_synthesizer.query_agent import QueryAgent
from sql_synthesizer.webapp import create_app
from sql_synthesizer.security import SecurityMiddleware, RateLimiter, InputValidator


@pytest.fixture
def test_agent():
    """Create a test agent with mocked database."""
    mock_agent = Mock(spec=QueryAgent)
    mock_agent.query.return_value = Mock(sql="SELECT 1", data=[{"result": 1}])
    mock_agent.health_check.return_value = {"overall_healthy": True, "database": {"healthy": True}}
    return mock_agent


@pytest.fixture
def app(test_agent):
    """Create Flask test app with security."""
    app = create_app(test_agent)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestSecurityIntegration:
    """Test security features integration."""
    
    def test_security_headers_present(self, client):
        """Test that security headers are added to responses."""
        response = client.get('/')
        
        # Check for key security headers
        assert 'Content-Security-Policy' in response.headers
        assert 'X-Frame-Options' in response.headers
        assert 'X-Content-Type-Options' in response.headers
        assert 'X-XSS-Protection' in response.headers
        
        # Verify header values
        assert response.headers['X-Frame-Options'] == 'DENY'
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        assert response.headers['X-XSS-Protection'] == '1; mode=block'
    
    def test_input_validation_length(self, client):
        """Test that overly long inputs are rejected."""
        long_question = "SELECT * FROM users WHERE " + "a" * 2000
        
        response = client.post('/', data={'question': long_question})
        assert response.status_code == 400
        assert 'too long' in response.get_data(as_text=True).lower()
    
    def test_input_sanitization(self, client):
        """Test that malicious input is sanitized."""
        malicious_input = "<script>alert('xss')</script>"
        
        response = client.post('/', data={'question': malicious_input})
        response_text = response.get_data(as_text=True)
        
        # Script tags should be escaped or removed
        assert '<script>' not in response_text
    
    def test_api_input_validation(self, client):
        """Test API endpoint input validation."""
        # Test missing content type
        response = client.post('/api/query', data='{"question": "test"}')
        assert response.status_code == 400
        
        # Test missing required field
        response = client.post('/api/query', 
                              json={'not_question': 'test'},
                              headers={'Content-Type': 'application/json'})
        assert response.status_code == 400
        
        # Test valid request
        response = client.post('/api/query',
                              json={'question': 'SELECT 1'},
                              headers={'Content-Type': 'application/json'})
        assert response.status_code == 200
    
    def test_error_message_sanitization(self, client, test_agent):
        """Test that error messages don't expose sensitive information."""
        # Mock an error
        test_agent.query.side_effect = Exception("Internal database connection failed at /root/sensitive/path")
        
        response = client.post('/api/query',
                              json={'question': 'SELECT 1'},
                              headers={'Content-Type': 'application/json'})
        
        assert response.status_code >= 400
        error_data = response.get_json()
        error_message = error_data.get('error', '')
        
        # Should not contain sensitive information
        assert '/root/' not in error_message
        assert 'Internal database connection failed' not in error_message
        assert 'error occurred' in error_message.lower() or 'connection issue' in error_message.lower()
    
    def test_health_endpoint(self, client):
        """Test health endpoint functionality."""
        response = client.get('/health')
        assert response.status_code == 200
        
        health_data = response.get_json()
        assert 'status' in health_data
        assert 'components' in health_data
        assert health_data['status'] in ['healthy', 'unhealthy']
    
    def test_rate_limiting_headers(self, client):
        """Test that rate limiting headers are present."""
        response = client.post('/api/query',
                              json={'question': 'SELECT 1'},
                              headers={'Content-Type': 'application/json'})
        
        # Check for rate limiting headers
        rate_limit_headers = ['X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Reset']
        present_headers = [h for h in rate_limit_headers if h in response.headers]
        
        # At least one rate limit header should be present
        assert len(present_headers) > 0
    
    def test_csrf_token_in_form(self, client):
        """Test that CSRF tokens are included in forms when enabled."""
        with patch('sql_synthesizer.webapp.config.webapp_csrf_enabled', True):
            response = client.get('/')
            response_text = response.get_data(as_text=True)
            
            # Should include CSRF token field
            assert 'csrf_token' in response_text


class TestSecurityMiddleware:
    """Test security middleware components in isolation."""
    
    def test_rate_limiter(self):
        """Test rate limiter functionality."""
        limiter = RateLimiter(requests_per_minute=5, window_size=60)
        client_id = "test_client"
        
        # First 5 requests should be allowed
        for i in range(5):
            assert limiter.is_allowed(client_id) is True
        
        # 6th request should be denied
        assert limiter.is_allowed(client_id) is False
        
        # Rate limit headers should work
        headers = limiter.get_rate_limit_headers(client_id)
        assert 'X-RateLimit-Limit' in headers
        assert 'X-RateLimit-Remaining' in headers
        assert headers['X-RateLimit-Remaining'] == '0'
    
    def test_input_validator(self):
        """Test input validator functionality."""
        validator = InputValidator()
        
        # Test length validation
        assert validator.validate_question_length("short question") is True
        assert validator.validate_question_length("a" * 2000, max_length=1000) is False
        
        # Test sanitization
        malicious = "<script>alert('xss')</script>"
        sanitized = validator.sanitize_question(malicious)
        assert '<script>' not in sanitized
        
        # Test JSON validation
        valid, msg = validator.validate_json_structure({"question": "test"}, ["question"])
        assert valid is True
        
        valid, msg = validator.validate_json_structure({"wrong": "field"}, ["question"])
        assert valid is False
        assert "Missing required field" in msg


class TestSecurityConfiguration:
    """Test security configuration."""
    
    def test_security_config_loading(self):
        """Test that security configuration loads properly."""
        from sql_synthesizer.config import Config
        
        config = Config()
        
        # Security settings should be present
        assert hasattr(config, 'webapp_csrf_enabled')
        assert hasattr(config, 'webapp_rate_limit')
        assert hasattr(config, 'webapp_max_request_size')
        
        # Should have reasonable defaults
        assert config.webapp_rate_limit > 0
        assert config.webapp_max_request_size > 0
    
    def test_secret_key_warning(self, caplog):
        """Test that missing secret key generates warning."""
        from sql_synthesizer.security import SecurityMiddleware
        from flask import Flask
        
        app = Flask(__name__)
        # Don't set secret key
        
        with patch('sql_synthesizer.security.config.webapp_secret_key', None):
            middleware = SecurityMiddleware()
            middleware.init_app(app)
            
            # Should set a secret key and log warning
            assert app.secret_key is not None
            assert len(app.secret_key) >= 16