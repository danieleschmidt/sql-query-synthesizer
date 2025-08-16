"""Test for enhanced health check endpoint - OpenAI API availability check."""

from unittest.mock import Mock

import pytest

from sql_synthesizer.query_agent import QueryAgent
from sql_synthesizer.webapp import create_app


class TestEnhancedHealthCheck:
    """Test enhanced health check functionality."""

    def test_health_check_includes_openai_api_status(self):
        """Test that health check includes OpenAI API availability status."""
        # Arrange
        mock_agent = Mock(spec=QueryAgent)
        mock_agent.health_check.return_value = {
            "database": {"healthy": True, "error": None, "timestamp": 1234567890.0},
            "caches": {
                "schema_cache": {"size": 0, "ttl": 3600, "healthy": True},
                "query_cache": {"size": 0, "ttl": 3600, "healthy": True}
            },
            "services": {
                "validator": {"healthy": True},
                "generator": {"healthy": True, "llm_provider_available": True},
                "openai_api": {"healthy": True, "available": True, "response_time_ms": 150}
            },
            "overall_healthy": True,
            "timestamp": 1234567890.0
        }

        app = create_app(mock_agent)

        # Act
        with app.test_client() as client:
            response = client.get('/health')

        # Assert
        assert response.status_code == 200
        data = response.get_json()

        # Verify structure matches acceptance criteria
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'components' in data

        # Verify all required components are present
        components = data['components']
        assert 'database' in components
        assert 'cache' in components
        assert 'openai_api' in components

    def test_health_check_openai_api_unavailable(self):
        """Test health check when OpenAI API is unavailable."""
        # Arrange
        mock_agent = Mock(spec=QueryAgent)
        mock_agent.health_check.return_value = {
            "database": {"healthy": True, "error": None, "timestamp": 1234567890.0},
            "caches": {
                "schema_cache": {"size": 0, "ttl": 3600, "healthy": True},
                "query_cache": {"size": 0, "ttl": 3600, "healthy": True}
            },
            "services": {
                "validator": {"healthy": True},
                "generator": {"healthy": True, "llm_provider_available": False},
                "openai_api": {"healthy": False, "available": False, "error": "API key not configured"}
            },
            "overall_healthy": False,  # Should be false when OpenAI is down
            "timestamp": 1234567890.0
        }

        app = create_app(mock_agent)

        # Act
        with app.test_client() as client:
            response = client.get('/health')

        # Assert
        assert response.status_code == 503  # Service unavailable
        data = response.get_json()
        assert data['status'] == 'unhealthy'
        assert data['components']['openai_api'] is False

    def test_health_check_comprehensive_cache_status(self):
        """Test that cache status includes detailed backend information."""
        # Arrange
        mock_agent = Mock(spec=QueryAgent)
        mock_agent.health_check.return_value = {
            "database": {"healthy": True, "error": None, "timestamp": 1234567890.0},
            "caches": {
                "schema_cache": {"size": 5, "ttl": 3600, "healthy": True, "backend": "memory"},
                "query_cache": {"size": 10, "ttl": 3600, "healthy": True, "backend": "redis"}
            },
            "services": {
                "validator": {"healthy": True},
                "generator": {"healthy": True, "llm_provider_available": True},
                "openai_api": {"healthy": True, "available": True, "response_time_ms": 150}
            },
            "overall_healthy": True,
            "timestamp": 1234567890.0
        }

        app = create_app(mock_agent)

        # Act
        with app.test_client() as client:
            response = client.get('/health')

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert data['components']['cache'] is True  # Should aggregate cache health


if __name__ == "__main__":
    pytest.main([__file__])
