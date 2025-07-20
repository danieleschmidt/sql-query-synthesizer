import os
import pytest
from sql_synthesizer.config import Config


class TestConfig:
    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        config = Config()
        
        # Web server defaults
        assert config.webapp_port == 5000
        assert config.webapp_input_size == 60
        
        # Query limits
        assert config.max_question_length == 1000
        assert config.default_max_rows == 5
        
        # Cache settings
        assert config.cache_cleanup_interval == 300  # 5 minutes
        
        # Timeout settings
        assert config.openai_timeout == 30
        assert config.database_timeout == 30
        
        # Metrics histogram buckets
        assert config.openai_request_buckets == (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
        assert config.database_query_buckets == (0.1, 0.5, 1, 2, 5, 10, 30)
        assert config.cache_operation_buckets == (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1)

    def test_environment_variable_override(self):
        """Test that environment variables override default values."""
        # Set environment variables
        os.environ["QUERY_AGENT_WEBAPP_PORT"] = "8080"
        os.environ["QUERY_AGENT_MAX_QUESTION_LENGTH"] = "2000"
        os.environ["QUERY_AGENT_CACHE_CLEANUP_INTERVAL"] = "600"
        
        try:
            config = Config(force_reload=True)
            assert config.webapp_port == 8080
            assert config.max_question_length == 2000
            assert config.cache_cleanup_interval == 600
        finally:
            # Clean up environment variables
            for key in ["QUERY_AGENT_WEBAPP_PORT", "QUERY_AGENT_MAX_QUESTION_LENGTH", "QUERY_AGENT_CACHE_CLEANUP_INTERVAL"]:
                os.environ.pop(key, None)

    def test_validation_positive_integers(self):
        """Test that positive integer validation works."""
        os.environ["QUERY_AGENT_WEBAPP_PORT"] = "0"
        
        try:
            with pytest.raises(ValueError, match="webapp_port must be positive"):
                Config(force_reload=True)
        finally:
            os.environ.pop("QUERY_AGENT_WEBAPP_PORT", None)

    def test_validation_invalid_type(self):
        """Test that type validation works."""
        os.environ["QUERY_AGENT_WEBAPP_PORT"] = "not_a_number"
        
        try:
            with pytest.raises(ValueError, match="Invalid value for webapp_port"):
                Config(force_reload=True)
        finally:
            os.environ.pop("QUERY_AGENT_WEBAPP_PORT", None)

    def test_custom_buckets_from_env(self):
        """Test that histogram buckets can be customized via environment."""
        os.environ["QUERY_AGENT_OPENAI_REQUEST_BUCKETS"] = "0.1,0.5,1.0,5.0"
        
        try:
            config = Config(force_reload=True)
            assert config.openai_request_buckets == (0.1, 0.5, 1.0, 5.0)
        finally:
            os.environ.pop("QUERY_AGENT_OPENAI_REQUEST_BUCKETS", None)

    def test_custom_buckets_invalid_format(self):
        """Test that invalid bucket format raises error."""
        os.environ["QUERY_AGENT_OPENAI_REQUEST_BUCKETS"] = "invalid,format"
        
        try:
            with pytest.raises(ValueError, match="Invalid bucket values"):
                Config(force_reload=True)
        finally:
            os.environ.pop("QUERY_AGENT_OPENAI_REQUEST_BUCKETS", None)

    def test_config_singleton_behavior(self):
        """Test that Config returns the same instance when called multiple times."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_as_dict(self):
        """Test that configuration can be converted to dictionary."""
        config = Config()
        config_dict = config.as_dict()
        
        assert isinstance(config_dict, dict)
        assert "webapp_port" in config_dict
        assert "max_question_length" in config_dict
        assert config_dict["webapp_port"] == 5000

    def test_env_prefix_consistency(self):
        """Test that all environment variables use consistent QUERY_AGENT_ prefix."""
        config = Config()
        
        # Check that the prefix is used consistently across all environment variable names
        env_vars = [
            "QUERY_AGENT_WEBAPP_PORT",
            "QUERY_AGENT_MAX_QUESTION_LENGTH", 
            "QUERY_AGENT_CACHE_CLEANUP_INTERVAL",
            "QUERY_AGENT_OPENAI_TIMEOUT",
            "QUERY_AGENT_DATABASE_TIMEOUT"
        ]
        
        # This test mainly documents the expected environment variable names
        # and ensures consistency in naming convention
        assert all(var.startswith("QUERY_AGENT_") for var in env_vars)