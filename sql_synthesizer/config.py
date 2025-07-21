"""
Centralized configuration management for SQL Query Synthesizer.

This module provides a Configuration class that consolidates all configurable
values with environment variable overrides and validation.
"""

import os
from typing import Dict, Any, Tuple


class Config:
    """
    Centralized configuration class with environment variable support.
    
    This class consolidates all hardcoded values from the application into
    a single, configurable location with environment variable overrides.
    
    Environment variables use the QUERY_AGENT_ prefix for consistency.
    """
    
    _instance = None
    
    def __new__(cls, force_reload=False):
        """Implement singleton pattern for configuration."""
        if cls._instance is None or force_reload:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, force_reload=False):
        """Initialize configuration with defaults and environment overrides."""
        if self._initialized and not force_reload:
            return
        
        # Web Application Settings
        self.webapp_port = self._get_int_env("QUERY_AGENT_WEBAPP_PORT", 5000, min_value=1)
        self.webapp_input_size = self._get_int_env("QUERY_AGENT_WEBAPP_INPUT_SIZE", 60, min_value=1)
        
        # Query Processing Limits
        self.max_question_length = self._get_int_env("QUERY_AGENT_MAX_QUESTION_LENGTH", 1000, min_value=1)
        self.default_max_rows = self._get_int_env("QUERY_AGENT_DEFAULT_MAX_ROWS", 5, min_value=1)
        
        # Cache Configuration
        self.cache_cleanup_interval = self._get_int_env("QUERY_AGENT_CACHE_CLEANUP_INTERVAL", 300, min_value=1)
        
        # Timeout Settings (in seconds)
        self.openai_timeout = self._get_int_env("QUERY_AGENT_OPENAI_TIMEOUT", 30, min_value=1)
        self.database_timeout = self._get_int_env("QUERY_AGENT_DATABASE_TIMEOUT", 30, min_value=1)
        
        # Database Connection Pool Settings
        self.db_pool_size = self._get_int_env("QUERY_AGENT_DB_POOL_SIZE", 10, min_value=1)
        self.db_max_overflow = self._get_int_env("QUERY_AGENT_DB_MAX_OVERFLOW", 20, min_value=0)
        self.db_pool_recycle = self._get_int_env("QUERY_AGENT_DB_POOL_RECYCLE", 3600, min_value=1)
        self.db_pool_pre_ping = self._get_bool_env("QUERY_AGENT_DB_POOL_PRE_PING", True)
        self.db_connect_retries = self._get_int_env("QUERY_AGENT_DB_CONNECT_RETRIES", 3, min_value=0)
        self.db_retry_delay = self._get_float_env("QUERY_AGENT_DB_RETRY_DELAY", 1.0, min_value=0.1)
        
        # Circuit Breaker Settings for LLM Provider Resilience
        self.circuit_breaker_failure_threshold = self._get_int_env("QUERY_AGENT_CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5, min_value=1)
        self.circuit_breaker_recovery_timeout = self._get_float_env("QUERY_AGENT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", 60.0, min_value=1.0)
        
        # Enhanced SQL Injection Prevention
        self.use_enhanced_sql_validation = self._get_bool_env("QUERY_AGENT_USE_ENHANCED_SQL_VALIDATION", True)
        
        # Pagination Configuration
        self.default_page_size = self._get_int_env("QUERY_AGENT_DEFAULT_PAGE_SIZE", 10, min_value=1)
        self.max_page_size = self._get_int_env("QUERY_AGENT_MAX_PAGE_SIZE", 1000, min_value=1)
        
        # Security Settings
        self.webapp_secret_key = os.environ.get("QUERY_AGENT_SECRET_KEY", None)
        self.webapp_csrf_enabled = self._get_bool_env("QUERY_AGENT_CSRF_ENABLED", True)
        self.webapp_rate_limit = self._get_int_env("QUERY_AGENT_RATE_LIMIT_PER_MINUTE", 60, min_value=1)
        self.webapp_enable_hsts = self._get_bool_env("QUERY_AGENT_ENABLE_HSTS", False)
        self.webapp_api_key_required = self._get_bool_env("QUERY_AGENT_API_KEY_REQUIRED", False)
        self.webapp_api_key = os.environ.get("QUERY_AGENT_API_KEY", None)
        self.webapp_max_request_size = self._get_int_env("QUERY_AGENT_MAX_REQUEST_SIZE_MB", 1, min_value=1)
        
        # Prometheus Metrics Histogram Buckets
        self.openai_request_buckets = self._get_bucket_env(
            "QUERY_AGENT_OPENAI_REQUEST_BUCKETS",
            (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
        )
        
        self.database_query_buckets = self._get_bucket_env(
            "QUERY_AGENT_DATABASE_QUERY_BUCKETS",
            (0.1, 0.5, 1, 2, 5, 10, 30)
        )
        
        self.cache_operation_buckets = self._get_bucket_env(
            "QUERY_AGENT_CACHE_OPERATION_BUCKETS",
            (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1)
        )
        
        self._initialized = True
    
    def _get_int_env(self, env_var: str, default: int, min_value: int = None) -> int:
        """Get integer value from environment with validation."""
        value_str = os.environ.get(env_var)
        if value_str is None:
            return default
        
        try:
            value = int(value_str)
        except ValueError:
            raise ValueError(f"Invalid value for {env_var.lower().replace('query_agent_', '')}: '{value_str}' is not a valid integer")
        
        if min_value is not None and value < min_value:
            config_name = env_var.lower().replace('query_agent_', '')
            raise ValueError(f"{config_name} must be positive (got {value})")
        
        return value
    
    def _get_bool_env(self, env_var: str, default: bool) -> bool:
        """Get boolean value from environment with validation."""
        value_str = os.environ.get(env_var)
        if value_str is None:
            return default
        
        value_lower = value_str.lower()
        if value_lower in ('true', '1', 'yes', 'on'):
            return True
        elif value_lower in ('false', '0', 'no', 'off'):
            return False
        else:
            config_name = env_var.lower().replace('query_agent_', '')
            raise ValueError(f"Invalid boolean value for {config_name}: '{value_str}' (use true/false)")
    
    def _get_float_env(self, env_var: str, default: float, min_value: float = None) -> float:
        """Get float value from environment with validation."""
        value_str = os.environ.get(env_var)
        if value_str is None:
            return default
        
        try:
            value = float(value_str)
        except ValueError:
            config_name = env_var.lower().replace('query_agent_', '')
            raise ValueError(f"Invalid value for {config_name}: '{value_str}' is not a valid number")
        
        if min_value is not None and value < min_value:
            config_name = env_var.lower().replace('query_agent_', '')
            raise ValueError(f"{config_name} must be >= {min_value} (got {value})")
        
        return value
    
    def _get_bucket_env(self, env_var: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
        """Get histogram bucket values from environment."""
        value_str = os.environ.get(env_var)
        if value_str is None:
            return default
        
        try:
            buckets = tuple(float(x.strip()) for x in value_str.split(','))
            # Validate that buckets are in ascending order
            if buckets != tuple(sorted(buckets)):
                raise ValueError("Bucket values must be in ascending order")
            return buckets
        except ValueError as e:
            raise ValueError(f"Invalid bucket values for {env_var}: {e}")
    
    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary for serialization or logging."""
        return {
            "webapp_port": self.webapp_port,
            "webapp_input_size": self.webapp_input_size,
            "max_question_length": self.max_question_length,
            "default_max_rows": self.default_max_rows,
            "cache_cleanup_interval": self.cache_cleanup_interval,
            "openai_timeout": self.openai_timeout,
            "database_timeout": self.database_timeout,
            "db_pool_size": self.db_pool_size,
            "db_max_overflow": self.db_max_overflow,
            "db_pool_recycle": self.db_pool_recycle,
            "db_pool_pre_ping": self.db_pool_pre_ping,
            "db_connect_retries": self.db_connect_retries,
            "db_retry_delay": self.db_retry_delay,
            "webapp_csrf_enabled": self.webapp_csrf_enabled,
            "webapp_rate_limit": self.webapp_rate_limit,
            "webapp_enable_hsts": self.webapp_enable_hsts,
            "webapp_api_key_required": self.webapp_api_key_required,
            "webapp_max_request_size": self.webapp_max_request_size,
            "openai_request_buckets": self.openai_request_buckets,
            "database_query_buckets": self.database_query_buckets,
            "cache_operation_buckets": self.cache_operation_buckets,
        }
    
    def __repr__(self) -> str:
        """Return string representation of configuration."""
        return f"Config({self.as_dict()})"


# Global configuration instance
config = Config()