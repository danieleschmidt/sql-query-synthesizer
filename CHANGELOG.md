# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2025-07-21
- **LLM Provider Resilience**: Circuit breaker pattern implementation for improved reliability
  - Added circuit breaker pattern to prevent cascading failures when OpenAI API is unavailable
  - Implemented automatic state management: CLOSED → OPEN → HALF_OPEN → CLOSED transitions
  - Added configurable failure threshold (default: 5) and recovery timeout (default: 60s)
  - Fast-fail behavior prevents repeated API calls during outages, enabling graceful degradation
  - Automatic recovery testing with single request validation in HALF_OPEN state
  - Integrated with centralized configuration system via QUERY_AGENT_CIRCUIT_BREAKER_* variables
  - Enhanced OpenAI adapter with circuit breaker status monitoring and metrics
  - Added comprehensive logging for circuit state changes and failure tracking
  - Created 15+ tests covering all circuit breaker states, transitions, and error scenarios
  - Enhanced documentation with new configuration options and feature descriptions
  - Maintained 100% backward compatibility with existing QueryAgent and OpenAI adapter APIs

## [0.7.0] - 2025-07-20
- **Enterprise Web Application Security**: Comprehensive security hardening for production deployments
  - Added SecurityMiddleware with CSRF protection, rate limiting, and security headers
  - Implemented comprehensive input validation and sanitization to prevent XSS and injection attacks
  - Added configurable rate limiting with sliding window algorithm (60 req/min default)
  - Enhanced error handling with sanitized error messages preventing information leakage
  - Added optional API key authentication for programmatic access
  - Implemented security headers (CSP, XSS protection, frame options, HSTS support)
  - Added request size limits and JSON validation for API endpoints
  - Created health check endpoint (/health) with sanitized status information
  - Added 7 new security configuration options with environment variable support
  - Enhanced logging for security events and rate limiting violations
  - Added 20+ comprehensive security tests covering all attack vectors
  - Maintained 100% backward compatibility while significantly improving security posture

## [0.6.0] - 2025-07-20
- **Enterprise Database Connection Management**: Comprehensive connection pooling and reliability improvements
  - Added DatabaseConnectionManager with configurable connection pooling (pool_size, max_overflow, recycle)
  - Implemented automatic connection retry logic with exponential backoff and configurable retry attempts
  - Added connection health checks with pre_ping validation and automatic connection recovery
  - Enhanced error handling with specific DatabaseConnectionError types and graceful degradation
  - Added comprehensive connection statistics and monitoring (total_connections, pool status, failures)
  - Integrated health monitoring with QueryAgent.health_check() for operational visibility
  - Added 15+ new configuration options for fine-tuning database connection behavior
  - Enhanced logging with connection lifecycle events and structured error reporting
  - Maintained 100% backward compatibility - existing QueryAgent API unchanged
  - Added 25+ comprehensive tests covering connection pooling, error scenarios, and configuration validation

## [0.5.0] - 2025-07-20
- **Service Layer Architecture Implementation**: Major refactoring with clean architecture principles
  - Split QueryAgent (489 lines) into focused, single-responsibility services for improved maintainability
  - Created QueryValidatorService for input sanitization and SQL injection prevention with comprehensive patterns
  - Created SQLGeneratorService for SQL generation with OpenAI fallback and error handling
  - Created QueryService for query orchestration, caching, execution, and metrics integration
  - Extracted QueryResult and common types to eliminate circular imports and improve modularity
  - Added 33 comprehensive service-specific tests with full mock coverage and edge case testing
  - Maintained 100% backward compatibility - all existing 45 tests pass without modification
  - Enhanced error handling with database-level exception catching and user-friendly error conversion
  - Improved separation of concerns: validation, generation, orchestration, and execution are now isolated
  - Reduced tight coupling and enhanced testability with dependency injection patterns (WSJF Score: 7/10)

## [0.4.0] - 2025-07-20
- **Centralized Configuration Management**: Implemented comprehensive configuration system with environment variable overrides
  - Created centralized Config class consolidating all hardcoded values into configurable settings
  - Added support for QUERY_AGENT_ prefixed environment variables with validation
  - Extracted hardcoded values from webapp (port, input size), query_agent (timeouts, limits), and metrics (histogram buckets)
  - Added comprehensive test suite for configuration management with 9 test cases
  - Updated documentation with detailed configuration table and examples
  - Improved deployment flexibility and reduced maintenance burden (WSJF Score: 8/10)
- **Enhanced Web Templates with Security**: Extracted HTML templates from Python code with security improvements
  - Moved embedded HTML to separate Jinja2 template files for better maintainability
  - Added Content Security Policy headers to prevent XSS attacks
  - Implemented proper template escaping for all user input
  - Enhanced UX with responsive design, examples, and modern styling
  - Added graceful error handling with styled error messages
  - Created comprehensive test suite covering security and UX functionality (WSJF Score: 7/10)

## [0.3.0] - 2025-07-19
- **Comprehensive Metrics and Monitoring**: Enhanced observability with detailed performance and error tracking
  - Added error rate metrics for input validation, SQL validation, and query execution failures
  - Implemented detailed database connection and query performance metrics with appropriate histogram buckets
  - Added OpenAI API request tracking with duration and status metrics
  - Enhanced metrics with granular error categorization (SQL injection attempts, invalid queries, etc.)
  - Added comprehensive test suite for all new metrics functionality (13 new tests)
  - Improved production monitoring capabilities with actionable error insights
- **Cache Performance Optimization**: Enhanced TTLCache with comprehensive metrics and memory management
  - Added cache hit/miss metrics tracking with Prometheus integration
  - Implemented LRU (Least Recently Used) eviction strategy for memory efficiency
  - Added automatic expired entry cleanup with configurable scheduling
  - Enhanced thread safety with RLock for concurrent access
  - Added comprehensive cache statistics and monitoring endpoints
  - Integrated cache metrics with QueryAgent for production monitoring
- Enhanced security with comprehensive input validation and SQL injection prevention
- Added structured logging with trace IDs for request correlation
- Implemented JSON log formatting option
- Added CLI options for logging configuration (--log-level, --log-format, --enable-structured-logging)
- Improved test coverage to 86% with comprehensive security test suite
- Fixed missing python-dotenv dependency declaration
- Installed pre-commit hooks for secret scanning
- Added environment variable support for logging configuration
- Implemented user-friendly error messages with helpful suggestions
- Enhanced CLI help with examples and usage patterns
- Added interactive mode improvements with better error handling
- Created comprehensive user experience test suite (9 tests)
- Improved error correlation and debugging with structured error handling

## [0.2.0] - 2025-06-29
- Modularized QueryAgent with TTLCache and OpenAIAdapter
- Added batch row count support
- Introduced Flask web UI via `create_app`
- Expanded test suite covering new modules

## [0.2.1] - 2025-06-29
- Export Prometheus metrics for query counts and latency
- Logged metrics from QueryAgent

## [0.2.2] - 2025-06-29
- Updated OpenAI integration to use v1 ``openai.chat.completions`` API
- Added ``openai_timeout`` option and ``--openai-timeout`` CLI flag

## [0.1.0] - 2024-06-01
- Initial release.

