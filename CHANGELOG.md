# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2025-07-19
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

