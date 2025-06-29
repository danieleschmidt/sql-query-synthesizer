# Changelog

All notable changes to this project will be documented in this file.

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

