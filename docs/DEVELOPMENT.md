# Development Guide

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Development Environment

* **Python**: 3.8+
* **Testing**: pytest with coverage
* **Linting**: ruff for code quality
* **Database**: SQLite for local development

## Running Tests

```bash
# Run all tests with coverage
pytest --cov=sql_synthesizer --cov=query_agent.py

# Run specific test modules
pytest tests/test_security.py
```

## Code Quality

```bash
# Format and lint code
ruff .
```

## Architecture

See [ARCHITECTURE.md](../ARCHITECTURE.md) and [ADR documentation](adr/) for system design details.

For deployment options, check [Dockerfile](../Dockerfile) and [docker-compose.yml](../docker-compose.yml).