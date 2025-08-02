# Unit Tests

Unit tests for individual components and functions of the SQL Query Synthesizer.

## Guidelines

- Test individual functions and classes in isolation
- Use mocks for external dependencies
- Focus on testing business logic and edge cases
- Keep tests fast and independent
- Use descriptive test names that explain the scenario

## Structure

Tests are organized by module:
- `test_query_*` - Query generation and processing
- `test_cache_*` - Caching functionality
- `test_security_*` - Security features
- `test_database_*` - Database operations
- `test_webapp_*` - Web interface components

## Running Unit Tests

```bash
# Run all unit tests
make test-unit

# Run specific test file
pytest tests/unit/test_query_agent.py -v

# Run tests with coverage
pytest tests/unit/ --cov=sql_synthesizer --cov-report=html
```