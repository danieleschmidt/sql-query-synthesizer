# Testing Documentation

Comprehensive testing guide for the SQL Query Synthesizer.

## Testing Philosophy

The SQL Query Synthesizer follows a comprehensive testing strategy with multiple layers:

1. **Unit Tests** - Fast, isolated tests for individual components
2. **Integration Tests** - Tests for component interactions and external dependencies
3. **End-to-End Tests** - Complete workflow testing
4. **Security Tests** - Specialized security and vulnerability testing
5. **Performance Tests** - Load testing and performance benchmarking

## Test Structure

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Component interaction tests
├── e2e/           # End-to-end workflow tests
├── fixtures/      # Test data and fixtures
└── *.py          # Legacy test files (to be migrated)
```

## Running Tests

### All Tests
```bash
make test                    # Run complete test suite
make test-coverage          # Run with detailed coverage report
```

### By Category
```bash
make test-unit              # Unit tests only
make test-integration       # Integration tests only
pytest tests/e2e/ -v       # End-to-end tests
make test-security          # Security-focused tests
make test-performance       # Performance benchmarks
```

### Specific Tests
```bash
pytest tests/unit/test_query_agent.py -v
pytest tests/integration/ -k "database" -v
pytest tests/ -m "not slow" -v    # Skip slow tests
```

## Test Configuration

### Environment Variables
```bash
# Test database configuration
export QUERY_AGENT_TEST_DB_URL="postgresql://test:test@localhost/test_db"
export QUERY_AGENT_TEST_CACHE_BACKEND="memory"

# External service testing
export OPENAI_API_KEY="test-key-for-testing"
export QUERY_AGENT_TEST_MODE=true

# Performance testing
export QUERY_AGENT_PERFORMANCE_TESTS=true
export QUERY_AGENT_LOAD_TEST_USERS=10
```

### Pytest Markers
```bash
pytest -m unit              # Run unit tests
pytest -m integration       # Run integration tests
pytest -m security          # Run security tests
pytest -m slow              # Run slow tests
pytest -m "not slow"        # Skip slow tests
```

## Test Data and Fixtures

### Using Fixtures
```python
from tests.fixtures import load_fixture

def test_query_generation():
    test_cases = load_fixture("sample_queries.json")
    for case in test_cases["test_queries"]:
        # Test with fixture data
        pass
```

### Database Fixtures
- Sample schemas in `tests/fixtures/database_schemas/`
- Test data sets for common scenarios
- Mock external service responses

## Mocking and Test Doubles

### Database Mocking
```python
import pytest
from unittest.mock import patch

@patch('sql_synthesizer.database.get_engine')
def test_database_operation(mock_engine):
    # Test with mocked database
    pass
```

### External Service Mocking
```python
@patch('openai.ChatCompletion.create')
def test_llm_integration(mock_openai):
    mock_openai.return_value = {"choices": [{"message": {"content": "SELECT * FROM users;"}}]}
    # Test with mocked OpenAI response
```

## Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Critical Paths**: 95% coverage required
- **Security Functions**: 100% coverage required

### Generating Coverage Reports
```bash
make test-coverage          # HTML and terminal report
pytest --cov-report=xml     # XML report for CI
```

## Security Testing

### SQL Injection Testing
```python
def test_sql_injection_prevention():
    malicious_input = "'; DROP TABLE users; --"
    # Verify input is sanitized
```

### Authentication Testing
```python
def test_api_authentication():
    # Test API key validation
    # Test rate limiting
    # Test authorization
```

## Performance Testing

### Benchmarking
```python
def test_query_performance(benchmark):
    result = benchmark(agent.query, "SELECT * FROM users")
    assert result.execution_time < 0.1
```

### Load Testing
```bash
# Using locust for load testing
locust -f tests/e2e/load_test.py --host http://localhost:5000
```

## Continuous Integration

### GitHub Actions Integration
```yaml
- name: Run Tests
  run: |
    make ci-test
    make ci-security
```

### Test Reports
- Coverage reports uploaded to Codecov
- Security scan results in CI artifacts
- Performance regression detection

## Debugging Tests

### Verbose Output
```bash
pytest -v -s                # Verbose with stdout
pytest --tb=long            # Detailed tracebacks
pytest --pdb               # Drop into debugger on failure
```

### Test Isolation
```bash
pytest --lf                # Last failed tests only
pytest --ff                # Failed first, then rest
pytest -x                  # Stop on first failure
```

## Adding New Tests

### Test Naming Conventions
- `test_<component>_<scenario>.py` for test files
- `test_<action>_<expected_result>()` for test functions
- Use descriptive names that explain the test scenario

### Test Structure
```python
def test_query_generation_with_valid_input():
    """Test that valid natural language generates correct SQL."""
    # Arrange
    agent = QueryAgent()
    question = "Show all users"
    
    # Act
    result = agent.query(question)
    
    # Assert
    assert result.sql == "SELECT * FROM users;"
    assert result.error is None
```

### Documentation
- Document complex test scenarios
- Include setup requirements for integration tests
- Explain any special test data or fixtures needed

For more detailed testing information, see the specific README files in each test directory.