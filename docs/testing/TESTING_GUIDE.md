# Testing Guide - SQL Query Synthesizer

This guide covers the comprehensive testing strategy and tools for the SQL Query Synthesizer project.

## Testing Philosophy

Our testing approach follows the testing pyramid with a focus on:
- **Security First**: All tests include security validation
- **Performance Awareness**: Tests include performance benchmarks
- **Integration Focus**: Comprehensive integration testing
- **Production Readiness**: Tests mirror production scenarios

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for component interaction
├── e2e/                    # End-to-end user workflow tests
├── fixtures/               # Test data and schemas
├── conftest.py             # Pytest configuration and fixtures
├── test_runner.py          # Comprehensive test runner
└── test_*.py               # Individual test modules
```

## Test Categories

### 1. Unit Tests
- **Core Functionality**: `test_core.py`, `test_utils.py`
- **Database Layer**: `test_database_layer.py`
- **Security Components**: `test_security.py`, `test_enhanced_sql_injection.py`
- **Cache Systems**: `test_cache.py`, `test_cache_backends.py`
- **Services**: `test_query_service.py`, `test_sql_generator_service.py`

### 2. Integration Tests
- **API Endpoints**: `test_api_endpoints.py`
- **End-to-End Workflows**: `test_end_to_end.py`
- **Database Integration**: `test_query_agent_db_integration.py`
- **Security Integration**: `test_security_integration.py`

### 3. Performance Tests
- **Cache Metrics**: `test_cache_metrics.py`
- **Comprehensive Metrics**: `test_comprehensive_metrics.py`
- **Query Performance**: Performance markers in various tests

### 4. Security Tests
- **SQL Injection Prevention**: `test_enhanced_sql_injection.py`
- **Security Audit**: `test_security_audit.py`
- **Web Application Security**: `test_webapp_security.py`

## Running Tests

### Quick Start
```bash
# Run all tests
python3 tests/test_runner.py --all

# Run specific test categories
python3 tests/test_runner.py --unit --verbose
python3 tests/test_runner.py --integration
python3 tests/test_runner.py --security

# Run with coverage
python3 tests/test_runner.py --unit --coverage
```

### Using NPM Scripts
```bash
# Run all tests
npm run test

# Run with coverage
npm run test:coverage

# Run linting and formatting
npm run lint
npm run format:check

# Run type checking
npm run typecheck

# Run security scan
npm run security
```

### Using Pytest Directly
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_core.py -v

# Run tests with markers
python3 -m pytest -m "security" -v
python3 -m pytest -m "integration" -v
python3 -m pytest -m "performance" -v

# Run tests with coverage
python3 -m pytest tests/ --cov=sql_synthesizer --cov-report=html
```

## Test Configuration

### Environment Variables
The following environment variables are set automatically for tests:
```bash
TESTING=true
DATABASE_URL=sqlite:///:memory:
OPENAI_API_KEY=test-key-123
QUERY_AGENT_CACHE_BACKEND=memory
QUERY_AGENT_RATE_LIMIT_PER_MINUTE=1000
QUERY_AGENT_DEBUG=true
```

### Test Fixtures
Common fixtures available in `conftest.py`:
- `mock_database_manager`: Mock database operations
- `mock_query_agent`: Mock query agent functionality
- `sample_query_result`: Sample query response data
- `mock_llm_provider`: Mock LLM provider for testing
- `test_config`: Test-specific configuration

## Writing Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*` (e.g., `TestQueryAgent`)
- Test methods: `test_*` (e.g., `test_query_processing`)

### Test Structure
```python
class TestComponent:
    """Test the Component functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        pass
    
    def test_functionality_success(self):
        """Test successful functionality."""
        # Arrange
        # Act
        # Assert
        pass
    
    def test_functionality_error(self):
        """Test error handling."""
        pass
    
    def teardown_method(self):
        """Clean up after tests."""
        pass
```

### Async Test Example
```python
import pytest

class TestAsyncComponent:
    """Test async component functionality."""
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operation."""
        async with mock_database_manager.session() as session:
            result = await component.async_method(session)
            assert result is not None
```

### Security Test Example
```python
@pytest.mark.security
class TestSecurityValidation:
    """Test security validation."""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT password FROM admin"
        ]
        
        for malicious_input in malicious_inputs:
            result = validator.validate_input(malicious_input)
            assert not result.is_valid
            assert 'injection' in result.reason.lower()
```

### Performance Test Example
```python
@pytest.mark.performance
class TestPerformance:
    """Test performance characteristics."""
    
    def test_query_response_time(self, performance_benchmarks):
        """Test query response time meets benchmarks."""
        start_time = time.time()
        result = query_agent.query("Show me users")
        duration = time.time() - start_time
        
        assert duration <= performance_benchmarks['simple_query_max_time']
```

## Test Markers

Use pytest markers to categorize tests:
```python
@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.security
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.asyncio
```

Run specific markers:
```bash
pytest -m "security and not slow"
pytest -m "unit or integration"
```

## Code Coverage

### Coverage Requirements
- **Minimum Coverage**: 85% overall
- **Critical Components**: 95% coverage required
  - Security validators
  - SQL injection prevention
  - Core query processing
- **Documentation**: Coverage reports generated in `htmlcov/`

### Coverage Commands
```bash
# Generate coverage report
python3 -m pytest --cov=sql_synthesizer --cov-report=html

# View coverage in browser
open htmlcov/index.html

# Coverage with missing lines
python3 -m pytest --cov=sql_synthesizer --cov-report=term-missing
```

## Performance Benchmarks

### Expected Performance
- **Simple Query**: < 2 seconds
- **Complex Query**: < 5 seconds  
- **Health Check**: < 100ms
- **Cache Operation**: < 10ms
- **Database Connection**: < 1 second

### Memory Usage
- **Baseline**: < 512MB
- **Under Load**: < 1GB
- **Cache Size**: Configurable, monitored

## Best Practices

### Test Independence
- Tests should not depend on each other
- Use fixtures for setup/teardown
- Mock external dependencies

### Test Clarity
- Descriptive test names
- Clear arrange/act/assert structure
- Good error messages

### Performance
- Mark slow tests with `@pytest.mark.slow`
- Use mocking to avoid real I/O
- Parallel test execution when possible

### Security
- Always test security boundaries
- Include negative test cases
- Validate error messages don't leak information

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Include both positive and negative test cases
3. Add appropriate test markers
4. Update this documentation if needed
5. Ensure tests pass in CI environment