# Integration Tests

Integration tests that verify the interaction between different components of the SQL Query Synthesizer.

## Guidelines

- Test component interactions and data flow
- Use real database connections (test databases)
- Test API endpoints and external service integrations
- Verify system behavior under realistic conditions
- Include performance and reliability testing

## Test Categories

- **Database Integration** - Tests with real database connections
- **API Integration** - Tests of REST API endpoints
- **Service Integration** - Tests of service layer interactions
- **Cache Integration** - Tests of caching with real backends
- **Security Integration** - End-to-end security testing

## Running Integration Tests

```bash
# Run all integration tests
make test-integration

# Run specific integration test
pytest tests/integration/ -m integration -v

# Run with database setup
QUERY_AGENT_TEST_DB=true pytest tests/integration/
```

## Requirements

Integration tests require:
- Test database instances
- External service configurations
- Appropriate environment variables
- Network access for external APIs