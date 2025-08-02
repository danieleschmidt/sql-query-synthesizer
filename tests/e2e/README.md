# End-to-End Tests

End-to-end tests that verify complete user workflows and system functionality.

## Guidelines

- Test complete user journeys
- Use real external services when possible
- Test browser interactions for web interface
- Verify system behavior under production-like conditions
- Include performance and load testing scenarios

## Test Scenarios

- **Query Generation Workflow** - Complete natural language to SQL flow
- **Web Interface Testing** - Browser-based interaction testing
- **API Workflow Testing** - Complete API request/response cycles
- **Multi-user Scenarios** - Concurrent user testing
- **Error Recovery Testing** - System resilience testing

## Running E2E Tests

```bash
# Run all e2e tests
pytest tests/e2e/ -v

# Run with browser testing
BROWSER_TESTS=true pytest tests/e2e/

# Run performance tests
pytest tests/e2e/ --benchmark-only
```

## Requirements

E2E tests require:
- Complete system deployment
- External service access
- Browser testing tools (for web interface)
- Load testing tools
- Production-like environment configuration

## Tools

- **Playwright/Selenium** - Browser automation
- **pytest-benchmark** - Performance testing
- **locust** - Load testing
- **requests** - API testing