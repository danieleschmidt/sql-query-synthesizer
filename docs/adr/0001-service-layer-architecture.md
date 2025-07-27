# ADR-0001: Service Layer Architecture

## Status
Accepted

## Context
The original SQL Synthesizer was implemented as a monolithic `QueryAgent` class with 489 lines of code handling query validation, SQL generation, database execution, caching, and metrics collection. This design made the code difficult to test, maintain, and extend.

Key issues with the monolithic approach:
- Single responsibility principle violations
- Difficult unit testing (mocking entire system)
- Tight coupling between concerns
- Code duplication across similar functionality
- Hard to add new features without affecting existing code

## Decision
Implement a service layer architecture that separates concerns into focused services:

1. **QueryValidatorService**: Input validation and security
2. **SQLGeneratorService**: SQL generation from natural language
3. **QueryService**: Core orchestration and execution
4. **QueryAgent**: Public facade maintaining backward compatibility

### Service Responsibilities

**QueryValidatorService**:
- User question sanitization
- SQL injection pattern detection (15+ patterns)
- SQL syntax validation
- Table name validation
- Input length limits

**SQLGeneratorService**:
- OpenAI LLM integration with error handling
- Naive keyword-based SQL generation fallback
- Circuit breaker pattern for API failures
- Error recovery with helpful messages

**QueryService**:
- Query lifecycle orchestration
- Multi-level caching (schema & results)
- Database connection management
- Metrics collection and tracing
- Performance monitoring

**QueryAgent**:
- Public API facade (backward compatible)
- Service dependency injection
- Configuration management

## Consequences

### Positive
- **Single Responsibility**: Each service has one focused purpose
- **Testability**: 33 comprehensive service tests with 100% mock coverage
- **Maintainability**: Clear separation of concerns, easier to modify
- **Security**: Centralized validation with comprehensive threat detection
- **Performance**: Optimized caching and connection management
- **Backward Compatibility**: Zero breaking changes to public API
- **Extensibility**: Easy to add new services or modify existing ones

### Negative
- **Complexity**: More files and classes to manage
- **Indirection**: Additional layer between public API and implementation
- **Learning Curve**: Developers need to understand service interactions

### Mitigations
- Comprehensive documentation of service interactions
- Clear dependency injection patterns
- Facade pattern maintains simple public API
- Extensive test coverage ensures reliability

## Implementation Notes
- Services use dependency injection for testability
- All existing tests pass without modification
- Performance metrics show no regression
- Memory usage remains comparable to monolithic version