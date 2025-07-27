# SQL Synthesizer Architecture

## Service Layer Pattern (v0.5.0+)

The SQL Synthesizer follows a clean service layer architecture that separates concerns and improves maintainability:

```
┌─────────────────────┐
│    QueryAgent       │  ← Public API (Backward Compatible)
│   (Facade/API)      │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│   QueryService      │  ← Core Orchestration
│ - Query processing  │
│ - Caching           │
│ - Database execution│
│ - Metrics           │
└─────┬───────┬───────┘
      │       │
      │       │
┌─────▼─────┐ │ ┌────▼──────────┐
│QueryValid-│ │ │SQLGenerator   │
│atorService│ │ │Service        │
│- Input    │ │ │- OpenAI calls │
│  sanitation│ │ │- Naive        │
│- SQL      │ │ │  generation   │
│  injection│ │ │- Fallback     │
│  prevention│ │ │  handling     │
│- Table    │ │ └───────────────┘
│  validation│ │
└───────────┘ │
              │
        ┌─────▼─────┐
        │  Cache    │
        │ TTLCache  │
        │ (Schema & │
        │  Query)   │
        └───────────┘
```

## Core Services

### QueryValidatorService
- **Purpose**: Input validation and security
- **Responsibilities**:
  - User question sanitization
  - SQL injection pattern detection  
  - SQL syntax validation
  - Table name validation
- **Security Features**: 15+ SQL injection patterns, input length limits, malicious content detection

### SQLGeneratorService  
- **Purpose**: SQL generation from natural language
- **Responsibilities**:
  - OpenAI LLM integration with error handling
  - Naive keyword-based SQL generation fallback
  - Error recovery and helpful error messages
- **Fallback Strategy**: OpenAI → Naive Generation → Helpful Error Comments

### QueryService
- **Purpose**: Core query orchestration and execution
- **Responsibilities**:
  - Query lifecycle management
  - Caching (schema & query results)
  - Database execution with connection management
  - Metrics collection and performance tracking
  - Trace ID correlation for structured logging

## Key Benefits

1. **Single Responsibility**: Each service has one focused purpose
2. **Testability**: 33 comprehensive service tests with 100% mock coverage  
3. **Maintainability**: Reduced from 489-line monolith to focused services
4. **Security**: Centralized validation with comprehensive threat detection
5. **Performance**: Optimized caching and connection management
6. **Backward Compatibility**: Zero breaking changes to public API

## Design Patterns Used

- **Service Layer**: Encapsulates business logic in focused services
- **Facade Pattern**: QueryAgent provides simplified interface to complex subsystem
- **Dependency Injection**: Services receive dependencies via constructor
- **Strategy Pattern**: SQL generation with OpenAI/naive strategies
- **Template Method**: Common query execution patterns with customization points

## Testing Strategy

- **Unit Tests**: Each service tested in isolation with mocks
- **Integration Tests**: QueryAgent tested with real database interactions
- **Security Tests**: Comprehensive SQL injection and validation testing
- **Backward Compatibility**: All 45 existing tests pass without modification

## File Structure

```
sql_synthesizer/
├── query_agent.py          # Public facade (refactored)
├── types.py                # Common types (QueryResult, etc.)
├── services/
│   ├── __init__.py
│   ├── query_validator_service.py
│   ├── sql_generator_service.py
│   └── query_service.py
└── tests/
    ├── test_query_validator_service.py  # 11 tests
    ├── test_sql_generator_service.py    # 11 tests
    ├── test_query_service.py            # 11 tests
    └── test_agent.py                    # 12 integration tests
```

This architecture enables easier feature development, better testing, and improved code quality while maintaining the simple, user-friendly API that users expect.

## System Design Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SQL Synthesizer System                      │
├─────────────────────────────────────────────────────────────────┤
│  Web Interface (Flask)          API Layer (REST)                │
│  ┌─────────────────┐           ┌─────────────────┐              │
│  │ Interactive UI  │           │ RESTful APIs    │              │
│  │ - Query Form    │           │ - /api/query    │              │
│  │ - Results View  │           │ - /health       │              │
│  │ - Health Check  │           │ - /metrics      │              │
│  └─────────────────┘           └─────────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                        Service Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Query Service   │  │ Validator Svc   │  │ SQL Gen Service │ │
│  │ - Orchestration │  │ - Input Valid   │  │ - LLM Interface │ │
│  │ - Caching       │  │ - SQL Injection │  │ - Fallback Gen  │ │
│  │ - Metrics       │  │ - Security      │  │ - Error Handling│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                        Data Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Cache Layer     │  │ Database Layer  │  │ External APIs   │ │
│  │ - Redis/Memory  │  │ - PostgreSQL    │  │ - OpenAI LLM    │ │
│  │ - Memcached     │  │ - MySQL         │  │ - Prometheus    │ │
│  │ - TTL Management│  │ - SQLite        │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
User Input → Validation → SQL Generation → Database Execution → Result Formatting
     ↓             ↓            ↓                ↓                    ↓
 Web UI/API → Security Check → LLM/Fallback → Connection Pool → JSON/HTML Response
     ↓             ↓            ↓                ↓                    ↓
  Rate Limit → Injection Detect → Cache Check → Health Monitor → Metrics Collection
```

### Component Interaction Patterns

1. **Request Flow**: Web/API → QueryAgent → QueryService → Validator/Generator → Database
2. **Caching Strategy**: Multi-level caching (schema, query results, metadata)
3. **Error Handling**: Circuit breaker pattern for external dependencies
4. **Security**: Input validation → SQL injection prevention → Safe execution
5. **Monitoring**: Request tracing → Performance metrics → Health checks

## Deployment Architecture

### Production Deployment Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                      Load Balancer                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ App Instance 1  │  │ App Instance 2  │  │ App Instance N  │ │
│  │ - Health Check  │  │ - Health Check  │  │ - Health Check  │ │
│  │ - Metrics       │  │ - Metrics       │  │ - Metrics       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Redis Cluster   │  │ Database Pool   │  │ Monitoring      │ │
│  │ - Cache Layer   │  │ - Connection Mgmt│  │ - Prometheus    │ │
│  │ - Session Store │  │ - Read Replicas │  │ - Grafana       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Container Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Container                             │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer (Python 3.11)                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Flask App       │  │ Worker Processes│  │ Health Monitor  │ │
│  │ - Gunicorn      │  │ - Async Tasks   │  │ - Readiness     │ │
│  │ - HTTP Server   │  │ - Background    │  │ - Liveness      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Base Image: python:3.11-slim                                  │
│  Security: Non-root user, minimal dependencies                 │
└─────────────────────────────────────────────────────────────────┘
```

## Security Architecture

### Defense in Depth Strategy

1. **Input Layer Security**
   - Request size limits
   - Rate limiting per client
   - Input sanitization and validation

2. **Application Layer Security**
   - SQL injection prevention (pattern + AST analysis)
   - CSRF protection
   - Security headers (CSP, HSTS, X-Frame-Options)
   - API key authentication

3. **Data Layer Security**
   - Database connection encryption
   - Prepared statements only
   - Connection pool isolation
   - Audit logging

4. **Infrastructure Security**
   - Container security scanning
   - Dependency vulnerability monitoring
   - Secrets management
   - Network isolation

### Security Event Flow

```
Request → Rate Limit Check → Authentication → Input Validation → SQL Analysis → Audit Log
   ↓           ↓                  ↓               ↓                ↓             ↓
  Block      Block              Block           Block           Block         Store
Excessive   Unauthorized      Malformed       Injection       Suspicious    Events
```

## Performance Architecture

### Caching Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-Level Caching                         │
├─────────────────────────────────────────────────────────────────┤
│  L1: In-Memory Cache (TTLCache)                                 │
│  ├─ Schema metadata (tables, columns)                           │
│  ├─ Query results (recent queries)                              │
│  └─ Connection pool statistics                                  │
├─────────────────────────────────────────────────────────────────┤
│  L2: Redis Cache (Optional)                                     │
│  ├─ Shared query results across instances                       │
│  ├─ Session data and user preferences                           │
│  └─ Rate limiting counters                                      │
├─────────────────────────────────────────────────────────────────┤
│  L3: Database Query Cache                                       │
│  ├─ Database-level query result caching                         │
│  ├─ Materialized views for complex queries                      │
│  └─ Index optimization                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Monitoring Points

1. **Request Latency**: End-to-end response times
2. **Database Performance**: Query execution times, connection pool metrics
3. **Cache Hit Rates**: Memory, Redis, and database cache effectiveness
4. **External API Latency**: OpenAI API response times
5. **Resource Utilization**: CPU, memory, network usage

This comprehensive architecture supports scalable, secure, and maintainable SQL query synthesis with robust monitoring and operational capabilities.