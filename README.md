# SQL-Query-Synthesizer

Natural-language-to-SQL agent with automatic schema discovery and query validation.

## Features
- Schema introspection across multiple database types (PostgreSQL, MySQL, SQLite)
- Context-aware query generation with relationship understanding
- Query validation and optimization suggestions
- Interactive query refinement through conversational interface
- Support for complex joins, aggregations, and subqueries
- Built-in query explanation and performance analysis
- YAML-based configuration for multiple environments
- **Enterprise-grade database connection pooling** with automatic retry logic
- **Health monitoring** with comprehensive diagnostics and connection statistics
- **Production-ready error handling** with graceful degradation and recovery
- **LLM provider resilience** with circuit breaker pattern for OpenAI API failures
- **High-performance async I/O operations** for improved scalability and concurrent request handling

## Quick Start
```bash
pip install -e .
query-agent --interactive
```
The agent reads connection details from `config/databases.yaml` or the `DATABASE_URL` environment variable. Use `QUERY_AGENT_CONFIG` and `QUERY_AGENT_ENV` to override the configuration file and environment name. Table names are cached for the number of seconds specified by the `schema_cache_ttl` field.
Set `QUERY_AGENT_SCHEMA_CACHE_TTL` or `--schema-cache-ttl` to override this value.
Set `QUERY_AGENT_CACHE_TTL` or `--cache-ttl` to enable caching of query results.
Provide an OpenAI API key via the ``OPENAI_API_KEY`` environment variable to
generate SQL using a large language model. The model can be customized with
``--openai-model`` or ``QUERY_AGENT_OPENAI_MODEL``.

## Usage

### Basic Query Execution
```python
from sql_synthesizer import QueryAgent

agent = QueryAgent(database_url="your_db_connection")
result = agent.query("Show me the top 5 customers by revenue last quarter")
print(result.sql)
print(result.explanation)
print(result.data)
```

### Async Query Execution
```python
import asyncio
from sql_synthesizer import AsyncQueryAgent

async def main():
    async with AsyncQueryAgent(database_url="postgresql+asyncpg://...") as agent:
        result = await agent.query("Show me the top 5 customers by revenue last quarter")
        print(result.sql)
        print(result.explanation)
        print(result.data)

asyncio.run(main())
```

### Paginated Query Results
```python
from sql_synthesizer import QueryAgent

agent = QueryAgent(database_url="your_db_connection")

# Query with pagination
result = agent.query_paginated("Show all users", page=2, page_size=20)
print(f"Page {result.pagination.page} of {result.pagination.total_pages}")
print(f"Showing {len(result.data)} of {result.pagination.total_count} total results")
print(f"Has next page: {result.pagination.has_next}")

# Execute raw SQL with pagination
result = agent.execute_sql_paginated(
    "SELECT * FROM orders ORDER BY created_at DESC", 
    page=1, 
    page_size=10
)
```

### Async Paginated Queries
```python
import asyncio
from sql_synthesizer import AsyncQueryAgent

async def main():
    async with AsyncQueryAgent(database_url="postgresql+asyncpg://...") as agent:
        # Async query with pagination
        result = await agent.query_paginated("Show all users", page=2, page_size=20)
        print(f"Page {result.pagination.page} of {result.pagination.total_pages}")
        
        # Concurrent queries for better performance
        tasks = [
            agent.query("Count users by region"),
            agent.query("Show recent orders"),
            agent.execute_sql("SELECT COUNT(*) FROM products")
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            print(f"SQL: {result.sql}")
            print(f"Data: {result.data}")

asyncio.run(main())
```
Alternatively start an interactive session (database configured in `config/databases.yaml`):
```bash
query-agent --interactive
```
Run `query-agent --list-tables` to see available tables with row counts. Use `--max-rows` (or `QUERY_AGENT_MAX_ROWS`) to control how many rows are returned for a simple ``SELECT *``.
Run `query-agent --describe-table users` to list columns for the ``users`` table.
Run `query-agent --explain "select * from users"` to see the execution plan for a query.
Use `--output-csv results.csv` to write returned rows to ``results.csv``.
Run `query-agent --execute-sql "SELECT * FROM users"` to run raw SQL directly.
Use `--sql-only` to print generated SQL without executing it.
Set `--cache-ttl` (or `QUERY_AGENT_CACHE_TTL`) to reuse results for repeated questions.
Run `query-agent --clear-cache` to reset cached schema and query results.
Provide `--openai-api-key` (or set ``OPENAI_API_KEY``) to use an LLM for query
generation. Run ``python -m sql_synthesizer.webapp --database-url <db>`` to start
a simple web UI. Metrics are exposed at ``/metrics`` for Prometheus scraping and
available programmatically from ``prometheus_client.REGISTRY``. Use
``--openai-timeout`` (or ``QUERY_AGENT_OPENAI_TIMEOUT``) to specify a request
timeout for OpenAI calls.

## Supported Queries
- Aggregations: "What's the average order value by region?"
- Joins: "List customers with their recent orders"
- Time series: "Show monthly sales trends for 2024"
- Complex filters: "Find products with declining sales but high ratings"

## Configuration

### Database Configuration
Configure your database connections in `config/databases.yaml`:
```yaml
databases:
  production:
    url: "postgresql://..."
    schema_cache_ttl: 3600
  staging:
    url: "mysql://..."
```

### Environment Variables
All configuration options can be customized via environment variables with the `QUERY_AGENT_` prefix:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QUERY_AGENT_WEBAPP_PORT` | 5000 | Web server port |
| `QUERY_AGENT_WEBAPP_INPUT_SIZE` | 60 | HTML input field size |
| `QUERY_AGENT_MAX_QUESTION_LENGTH` | 1000 | Maximum question length in characters |
| `QUERY_AGENT_DEFAULT_MAX_ROWS` | 5 | Default maximum rows to return |
| `QUERY_AGENT_CACHE_CLEANUP_INTERVAL` | 300 | Cache cleanup interval in seconds |
| `QUERY_AGENT_OPENAI_TIMEOUT` | 30 | OpenAI API timeout in seconds |
| `QUERY_AGENT_DATABASE_TIMEOUT` | 30 | Database query timeout in seconds |
| `QUERY_AGENT_SCHEMA_CACHE_TTL` | - | Schema cache TTL override |
| `QUERY_AGENT_OPENAI_MODEL` | gpt-3.5-turbo | OpenAI model to use |

### Database Connection Pool Configuration
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QUERY_AGENT_DB_POOL_SIZE` | 10 | Connection pool size |
| `QUERY_AGENT_DB_MAX_OVERFLOW` | 20 | Maximum overflow connections |
| `QUERY_AGENT_DB_POOL_RECYCLE` | 3600 | Connection recycle time (seconds) |
| `QUERY_AGENT_DB_POOL_PRE_PING` | true | Enable connection health checks |
| `QUERY_AGENT_DB_CONNECT_RETRIES` | 3 | Number of connection retry attempts |
| `QUERY_AGENT_DB_RETRY_DELAY` | 1.0 | Base delay between retries (seconds) |

### Circuit Breaker Configuration (LLM Provider Resilience)
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QUERY_AGENT_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 5 | Number of failures before opening circuit |
| `QUERY_AGENT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | 60.0 | Seconds to wait before attempting recovery |

### Enhanced SQL Injection Prevention
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QUERY_AGENT_USE_ENHANCED_SQL_VALIDATION` | true | Enable enhanced SQL injection prevention with AST-based validation |

### Pagination Configuration
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QUERY_AGENT_DEFAULT_PAGE_SIZE` | 10 | Default number of items per page |
| `QUERY_AGENT_MAX_PAGE_SIZE` | 1000 | Maximum allowed page size |

### Security Configuration
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QUERY_AGENT_SECRET_KEY` | auto-generated | Flask secret key for session security |
| `QUERY_AGENT_CSRF_ENABLED` | true | Enable CSRF protection for forms |
| `QUERY_AGENT_RATE_LIMIT_PER_MINUTE` | 60 | API rate limit per client per minute |
| `QUERY_AGENT_ENABLE_HSTS` | false | Enable HTTP Strict Transport Security |
| `QUERY_AGENT_API_KEY_REQUIRED` | false | Require API key for API endpoints |
| `QUERY_AGENT_API_KEY` | - | API key for authentication |
| `QUERY_AGENT_MAX_REQUEST_SIZE_MB` | 1 | Maximum request size in megabytes |

### Prometheus Metrics Configuration
Histogram buckets for metrics can be customized via environment variables (comma-separated values):

- `QUERY_AGENT_OPENAI_REQUEST_BUCKETS`: OpenAI request duration buckets
- `QUERY_AGENT_DATABASE_QUERY_BUCKETS`: Database query duration buckets  
- `QUERY_AGENT_CACHE_OPERATION_BUCKETS`: Cache operation duration buckets

Set `OPENAI_API_KEY` and optionally `QUERY_AGENT_OPENAI_MODEL` to enable LLM-based SQL generation.
Environment variables may also be loaded from a `.env` file if present.

## Health Monitoring

The query agent provides comprehensive health monitoring capabilities:

```python
from sql_synthesizer import QueryAgent

agent = QueryAgent(database_url="your_db_connection")

# Get comprehensive health status
health = agent.health_check()
print(f"Overall healthy: {health['overall_healthy']}")
print(f"Database status: {health['database']['healthy']}")

# Monitor connection pool statistics
stats = agent.get_connection_stats()
print(f"Active connections: {stats['checked_out']}")
print(f"Pool size: {stats['pool_size']}")
```

### Health Check Endpoints
When using the web interface, health status is available at:
- `/health` - Basic health check
- `/metrics` - Prometheus metrics including connection pool stats

## Security Features

The SQL Query Synthesizer includes comprehensive security features for production deployments:

### Web Application Security
- **CSRF Protection**: Automatic Cross-Site Request Forgery protection for forms
- **Input Validation**: Length limits and sanitization of user input
- **Security Headers**: Comprehensive security headers (CSP, XSS protection, frame options)
- **Rate Limiting**: Configurable API rate limiting per client
- **Error Sanitization**: Prevents information leakage in error messages

### API Security
- **Optional API Key Authentication**: Secure API access with configurable API keys
- **Request Size Limits**: Configurable maximum request size
- **JSON Validation**: Strict validation of API request structure
- **Rate Limiting**: Per-client rate limiting with headers

### Example Security Configuration
```bash
# Enable production security features
export QUERY_AGENT_SECRET_KEY="your-secure-secret-key-here"
export QUERY_AGENT_CSRF_ENABLED=true
export QUERY_AGENT_RATE_LIMIT_PER_MINUTE=30
export QUERY_AGENT_ENABLE_HSTS=true
export QUERY_AGENT_API_KEY_REQUIRED=true
export QUERY_AGENT_API_KEY="your-api-key-here"
```

### Security Event Logging and Audit Trail

The SQL Query Synthesizer includes comprehensive security event logging for monitoring and compliance:

#### Event Types Logged
- **SQL Injection Attempts**: Pattern-based, AST analysis, and semantic detection
- **Authentication Failures**: API key validation failures
- **Rate Limiting Violations**: Exceeded request limits per client
- **Query Executions**: All successful SQL query executions with metadata
- **Schema Access**: Database schema discovery events
- **Input Validation**: Unsafe input detection and sanitization

#### Audit Log Format
All security events are logged in structured JSON format:

```json
{
  "event_type": "sql_injection_attempt",
  "severity": "high",
  "message": "SQL injection attempt detected: pattern_matching",
  "timestamp": "2025-07-21T15:30:45.123456Z",
  "client_ip": "192.168.1.100",
  "trace_id": "trace-sql-123",
  "additional_data": {
    "malicious_input": "'; DROP TABLE users; --",
    "detection_method": "pattern_matching",
    "input_length": 25
  }
}
```

#### Event Severity Levels
- **LOW**: Normal operations (query executions, schema access)
- **MEDIUM**: Suspicious activity (rate limiting, failed authentication)
- **HIGH**: Confirmed security violations (SQL injection, unauthorized access)
- **CRITICAL**: Immediate threats requiring response

#### Security Audit Statistics
Access real-time security event statistics programmatically:

```python
from sql_synthesizer.security_audit import security_audit_logger

# Get comprehensive security event statistics
stats = security_audit_logger.get_event_statistics()
print(f"Total security events: {stats['total_events']}")
print(f"SQL injection attempts: {stats['events_by_type'].get('sql_injection_attempt', 0)}")
print(f"High severity events: {stats['events_by_severity'].get('high', 0)}")
```

### Security Best Practices
1. **Always set a strong SECRET_KEY** in production
2. **Enable HSTS** when using HTTPS
3. **Use API keys** for programmatic access
4. **Monitor rate limiting** metrics
5. **Review security logs** regularly
6. **Set up alerting** on high/critical severity security events
7. **Implement log aggregation** for centralized security monitoring

## License
MIT

## Development
Install pre-commit hooks to scan for secrets before commits:
```bash
pip install pre-commit detect-secrets
pre-commit install
```
Run `pre-commit run --all-files` to check manually.
