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
```python
from sql_synthesizer import QueryAgent

agent = QueryAgent(database_url="your_db_connection")
result = agent.query("Show me the top 5 customers by revenue last quarter")
print(result.sql)
print(result.explanation)
print(result.data)
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

## License
MIT

## Development
Install pre-commit hooks to scan for secrets before commits:
```bash
pip install pre-commit detect-secrets
pre-commit install
```
Run `pre-commit run --all-files` to check manually.
