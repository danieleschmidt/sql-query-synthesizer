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
generation.

## Supported Queries
- Aggregations: "What's the average order value by region?"
- Joins: "List customers with their recent orders"
- Time series: "Show monthly sales trends for 2024"
- Complex filters: "Find products with declining sales but high ratings"

## Configuration
Configure your database connections in `config/databases.yaml`:
```yaml
databases:
  production:
    url: "postgresql://..."
    schema_cache_ttl: 3600
  staging:
    url: "mysql://..."
```
Set `schema_cache_ttl` to control how long table names are cached when using the CLI.
Use `query_cache_ttl` to enable query result caching.
You can override the schema TTL via the `QUERY_AGENT_SCHEMA_CACHE_TTL` environment variable.
Set `OPENAI_API_KEY` and optionally `QUERY_AGENT_OPENAI_MODEL` to enable LLM-based SQL generation.

## Roadmap
1. Add support for NoSQL databases
2. Implement query caching and performance monitoring
3. Build web UI for non-technical users

## License
MIT
