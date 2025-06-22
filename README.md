# SQL-Query-Synthesizer

Natural-language-to-SQL agent with automatic schema discovery and query validation.

## Features
- Schema introspection across multiple database types (PostgreSQL, MySQL, SQLite)
- Context-aware query generation with relationship understanding
- Query validation and optimization suggestions
- Interactive query refinement through conversational interface
- Support for complex joins, aggregations, and subqueries
- Built-in query explanation and performance analysis

## Quick Start
```bash
pip install -r requirements.txt
export DATABASE_URL="postgresql://user:pass@localhost/db"
python setup.py
python query_agent.py --interactive
```

## Usage
```python
from sql_synthesizer import QueryAgent

agent = QueryAgent(database_url="your_db_connection")
result = agent.query("Show me the top 5 customers by revenue last quarter")
print(result.sql)
print(result.explanation)
print(result.data)
```

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

## Roadmap
1. Add support for NoSQL databases
2. Implement query caching and performance monitoring
3. Build web UI for non-technical users

## License
MIT
