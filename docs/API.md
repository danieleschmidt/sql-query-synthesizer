# API Documentation

## Overview

The SQL Query Synthesizer provides both REST API endpoints and Python SDK interfaces for programmatic access to natural language SQL generation capabilities.

## REST API

### Base URL
```
http://localhost:5000
```

### Authentication

#### API Key Authentication (Optional)
```bash
# Enable API key authentication
export QUERY_AGENT_API_KEY_REQUIRED=true
export QUERY_AGENT_API_KEY="your-secure-api-key"

# Use API key in requests
curl -H "X-API-Key: your-secure-api-key" \
  http://localhost:5000/api/query
```

#### Rate Limiting
- **Default Limit**: 60 requests per minute per client
- **Headers**: Rate limit information in response headers
- **Configuration**: `QUERY_AGENT_RATE_LIMIT_PER_MINUTE`

### Endpoints

#### POST /api/query
Generate SQL from natural language query.

**Request:**
```json
{
  "question": "Show me the top 5 customers by revenue last quarter",
  "database_url": "postgresql://user:pass@host:port/db",
  "max_rows": 100,
  "explain": false
}
```

**Response:**
```json
{
  "sql": "SELECT customer_name, SUM(order_total) as revenue FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.created_at >= '2024-04-01' AND o.created_at < '2024-07-01' GROUP BY customer_name ORDER BY revenue DESC LIMIT 5",
  "explanation": "This query joins customers and orders tables, filters for Q2 2024, groups by customer, and returns the top 5 by revenue.",
  "data": [
    {"customer_name": "Acme Corp", "revenue": 150000},
    {"customer_name": "Beta LLC", "revenue": 125000}
  ],
  "row_count": 5,
  "execution_time": 0.245,
  "cached": false,
  "trace_id": "trace-sql-123"
}
```

**Parameters:**
- `question` (string, required): Natural language query
- `database_url` (string, optional): Database connection string
- `max_rows` (integer, optional): Maximum rows to return (default: 100)
- `explain` (boolean, optional): Include query explanation (default: false)

**Status Codes:**
- `200`: Success
- `400`: Bad request (invalid parameters)
- `401`: Unauthorized (API key required)
- `429`: Rate limit exceeded
- `500`: Internal server error

#### POST /api/query/paginated
Execute paginated queries for large result sets.

**Request:**
```json
{
  "question": "Show all users created this year",
  "page": 2,
  "page_size": 50,
  "database_url": "postgresql://user:pass@host:port/db"
}
```

**Response:**
```json
{
  "sql": "SELECT * FROM users WHERE created_at >= '2024-01-01' ORDER BY created_at DESC",
  "data": [...],
  "pagination": {
    "page": 2,
    "page_size": 50,
    "total_count": 1250,
    "total_pages": 25,
    "has_next": true,
    "has_previous": true
  },
  "execution_time": 0.156
}
```

#### POST /api/sql/execute
Execute raw SQL queries directly.

**Request:**
```json
{
  "sql": "SELECT COUNT(*) FROM users WHERE active = true",
  "database_url": "postgresql://user:pass@host:port/db",
  "max_rows": 1000
}
```

**Response:**
```json
{
  "data": [{"count": 1250}],
  "row_count": 1,
  "execution_time": 0.023,
  "columns": ["count"]
}
```

#### GET /api/schema
Retrieve database schema information.

**Parameters:**
- `database_url` (query parameter): Database connection string
- `table_name` (query parameter, optional): Specific table schema

**Response:**
```json
{
  "tables": [
    {
      "name": "users",
      "columns": [
        {"name": "id", "type": "integer", "nullable": false, "primary_key": true},
        {"name": "email", "type": "varchar", "nullable": false, "unique": true},
        {"name": "created_at", "type": "timestamp", "nullable": false}
      ],
      "row_count": 1250,
      "indexes": ["idx_users_email", "idx_users_created_at"]
    }
  ],
  "cached": true,
  "cache_ttl": 3600
}
```

#### GET /health
System health check with detailed status information.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.2.2",
  "timestamp": "2025-07-29T10:30:00Z",
  "checks": {
    "database": {
      "healthy": true,
      "response_time": 0.012,
      "pool_status": {
        "checked_out": 2,
        "pool_size": 20,
        "overflow": 0
      }
    },
    "cache": {
      "healthy": true,
      "backend": "redis",
      "hit_rate": 0.85,
      "total_operations": 15620
    },
    "openai": {
      "healthy": true,
      "circuit_breaker_state": "closed",
      "last_request_time": 0.856
    }
  }
}
```

#### GET /metrics
Prometheus metrics endpoint for monitoring.

**Response:**
```
# HELP query_requests_total Total number of query requests
# TYPE query_requests_total counter
query_requests_total{status="success"} 1250
query_requests_total{status="error"} 23

# HELP query_duration_seconds Query execution duration
# TYPE query_duration_seconds histogram
query_duration_seconds_bucket{le="0.1"} 890
query_duration_seconds_bucket{le="0.5"} 1200
query_duration_seconds_bucket{le="1.0"} 1250
```

### Error Handling

#### Error Response Format
```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "The provided query contains invalid SQL syntax",
    "details": {
      "input": "user input that caused error",
      "suggestion": "Try rephrasing your question more specifically"
    },
    "trace_id": "trace-error-456"
  }
}
```

#### Common Error Codes
- `INVALID_QUERY`: Malformed or invalid query
- `SQL_INJECTION_DETECTED`: Security violation detected
- `DATABASE_CONNECTION_ERROR`: Database connectivity issues
- `OPENAI_API_ERROR`: LLM service unavailable
- `CACHE_ERROR`: Cache backend error
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `AUTHENTICATION_REQUIRED`: API key required
- `AUTHORIZATION_FAILED`: Invalid API key

## Python SDK

### Installation
```bash
pip install sql-synthesizer
```

### Basic Usage

#### Synchronous Client
```python
from sql_synthesizer import QueryAgent

# Initialize client
agent = QueryAgent(
    database_url="postgresql://user:pass@host:port/db",
    openai_api_key="your-openai-key"
)

# Execute query
result = agent.query("Show me the top 5 products by sales")
print(f"SQL: {result.sql}")
print(f"Data: {result.data}")
print(f"Execution time: {result.execution_time}s")

# Paginated queries
paginated_result = agent.query_paginated(
    "Show all orders from this year",
    page=1,
    page_size=100
)
print(f"Page {paginated_result.pagination.page} of {paginated_result.pagination.total_pages}")

# Raw SQL execution
sql_result = agent.execute_sql("SELECT COUNT(*) FROM users")
print(f"Count: {sql_result.data[0]['count']}")

# Schema introspection
schema = agent.get_schema()
for table in schema.tables:
    print(f"Table: {table.name} ({table.row_count} rows)")
```

#### Asynchronous Client
```python
import asyncio
from sql_synthesizer import AsyncQueryAgent

async def main():
    async with AsyncQueryAgent(
        database_url="postgresql+asyncpg://user:pass@host:port/db",
        openai_api_key="your-openai-key"
    ) as agent:
        # Concurrent queries
        tasks = [
            agent.query("Show user count by region"),
            agent.query("Show top products"),
            agent.query("Show recent orders")
        ]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            print(f"SQL: {result.sql}")
            print(f"Rows: {len(result.data)}")

asyncio.run(main())
```

### Configuration Options

#### QueryAgent Configuration
```python
from sql_synthesizer import QueryAgent, Config

# Custom configuration
config = Config(
    cache_backend="redis",
    cache_ttl=7200,
    database_timeout=60,
    openai_timeout=30,
    max_rows=1000,
    log_level="DEBUG"
)

agent = QueryAgent(
    database_url="your-db-url",
    config=config
)
```

#### Environment Variables
```python
import os

# Set configuration via environment
os.environ["QUERY_AGENT_CACHE_BACKEND"] = "redis"
os.environ["QUERY_AGENT_CACHE_TTL"] = "3600"
os.environ["QUERY_AGENT_MAX_ROWS"] = "500"

from sql_synthesizer import QueryAgent
agent = QueryAgent()  # Uses environment configuration
```

### Advanced Features

#### Custom Cache Backend
```python
from sql_synthesizer.cache import Cache
from sql_synthesizer import QueryAgent

# Redis cache
cache = Cache.create(
    backend="redis",
    host="redis.example.com",
    port=6379,
    db=0,
    password="redis-password"
)

agent = QueryAgent(cache=cache)
```

#### Health Monitoring
```python
# Get detailed health status
health = agent.health_check()
print(f"Overall healthy: {health['overall_healthy']}")
print(f"Database status: {health['database']['healthy']}")
print(f"Cache hit rate: {health['cache']['hit_rate']:.2%}")

# Connection pool statistics
stats = agent.get_connection_stats()
print(f"Active connections: {stats['checked_out']}")
print(f"Pool utilization: {stats['checked_out']/stats['pool_size']:.2%}")
```

#### Metrics Collection
```python
from sql_synthesizer.metrics import get_metrics

# Get Prometheus metrics
metrics = get_metrics()
print(f"Total queries: {metrics.query_total}")
print(f"Average duration: {metrics.query_duration_avg}s")
print(f"Error rate: {metrics.error_rate:.2%}")
```

### Error Handling

#### Exception Types
```python
from sql_synthesizer.exceptions import (
    QuerySynthesizerError,
    DatabaseConnectionError,
    SQLInjectionError,
    OpenAIAPIError,
    CacheError
)

try:
    result = agent.query("malicious query'; DROP TABLE users; --")
except SQLInjectionError as e:
    print(f"Security violation: {e}")
except DatabaseConnectionError as e:
    print(f"Database error: {e}")
except OpenAIAPIError as e:
    print(f"LLM service error: {e}")
except QuerySynthesizerError as e:
    print(f"General error: {e}")
```

#### Retry Logic
```python
from sql_synthesizer.retry import with_retry

@with_retry(max_attempts=3, backoff_factor=2.0)
def robust_query(agent, question):
    return agent.query(question)

# Automatic retry on transient failures
result = robust_query(agent, "Show user statistics")
```

## WebSocket API (Experimental)

### Real-time Query Streaming
```javascript
// JavaScript WebSocket client
const ws = new WebSocket('ws://localhost:5000/ws');

ws.onopen = function() {
    // Send query request
    ws.send(JSON.stringify({
        type: 'query',
        question: 'Show sales trends over time',
        stream: true
    }));
};

ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    if (response.type === 'partial_result') {
        console.log('Partial data:', response.data);
    } else if (response.type === 'complete') {
        console.log('Query complete:', response.sql);
    }
};
```

## GraphQL API (Planned)

### Query Interface
```graphql
type Query {
  generateSQL(
    question: String!
    databaseUrl: String
    maxRows: Int = 100
  ): SQLResult!
  
  executeSQL(
    sql: String!
    databaseUrl: String
  ): QueryResult!
  
  schema(databaseUrl: String!): DatabaseSchema!
  
  health: HealthStatus!
}

type SQLResult {
  sql: String!
  explanation: String
  data: [JSON!]!
  executionTime: Float!
  cached: Boolean!
}
```

## Rate Limiting and Quotas

### Request Limits
```yaml
rate_limits:
  api_key_authenticated:
    requests_per_minute: 300
    requests_per_hour: 10000
    requests_per_day: 100000
  
  unauthenticated:
    requests_per_minute: 60
    requests_per_hour: 1000
    requests_per_day: 5000
```

### Usage Tracking
```python
# Monitor API usage
from sql_synthesizer.usage import get_usage_stats

stats = get_usage_stats(api_key="your-api-key")
print(f"Requests today: {stats['requests_today']}")
print(f"Quota remaining: {stats['quota_remaining']}")
```

## SDK Examples

### Batch Processing
```python
# Process multiple queries efficiently
questions = [
    "Show user count by country",
    "Show revenue by product category",
    "Show monthly growth rate"
]

# Synchronous batch
results = agent.batch_query(questions)

# Asynchronous batch
async with AsyncQueryAgent() as agent:
    results = await agent.batch_query(questions)
```

### Data Pipeline Integration
```python
import pandas as pd
from sql_synthesizer import QueryAgent

def query_to_dataframe(agent, question):
    """Convert query result to pandas DataFrame"""
    result = agent.query(question)
    return pd.DataFrame(result.data)

# Example usage
agent = QueryAgent()
df = query_to_dataframe(agent, "Show all active users")
print(df.head())
```

### Custom Validation
```python
from sql_synthesizer.validation import SQLValidator

# Custom validator
validator = SQLValidator(
    allowed_tables=["users", "orders", "products"],
    allowed_operations=["SELECT", "WITH"],
    max_query_length=1000
)

agent = QueryAgent(validator=validator)
```

This API documentation provides comprehensive coverage of all available interfaces and usage patterns for the SQL Query Synthesizer.