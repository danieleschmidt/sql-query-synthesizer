# SQL Query Synthesizer API Documentation

## Overview

The SQL Query Synthesizer provides a RESTful API for natural language to SQL conversion with autonomous SDLC capabilities.

## Base URL

```
Production: https://api.sql-synthesizer.com
Staging: https://staging-api.sql-synthesizer.com
Development: http://localhost:5000
```

## Authentication

All API requests require authentication via API key:

```http
X-API-Key: your-api-key-here
```

## Endpoints

### Health Check

```http
GET /health
```

Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00Z",
  "checks": {
    "database": "healthy",
    "cache": "healthy",
    "memory": "healthy"
  }
}
```

### Query Synthesis

```http
POST /api/query
```

Convert natural language to SQL query.

**Request Body:**
```json
{
  "question": "Show me all users who registered last month",
  "schema": {
    "tables": ["users", "orders"],
    "columns": {
      "users": ["id", "email", "created_at"],
      "orders": ["id", "user_id", "amount"]
    }
  },
  "options": {
    "explain": true,
    "validate": true
  }
}
```

**Response:**
```json
{
  "sql": "SELECT * FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH)",
  "explanation": "This query selects all users created in the last month",
  "confidence": 0.95,
  "execution_plan": "...",
  "metadata": {
    "processing_time_ms": 150,
    "complexity": "simple"
  }
}
```

### Schema Discovery

```http
GET /api/schema
```

Discover database schema information.

**Response:**
```json
{
  "tables": [
    {
      "name": "users",
      "columns": [
        {"name": "id", "type": "INTEGER", "primary_key": true},
        {"name": "email", "type": "VARCHAR(255)", "nullable": false}
      ]
    }
  ]
}
```

### Metrics

```http
GET /metrics
```

Prometheus-compatible metrics endpoint.

## Error Handling

The API uses standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `422` - Validation Error
- `500` - Internal Server Error

**Error Response Format:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid query format",
    "details": {
      "field": "question",
      "reason": "Required field missing"
    }
  }
}
```

## Rate Limiting

- **Free Tier**: 100 requests/hour
- **Premium**: 1000 requests/hour
- **Enterprise**: Unlimited

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## SDKs and Examples

### Python

```python
import requests

response = requests.post(
    "http://localhost:5000/api/query",
    headers={"X-API-Key": "your-key"},
    json={"question": "Show me recent orders"}
)

result = response.json()
print(result["sql"])
```

### JavaScript

```javascript
const response = await fetch('/api/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-key'
  },
  body: JSON.stringify({
    question: 'Show me recent orders'
  })
});

const result = await response.json();
console.log(result.sql);
```

### cURL

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"question": "Show me recent orders"}'
```

## Autonomous SDLC Integration

### Quality Metrics

```http
GET /api/quality/metrics
```

Get real-time quality metrics from the autonomous SDLC system.

**Response:**
```json
{
  "quality_score": 0.95,
  "gates": {
    "code_quality": {"score": 9.2, "status": "passed"},
    "security": {"issues": 0, "status": "passed"},
    "test_coverage": {"percentage": 87.5, "status": "passed"}
  }
}
```

### Performance Metrics

```http
GET /api/performance/metrics
```

Get performance optimization data.

**Response:**
```json
{
  "response_time_p95": 180,
  "throughput_qps": 95,
  "error_rate": 0.002,
  "recommendations": [
    "Consider query caching for repeated patterns"
  ]
}
```

### Deployment Status

```http
GET /api/deployment/status
```

Get current deployment status.

**Response:**
```json
{
  "environment": "production",
  "version": "v1.2.3",
  "health": "healthy",
  "deployment_time": "2024-01-01T00:00:00Z",
  "blue_green_status": {
    "active_slot": "blue",
    "traffic_percentage": 100
  }
}
```

## Webhooks

Configure webhooks for deployment events:

```json
{
  "url": "https://your-app.com/webhooks/deployment",
  "events": ["deployment.started", "deployment.completed", "deployment.failed"],
  "secret": "webhook-secret"
}
```

## Monitoring and Observability

### OpenTelemetry Integration

The API supports OpenTelemetry for distributed tracing:

```http
X-Trace-Id: 1234567890abcdef
X-Span-Id: abcdef1234567890
```

### Structured Logging

All API requests are logged with structured data:

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "level": "INFO",
  "message": "Query processed successfully",
  "request_id": "req_123",
  "user_id": "user_456",
  "processing_time_ms": 150,
  "query_complexity": "medium"
}
```

## Best Practices

### Query Optimization

- Provide complete schema information
- Use specific, clear natural language
- Test queries in development first
- Monitor performance metrics

### Error Handling

- Implement retry logic with exponential backoff
- Handle rate limiting gracefully
- Log errors for debugging
- Provide user-friendly error messages

### Security

- Rotate API keys regularly
- Use HTTPS in production
- Validate all inputs
- Monitor for suspicious activity

## Support

- **Documentation**: https://docs.sql-synthesizer.com
- **Status Page**: https://status.sql-synthesizer.com
- **Support**: support@sql-synthesizer.com