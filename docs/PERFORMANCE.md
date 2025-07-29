# Performance Guide

## Overview

This document outlines performance optimization strategies, monitoring approaches, and benchmarking practices for the SQL Query Synthesizer.

## Performance Architecture

### Async-First Design
- **Async Query Processing**: All database operations use async/await patterns
- **Connection Pooling**: SQLAlchemy connection pooling with configurable limits
- **Concurrent Request Handling**: FastAPI/Flask async support for high throughput

### Caching Strategy
- **Multi-Backend Support**: Memory, Redis, and Memcached backends
- **Smart Cache Invalidation**: TTL-based with manual invalidation capabilities
- **Cache Warming**: Proactive schema and frequently-used query caching

### Database Optimization
- **Connection Pool Configuration**:
  ```python
  # Recommended production settings
  QUERY_AGENT_DB_POOL_SIZE=20
  QUERY_AGENT_DB_MAX_OVERFLOW=40
  QUERY_AGENT_DB_POOL_RECYCLE=3600
  QUERY_AGENT_DB_POOL_PRE_PING=true
  ```

## Performance Monitoring

### Metrics Collection
Built-in Prometheus metrics for monitoring:
- Request duration histograms
- Database query performance
- Cache hit/miss ratios
- Error rates and circuit breaker status

### Key Performance Indicators (KPIs)
- **Response Time**: P95 < 500ms for cached queries
- **Throughput**: 1000+ requests/minute sustained
- **Cache Hit Rate**: > 80% for schema queries
- **Database Connection Utilization**: < 70% under normal load

### Monitoring Endpoints
- `/metrics` - Prometheus metrics
- `/health` - Health check with performance diagnostics
- Connection pool statistics via API

## Performance Testing

### Load Testing
```bash
# Install dependencies
pip install locust

# Run load tests
locust -f tests/performance/locustfile.py --host=http://localhost:5000
```

### Benchmark Scripts
```bash
# Database query performance
python benchmarks/query_performance.py

# Cache backend comparison
python benchmarks/cache_comparison.py

# Connection pool stress test
python benchmarks/connection_pool_test.py
```

### Performance Regression Testing
- Automated performance tests in CI/CD pipeline
- Performance budgets with failure thresholds
- Historical performance tracking

## Optimization Strategies

### Query Optimization
1. **Schema Caching**: Cache database schema for configurable TTL
2. **Query Result Caching**: Cache expensive query results
3. **Prepared Statements**: Use parameterized queries for better performance
4. **Connection Reuse**: Leverage connection pooling effectively

### Application Performance
1. **Async Processing**: Use async/await for I/O operations
2. **Batch Operations**: Batch multiple queries when possible
3. **Circuit Breakers**: Prevent cascading failures with OpenAI API
4. **Resource Limits**: Configure appropriate memory and connection limits

### Infrastructure Optimization
1. **Database Tuning**: Optimize database configuration for workload
2. **Cache Deployment**: Deploy Redis/Memcached for distributed caching
3. **Load Balancing**: Distribute traffic across multiple instances
4. **Resource Monitoring**: Monitor CPU, memory, and network usage

## Performance Configuration

### Environment Variables
```bash
# Database Performance
export QUERY_AGENT_DB_POOL_SIZE=20
export QUERY_AGENT_DB_MAX_OVERFLOW=40
export QUERY_AGENT_DB_POOL_RECYCLE=3600
export QUERY_AGENT_DB_CONNECT_RETRIES=3
export QUERY_AGENT_DB_RETRY_DELAY=1.0
export QUERY_AGENT_DATABASE_TIMEOUT=30

# Cache Performance
export QUERY_AGENT_CACHE_BACKEND=redis
export QUERY_AGENT_CACHE_TTL=3600
export QUERY_AGENT_CACHE_MAX_SIZE=10000
export QUERY_AGENT_CACHE_CLEANUP_INTERVAL=300

# Application Performance
export QUERY_AGENT_OPENAI_TIMEOUT=30
export QUERY_AGENT_MAX_REQUEST_SIZE_MB=10
export QUERY_AGENT_DEFAULT_PAGE_SIZE=50
export QUERY_AGENT_MAX_PAGE_SIZE=1000

# Circuit Breaker
export QUERY_AGENT_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
export QUERY_AGENT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
```

### Production Recommendations
1. **Use Redis for Caching**: Better performance than memory cache
2. **Enable Connection Pre-ping**: Detect stale connections
3. **Set Appropriate Timeouts**: Prevent hanging requests
4. **Monitor Resource Usage**: Set up alerts for high utilization
5. **Use Database Read Replicas**: Distribute read load

## Troubleshooting Performance Issues

### Common Issues
1. **High Response Times**:
   - Check database query performance
   - Verify cache hit rates
   - Monitor connection pool utilization

2. **Memory Issues**:
   - Review cache size configuration
   - Check for memory leaks in long-running processes
   - Monitor connection pool size

3. **Database Connection Exhaustion**:
   - Increase pool size if needed
   - Check for connection leaks
   - Monitor connection lifecycle

### Performance Debugging
```python
# Enable detailed logging
export QUERY_AGENT_LOG_LEVEL=DEBUG

# Monitor connection pool
from sql_synthesizer import QueryAgent
agent = QueryAgent()
stats = agent.get_connection_stats()
print(f"Pool utilization: {stats['checked_out']}/{stats['pool_size']}")

# Cache performance analysis
cache_stats = agent.cache.get_statistics()
print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
```

## Performance Benchmarks

### Baseline Performance (Single Instance)
- **Simple Queries**: ~100ms response time
- **Complex Queries**: ~300ms response time  
- **Cached Queries**: ~10ms response time
- **Throughput**: 500 requests/minute sustained

### Scaling Characteristics
- **Linear Scaling**: Up to 4 instances
- **Database Bottleneck**: Above 8 instances without read replicas
- **Cache Effectiveness**: 80%+ hit rate under normal load

## Future Optimizations
- **GraphQL Support**: Reduce over-fetching
- **Query Plan Caching**: Cache execution plans
- **Predictive Caching**: ML-based cache warming
- **Auto-scaling**: Dynamic resource allocation
- **Database Sharding**: Horizontal database scaling