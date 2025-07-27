# ADR-0002: Multi-Backend Caching Strategy

## Status
Accepted

## Context
The SQL Synthesizer needs to cache multiple types of data to improve performance:
- Database schema metadata (tables, columns, relationships)
- Query results for repeated questions
- Connection pool statistics
- User session data

Initial implementation used only in-memory TTLCache, which has limitations:
- Data lost on application restart
- No sharing between multiple instances
- Limited scalability for high-traffic scenarios
- No persistence for expensive-to-compute data

## Decision
Implement a multi-backend caching strategy with three levels:

### Level 1: In-Memory Cache (TTLCache)
- **Purpose**: Ultra-fast access for frequently used data
- **Data**: Schema metadata, recent query results, connection stats
- **TTL**: Configurable per cache type
- **Benefits**: Zero network latency, built-in cleanup

### Level 2: Redis Cache (Optional)
- **Purpose**: Shared cache across multiple instances
- **Data**: Query results, session data, rate limiting counters
- **TTL**: Configurable with Redis expiration
- **Benefits**: Persistence, clustering, atomic operations

### Level 3: Memcached Cache (Optional)
- **Purpose**: High-performance distributed caching
- **Data**: Large query results, computed aggregations
- **TTL**: Memcached native expiration
- **Benefits**: Memory efficiency, horizontal scaling

### Cache Selection Strategy
```python
def get_cache_backend():
    backend = os.getenv('QUERY_AGENT_CACHE_BACKEND', 'memory')
    if backend == 'redis' and redis_available():
        return RedisCache()
    elif backend == 'memcached' and memcached_available():
        return MemcachedCache()
    else:
        return MemoryCache()  # Always available fallback
```

## Consequences

### Positive
- **Performance**: Faster response times through multi-level caching
- **Scalability**: Redis/Memcached support horizontal scaling
- **Flexibility**: Choose appropriate backend for deployment scenario
- **Resilience**: Graceful fallback to memory cache if external systems fail
- **Efficiency**: Optimal cache placement reduces database load

### Negative
- **Complexity**: Multiple cache backends to configure and monitor
- **Dependencies**: Optional dependencies on Redis/Memcached
- **Consistency**: Potential cache coherence issues across levels
- **Debugging**: More complex troubleshooting with multiple cache layers

### Configuration Requirements
```bash
# Memory Cache (default)
export QUERY_AGENT_CACHE_BACKEND=memory
export QUERY_AGENT_CACHE_TTL=3600

# Redis Cache
export QUERY_AGENT_CACHE_BACKEND=redis
export QUERY_AGENT_REDIS_HOST=localhost
export QUERY_AGENT_REDIS_PORT=6379
export QUERY_AGENT_REDIS_DB=0

# Memcached Cache
export QUERY_AGENT_CACHE_BACKEND=memcached
export QUERY_AGENT_MEMCACHED_SERVERS=localhost:11211
```

### Monitoring
- Cache hit rates per backend
- Cache size and memory usage
- Eviction rates and TTL effectiveness
- Backend availability and latency

## Implementation Notes
- Cache backends implement common interface for consistency
- Automatic fallback ensures system remains functional
- Cache warming strategies for critical data
- Comprehensive testing for all backend types