# Troubleshooting Guide

## Overview

This guide provides solutions for common issues, debugging techniques, and diagnostic procedures for the SQL Query Synthesizer.

## Quick Diagnostics

### Health Check
```bash
# Basic health check
curl -f http://localhost:5000/health

# Detailed health status
curl http://localhost:5000/health | jq '.'

# Metrics overview
curl http://localhost:5000/metrics | grep -E "(up|error|duration)"
```

### Log Analysis
```bash
# View recent logs
tail -f logs/sql-synthesizer.log

# Search for errors
grep -i error logs/sql-synthesizer.log | tail -20

# Parse JSON logs
tail -f logs/sql-synthesizer.log | jq 'select(.level == "ERROR")'
```

## Common Issues and Solutions

### 1. Application Won't Start

#### Symptoms
- Container exits immediately
- Import errors on startup
- Port binding failures

#### Diagnostic Steps
```bash
# Check container logs
docker logs sql-synthesizer-container

# Verify dependencies
pip check

# Test import
python -c "import sql_synthesizer; print('OK')"

# Check port availability
netstat -tulpn | grep 5000
```

#### Solutions
- **Missing Dependencies**: `pip install -r requirements.txt`
- **Port Conflicts**: Change `QUERY_AGENT_WEBAPP_PORT`
- **Permission Issues**: Check file permissions and user context
- **Database Connection**: Verify `DATABASE_URL` is accessible

### 2. Database Connection Problems

#### Symptoms
- Connection timeouts
- "Connection refused" errors
- High connection pool utilization

#### Diagnostic Commands
```python
# Test database connectivity
from sql_synthesizer import QueryAgent
agent = QueryAgent()
health = agent.health_check()
print(health['database'])

# Check connection pool status
stats = agent.get_connection_stats()
print(f"Pool: {stats['checked_out']}/{stats['pool_size']}")
```

#### Solutions
```bash
# Increase connection pool size
export QUERY_AGENT_DB_POOL_SIZE=25
export QUERY_AGENT_DB_MAX_OVERFLOW=50

# Adjust timeouts
export QUERY_AGENT_DATABASE_TIMEOUT=60
export QUERY_AGENT_DB_CONNECT_RETRIES=5

# Enable connection pre-ping
export QUERY_AGENT_DB_POOL_PRE_PING=true
```

### 3. High Memory Usage

#### Symptoms
- OOM kills in containers
- Gradual memory increase
- Slow response times

#### Diagnostic Steps
```bash
# Monitor memory usage
docker stats sql-synthesizer-container

# Profile memory usage
python -m memory_profiler sql_synthesizer/webapp.py

# Check cache size
redis-cli info memory  # if using Redis cache
```

#### Solutions
```bash
# Limit cache size
export QUERY_AGENT_CACHE_MAX_SIZE=1000

# Reduce connection pool
export QUERY_AGENT_DB_POOL_SIZE=10
export QUERY_AGENT_DB_MAX_OVERFLOW=20

# Enable garbage collection tuning
export PYTHONHASHSEED=random
export PYTHONOPTIMIZE=1
```

### 4. OpenAI API Issues

#### Symptoms
- "API key not found" errors
- Rate limit exceeded
- High response times from LLM

#### Diagnostics
```python
# Test OpenAI connectivity
import openai
try:
    response = openai.Model.list()
    print("OpenAI API accessible")
except Exception as e:
    print(f"OpenAI API error: {e}")

# Check circuit breaker status
from sql_synthesizer.circuit_breaker import circuit_breaker
print(f"Circuit breaker state: {circuit_breaker.state}")
```

#### Solutions
```bash
# Verify API key
export OPENAI_API_KEY="your-valid-key"

# Adjust timeouts
export QUERY_AGENT_OPENAI_TIMEOUT=60

# Configure circuit breaker
export QUERY_AGENT_CIRCUIT_BREAKER_FAILURE_THRESHOLD=10
export QUERY_AGENT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=120

# Use different model
export QUERY_AGENT_OPENAI_MODEL=gpt-3.5-turbo
```

### 5. Cache Performance Issues

#### Symptoms
- Low cache hit rates
- Cache connection errors
- Slow query responses

#### Diagnostics
```python
# Check cache statistics
from sql_synthesizer.cache import cache
stats = cache.get_statistics()
print(f"Hit rate: {stats.get('hit_rate', 0):.2%}")
print(f"Total operations: {stats.get('total_operations', 0)}")

# Test cache backend
cache.set('test_key', 'test_value', ttl=10)
value = cache.get('test_key')
print(f"Cache test: {'PASS' if value == 'test_value' else 'FAIL'}")
```

#### Solutions
```bash
# Switch to Redis cache
export QUERY_AGENT_CACHE_BACKEND=redis
export QUERY_AGENT_REDIS_HOST=localhost
export QUERY_AGENT_REDIS_PORT=6379

# Adjust TTL settings
export QUERY_AGENT_CACHE_TTL=7200
export QUERY_AGENT_SCHEMA_CACHE_TTL=3600

# Clear cache if corrupted
curl -X POST http://localhost:5000/api/cache/clear
```

### 6. Security and Authentication Issues

#### Symptoms
- CSRF token validation errors
- Rate limiting false positives
- Authentication failures

#### Diagnostics
```bash
# Check security event logs
grep -i "security" logs/sql-synthesizer.log | tail -10

# Test API key authentication
curl -H "X-API-Key: your-api-key" http://localhost:5000/api/query

# Check rate limiting
curl -v http://localhost:5000/api/query 2>&1 | grep -i "rate"
```

#### Solutions
```bash
# Disable CSRF for API testing
export QUERY_AGENT_CSRF_ENABLED=false

# Adjust rate limits
export QUERY_AGENT_RATE_LIMIT_PER_MINUTE=200

# Regenerate secret key
export QUERY_AGENT_SECRET_KEY="$(openssl rand -hex 32)"

# Configure API key
export QUERY_AGENT_API_KEY_REQUIRED=true
export QUERY_AGENT_API_KEY="your-secure-api-key"
```

## Performance Troubleshooting

### Slow Query Response Times

#### Investigation Steps
1. **Check Query Complexity**:
   ```sql
   EXPLAIN ANALYZE your-generated-query;
   ```

2. **Monitor Database Performance**:
   ```bash
   # PostgreSQL
   SELECT * FROM pg_stat_activity WHERE state = 'active';
   
   # MySQL
   SHOW PROCESSLIST;
   ```

3. **Analyze Cache Performance**:
   ```python
   # Cache hit rate analysis
   from sql_synthesizer.metrics import cache_metrics
   print(cache_metrics.get_hit_rate_last_hour())
   ```

#### Optimization Solutions
```bash
# Enable query result caching
export QUERY_AGENT_CACHE_TTL=1800

# Optimize database settings
export QUERY_AGENT_DB_POOL_SIZE=15
export QUERY_AGENT_DB_POOL_RECYCLE=1800

# Use database indexes
# Add indexes for frequently queried columns
```

### High CPU Usage

#### Monitoring Commands
```bash
# Process-level monitoring
top -p $(pgrep -f sql_synthesizer)

# Container monitoring
docker exec sql-synthesizer-container top

# Profiling
python -m cProfile -o profile.stats sql_synthesizer/webapp.py
```

#### Solutions
- Optimize database queries
- Increase cache usage
- Scale horizontally
- Review algorithm complexity

## Debugging Techniques

### Enable Debug Mode
```bash
# Application debug mode
export QUERY_AGENT_LOG_LEVEL=DEBUG
export FLASK_DEBUG=1

# Database query logging
export QUERY_AGENT_LOG_SQL_QUERIES=true

# Cache operation logging
export QUERY_AGENT_LOG_CACHE_OPERATIONS=true
```

### Interactive Debugging
```python
# Debug specific functionality
from sql_synthesizer import QueryAgent
import logging

logging.basicConfig(level=logging.DEBUG)
agent = QueryAgent(database_url="your-db-url")

# Test query generation
result = agent.query("test query", debug=True)
print(f"Generated SQL: {result.sql}")
print(f"Execution time: {result.execution_time}")
```

### Log Analysis Techniques
```bash
# Error rate analysis
grep -c "ERROR" logs/sql-synthesizer.log

# Response time analysis
grep "response_time" logs/sql-synthesizer.log | \
  awk '{print $5}' | sort -n | \
  awk 'END {print "P95:", $(int(NR*0.95))}'

# Most common errors
grep "ERROR" logs/sql-synthesizer.log | \
  awk '{print $6}' | sort | uniq -c | sort -nr
```

## Container-Specific Issues

### Docker Troubleshooting
```bash
# Check container resource limits
docker inspect sql-synthesizer-container | jq '.[0].HostConfig'

# Access container shell
docker exec -it sql-synthesizer-container /bin/bash

# View container logs with timestamps
docker logs -t sql-synthesizer-container

# Check container health
docker inspect --format='{{.State.Health.Status}}' sql-synthesizer-container
```

### Kubernetes Troubleshooting
```bash
# Pod status and events
kubectl describe pod sql-synthesizer-pod

# Container logs
kubectl logs sql-synthesizer-deployment-xxx -c sql-synthesizer

# Resource usage
kubectl top pod sql-synthesizer-pod

# Network connectivity
kubectl exec -it sql-synthesizer-pod -- nslookup database-service
```

## Emergency Procedures

### System Recovery
1. **Immediate Steps**:
   ```bash
   # Check system status
   systemctl status sql-synthesizer
   
   # Restart service
   systemctl restart sql-synthesizer
   
   # Check logs
   journalctl -u sql-synthesizer -f
   ```

2. **Rollback Procedure**:
   ```bash
   # Docker rollback
   docker tag sql-synthesizer:current sql-synthesizer:backup
   docker pull sql-synthesizer:previous-stable
   docker-compose up -d
   
   # Kubernetes rollback
   kubectl rollout undo deployment/sql-synthesizer
   ```

### Data Recovery
```bash
# Database backup restoration
pg_restore -d database_name backup_file.sql

# Cache restoration (if needed)
redis-cli --rdb dump.rdb

# Configuration backup
cp /backup/config/databases.yaml config/databases.yaml
```

## Getting Help

### Diagnostic Information Collection
```bash
#!/bin/bash
# collect_diagnostics.sh
echo "=== System Information ===" > diagnostics.txt
uname -a >> diagnostics.txt
date >> diagnostics.txt

echo "=== Application Status ===" >> diagnostics.txt
curl -s http://localhost:5000/health >> diagnostics.txt

echo "=== Recent Logs ===" >> diagnostics.txt
tail -100 logs/sql-synthesizer.log >> diagnostics.txt

echo "=== Configuration ===" >> diagnostics.txt
env | grep QUERY_AGENT >> diagnostics.txt

echo "=== Resource Usage ===" >> diagnostics.txt
free -h >> diagnostics.txt
df -h >> diagnostics.txt
```

### Support Channels
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check latest documentation updates
- **Community Forums**: Discussion and community support
- **Professional Support**: Enterprise support options

### Information to Include in Bug Reports
1. Version information
2. Environment configuration
3. Error logs and stack traces
4. Steps to reproduce
5. Expected vs actual behavior
6. System specifications
7. Network configuration
8. Database schema (if relevant)