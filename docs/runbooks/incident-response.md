# Incident Response Runbooks

Standard operating procedures for responding to incidents in the SQL Query Synthesizer.

## General Incident Response Process

### 1. Detection & Triage (0-5 minutes)
- Acknowledge alert or report
- Assess severity level
- Open incident channel
- Assign incident commander

### 2. Initial Response (5-15 minutes)
- Execute immediate containment
- Gather initial information
- Post status update
- Escalate if needed

### 3. Investigation & Resolution (15+ minutes)
- Follow specific runbook procedures
- Implement fixes or workarounds
- Monitor for improvement
- Document actions taken

### 4. Recovery & Monitoring (Post-fix)
- Verify full service restoration
- Monitor for recurrence
- Close incident
- Schedule post-mortem

---

## High Error Rate Response

### Symptoms
- Alert: `HighErrorRate` firing
- Error rate > 10% for 5+ minutes
- User reports of failures

### Immediate Actions (0-5 minutes)
```bash
# 1. Check current error rate
curl -s http://localhost:5000/metrics | grep api_requests_total

# 2. View recent application logs
docker logs --tail=100 sql-synthesizer-app | grep ERROR

# 3. Check service health
curl -f http://localhost:5000/health
```

### Investigation Steps (5-15 minutes)
```bash
# 4. Check database connectivity
docker exec sql-synthesizer-db pg_isready -U postgres

# 5. Verify cache service
docker exec sql-synthesizer-cache redis-cli ping

# 6. Check resource utilization
docker stats sql-synthesizer-app

# 7. Review detailed logs
docker logs --since=15m sql-synthesizer-app | grep -E "(ERROR|CRITICAL)"
```

### Common Causes & Solutions

#### Database Connection Issues
```bash
# Check connection pool status
curl http://localhost:5000/health/detailed | jq '.checks.database'

# If connection pool exhausted:
# Option 1: Restart application (quick fix)
docker restart sql-synthesizer-app

# Option 2: Scale up replicas
kubectl scale deployment sql-synthesizer --replicas=3

# Option 3: Investigate long-running queries
docker exec sql-synthesizer-db psql -U postgres -c "
SELECT pid, state, query_start, query 
FROM pg_stat_activity 
WHERE state = 'active' AND query_start < now() - interval '1 minute';"
```

#### OpenAI API Failures
```bash
# Check OpenAI service status
curl -s "https://status.openai.com/api/v2/status.json" | jq '.status.indicator'

# Verify API key and rate limits
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models | jq '.error'

# Enable circuit breaker if needed
export QUERY_AGENT_CIRCUIT_BREAKER_ENABLED=true
docker restart sql-synthesizer-app
```

#### Cache Service Failures
```bash
# Check Redis health
docker exec sql-synthesizer-cache redis-cli info replication

# Switch to memory cache temporarily
export QUERY_AGENT_CACHE_BACKEND=memory
docker restart sql-synthesizer-app

# Clear corrupted cache data
docker exec sql-synthesizer-cache redis-cli flushall
```

### Verification
```bash
# Confirm error rate has decreased
curl -s http://localhost:5000/metrics | grep api_requests_total

# Monitor for 10 minutes
watch "curl -s http://localhost:5000/health | jq '.overall_healthy'"
```

---

## Database Connection Failures

### Symptoms
- Alert: `DatabaseConnectionFailure` firing
- Database health check failing
- Connection timeout errors in logs

### Immediate Actions (0-5 minutes)
```bash
# 1. Check database container status
docker ps | grep postgres

# 2. Verify database process
docker exec sql-synthesizer-db pg_isready -U postgres -d sqlsynth_dev

# 3. Check connection count
docker exec sql-synthesizer-db psql -U postgres -c "
SELECT count(*) as active_connections 
FROM pg_stat_activity 
WHERE state = 'active';"
```

### Resolution Steps

#### Database Container Down
```bash
# Check container logs
docker logs sql-synthesizer-db

# Restart database service
docker restart sql-synthesizer-db

# Wait for startup and verify
sleep 30
docker exec sql-synthesizer-db pg_isready -U postgres
```

#### Connection Pool Exhaustion
```bash
# Kill long-running connections
docker exec sql-synthesizer-db psql -U postgres -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle in transaction' 
AND query_start < now() - interval '5 minutes';"

# Increase connection pool temporarily
export QUERY_AGENT_DB_POOL_SIZE=20
export QUERY_AGENT_DB_MAX_OVERFLOW=40
docker restart sql-synthesizer-app
```

#### Disk Space Issues
```bash
# Check disk usage
docker exec sql-synthesizer-db df -h

# Clean up old WAL files if needed
docker exec sql-synthesizer-db psql -U postgres -c "SELECT pg_switch_wal();"

# Vacuum large tables
docker exec sql-synthesizer-db psql -U postgres -d sqlsynth_dev -c "VACUUM ANALYZE;"
```

---

## Security Incident Response

### Symptoms
- Alert: `SecurityIncident` firing
- SQL injection attempts detected
- Unusual authentication patterns
- High-severity security events

### Immediate Actions (0-5 minutes)
```bash
# 1. Check recent security events
curl http://localhost:5000/api/security/events/recent

# 2. Review security audit logs
docker logs sql-synthesizer-app | grep "SECURITY"

# 3. Check for active threats
curl http://localhost:5000/health | jq '.security_status'
```

### Investigation & Containment

#### SQL Injection Attack
```bash
# Block suspicious IP immediately
iptables -A INPUT -s SUSPICIOUS_IP -j DROP

# Review injection attempts
grep "sql_injection_attempt" /var/log/security_audit.log

# Verify input validation is working
curl -X POST http://localhost:5000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "test; DROP TABLE users;"}'
```

#### Authentication Bypass Attempt
```bash
# Disable API key if compromised
export QUERY_AGENT_API_KEY_REQUIRED=false
docker restart sql-synthesizer-app

# Generate new API key
NEW_API_KEY=$(openssl rand -hex 32)
export QUERY_AGENT_API_KEY=$NEW_API_KEY

# Notify all API consumers of key rotation
```

#### Rate Limit Violation
```bash
# Identify violating IPs
grep "rate_limit_exceeded" /var/log/security_audit.log | \
  awk '{print $5}' | sort | uniq -c | sort -nr

# Temporarily lower rate limits
export QUERY_AGENT_RATE_LIMIT_PER_MINUTE=10
docker restart sql-synthesizer-app
```

### Post-Incident Security Hardening
```bash
# Update security configuration
export QUERY_AGENT_ENABLE_HSTS=true
export QUERY_AGENT_CSRF_ENABLED=true
export QUERY_AGENT_MAX_REQUEST_SIZE_MB=0.5

# Review and rotate all secrets
make security-audit
make rotate-secrets
```

---

## Performance Degradation

### Symptoms
- Alert: `HighResponseTime` firing
- Query latency > 100ms (95th percentile)
- User reports of slow responses

### Immediate Actions (0-5 minutes)
```bash
# 1. Check current response times
curl -w "@curl-format.txt" -s http://localhost:5000/health

# 2. Monitor resource usage
docker stats --no-stream

# 3. Check active queries
curl http://localhost:5000/metrics | grep query_duration
```

### Performance Investigation

#### Database Performance
```bash
# Check slow queries
docker exec sql-synthesizer-db psql -U postgres -c "
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"

# Check for lock contention
docker exec sql-synthesizer-db psql -U postgres -c "
SELECT blocked_locks.pid AS blocked_pid,
       blocking_locks.pid AS blocking_pid,
       blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_locks blocking_locks 
  ON blocking_locks.locktype = blocked_locks.locktype;"
```

#### Cache Performance
```bash
# Check cache hit ratio
curl http://localhost:5000/metrics | grep cache_hit_rate

# Review cache configuration
docker exec sql-synthesizer-cache redis-cli info memory

# Clear cache if needed
docker exec sql-synthesizer-cache redis-cli flushall
```

#### Application Performance
```bash
# Check for memory leaks
docker exec sql-synthesizer-app python -c "
import psutil, os
p = psutil.Process(os.getpid())
print(f'Memory: {p.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'Threads: {p.num_threads()}')"

# Review garbage collection
docker logs sql-synthesizer-app | grep -i "garbage"

# Scale horizontally if needed
kubectl scale deployment sql-synthesizer --replicas=5
```

### Recovery Actions
```bash
# Restart application if memory leak suspected
docker restart sql-synthesizer-app

# Increase resource limits temporarily
docker update --memory=2g --cpus=2 sql-synthesizer-app

# Monitor recovery
watch "curl -w '%{time_total}' -s http://localhost:5000/health"
```

---

## Communication During Incidents

### Status Page Updates
```bash
# Update status page (if using statuspage.io)
curl -X PATCH "https://api.statuspage.io/v1/pages/PAGE_ID/incidents/INCIDENT_ID" \
     -H "Authorization: OAuth TOKEN" \
     -d "incident[status]=investigating"
```

### Slack Notifications
```bash
# Post to incident channel
curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"🚨 Investigating high error rate in SQL Synthesizer"}' \
     $SLACK_WEBHOOK_URL
```

### Email Notifications
```bash
# Send email update (using sendmail)
echo "Subject: SQL Synthesizer Incident Update
High error rate detected and under investigation.
ETA for resolution: 30 minutes." | sendmail oncall@company.com
```

Remember: Always follow up incidents with a post-mortem to prevent recurrence!