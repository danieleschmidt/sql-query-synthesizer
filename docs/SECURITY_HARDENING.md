# Security Hardening Guide

This document provides comprehensive security hardening guidelines for the SQL Synthesizer project.

## Overview

The SQL Synthesizer processes natural language queries and executes SQL against databases, making security critical at every layer.

## Application Security

### 1. SQL Injection Prevention

**Current Protections:**
- Parameterized queries using SQLAlchemy ORM
- Input validation and sanitization
- AST-based SQL parsing validation
- Query pattern allow/deny lists

**Additional Hardening:**
```python
# Enable enhanced SQL injection prevention
export QUERY_AGENT_USE_ENHANCED_SQL_VALIDATION=true

# Restrict allowed SQL operations
export QUERY_AGENT_ALLOWED_SQL_OPERATIONS="SELECT,WITH"
export QUERY_AGENT_DENY_DDL_OPERATIONS=true
```

### 2. Authentication & Authorization

**API Key Configuration:**
```bash
# Enable API key authentication
export QUERY_AGENT_API_KEY_REQUIRED=true
export QUERY_AGENT_API_KEY="your-secure-api-key-here"

# Configure API key headers
export QUERY_AGENT_API_KEY_HEADER="X-API-Key"
```

**Session Security:**
```bash
# Flask session configuration
export QUERY_AGENT_SECRET_KEY="your-256-bit-secret-key-here"
export QUERY_AGENT_SESSION_COOKIE_SECURE=true
export QUERY_AGENT_SESSION_COOKIE_HTTPONLY=true
export QUERY_AGENT_SESSION_COOKIE_SAMESITE="Strict"
```

### 3. Rate Limiting & DDoS Protection

```bash
# Configure rate limiting
export QUERY_AGENT_RATE_LIMIT_PER_MINUTE=30
export QUERY_AGENT_RATE_LIMIT_PER_HOUR=500
export QUERY_AGENT_RATE_LIMIT_STORAGE="redis://localhost:6379/1"
```

### 4. Input Validation

```bash
# Request size limits
export QUERY_AGENT_MAX_REQUEST_SIZE_MB=1
export QUERY_AGENT_MAX_QUESTION_LENGTH=1000
export QUERY_AGENT_MAX_SQL_LENGTH=10000

# Content filtering
export QUERY_AGENT_ENABLE_CONTENT_FILTERING=true
export QUERY_AGENT_BLOCKED_PATTERNS="DROP,DELETE,UPDATE,INSERT,ALTER,CREATE"
```

## Infrastructure Security

### 1. Container Security

**Dockerfile Security Best Practices:**
- Non-root user execution ✅
- Multi-stage builds ✅
- Minimal base images ✅
- No secrets in layers ✅

**Runtime Security:**
```bash
# Run with security options
docker run --security-opt no-new-privileges:true \
           --cap-drop ALL \
           --cap-add NET_BIND_SERVICE \
           --read-only \
           --tmpfs /tmp \
           sql-synthesizer:latest
```

### 2. Database Security

**Connection Security:**
```bash
# Use SSL connections
export DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require"

# Connection pool limits
export QUERY_AGENT_DB_POOL_SIZE=5
export QUERY_AGENT_DB_MAX_OVERFLOW=0
export QUERY_AGENT_DB_POOL_TIMEOUT=30
```

**Database Permissions:**
```sql
-- Create restricted user for SQL Synthesizer
CREATE USER sql_synthesizer_app WITH PASSWORD 'secure_password';

-- Grant minimal required permissions
GRANT CONNECT ON DATABASE your_database TO sql_synthesizer_app;
GRANT USAGE ON SCHEMA public TO sql_synthesizer_app;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO sql_synthesizer_app;

-- Revoke dangerous permissions
REVOKE CREATE ON DATABASE your_database FROM sql_synthesizer_app;
REVOKE ALL ON SCHEMA information_schema FROM sql_synthesizer_app;
```

### 3. Network Security

**TLS Configuration:**
```bash
# Enable HTTPS and security headers
export QUERY_AGENT_ENABLE_HSTS=true
export QUERY_AGENT_HSTS_MAX_AGE=31536000
export QUERY_AGENT_ENABLE_CSP=true
export QUERY_AGENT_CSP_POLICY="default-src 'self'; script-src 'self' 'unsafe-inline'"
```

**Firewall Rules:**
```bash
# Allow only necessary ports
ufw allow 443/tcp  # HTTPS
ufw allow 80/tcp   # HTTP (redirect to HTTPS)
ufw deny 22/tcp    # SSH (use VPN or bastion)
ufw enable
```

## Monitoring & Alerting

### 1. Security Event Monitoring

**Configure Audit Logging:**
```bash
export QUERY_AGENT_SECURITY_AUDIT_ENABLED=true
export QUERY_AGENT_AUDIT_LOG_LEVEL="INFO"
export QUERY_AGENT_AUDIT_LOG_FILE="/var/log/sql-synthesizer/security.log"
```

**Key Events to Monitor:**
- Failed authentication attempts
- SQL injection attempts
- Rate limit violations
- Unusual query patterns
- Database connection failures

### 2. Prometheus Metrics

**Security Metrics:**
```python
# Monitor these metrics for security events
security_events_total{event_type="sql_injection_attempt"}
authentication_failures_total
rate_limit_exceeded_total
database_connection_failures_total
```

### 3. Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: sql_synthesizer_security
    rules:
      - alert: SQLInjectionAttack
        expr: increase(security_events_total{event_type="sql_injection_attempt"}[5m]) > 5
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "SQL injection attack detected"
          
      - alert: AuthenticationFailures
        expr: increase(authentication_failures_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High number of authentication failures"
```

## Development Security

### 1. Pre-commit Security Hooks

The repository includes comprehensive pre-commit hooks:
- `bandit` - Python security linter
- `detect-secrets` - Secret detection
- `safety` - Dependency vulnerability scanning

**Installation:**
```bash
pip install pre-commit
pre-commit install
```

### 2. Dependency Management

**Security Scanning:**
```bash
# Regular dependency auditing
pip-audit
safety check
bandit -r sql_synthesizer/

# Update dependencies regularly
pip-compile --upgrade
```

**Lock File Management:**
- Keep `requirements.txt` locked to specific versions
- Regular security updates via Dependabot
- Test updates in staging before production

### 3. Secret Management

**Environment Variables:**
- Never commit secrets to version control
- Use environment-specific `.env` files
- Rotate secrets regularly
- Use secret management tools (AWS Secrets Manager, Azure Key Vault)

**Example `.env.production`:**
```bash
# Database
DATABASE_URL="postgresql://user:pass@prod-db:5432/dbname"

# OpenAI API
OPENAI_API_KEY="sk-your-production-key-here"

# Application Security
QUERY_AGENT_SECRET_KEY="your-256-bit-secret-key"
QUERY_AGENT_API_KEY="your-api-key"

# Redis Cache
QUERY_AGENT_REDIS_PASSWORD="your-redis-password"
```

## Compliance & Auditing

### 1. GDPR Compliance

**Data Processing:**
- Log data retention policies
- Right to deletion implementation
- Data anonymization for analytics
- Privacy by design principles

### 2. SOC 2 Type II

**Access Controls:**
- Multi-factor authentication
- Role-based access control
- Regular access reviews
- Privileged access monitoring

### 3. Audit Trail

**Required Logging:**
- All database queries executed
- User authentication events
- Administrative actions
- Configuration changes
- Security events

## Incident Response

### 1. Security Incident Playbook

**Detection:**
1. Monitor security metrics and alerts
2. Review audit logs regularly
3. Implement anomaly detection

**Response:**
1. Isolate affected systems
2. Preserve evidence
3. Assess impact and scope
4. Notify stakeholders
5. Implement containment measures

**Recovery:**
1. Remove malicious access
2. Patch vulnerabilities
3. Restore from clean backups
4. Monitor for reoccurrence

### 2. Emergency Contacts

- **Security Team:** security@sqlsynthesizer.com
- **On-call Engineer:** +1-555-SEC-TEAM
- **Management:** cto@sqlsynthesizer.com

## Security Checklist

### Pre-deployment Security Review

- [ ] All secrets removed from code
- [ ] Database connections use SSL
- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] Security headers enabled
- [ ] Container runs as non-root
- [ ] Firewall rules configured
- [ ] Monitoring and alerting active
- [ ] Backup and recovery tested
- [ ] Incident response plan documented

### Monthly Security Review

- [ ] Dependency vulnerabilities scanned
- [ ] Access permissions reviewed
- [ ] Log analysis completed
- [ ] Security metrics reviewed
- [ ] Penetration testing results
- [ ] Security training completed
- [ ] Compliance status verified

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.org/dev/security/)
- [Flask Security Guidelines](https://flask.palletsprojects.com/en/2.3.x/security/)
- [Container Security Guide](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)