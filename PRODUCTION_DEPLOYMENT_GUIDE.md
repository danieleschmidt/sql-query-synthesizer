# SQL Query Synthesizer - Production Deployment Guide

## 🚀 Production-Ready Quantum-Enhanced SQL Synthesizer

This guide provides comprehensive instructions for deploying the quantum-enhanced SQL Query Synthesizer in production environments with enterprise-grade security, scalability, and monitoring.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or Docker environment
- **Python**: 3.8+ with virtual environment support
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: 10GB free space for logs and caching
- **Network**: HTTPS support with valid SSL certificates

### Dependencies
- PostgreSQL/MySQL/SQLite database access
- Redis or Memcached (for distributed caching)
- OpenAI API key (for LLM-based SQL generation)
- Prometheus + Grafana (for monitoring)

## 🔧 Installation & Configuration

### 1. Environment Setup

```bash
# Create production user
sudo useradd -m -s /bin/bash sqlsynth
sudo usermod -aG sudo sqlsynth

# Clone repository
git clone https://github.com/terragon-labs/sql-query-synthesizer.git
cd sql-query-synthesizer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Production Configuration

Create `/opt/sqlsynth/config/production.env`:

```bash
# Core Configuration
QUERY_AGENT_ENV=production
QUERY_AGENT_CONFIG=/opt/sqlsynth/config/databases.yaml
DATABASE_URL=postgresql://user:pass@localhost:5432/prod_db

# Security Configuration
QUERY_AGENT_SECRET_KEY=your-super-secure-256-bit-key-here
QUERY_AGENT_API_KEY=your-production-api-key
QUERY_AGENT_API_KEY_REQUIRED=true
QUERY_AGENT_CSRF_ENABLED=true
QUERY_AGENT_ENABLE_HSTS=true
QUERY_AGENT_RATE_LIMIT_PER_MINUTE=100

# Performance Configuration
QUERY_AGENT_DB_POOL_SIZE=20
QUERY_AGENT_DB_MAX_OVERFLOW=50
QUERY_AGENT_DB_POOL_RECYCLE=3600
QUERY_AGENT_CACHE_BACKEND=redis
QUERY_AGENT_REDIS_HOST=localhost
QUERY_AGENT_REDIS_PORT=6379
QUERY_AGENT_CACHE_TTL=3600

# Monitoring & Logging
QUERY_AGENT_LOG_LEVEL=INFO
QUERY_AGENT_ENABLE_METRICS=true
QUERY_AGENT_SECURITY_AUDIT=true

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
QUERY_AGENT_OPENAI_MODEL=gpt-4
QUERY_AGENT_OPENAI_TIMEOUT=30

# Quantum Features
QUANTUM_OPTIMIZATION_ENABLED=true
QUANTUM_SUPERPOSITION_COHERENCE=0.8
QUANTUM_ENTANGLEMENT_STRENGTH=0.6
```

### 3. Database Configuration

Create `/opt/sqlsynth/config/databases.yaml`:

```yaml
databases:
  production:
    url: "${DATABASE_URL}"
    schema_cache_ttl: 3600
    pool_size: 20
    max_overflow: 50
    pool_recycle: 3600
    
  analytics:
    url: "postgresql://analytics:pass@analytics-db:5432/analytics"
    schema_cache_ttl: 7200
    read_only: true
    
security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
  audit_logging:
    enabled: true
    retention_days: 90
    
performance:
  query_timeout: 30
  max_result_size: 10000
  enable_caching: true
```

## 🐳 Docker Deployment

### 1. Production Dockerfile

```dockerfile
FROM python:3.11-slim

# Security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    postgresql-client redis-tools && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash sqlsynth
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Security configurations
RUN chown -R sqlsynth:sqlsynth /app
USER sqlsynth

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

CMD ["gunicorn", "--config", "deployment/gunicorn.conf.py", "sql_synthesizer.webapp:create_app()"]
```

### 2. Docker Compose Production Stack

```yaml
version: '3.8'

services:
  sql-synthesizer:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/sqlsynth
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    restart: unless-stopped
    
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: sqlsynth
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

## 🔒 Security Hardening

### 1. SSL/TLS Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourcompany.com;
    
    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
        add_header Content-Security-Policy "default-src 'self'";
    }
}
```

### 2. Firewall Configuration

```bash
# UFW firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow from 10.0.0.0/8 to any port 9090  # Prometheus (internal)
sudo ufw enable
```

### 3. Security Audit Configuration

```python
# Enable comprehensive security logging
SECURITY_AUDIT_CONFIG = {
    "enabled": True,
    "log_level": "INFO",
    "events": [
        "sql_injection_attempt",
        "authentication_failure", 
        "rate_limit_exceeded",
        "unauthorized_access",
        "query_execution",
        "schema_access"
    ],
    "retention_days": 90,
    "alert_thresholds": {
        "sql_injection_attempts_per_hour": 5,
        "failed_auth_attempts_per_hour": 10
    }
}
```

## 📊 Monitoring & Observability

### 1. Prometheus Metrics

Key metrics automatically collected:
- `sql_synthesizer_query_total`: Total queries processed
- `sql_synthesizer_query_duration_seconds`: Query processing time
- `sql_synthesizer_cache_hits_total`: Cache hit statistics
- `sql_synthesizer_database_connections`: Active database connections
- `sql_synthesizer_security_events_total`: Security event counts

### 2. Grafana Dashboard

Pre-configured dashboards include:
- **Application Performance**: Response times, throughput, error rates
- **Security Monitoring**: Failed authentications, injection attempts
- **Resource Utilization**: Memory, CPU, database connections
- **Quantum Algorithm Performance**: Algorithm-specific metrics

### 3. Health Checks

```bash
# Application health
curl -f https://api.yourcompany.com/health

# Detailed health with auth
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.yourcompany.com/health/detailed
```

## 🔄 High Availability Setup

### 1. Load Balancer Configuration

```yaml
# HAProxy configuration
backend sql_synthesizer
    balance roundrobin
    option httpchk GET /health
    server app1 10.0.1.10:5000 check
    server app2 10.0.1.11:5000 check
    server app3 10.0.1.12:5000 check
```

### 2. Database Clustering

```yaml
# PostgreSQL cluster configuration
postgresql:
  primary:
    host: pg-primary.internal
    port: 5432
  replicas:
    - host: pg-replica-1.internal
      port: 5432
    - host: pg-replica-2.internal  
      port: 5432
  failover:
    enabled: true
    timeout: 30
```

## 🚀 Deployment Automation

### 1. CI/CD Pipeline

```yaml
# GitHub Actions deployment
name: Production Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          python -m pytest tests/ -v
          python -m bandit -r sql_synthesizer/
      - name: Deploy to production
        run: |
          docker build -t sql-synthesizer:${{ github.sha }} .
          docker push registry.yourcompany.com/sql-synthesizer:${{ github.sha }}
          kubectl set image deployment/sql-synthesizer \
            app=registry.yourcompany.com/sql-synthesizer:${{ github.sha }}
```

### 2. Zero-Downtime Deployment

```bash
#!/bin/bash
# Blue-green deployment script
NEW_VERSION=${1:-latest}

# Deploy new version to green environment
docker-compose -f docker-compose.green.yml up -d
sleep 30

# Health check green environment
if curl -f http://green.internal:5000/health; then
    # Switch traffic to green
    docker-compose -f docker-compose.blue.yml down
    mv docker-compose.green.yml docker-compose.yml
    echo "Deployment successful"
else
    echo "Health check failed, rolling back"
    docker-compose -f docker-compose.green.yml down
    exit 1
fi
```

## 📈 Performance Optimization

### 1. Redis Clustering

```yaml
redis_cluster:
  nodes:
    - redis-1.internal:7000
    - redis-2.internal:7000
    - redis-3.internal:7000
  settings:
    cluster-enabled: yes
    cluster-config-file: nodes.conf
    cluster-node-timeout: 5000
```

### 2. Database Optimization

```sql
-- Production database indexes
CREATE INDEX CONCURRENTLY idx_queries_timestamp ON query_logs(timestamp);
CREATE INDEX CONCURRENTLY idx_schemas_cache_key ON schema_cache(cache_key);
CREATE INDEX CONCURRENTLY idx_security_events_severity ON security_events(severity, timestamp);

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
```

## 🔍 Troubleshooting

### Common Issues

1. **High Memory Usage**: Check cache settings and database connection pools
2. **Slow Query Performance**: Analyze database indexes and optimize queries
3. **Security Alerts**: Review audit logs and update security policies
4. **Connection Timeouts**: Adjust database and Redis timeout settings

### Diagnostic Commands

```bash
# Check application logs
docker logs sql-synthesizer-app

# Monitor resource usage
docker stats

# Check database connections
psql -h localhost -U postgres -c "SELECT * FROM pg_stat_activity;"

# Redis cache status
redis-cli info memory
```

## 📞 Support & Maintenance

### Backup Strategy
- **Database**: Daily automated backups with 30-day retention
- **Configuration**: Version-controlled in Git
- **Logs**: Centralized logging with 90-day retention

### Update Schedule
- **Security patches**: Weekly automated updates
- **Feature releases**: Monthly with staging validation
- **Major versions**: Quarterly with comprehensive testing

---

**Production Support**: production-support@terragonlabs.com  
**Emergency Contact**: +1-XXX-XXX-XXXX (24/7)  
**Documentation**: https://docs.sqlsynthesizer.com