# 🚀 Quantum SDLC Systems - Production Deployment Guide

## Overview

This guide covers the complete production deployment of the Quantum SDLC Systems, including the SQL Synthesizer, Autonomous SDLC Engine, and Quantum-inspired optimization components.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Web App       │────│   Database      │
│   (Nginx)       │    │   (Gunicorn)    │    │   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Cache         │────│   Monitoring    │
                       │   (Redis)       │    │   (Prometheus)  │
                       └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Quantum SDLC  │
                       │   Master        │
                       └─────────────────┘
```

## Prerequisites

### System Requirements
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 50GB SSD
- **Network**: High-speed internet connection
- **OS**: Ubuntu 20.04 LTS or later, RHEL 8+, or Docker-compatible system

### Software Dependencies
- Docker Engine 24.0+
- Docker Compose 2.0+
- Git 2.30+
- OpenSSL for SSL certificates

### External Services
- **Database**: PostgreSQL 15+ (can be containerized or external)
- **Cache**: Redis 7+ (can be containerized or external)  
- **Monitoring**: Prometheus + Grafana (included in docker-compose)
- **LLM API**: OpenAI API key (for SQL generation features)

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/terragon-labs/sql-synthesizer.git
cd sql-synthesizer
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env.production

# Edit configuration
nano .env.production
```

Required environment variables:
```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@postgres:5432/sql_synthesizer
POSTGRES_DB=sql_synthesizer
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password

# Redis Configuration
REDIS_PASSWORD=your_redis_password

# Security
SECRET_KEY=your_very_long_and_random_secret_key
GRAFANA_PASSWORD=your_grafana_admin_password

# OpenAI API (optional, for LLM features)
OPENAI_API_KEY=your_openai_api_key

# Version and Build Info
VERSION=v1.0.0
BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
VCS_REF=$(git rev-parse HEAD)
```

### 3. Deploy with Docker Compose
```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check service status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

### 4. Verify Deployment
```bash
# Check web application
curl http://localhost/health

# Check quantum master status
docker logs quantum-master

# Access monitoring
# Grafana: http://localhost:3000 (admin/your_password)
# Prometheus: http://localhost:9090
```

## Detailed Configuration

### SSL/TLS Configuration
```bash
# Create SSL directory
mkdir -p ssl

# Generate self-signed certificates (for testing)
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes

# Or place your certificates
# ssl/cert.pem - Certificate file
# ssl/key.pem - Private key file
```

### Nginx Configuration
Create `nginx/nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server sql-synthesizer-web:8000;
    }
    
    upstream grafana {
        server grafana:3000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

    server {
        listen 80;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Main application
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Rate limiting
            limit_req zone=api burst=20 nodelay;
        }

        # API endpoints with stricter limits
        location /api/ {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Stricter rate limiting for API
            limit_req zone=api burst=10 nodelay;
        }

        # Monitoring dashboard
        location /monitoring/ {
            proxy_pass http://grafana/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health checks (no rate limiting)
        location /health {
            proxy_pass http://app;
            access_log off;
        }
    }
}
```

### Database Initialization
Create `scripts/init-db.sql`:
```sql
-- Create additional databases if needed
CREATE DATABASE sql_synthesizer_dev;
CREATE DATABASE sql_synthesizer_test;

-- Create application user with limited privileges
CREATE USER app_user WITH PASSWORD 'app_password';
GRANT CONNECT ON DATABASE sql_synthesizer TO app_user;
GRANT ALL PRIVILEGES ON DATABASE sql_synthesizer TO app_user;

-- Create extensions
\c sql_synthesizer;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO app_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app_user;
```

### Redis Configuration
Create `config/redis.conf`:
```conf
# Network
bind 0.0.0.0
port 6379

# Security
requirepass your_redis_password

# Memory
maxmemory 1gb
maxmemory-policy allkeys-lru

# Persistence
appendonly yes
appendfsync everysec

# Performance
tcp-keepalive 300
timeout 300
```

## Monitoring Configuration

### Prometheus Configuration
Create `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'sql-synthesizer'
    static_configs:
      - targets: ['sql-synthesizer-web:8000']
    scrape_interval: 30s
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 30s
```

### Alert Rules
Create `monitoring/alert_rules.yml`:
```yaml
groups:
  - name: sql_synthesizer_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: High error rate detected
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High latency detected
          description: "95th percentile latency is {{ $value }} seconds"

      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Database is down
          description: "PostgreSQL database is not responding"

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: Redis is down
          description: "Redis cache is not responding"
```

## Health Checks and Monitoring

### Application Health Endpoints
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health with dependencies
- `GET /metrics` - Prometheus metrics

### Key Metrics to Monitor
- **Response Time**: 95th percentile < 2 seconds
- **Error Rate**: < 1% of total requests
- **Throughput**: Requests per second
- **Database Connection Pool**: Active connections
- **Memory Usage**: < 80% of allocated memory
- **CPU Usage**: < 70% average utilization

### Alerting Thresholds
```yaml
Critical Alerts:
  - Database Down: Immediate
  - Application Down: Immediate
  - Error Rate > 5%: Within 2 minutes

Warning Alerts:
  - Response Time > 2s: Within 5 minutes
  - Memory Usage > 80%: Within 5 minutes
  - CPU Usage > 70%: Within 10 minutes
  - Error Rate > 1%: Within 5 minutes
```

## Security Hardening

### 1. Container Security
```bash
# Run containers as non-root
USER appuser

# Use minimal base images
FROM python:3.11-slim

# Scan images for vulnerabilities
docker scan your-image:tag
```

### 2. Network Security
```bash
# Use custom networks
networks:
  quantum-network:
    driver: bridge
    internal: true  # No external access

# Expose only necessary ports
ports:
  - "443:443"  # HTTPS only
```

### 3. Secret Management
```bash
# Use Docker secrets or external secret management
echo "your_secret" | docker secret create db_password -

# Or use environment files with restricted permissions
chmod 600 .env.production
```

### 4. Database Security
```sql
-- Use specific user with minimal permissions
CREATE USER app_user WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE sql_synthesizer TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;

-- Enable SSL
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

## Backup and Recovery

### Database Backup
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker exec sql-synthesizer-db pg_dump -U postgres sql_synthesizer | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/backup_$DATE.sql.gz s3://your-backup-bucket/db-backups/
```

### Application Data Backup
```bash
# Backup volumes
docker run --rm -v sql-synthesizer-postgres-data:/source -v $(pwd)/backups:/backup alpine tar czf /backup/postgres-data-$DATE.tar.gz -C /source .
docker run --rm -v sql-synthesizer-redis-data:/source -v $(pwd)/backups:/backup alpine tar czf /backup/redis-data-$DATE.tar.gz -C /source .
```

### Recovery Procedures
```bash
# Restore database
docker exec -i sql-synthesizer-db psql -U postgres sql_synthesizer < backup.sql

# Restore volumes
docker run --rm -v sql-synthesizer-postgres-data:/target -v $(pwd)/backups:/backup alpine tar xzf /backup/postgres-data.tar.gz -C /target
```

## Scaling and Performance Optimization

### Horizontal Scaling
```yaml
# Docker Compose with replicas
services:
  sql-synthesizer-web:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
```

### Database Optimization
```sql
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_queries_created_at ON queries(created_at);
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);

-- Analyze tables
ANALYZE queries;
ANALYZE users;

-- Connection pooling (PgBouncer)
docker run -d --name pgbouncer \
  -p 6432:6432 \
  -e DATABASES_HOST=postgres \
  -e DATABASES_PORT=5432 \
  -e DATABASES_USER=postgres \
  -e DATABASES_PASSWORD=password \
  -e DATABASES_DBNAME=sql_synthesizer \
  pgbouncer/pgbouncer
```

### Redis Optimization
```conf
# Redis configuration for production
maxmemory 4gb
maxmemory-policy allkeys-lru
tcp-keepalive 60
timeout 300

# Enable compression
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit slave 256mb 64mb 60
```

## Troubleshooting

### Common Issues

#### 1. Application Won't Start
```bash
# Check logs
docker-compose logs sql-synthesizer-web

# Check resource usage
docker stats

# Check database connection
docker exec sql-synthesizer-web python -c "
from sqlalchemy import create_engine
engine = create_engine('$DATABASE_URL')
engine.execute('SELECT 1')
print('Database connection OK')
"
```

#### 2. High Memory Usage
```bash
# Monitor memory usage
docker exec sql-synthesizer-web python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB')
"

# Check for memory leaks
docker exec sql-synthesizer-web python -c "
import gc
print(f'Objects: {len(gc.get_objects())}')
gc.collect()
"
```

#### 3. Database Performance Issues
```sql
-- Check slow queries
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Check connections
SELECT * FROM pg_stat_activity;

-- Check locks
SELECT * FROM pg_locks;
```

### Performance Tuning

#### Application Tuning
```python
# Gunicorn configuration
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 120
keepalive = 5
```

#### Database Tuning
```sql
-- PostgreSQL configuration
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.7
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
```

## Maintenance

### Regular Maintenance Tasks
```bash
# Weekly tasks
- Database backup verification
- Log rotation and cleanup
- Security updates check
- Performance metrics review

# Monthly tasks
- Full system backup
- Dependency updates
- Security audit
- Capacity planning review

# Quarterly tasks
- Disaster recovery test
- Performance benchmark
- Security penetration test
- Documentation update
```

### Update Procedures
```bash
# Rolling update
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d --no-deps sql-synthesizer-web

# Blue-green deployment
docker-compose -f docker-compose.production.yml up -d --scale sql-synthesizer-web=2
# Verify new instance
docker-compose -f docker-compose.production.yml up -d --scale sql-synthesizer-web=1 --no-deps
```

## Support and Troubleshooting

### Log Locations
- Application logs: `/app/logs/`
- Nginx logs: `/var/log/nginx/`
- Database logs: Docker logs for postgres container
- System logs: `journalctl -u docker`

### Key Commands
```bash
# Service status
docker-compose -f docker-compose.production.yml ps

# Restart service
docker-compose -f docker-compose.production.yml restart sql-synthesizer-web

# Scale service
docker-compose -f docker-compose.production.yml up -d --scale sql-synthesizer-web=3

# Emergency stop
docker-compose -f docker-compose.production.yml down
```

### Contact Information
- **Technical Support**: engineering@terragon.ai
- **Emergency Hotline**: +1-XXX-XXX-XXXX
- **Documentation**: https://docs.terragon.ai/sql-synthesizer
- **Issue Tracker**: https://github.com/terragon-labs/sql-synthesizer/issues

---

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Environment variables configured
- [ ] SSL certificates in place
- [ ] Database initialized
- [ ] Secrets secured
- [ ] Monitoring configured
- [ ] Backup procedures tested

### Deployment
- [ ] Services started successfully
- [ ] Health checks passing
- [ ] Database connectivity verified
- [ ] Cache connectivity verified
- [ ] External API connectivity tested
- [ ] Monitoring dashboards accessible

### Post-deployment
- [ ] Performance benchmarks run
- [ ] Security scan completed
- [ ] Backup procedures verified
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Team notified

### Production Readiness
- [ ] All services stable for 24 hours
- [ ] Performance meets SLAs
- [ ] Security posture verified
- [ ] Incident response procedures ready
- [ ] Scaling policies configured

---

**🎉 Congratulations! Your Quantum SDLC Systems are now production-ready!**