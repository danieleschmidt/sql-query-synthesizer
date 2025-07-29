# Deployment Guide

## Overview

This guide covers deployment strategies, infrastructure requirements, and operational considerations for the SQL Query Synthesizer in production environments.

## Deployment Architecture

### Container-Based Deployment (Recommended)
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  sql-synthesizer:
    image: sql-synthesizer:latest
    replicas: 3
    environment:
      - QUERY_AGENT_CACHE_BACKEND=redis
      - QUERY_AGENT_DB_POOL_SIZE=20
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sql-synthesizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sql-synthesizer
  template:
    metadata:
      labels:
        app: sql-synthesizer
    spec:
      containers:
      - name: sql-synthesizer
        image: sql-synthesizer:latest
        ports:
        - containerPort: 5000
        env:
        - name: QUERY_AGENT_CACHE_BACKEND
          value: "redis"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Infrastructure Requirements

### Minimum System Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB SSD
- **Network**: 1Gbps

### Recommended Production Requirements
- **CPU**: 4-8 cores
- **RAM**: 8-16GB
- **Storage**: 50GB SSD
- **Network**: 10Gbps
- **Load Balancer**: Required for HA

### External Dependencies
1. **Database**: PostgreSQL 12+, MySQL 8+, or SQLite 3.31+
2. **Cache**: Redis 6+ or Memcached 1.6+ (optional but recommended)
3. **Monitoring**: Prometheus + Grafana (recommended)
4. **Logging**: ELK Stack or similar centralized logging

## Deployment Strategies

### Blue-Green Deployment
```bash
# Deploy to green environment
docker-compose -f docker-compose.green.yml up -d

# Health check green environment
curl -f http://green.internal/health

# Switch traffic (load balancer configuration)
# Update DNS or load balancer to point to green

# Monitor for issues, rollback if needed
```

### Rolling Deployment
```bash
# Kubernetes rolling update
kubectl set image deployment/sql-synthesizer \
  sql-synthesizer=sql-synthesizer:new-version

# Monitor rollout
kubectl rollout status deployment/sql-synthesizer

# Rollback if needed
kubectl rollout undo deployment/sql-synthesizer
```

### Canary Deployment
```yaml
# Istio/Service Mesh canary configuration
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: sql-synthesizer
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: sql-synthesizer
        subset: v2
  - route:
    - destination:
        host: sql-synthesizer
        subset: v1
      weight: 90
    - destination:
        host: sql-synthesizer
        subset: v2
      weight: 10
```

## Environment Configuration

### Production Environment Variables
```bash
# Application Configuration
export QUERY_AGENT_ENV=production
export QUERY_AGENT_SECRET_KEY="$(openssl rand -hex 32)"
export QUERY_AGENT_WEBAPP_PORT=5000

# Database Configuration
export DATABASE_URL="postgresql://user:pass@db:5432/dbname"
export QUERY_AGENT_DB_POOL_SIZE=20
export QUERY_AGENT_DB_MAX_OVERFLOW=40
export QUERY_AGENT_DB_POOL_RECYCLE=3600

# Cache Configuration
export QUERY_AGENT_CACHE_BACKEND=redis
export QUERY_AGENT_REDIS_HOST=redis.internal
export QUERY_AGENT_REDIS_PORT=6379
export QUERY_AGENT_CACHE_TTL=3600

# Security Configuration
export QUERY_AGENT_CSRF_ENABLED=true
export QUERY_AGENT_ENABLE_HSTS=true
export QUERY_AGENT_API_KEY_REQUIRED=true
export QUERY_AGENT_API_KEY="$(openssl rand -hex 32)"
export QUERY_AGENT_RATE_LIMIT_PER_MINUTE=100

# Monitoring Configuration
export QUERY_AGENT_LOG_LEVEL=INFO
export QUERY_AGENT_ENABLE_METRICS=true

# OpenAI Configuration
export OPENAI_API_KEY="your-openai-api-key"
export QUERY_AGENT_OPENAI_MODEL=gpt-4
export QUERY_AGENT_OPENAI_TIMEOUT=30
```

### Configuration Management
```yaml
# Using Kubernetes ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: sql-synthesizer-config
data:
  QUERY_AGENT_CACHE_BACKEND: "redis"
  QUERY_AGENT_DB_POOL_SIZE: "20"
  QUERY_AGENT_LOG_LEVEL: "INFO"
---
apiVersion: v1
kind: Secret
metadata:
  name: sql-synthesizer-secrets
type: Opaque
data:
  DATABASE_URL: <base64-encoded-url>
  OPENAI_API_KEY: <base64-encoded-key>
  QUERY_AGENT_SECRET_KEY: <base64-encoded-key>
```

## Security Considerations

### Network Security
- Deploy behind a WAF (Web Application Firewall)
- Use TLS 1.3 for all external communication
- Implement network segmentation
- Restrict database access to application subnets only

### Container Security
```dockerfile
# Dockerfile security best practices
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . /app
WORKDIR /app

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "-m", "sql_synthesizer.webapp"]
```

### Secret Management
- Use external secret management (HashiCorp Vault, AWS Secrets Manager)
- Rotate secrets regularly
- Never commit secrets to version control
- Use environment-specific secret stores

## Monitoring and Observability

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sql-synthesizer'
    static_configs:
      - targets: ['sql-synthesizer:5000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard
Key metrics to monitor:
- Request rate and response times
- Error rates and status codes
- Database connection pool utilization
- Cache hit rates
- OpenAI API response times
- Resource utilization (CPU, memory, disk)

### Logging Configuration
```yaml
# Vector or Fluentd configuration
sources:
  sql_synthesizer:
    type: file
    include:
      - /var/log/sql-synthesizer/*.log
    
transforms:
  parse_logs:
    type: json_parser
    inputs:
      - sql_synthesizer
    
sinks:
  elasticsearch:
    type: elasticsearch
    inputs:
      - parse_logs
    endpoint: https://elasticsearch.internal:9200
```

### Health Checks
```python
# Custom health check endpoint
@app.route('/health/detailed')
def detailed_health():
    return {
        'status': 'healthy',
        'version': __version__,
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {
            'database': check_database_health(),
            'cache': check_cache_health(),
            'openai': check_openai_health(),
            'memory': check_memory_usage(),
            'disk': check_disk_usage()
        }
    }
```

## Backup and Disaster Recovery

### Database Backup Strategy
```bash
# Automated PostgreSQL backups
#!/bin/bash
BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)

pg_dump $DATABASE_URL > "$BACKUP_DIR/backup_$DATE.sql"

# Retention policy (keep 30 days)
find $BACKUP_DIR -name "backup_*.sql" -mtime +30 -delete
```

### Application State Backup
- Configuration backups
- Cache state snapshots (if needed)
- Log retention policies
- Secret backup procedures

### Disaster Recovery Plan
1. **RTO (Recovery Time Objective)**: 15 minutes
2. **RPO (Recovery Point Objective)**: 1 hour
3. **Failover Procedures**: Automated with health checks
4. **Data Recovery**: Point-in-time recovery from backups
5. **Communication Plan**: Status page and notifications

## Performance Optimization

### Load Balancing
```nginx
# Nginx load balancer configuration
upstream sql_synthesizer {
    least_conn;
    server sql-synthesizer-1:5000 max_fails=3 fail_timeout=30s;
    server sql-synthesizer-2:5000 max_fails=3 fail_timeout=30s;
    server sql-synthesizer-3:5000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name sql-synthesizer.example.com;
    
    location / {
        proxy_pass http://sql_synthesizer;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        proxy_pass http://sql_synthesizer;
        access_log off;
    }
}
```

### Caching Strategy
- Application-level caching with Redis
- CDN for static assets
- Database query result caching
- Schema metadata caching

### Auto-scaling
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sql-synthesizer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sql-synthesizer
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Troubleshooting

### Common Issues
1. **High CPU Usage**: Check query complexity, enable caching
2. **Memory Leaks**: Monitor connection pools, review caching
3. **Database Timeouts**: Tune connection pool settings
4. **Cache Misses**: Review cache TTL and invalidation strategy

### Debug Mode
```bash
# Enable debug logging
export QUERY_AGENT_LOG_LEVEL=DEBUG

# Enable detailed health checks
export QUERY_AGENT_DETAILED_HEALTH=true

# Monitor real-time metrics
curl http://localhost:5000/metrics | grep -E "(request_duration|error_rate)"
```

### Support and Maintenance
- Regular security updates
- Dependency vulnerability scanning
- Performance monitoring and optimization
- Capacity planning and scaling
- Documentation updates and training