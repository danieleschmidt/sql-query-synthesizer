# Deployment Documentation

Comprehensive deployment guides for the SQL Query Synthesizer.

## Deployment Options

1. **[Docker Compose](../DEPLOYMENT.md)** - Local and development deployment
2. **[Kubernetes](kubernetes.md)** - Production container orchestration
3. **[Cloud Deployment](cloud-deployment.md)** - AWS, GCP, Azure deployment guides
4. **[Bare Metal](bare-metal.md)** - Traditional server deployment

## Build Process

### Local Build
```bash
# Build Python package
make build

# Build Docker image
make docker-build

# Build with security scanning
make docker-build && make docker-scan
```

### CI/CD Build
```bash
# CI-optimized build commands
make ci-install
make ci-test
make ci-security
make build
```

## Container Security

### Multi-stage Build
The Dockerfile uses multi-stage builds to minimize attack surface:
- Builder stage: Compiles dependencies
- Production stage: Runtime-only environment
- Non-root user execution
- Minimal base image (python:slim)

### Security Scanning
```bash
# Scan Docker image for vulnerabilities
make docker-scan

# SBOM generation
python scripts/generate_sbom.py
python scripts/generate_advanced_sbom.py
```

## Environment Configuration

### Required Environment Variables
```bash
# Database configuration
DATABASE_URL=postgresql://user:pass@host:port/db
QUERY_AGENT_CACHE_BACKEND=redis
QUERY_AGENT_REDIS_HOST=redis-host

# Security
QUERY_AGENT_SECRET_KEY=your-secret-key
OPENAI_API_KEY=your-openai-key

# Monitoring
QUERY_AGENT_PROMETHEUS_ENABLED=true
```

### Optional Configuration
```bash
# Performance tuning
QUERY_AGENT_DB_POOL_SIZE=10
QUERY_AGENT_DB_MAX_OVERFLOW=20
QUERY_AGENT_CACHE_TTL=3600

# Security hardening
QUERY_AGENT_RATE_LIMIT_PER_MINUTE=60
QUERY_AGENT_ENABLE_HSTS=true
```

## Health Checks

### Application Health
```bash
curl http://localhost:5000/health
```

### Container Health
```bash
docker ps  # Check container status
docker logs sql-synthesizer-app  # Check logs
```

## Troubleshooting

### Common Issues
1. **Database Connection Failures** - Check DATABASE_URL and network connectivity
2. **Cache Connection Issues** - Verify Redis configuration and connectivity
3. **Permission Errors** - Ensure proper file ownership and container user
4. **Resource Constraints** - Monitor CPU, memory, and disk usage

### Debugging
```bash
# Access container shell
docker exec -it sql-synthesizer-app /bin/bash

# View detailed logs
docker-compose logs -f sql-synthesizer

# Resource monitoring
docker stats sql-synthesizer-app
```

For detailed deployment instructions, see the specific guides in this directory.