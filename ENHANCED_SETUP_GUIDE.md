# Enhanced Setup Guide - Post Autonomous SDLC Enhancement

This guide provides comprehensive setup instructions for the enhanced SQL Query Synthesizer repository following autonomous SDLC optimization.

## 🚀 Quick Start

### VS Code Development (Recommended)

1. **Install VS Code** with Dev Containers extension
2. **Clone and open repository**:
   ```bash
   git clone <repository-url>
   code sql-query-synthesizer
   ```
3. **Open in Container**: VS Code will prompt "Reopen in Container"
4. **Done!** Full development environment with PostgreSQL + Redis

### Traditional Setup

```bash
# Clone repository
git clone <repository-url>
cd sql-query-synthesizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 🔧 Development Environment Features

### VS Code Integration (`.vscode/`)
- **Complete IDE configuration** with Python toolchain
- **Debugging profiles** for CLI, web app, and Docker
- **Task automation** for testing, linting, security checks
- **15+ recommended extensions** for optimal workflow
- **SQL tools integration** for database development

### Development Containers (`.devcontainer/`)
- **Full containerized environment** with all services
- **Automatic dependency installation** and tool setup
- **PostgreSQL and Redis** pre-configured and running
- **Port forwarding** for web app (5000) and databases

### Enhanced Development Tools
- **Advanced linting** with Ruff, Black, isort, MyPy, Pylint
- **Security scanning** with Bandit and Safety
- **Pre-commit hooks** with comprehensive checks
- **Test automation** with pytest and coverage reporting

## 🛡️ Security Hardening Features

### Comprehensive Security (`security_hardening.py`)

#### Multi-Layer Input Validation
- **SQL injection protection** with AST analysis and pattern matching
- **PII detection and masking** for GDPR compliance
- **Rate limiting** with IP-based enforcement
- **Threat detection** with real-time monitoring

#### Authentication & Authorization
- **API key management** with rotation policies
- **Session security** with encryption and timeouts
- **Multi-factor authentication** support (TOTP, email)
- **Role-based access control** with audit logging

#### Network Security Hardening
- **Security headers** (HSTS, CSP, XSS protection)
- **CORS policy** enforcement with origin validation
- **DDoS protection** with rate limiting and IP filtering
- **TLS 1.3 enforcement** with HTTPS redirect

#### Compliance Framework Integration
- **SOC 2 Type II** controls implementation
- **GDPR compliance** with data protection measures
- **Automated compliance reporting** and evidence collection
- **Comprehensive audit trail** with 365-day retention

### Production Security Configuration

```bash
# Environment Variables for Production Security
QUERY_AGENT_SECURITY_LEVEL=maximum
QUERY_AGENT_API_KEY_REQUIRED=true
QUERY_AGENT_ENABLE_HSTS=true
QUERY_AGENT_RATE_LIMIT_PER_MINUTE=30
QUERY_AGENT_ENCRYPT_DATA=true
QUERY_AGENT_GDPR_COMPLIANCE=true
QUERY_AGENT_SOC2_COMPLIANCE=true
```

## ⚡ Performance Optimization

### Advanced Performance Configuration (`performance_config.py`)

#### Adaptive Database Connection Pooling
- **Dynamic pool sizing** based on utilization (20-60 connections)
- **Multi-strategy support**: Fixed, Dynamic, Adaptive, Elastic
- **Health monitoring** with automatic retry logic
- **Environment-specific optimization** for dev/staging/prod

#### Multi-Tier Caching Architecture
- **L1 in-memory cache** for ultra-fast access (1000 items, 60s TTL)
- **L2 Redis/Memcached** for shared caching (3600s TTL)
- **Intelligent cache warming** on application startup
- **Cache compression** for objects >1KB
- **Hit rate monitoring** with alerting (<70% threshold)

#### Runtime Performance Optimization
- **Automatic resource monitoring** with CPU/memory thresholds
- **Dynamic optimization** based on real-time metrics
- **Auto-scaling triggers** for containerized deployments
- **Performance profiling** with 1% sampling rate

### Production Performance Configuration

```bash
# Environment Variables for Performance Optimization
QUERY_AGENT_DB_POOL_SIZE=30
QUERY_AGENT_DB_MAX_OVERFLOW=50
QUERY_AGENT_DB_POOL_STRATEGY=adaptive
QUERY_AGENT_CACHE_BACKEND=redis
QUERY_AGENT_CACHE_MAX_MEMORY_MB=2048
QUERY_AGENT_ENABLE_CACHE_WARMING=true
QUERY_AGENT_MAX_CONCURRENT=200
QUERY_AGENT_MEMORY_LIMIT_MB=4096
```

## 📊 Advanced Observability

### OpenTelemetry Integration (`observability.py`)

#### Distributed Tracing
- **OTLP export** to standard observability platforms
- **Auto-instrumentation** for Flask, SQLAlchemy, Redis, HTTP requests
- **Custom spans** for business logic tracing
- **Environment-specific configuration** (dev/staging/prod)

#### Custom Business Metrics
- **SQL generation duration** histograms
- **Query execution performance** tracking
- **Cache hit ratio** monitoring
- **Database connection pool** statistics
- **LLM API call** tracking and costs
- **Error rates** by category and severity

#### Structured Logging with Trace Correlation
- **JSON-structured logs** with trace IDs
- **Security event logging** with threat classification
- **Performance event logging** with optimization triggers
- **Compliance event logging** for audit requirements

### Observability Configuration

```bash
# Environment Variables for Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://your-collector:4317
OTEL_SERVICE_NAME=sql-query-synthesizer
OTEL_SERVICE_VERSION=1.0.0
QUERY_AGENT_ENABLE_METRICS=true
QUERY_AGENT_METRICS_EXPORT_INTERVAL=30
```

## 🔍 Supply Chain Security

### Advanced SBOM Generation (`scripts/generate_advanced_sbom.py`)

#### Multi-Format SBOM Support
- **SPDX-JSON** format for industry standard compliance
- **CycloneDX-JSON** format for enhanced security metadata
- **Custom summary reports** with risk assessment
- **Vulnerability reports** with remediation recommendations

#### Enhanced Security Analysis
- **License compliance** analysis with risk scoring
- **Dependency integrity** verification with cryptographic hashes
- **Transitive dependency** analysis with depth tracking
- **Supply chain risk** assessment with scoring

#### Integration Features
- **CI/CD pipeline** integration for automated generation
- **Container image** analysis support
- **Vulnerability database** integration for real-time scanning
- **Compliance reporting** for regulatory requirements

### SBOM Generation Usage

```bash
# Install SBOM dependencies
pip install cyclonedx-bom packagedcode requests

# Generate comprehensive SBOM
python scripts/generate_advanced_sbom.py --output-dir sbom --format all

# Output files:
# - sbom/sbom-cyclonedx.json
# - sbom/sbom-spdx.json  
# - sbom/sbom-summary.json
# - sbom/vulnerability-report.json
```

## 🏗️ GitHub Actions & CI/CD

### Comprehensive Workflow Documentation

The repository includes complete workflow documentation in `docs/GITHUB_WORKFLOWS.md` with templates for:

- **Core CI Pipeline** (`ci.yml`) - Multi-Python version testing
- **Security Scanning** (`security.yml`) - Bandit, Safety, Semgrep, Trivy
- **Code Quality** (`quality.yml`) - Pre-commit, SonarCloud analysis
- **Performance Testing** (`performance.yml`) - Benchmarking with regression detection
- **Container Security** (`container-security.yml`) - Docker image vulnerability scanning
- **Release Automation** (`release.yml`) - Automated PyPI publishing
- **Mutation Testing** (`mutation.yml`) - Test quality assessment
- **Dependency Scanning** (`dependency-check.yml`) - Continuous vulnerability monitoring

### Required Repository Secrets

Configure these secrets in GitHub repository settings:

#### Essential Secrets
```
CODECOV_TOKEN          # Coverage reporting
SONAR_TOKEN           # Code quality analysis
PYPI_API_TOKEN        # Package publishing
```

#### Enhanced Security Secrets
```
QUERY_AGENT_API_KEY           # Production API authentication
QUERY_AGENT_SECRET_KEY        # Flask session security
DATABASE_ENCRYPTION_KEY       # Data encryption (32-byte base64)
```

#### Observability Secrets
```
OTEL_EXPORTER_OTLP_ENDPOINT  # OpenTelemetry collector
SENTRY_DSN                   # Error tracking (optional)
```

## 🚀 Deployment Guide

### Production Environment Setup

#### 1. Infrastructure Requirements
- **PostgreSQL 13+** with connection pooling (PgBouncer)
- **Redis 6+** for L2 caching and session storage
- **Load balancer** with SSL termination and security headers
- **Container orchestration** (Kubernetes/Docker Swarm)

#### 2. Security Configuration
```bash
# Required security environment variables
QUERY_AGENT_ENV=production
QUERY_AGENT_SECURITY_LEVEL=maximum
QUERY_AGENT_API_KEY_REQUIRED=true
QUERY_AGENT_ENABLE_HSTS=true
QUERY_AGENT_ENCRYPT_DATA=true
QUERY_AGENT_GDPR_COMPLIANCE=true
QUERY_AGENT_SOC2_COMPLIANCE=true

# Database security
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/dbname?sslmode=require
QUERY_AGENT_DB_POOL_SIZE=30
QUERY_AGENT_DB_MAX_OVERFLOW=50

# Cache configuration
REDIS_URL=redis://redis:6379/0?ssl=true
QUERY_AGENT_CACHE_BACKEND=redis
QUERY_AGENT_ENABLE_CACHE_WARMING=true
```

#### 3. Observability Setup
```bash
# OpenTelemetry configuration
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-collector:4317
OTEL_SERVICE_NAME=sql-query-synthesizer
OTEL_SERVICE_VERSION=1.0.0

# Metrics and monitoring
QUERY_AGENT_ENABLE_METRICS=true
QUERY_AGENT_METRICS_EXPORT_INTERVAL=30
```

### Container Deployment

#### Docker Compose (Development/Staging)
```bash
# Use provided docker-compose.yml with dev containers
docker-compose up -d
```

#### Kubernetes (Production)
```yaml
# Example Kubernetes deployment with security and performance
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sql-query-synthesizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sql-query-synthesizer
  template:
    metadata:
      labels:
        app: sql-query-synthesizer
    spec:
      containers:
      - name: app
        image: sql-query-synthesizer:latest
        env:
        - name: QUERY_AGENT_ENV
          value: "production"
        - name: QUERY_AGENT_SECURITY_LEVEL
          value: "maximum"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## 🧪 Testing & Quality Assurance

### Comprehensive Test Suite
- **Unit tests** with pytest and comprehensive coverage
- **Integration tests** with real database connections
- **Security tests** with threat simulation
- **Performance tests** with benchmarking and regression detection
- **Mutation tests** for test quality assessment

### Quality Gates
- **85% code coverage** minimum requirement
- **Security scanning** with zero high/critical vulnerabilities
- **Performance benchmarks** with regression detection
- **Code quality** analysis with SonarCloud
- **Dependency vulnerability** scanning with Safety

### Running Tests Locally
```bash
# Run full test suite
pytest tests/ --cov=sql_synthesizer --cov-report=html -v

# Run security tests only
pytest tests/ -m security -v

# Run performance benchmarks
pytest tests/performance/ --benchmark-json=benchmark.json

# Generate SBOM and security reports
python scripts/generate_advanced_sbom.py --output-dir reports
python -m bandit -r sql_synthesizer/ -f json -o security-report.json
```

## 📋 Verification Checklist

### Development Environment
- [ ] VS Code opens with all recommended extensions
- [ ] Dev container builds and runs successfully
- [ ] All tests pass with coverage >85%
- [ ] Pre-commit hooks run without errors
- [ ] Security scans pass with no high-severity issues
- [ ] Performance benchmarks complete successfully

### Production Deployment
- [ ] All environment variables configured
- [ ] Database connections working with pooling
- [ ] Redis cache operational with multi-tier setup
- [ ] Security hardening features enabled
- [ ] Observability metrics being collected
- [ ] SBOM generation working in CI/CD
- [ ] All monitoring and alerting configured
- [ ] Backup and disaster recovery tested

### Security Verification
- [ ] API key authentication working
- [ ] Input validation blocking malicious inputs
- [ ] Security headers present in HTTP responses
- [ ] Audit logging functioning correctly
- [ ] Compliance features operational
- [ ] Threat detection generating appropriate alerts

### Performance Verification
- [ ] Connection pooling adapting to load
- [ ] Multi-tier caching showing hit rates >70%
- [ ] Query performance within acceptable thresholds
- [ ] Memory usage within configured limits
- [ ] Auto-scaling triggers responding appropriately

## 🆘 Troubleshooting

### Common Issues

#### Development Environment
```bash
# VS Code dev container not starting
# Check Docker is running and dev containers extension installed

# Pre-commit hooks failing
pre-commit clean
pre-commit install
pre-commit run --all-files

# Tests failing with database errors
# Ensure PostgreSQL is running (automatic in dev container)
docker-compose up -d postgres
```

#### Production Issues
```bash
# High memory usage
# Check cache configuration and enable monitoring
QUERY_AGENT_CACHE_MAX_MEMORY_MB=1024
QUERY_AGENT_MEMORY_LIMIT_MB=2048

# Security alerts triggering
# Review security logs for threat patterns
python -c "from security_hardening import load_security_config; print(load_security_config())"

# Performance degradation
# Enable performance monitoring and check metrics
QUERY_AGENT_ENABLE_PROFILING=true
QUERY_AGENT_ENABLE_METRICS=true
```

### Support Resources
- **Documentation**: Comprehensive docs in `docs/` directory
- **GitHub Issues**: For bug reports and feature requests
- **Security Issues**: Follow responsible disclosure in `SECURITY.md`
- **Performance Issues**: Use observability tools and metrics

## 🎯 Next Steps

### Immediate (Week 1)
1. **Set up development environment** with VS Code dev containers
2. **Configure GitHub repository** with secrets and branch protection
3. **Create GitHub Actions workflows** from provided templates
4. **Run comprehensive test suite** and verify all checks pass

### Short-term (Month 1)
1. **Deploy to staging environment** with full security hardening
2. **Set up monitoring and observability** with OpenTelemetry
3. **Conduct security testing** with threat simulation
4. **Performance optimization** based on production-like load

### Long-term (Quarter 1)
1. **Production deployment** with enterprise-grade configuration
2. **SOC 2 Type II audit** preparation and certification
3. **Advanced threat detection** with machine learning models
4. **Multi-region deployment** for high availability

---

This enhanced setup guide provides comprehensive instructions for leveraging all the advanced features implemented through autonomous SDLC enhancement. The repository now supports enterprise-grade development, security, performance, and operational requirements.