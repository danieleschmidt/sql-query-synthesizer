# Production Deployment Guide

## Overview

This directory contains production deployment configurations and scripts for the SQL Query Synthesizer with Autonomous SDLC capabilities.

## Files

- `deploy.sh` - Main deployment script with blue-green deployment
- `docker-compose.production.yml` - Production service orchestration
- `production.env.example` - Environment variables template
- `configs.yaml` - Deployment environment configurations

## Quick Deployment

1. **Setup Environment**
   ```bash
   cp production.env.example production.env
   # Edit production.env with your values
   ```

2. **Run Deployment**
   ```bash
   # Dry run first
   DRY_RUN=true ./deploy.sh production v1.2.3
   
   # Execute deployment
   ./deploy.sh production v1.2.3
   ```

## Deployment Process

The deployment script follows these steps:

1. **Prerequisites Check** - Verify required tools and configuration
2. **Quality Gates** - Run autonomous quality validation
3. **Backup** - Create database and configuration backups
4. **Build & Test** - Build and validate Docker image
5. **Blue-Green Deploy** - Deploy with zero downtime
6. **Health Checks** - Comprehensive health validation
7. **Traffic Switch** - Gradual traffic migration
8. **Rollback** - Automatic rollback on failure

## Environment Configuration

### Development
- Single replica
- Minimal resources
- Fast health checks

### Staging
- 2 replicas
- Moderate resources
- Production-like configuration

### Production
- 3+ replicas
- Full resources
- Comprehensive monitoring

## Health Checks

The system performs multi-dimensional health checks:

- **HTTP Endpoints** - Application availability
- **Database** - Connection and performance
- **Resources** - CPU, memory, disk usage
- **Metrics** - Error rates, response times

## Rollback Strategy

Automatic rollback triggers:
- Health check failures
- Performance degradation
- Error rate increases
- Manual intervention

## Monitoring

Post-deployment monitoring includes:
- Service health status
- Performance metrics
- Error tracking
- Resource utilization

## Troubleshooting

### Common Issues

1. **Health checks failing**
   - Check service logs
   - Verify configuration
   - Review network connectivity

2. **Performance issues**
   - Monitor resource usage
   - Check database performance
   - Review application metrics

3. **Deployment timeouts**
   - Increase timeout values
   - Check resource availability
   - Review startup dependencies

### Debug Commands

```bash
# Check service status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs sql-synthesizer

# Run health check manually
curl -f http://localhost:5000/health
```

## Security

- Environment variables for secrets
- Network isolation
- Regular security scans
- Access control

## Backup and Recovery

- Automated database backups
- Configuration backups
- Point-in-time recovery
- Disaster recovery procedures

For detailed information, see the main [AUTONOMOUS_SDLC_GUIDE.md](../AUTONOMOUS_SDLC_GUIDE.md).