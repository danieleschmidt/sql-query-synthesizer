# Automation Scripts

This directory contains automation scripts for the SQL Query Synthesizer project.

## Available Scripts

### Core Automation
- [`collect_metrics.py`](../collect_metrics.py) - Automated metrics collection and reporting
- [`repository_health.py`](../repository_health.py) - Repository health monitoring and assessment
- [`generate_sbom.py`](../generate_sbom.py) - Software Bill of Materials generation
- [`generate_advanced_sbom.py`](../generate_advanced_sbom.py) - Advanced SBOM with security analysis

### Maintenance Automation
- [`dependency_update.sh`](dependency_update.sh) - Automated dependency updates
- [`security_scan.sh`](security_scan.sh) - Comprehensive security scanning
- [`backup_artifacts.sh`](backup_artifacts.sh) - Artifact backup and archival
- [`cleanup_branches.sh`](cleanup_branches.sh) - Automated branch cleanup

### Monitoring Automation
- [`health_check.sh`](health_check.sh) - Application health monitoring
- [`performance_monitor.py`](performance_monitor.py) - Performance metrics collection
- [`log_analysis.py`](log_analysis.py) - Automated log analysis and alerting

## Usage

### Daily Automation
```bash
# Run daily metrics collection
./scripts/collect_metrics.py

# Check repository health
./scripts/repository_health.py --output daily_health_report.md

# Security scan
./scripts/automation/security_scan.sh
```

### Weekly Automation
```bash
# Dependency updates
./scripts/automation/dependency_update.sh

# Performance analysis
python ./scripts/automation/performance_monitor.py --weekly

# Cleanup old branches
./scripts/automation/cleanup_branches.sh --older-than 30
```

### On-Demand Automation
```bash
# Generate SBOM
python ./scripts/generate_advanced_sbom.py --format spdx

# Full health assessment
./scripts/repository_health.py --json --output health_report.json

# Performance benchmarking
python ./scripts/automation/performance_monitor.py --benchmark
```

## Configuration

Most scripts support configuration through:
- Environment variables
- Command-line arguments  
- Configuration files in `config/`

### Environment Variables
```bash
# GitHub integration
export GITHUB_TOKEN=your_github_token

# Monitoring endpoints
export PROMETHEUS_URL=http://localhost:9090
export GRAFANA_URL=http://localhost:3000

# Notification endpoints
export SLACK_WEBHOOK_URL=your_slack_webhook
export EMAIL_ALERTS=alerts@company.com
```

## CI/CD Integration

These scripts are integrated into GitHub Actions workflows:

- **Metrics Collection**: Runs daily via `collect_metrics.py`
- **Health Monitoring**: Runs on every PR via `repository_health.py`
- **Security Scanning**: Runs on schedule and PR via security scripts
- **Performance Monitoring**: Runs on release via performance scripts

## Monitoring and Alerting

Scripts support multiple notification channels:
- Slack webhooks for immediate alerts
- Email notifications for critical issues
- GitHub issue creation for tracking
- Prometheus metrics for monitoring dashboards

## Development

### Adding New Scripts

1. Create script in appropriate subdirectory
2. Follow naming convention: `action_subject.{py|sh}`
3. Include comprehensive help documentation
4. Add error handling and logging
5. Update this README

### Testing Scripts

```bash
# Test metrics collection (dry run)
python scripts/collect_metrics.py --dry-run

# Test health check with verbose output
python scripts/repository_health.py --verbose

# Validate automation scripts
./scripts/automation/validate_scripts.sh
```

### Script Requirements

All automation scripts should:
- Include shebang and be executable
- Support `--help` argument
- Include error handling
- Log activities appropriately
- Support dry-run mode where applicable
- Follow project coding standards

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure scripts are executable (`chmod +x`)
2. **Missing Dependencies**: Install required packages (`pip install -r requirements-dev.txt`)
3. **GitHub API Limits**: Set `GITHUB_TOKEN` environment variable
4. **Network Issues**: Check connectivity to external services

### Debug Mode

Most scripts support debug mode:
```bash
# Enable debug logging
export DEBUG=1
python scripts/collect_metrics.py

# Verbose output
./scripts/repository_health.py --verbose
```

### Log Files

Automation logs are written to:
- `logs/automation.log` - General automation logs
- `logs/metrics.log` - Metrics collection logs
- `logs/health.log` - Health monitoring logs
- `logs/security.log` - Security scan logs

## Security Considerations

- Scripts handle sensitive data (tokens, credentials)
- All secrets should be passed via environment variables
- Logs are sanitized to prevent credential leakage
- Network communications use TLS where possible
- Scripts validate inputs to prevent injection attacks

## Maintenance

### Regular Updates
- Update script dependencies monthly
- Review and optimize performance quarterly
- Update documentation when adding features
- Test scripts in staging before production deployment

### Monitoring Script Health
- Monitor script execution times
- Track success/failure rates
- Alert on script failures
- Regular performance reviews

For more detailed information about specific scripts, see their individual documentation or run with `--help`.