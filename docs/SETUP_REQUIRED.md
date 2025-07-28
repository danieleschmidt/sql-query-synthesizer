# Manual Setup Required

## GitHub Repository Settings

### 1. Branch Protection Rules
Navigate to Settings > Branches and configure:

```
Branch name pattern: main
✓ Restrict pushes that create files
✓ Require a pull request before merging
  - Require approvals: 1
  - Dismiss stale reviews when new commits are pushed
✓ Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Status checks: ci, security-scan
```

### 2. Repository Secrets
Add these secrets in Settings > Secrets and variables > Actions:

- `PYPI_API_TOKEN`: For automated package publishing
- `CODECOV_TOKEN`: For code coverage reporting
- `SECURITY_CONTACT`: Email for security notifications

### 3. Repository Topics
Add these topics in Settings > General:

```
python, sql, natural-language-processing, query-synthesis, 
security-first, flask, sqlalchemy, openai
```

## GitHub Actions Workflows

Create these workflow files in `.github/workflows/`:

1. **ci.yml** - Continuous Integration
2. **security.yml** - Security Scanning  
3. **release.yml** - Release Management
4. **docs.yml** - Documentation Deployment

## External Integrations

### Code Quality Tools
- **Codecov**: Enable for coverage reporting
- **CodeQL**: Enable GitHub's security analysis
- **Dependabot**: Configure for dependency updates

### Monitoring Setup  
- **Sentry**: Application error tracking
- **DataDog**: Performance monitoring (optional)

## Pre-commit Setup

Each developer should run:
```bash
pre-commit install
pre-commit run --all-files
```

## Development Environment

Follow instructions in [docs/DEVELOPMENT.md](DEVELOPMENT.md) for local setup.