# Extended Manual Setup Guide

This document provides comprehensive instructions for completing the SDLC implementation with workflow templates and detailed configuration steps.

## Workflow Template Installation

### 1. Copy Workflow Templates

Copy the comprehensive workflow templates from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy all workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/

# Commit the workflows
git add .github/workflows/
git commit -m "ci: add comprehensive GitHub Actions workflows for SDLC automation

- Add CI/CD pipeline with multi-Python version testing
- Add comprehensive security scanning with SARIF uploads
- Add automated release management with semantic versioning
- Add automated dependency updates with security validation

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin main
```

### 2. Workflow Configuration Details

#### CI Workflow (`ci.yml`)
- **Multi-Python Testing**: Tests across Python 3.8-3.11
- **Service Dependencies**: PostgreSQL and Redis for integration tests
- **Security Integration**: Bandit, Safety, and secret scanning
- **Container Scanning**: Trivy vulnerability scanning
- **Performance Testing**: Automated benchmarks on PRs
- **Artifact Management**: Build artifacts and reports

#### Security Scanning (`security-scan.yml`)
- **Daily Scheduling**: Automated security scans at 2 AM UTC
- **Comprehensive Coverage**: Dependencies, secrets, containers, and code
- **SARIF Integration**: Results uploaded to GitHub Security tab
- **SBOM Generation**: Software Bill of Materials for compliance
- **Automated Alerting**: Slack and GitHub issue creation

#### Release Management (`release.yml`)
- **Semantic Versioning**: Automated version management
- **Multi-Stage Validation**: Quality gates before release
- **Multi-Registry Publishing**: PyPI and container registries
- **Security Scanning**: Pre-release vulnerability checks
- **Automated Notifications**: Team notifications and changelog

#### Dependency Management (`dependency-update.yml`)
- **Automated Updates**: Weekly dependency updates
- **Security Prioritization**: Vulnerability-driven updates
- **Automated Testing**: Full test suite validation
- **PR Generation**: Automated pull requests with detailed summaries

## Advanced Configuration

### 1. CodeQL Analysis

Create `.github/workflows/codeql.yml`:

```yaml
name: "CodeQL Analysis"

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '20 14 * * 1'

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
        queries: +security-and-quality

    - name: Autobuild
      uses: github/codeql-action/autobuild@v2

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      with:
        category: "/language:${{matrix.language}}"
```

### 2. Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "UTC"
    reviewers:
      - "@security-team"
      - "@team-leads"
    assignees:
      - "@devops-team"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "chore"
      prefix-development: "chore"
      include: "scope"
    
  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "tuesday"
      time: "09:00"
    reviewers:
      - "@devops-team"
    labels:
      - "dependencies"
      - "docker"
    
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "wednesday"
      time: "09:00"
    reviewers:
      - "@devops-team"
    labels:
      - "dependencies"
      - "github-actions"
```

## Repository Secrets Configuration

### Required Secrets

Configure these secrets in `Settings > Secrets and variables > Actions`:

#### Production Secrets
```bash
# PyPI publishing
PYPI_API_TOKEN=pypi-xxxxxxxxxxxx

# GitHub operations
DEPENDENCY_UPDATE_TOKEN=ghp_xxxxxxxxxxxx

# Notifications
SECURITY_SLACK_WEBHOOK=https://hooks.slack.com/services/xxx/xxx/xxx
RELEASE_SLACK_WEBHOOK=https://hooks.slack.com/services/xxx/xxx/xxx

# Container registry (if using private registry)
CONTAINER_REGISTRY_USERNAME=username
CONTAINER_REGISTRY_PASSWORD=password
```

#### Optional Secrets
```bash
# Coverage reporting
CODECOV_TOKEN=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Monitoring
GRAFANA_API_KEY=eyJrIjoixxxxxxxx
PROMETHEUS_WEBHOOK=https://your-prometheus.com/api/v1/alerts

# Testing
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TEST_DATABASE_URL=postgresql://user:pass@host:port/testdb
```

## Security Configuration

### 1. Branch Protection Rules

Configure these rules for the `main` branch in `Settings > Branches`:

#### Protection Rules
- ✅ **Require pull request reviews before merging**
  - Required number of reviewers: `2` (for production repos)
  - Dismiss stale reviews when new commits are pushed: ✅
  - Require review from code owners: ✅
  - Restrict reviews to users with read access: ✅

- ✅ **Require status checks to pass before merging**
  - Require branches to be up to date before merging: ✅
  - Required status checks:
    - `lint`
    - `test (3.11)`
    - `security`
    - `build`
    - `docker`
    - `CodeQL`

- ✅ **Require conversation resolution before merging**
- ✅ **Require signed commits**
- ✅ **Require linear history**
- ✅ **Restrict pushes that create files with a path length > 100**
- ✅ **Restrict force pushes**
- ❌ **Allow deletions**

#### Administrator Settings
- ✅ **Include administrators** (enforce rules on admins)

### 2. Security Features

Enable in `Settings > Security`:

#### Security Advisories
- ✅ **Private vulnerability reporting**
- ✅ **Dependency graph**
- ✅ **Dependabot alerts**
- ✅ **Dependabot security updates**

#### Code Scanning
- ✅ **Code scanning alerts**
- ✅ **Push protection for secrets**
- ✅ **Secret scanning alerts**

## Team Configuration

### 1. Code Owners

Ensure `.github/CODEOWNERS` contains:

```
# Global code ownership
* @team-leads @senior-developers

# Security-sensitive files
/sql_synthesizer/security*.py @security-team @team-leads
/docs/security/ @security-team
/.github/workflows/ @devops-team @security-team
/Dockerfile @devops-team
/docker-compose*.yml @devops-team

# Documentation
/docs/ @tech-writers @team-leads
/README.md @tech-writers @team-leads
/*.md @tech-writers

# Configuration files
/pyproject.toml @team-leads
/requirements*.txt @security-team @team-leads
/.pre-commit-config.yaml @devops-team

# Testing
/tests/ @qa-team @developers
/benchmarks/ @performance-team

# Infrastructure
/infrastructure/ @devops-team @sre-team
/monitoring/ @sre-team
```

### 2. Team Permissions

Configure team access in `Settings > Manage access`:

#### Teams and Permissions
- **@team-leads**: Admin access
- **@senior-developers**: Maintain access
- **@developers**: Write access
- **@security-team**: Maintain access (security focus)
- **@devops-team**: Maintain access (infrastructure focus)
- **@qa-team**: Write access
- **@contributors**: Read access

## Monitoring and Alerting

### 1. GitHub Repository Settings

Configure in `Settings > General`:

#### Features
- ✅ **Issues**: For bug reports and feature requests
- ✅ **Projects**: For project management
- ✅ **Wiki**: For additional documentation
- ✅ **Discussions**: For community engagement
- ✅ **Sponsorships**: If accepting sponsorships

#### Data Services
- ✅ **Dependency graph**
- ✅ **Dependency insights**

### 2. Notification Configuration

#### Email Notifications
Configure email notifications for:
- Security vulnerability alerts
- Failed CI builds on main branch
- Release announcements
- Critical dependency updates

#### Slack Integration
Set up Slack channels for:
- `#security-alerts`: Security scan results
- `#releases`: Release notifications
- `#ci-cd`: Build failures and successes
- `#dependencies`: Dependency update notifications

## Validation and Testing

### 1. Workflow Testing

After configuration, validate the setup:

```bash
# Test CI pipeline
git checkout -b test-ci-pipeline
echo "# Test CI" >> test-ci.md
git add test-ci.md
git commit -m "test: validate CI pipeline functionality"
git push origin test-ci-pipeline
gh pr create --title "Test: CI Pipeline Validation" --body "Testing comprehensive CI/CD pipeline"
```

### 2. Security Testing

Validate security features:

```bash
# Test secret scanning (should be caught)
echo "password=secret123" >> test-secret.txt
git add test-secret.txt
git commit -m "test: trigger secret scanning"
# This should be blocked by push protection

# Test dependency scanning
# Add a known vulnerable dependency temporarily
echo "django==1.0" >> requirements.txt
git add requirements.txt
git commit -m "test: trigger dependency alert"
```

### 3. Release Testing

Test the release process:

```bash
# Create a test release
git tag v0.2.3-test
git push origin v0.2.3-test
# Monitor release workflow execution
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Workflow Permission Errors
```yaml
# Add to workflow if permissions issues occur
permissions:
  contents: read
  security-events: write
  actions: read
  checks: write
  pull-requests: write
```

#### 2. Secret Scanning False Positives
```yaml
# Add to .gitignore for test files
test_secrets.txt
*.example
docs/examples/
```

#### 3. Dependency Update Conflicts
```yaml
# Configure dependabot ignore patterns
ignore:
  - dependency-name: "package-name"
    versions: ["1.x", "2.x"]
```

#### 4. SARIF Upload Failures
- Ensure CodeQL and security scanning actions have `security-events: write` permission
- Verify SARIF files are properly formatted
- Check organization security settings

## Maintenance

### 1. Regular Reviews

Schedule regular reviews:
- **Weekly**: Security scan results
- **Monthly**: Dependency update effectiveness
- **Quarterly**: Workflow optimization
- **Annually**: Complete security audit

### 2. Updates and Improvements

Keep the SDLC implementation current:
- Update workflow action versions quarterly
- Review and update security policies
- Optimize build performance
- Add new security scanning tools as available

## Completion Verification

### Final Checklist

- [ ] All workflow templates copied and committed
- [ ] Branch protection rules configured
- [ ] Repository secrets configured
- [ ] Security features enabled
- [ ] Dependabot configuration added
- [ ] CodeQL analysis configured
- [ ] Team permissions set
- [ ] Code owners file updated
- [ ] Notification integrations configured
- [ ] Test PR created and validated
- [ ] Security scanning verified
- [ ] Release process tested
- [ ] Documentation updated

### Success Criteria

The SDLC implementation is complete when:
1. ✅ All CI checks pass on test PRs
2. ✅ Security scans run automatically and report results
3. ✅ Branch protection prevents direct pushes to main
4. ✅ Dependency updates create PRs automatically
5. ✅ Release process works end-to-end
6. ✅ Team receives notifications correctly
7. ✅ All documentation is accessible and current

Once all criteria are met, the repository has full enterprise-grade SDLC automation!