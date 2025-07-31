# GitHub Actions CI/CD Setup Guide

This guide provides template workflows for the SQL Synthesizer project. Since GitHub Actions workflows cannot be created automatically, this document provides the exact files you need to create manually.

## Quick Setup Instructions

1. Create the `.github/workflows/` directory in your repository root
2. Copy each workflow template below into separate `.yml` files
3. Customize the configuration variables as needed
4. Commit and push to activate the workflows

## Required Secrets

Configure these in your GitHub repository settings → Secrets and Variables → Actions:

```yaml
OPENAI_API_KEY: "your-openai-api-key-here"
DOCKER_USERNAME: "your-docker-hub-username"
DOCKER_PASSWORD: "your-docker-hub-token"
CODECOV_TOKEN: "your-codecov-token"  # Optional
```

## Core Workflows

### 1. Continuous Integration (.github/workflows/ci.yml)

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
    
    - name: Run tests with coverage
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        pytest tests/ --cov=sql_synthesizer --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      if: matrix.python-version == '3.11'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        token: ${{ secrets.CODECOV_TOKEN }}

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
    
    - name: Run Bandit security scan
      run: bandit -r sql_synthesizer/ -f json -o bandit-report.json
    
    - name: Run Safety dependency scan
      run: safety check --json --output safety-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
```

### 2. Release Automation (.github/workflows/release.yml)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (e.g., v1.2.3)'
        required: true
        type: string

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    uses: ./.github/workflows/ci.yml
    secrets: inherit

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    
    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  publish-pypi:
    needs: test
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
```

### 3. Dependency Updates (.github/workflows/dependencies.yml)

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 6 * * 1'  # Monday 6 AM UTC
  workflow_dispatch:

jobs:
  update-deps:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install pip-tools
      run: pip install pip-tools
    
    - name: Update requirements
      run: |
        pip-compile --upgrade requirements.in
        pip-compile --upgrade requirements-dev.in
    
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'deps: update Python dependencies'
        title: 'Automated dependency updates'
        body: |
          This PR updates Python dependencies to their latest compatible versions.
          
          - Review changes carefully before merging
          - Run tests to ensure compatibility
          - Check for any breaking changes in dependencies
        branch: automated/dependency-updates
        labels: |
          dependencies
          automated
```

## Advanced Workflows

### 4. Performance Testing (.github/workflows/performance.yml)

```yaml
name: Performance Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
    paths:
      - 'sql_synthesizer/**'
      - 'benchmarks/**'
  schedule:
    - cron: '0 4 * * *'  # Daily at 4 AM UTC

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: benchmark_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run benchmarks
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/benchmark_db
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        python benchmarks/performance_benchmark.py
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: benchmark-results.json
```

## Workflow Configuration Tips

1. **Environment Variables**: Store sensitive data in GitHub Secrets
2. **Matrix Builds**: Test across multiple Python versions
3. **Caching**: Use action caching for pip and other dependencies
4. **Parallel Jobs**: Run tests and security scans in parallel
5. **Conditional Steps**: Use `if` conditions for optional steps
6. **Artifact Upload**: Save test reports and coverage data

## Monitoring and Alerts

Set up repository alerts for:
- Failed CI builds
- Security vulnerabilities found
- Coverage drops below threshold
- Performance regression detected

## Next Steps

1. Create the workflow files from templates above
2. Configure required secrets in repository settings
3. Test workflows with a small PR
4. Monitor and adjust based on your team's needs
5. Consider adding custom deployment workflows for your infrastructure