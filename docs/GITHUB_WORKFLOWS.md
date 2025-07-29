# GitHub Actions Workflow Templates

This directory contains documentation for recommended GitHub Actions workflows for the SQL Query Synthesizer project.

## Core Workflows Required

### 1. Continuous Integration (`ci.yml`)
```yaml
name: CI Pipeline
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests with coverage
      run: |
        pytest --cov=sql_synthesizer --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        files: ./coverage.xml
```

### 2. Security Scanning (`security.yml`)
```yaml
name: Security Scan
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run Bandit Security Scanner
      uses: securecodewarrior/github-action-bandit@v1
      with:
        path: "sql_synthesizer"
    
    - name: Run Safety Check
      run: |
        pip install safety
        safety check --json --output safety-report.json
    
    - name: Run Semgrep
      uses: returntocorp/semgrep-action@v1
      with:
        config: auto
```

### 3. Code Quality (`quality.yml`)
```yaml
name: Code Quality
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    
    - name: Run pre-commit
      uses: pre-commit/action@v3.0.0
    
    - name: Run SonarCloud Scan
      uses: SonarSource/sonarcloud-github-action@master
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

### 4. Performance Testing (`performance.yml`)
```yaml
name: Performance Tests
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  performance:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v4
    - name: Run performance benchmarks
      run: |
        pip install -e .[dev]
        python -m pytest tests/performance/ --benchmark-json=benchmark.json
    
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

### 5. Container Security (`container-security.yml`)
```yaml
name: Container Security
on:
  push:
    branches: [ main ]
    paths: [ 'Dockerfile', 'docker-compose.yml' ]

jobs:
  container-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -t sql-synthesizer:latest .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'sql-synthesizer:latest'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

## Deployment Workflows

### 6. Release Management (`release.yml`)
```yaml
name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
    
    - name: Build and publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

## Advanced Workflows

### 7. Mutation Testing (`mutation.yml`)
```yaml
name: Mutation Testing
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  mutation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run Mutation Testing
      run: |
        pip install mutmut
        mutmut run --paths-to-mutate sql_synthesizer/
        mutmut html
    
    - name: Upload mutation report
      uses: actions/upload-artifact@v3
      with:
        name: mutation-report
        path: html/
```

### 8. Dependency Updates (`dependency-update.yml`)
```yaml
name: Dependency Update Check
on:
  schedule:
    - cron: '0 9 * * 1'  # Weekly on Monday

jobs:
  update-deps:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Check for outdated packages
      run: |
        pip install pip-check-updates
        pcu --show-outdated
    
    - name: Create update PR
      uses: peter-evans/create-pull-request@v5
      with:
        title: "chore: update dependencies"
        body: "Automated dependency updates"
        branch: dependency-updates
```

## Workflow Integration Notes

1. **Secrets Required**:
   - `CODECOV_TOKEN`: For coverage reporting
   - `SONAR_TOKEN`: For SonarCloud analysis
   - `PYPI_API_TOKEN`: For package publishing

2. **Branch Protection**:
   - Enable required status checks for all CI workflows
   - Require review from CODEOWNERS
   - Enable automatic security updates

3. **Workflow Triggers**:
   - All quality checks run on PRs
   - Security scans run weekly
   - Performance tests run on main branch changes
   - Release workflows trigger on version tags

4. **Caching Strategy**:
   - Cache pip dependencies between runs
   - Cache pre-commit environments
   - Cache Docker layers for faster builds

## Manual Setup Required

1. Configure repository secrets in GitHub Settings
2. Enable GitHub Advanced Security features
3. Set up SonarCloud project integration
4. Configure Codecov project settings
5. Set up PyPI trusted publishing (recommended)

See [SETUP_REQUIRED.md](../docs/SETUP_REQUIRED.md) for detailed configuration instructions.