# Workflow Requirements

## Overview

This document outlines the GitHub Actions workflows required for comprehensive SDLC automation.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)
- **Triggers**: Push to main, pull requests
- **Jobs**: lint, test, security scan, build
- **Required Actions**: 
  - `actions/checkout@v4`
  - `actions/setup-python@v4`
  - Run `make ci-test` and `make ci-security`

### 2. Security Scanning (`security.yml`)
- **Schedule**: Daily at 2 AM UTC
- **Jobs**: dependency audit, secret scanning, container scanning
- **Tools**: Bandit, Safety, Trivy

### 3. Release Management (`release.yml`)
- **Triggers**: Tag creation (`v*`)
- **Jobs**: build, publish to PyPI, create GitHub release
- **Requirements**: PyPI token in secrets

### 4. Documentation (`docs.yml`)
- **Triggers**: Push to main (docs changes)
- **Jobs**: build and deploy documentation
- **Target**: GitHub Pages

## Branch Protection Requirements

Configure these rules manually in GitHub settings:

- Require pull request reviews (1 reviewer minimum)
- Require status checks to pass before merging
- Require up-to-date branches before merging
- Restrict pushes to main branch

## Manual Setup Required

See [SETUP_REQUIRED.md](../SETUP_REQUIRED.md) for detailed manual configuration steps.