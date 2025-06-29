# 🧭 Project Vision

> A short 2–3 sentence description of what this repo does, for whom, and why.

# 📅 12-Week Roadmap

## I1 - Foundations

- **Themes**: Security, Refactoring
- **Goals / Epics**
  - Harden input validation and remove hardcoded credentials
  - Modularize query generation logic
  - Stabilize CI pipeline
- **Definition of Done**
  - Unit tests pass with >80% coverage
  - No plaintext secrets in repo
  - CI green on main for two consecutive runs

## I2 - Performance & UX

- **Themes**: Performance, Developer UX
- **Goals / Epics**
  - Optimize query caching and batch operations
  - Add user-friendly CLI prompts and web UI improvements
  - Introduce pre-commit hooks (ruff, pytest)
- **Definition of Done**
  - Average query latency <200ms on staging database
  - Pre-commit hooks required by CI
  - Web app templates responsive

## I3 - Observability & Expansion

- **Themes**: Observability, Features
- **Goals / Epics**
  - Expose structured logs and Prometheus metrics
  - Support multiple LLM providers behind adapter interface
  - Deploy demo environment
- **Definition of Done**
  - Metrics exported for all query paths
  - Adapter pattern documented and tested
  - Demo deploy script in /docs

# ✅ Epic & Task Checklist

### 🔒 Increment 1: Security & Refactoring
- [ ] [EPIC] Eliminate hardcoded secrets
  - [ ] Load from environment securely
  - [ ] Add `pre-commit` hook for scanning secrets
- [ ] [EPIC] Improve CI stability
  - [ ] Replace flaky integration tests
  - [ ] Enable parallel test execution

### 🚀 Increment 2: Performance & UX
- [ ] [EPIC] Optimize caching layer
  - [ ] Profile query timings
  - [ ] Add config for cache backend
- [ ] [EPIC] Enhance CLI & web UX
  - [ ] Better error messages
  - [ ] Responsive HTML templates

### 🔍 Increment 3: Observability & Features
- [ ] [EPIC] Structured logging
  - [ ] JSON log option
  - [ ] Trace IDs for each request
- [ ] [EPIC] Multi-provider LLM adapter
  - [ ] Abstract OpenAI adapter
  - [ ] Add tests for new provider

# ⚠️ Risks & Mitigation
- Limited test data → use containerized databases for CI
- LLM API quota issues → implement fallback and caching
- Security vulnerabilities from user SQL → strict validation and parameterization
- CI resource limits → run tests in parallel and cache deps
- Feature creep → lock scope per increment

# 📊 KPIs & Metrics
- [ ] >85% test coverage
- [ ] <15 min CI pipeline time
- [ ] <5% error rate on core service
- [ ] 100% secrets loaded from vault/env

# 👥 Ownership & Roles (Optional)
- **DevOps**: CI/CD, infrastructure
- **Backend**: QueryAgent, caching, adapters
- **Frontend**: Web UI templates
- **QA**: Tests and metrics validation
