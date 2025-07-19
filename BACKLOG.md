# Project Backlog - Impact-Ranked (WSJF)

## Current Status
- Test Coverage: 80% ✅ (Target: >80%)
- CI Pipeline: ✅ Configured
- Code Quality: Tests passing, ruff configured
- Security: Missing pre-commit hooks for secret scanning

## High Impact / Low Effort (WSJF Score: High)

### 🔒 Security & Compliance
1. **Fix missing python-dotenv in setup.py** (Impact: Medium, Effort: Low)
   - Missing dependency causing CI failures
   - Required for dotenv functionality in query_agent.py:6
   - File: setup.py:7

2. **Install pre-commit hooks for secret scanning** (Impact: High, Effort: Low)
   - Development plan calls for this (DEVELOPMENT_PLAN.md:48)
   - Prevents secrets from being committed
   - CLI docs mention this is configured

3. **Add input validation and sanitization** (Impact: High, Effort: Medium)
   - Critical for SQL injection prevention
   - Current query generation lacks proper validation
   - File: sql_synthesizer/query_agent.py

### 🚀 Performance & UX
4. **Improve test coverage to >85%** (Impact: Medium, Effort: Medium)
   - Current: 80%, Target: >85% (DEVELOPMENT_PLAN.md:77)
   - Missing coverage in webapp.py:56-65,69 and query_agent.py critical paths

5. **Add structured logging with trace IDs** (Impact: Medium, Effort: Medium)
   - Development plan requirement (DEVELOPMENT_PLAN.md:64)
   - Improves observability and debugging

## Medium Impact / Medium Effort (WSJF Score: Medium)

### 🔧 Architecture & Refactoring
6. **Abstract OpenAI adapter interface** (Impact: High, Effort: High)
   - Development plan epic (DEVELOPMENT_PLAN.md:66)
   - Enables multi-provider LLM support
   - File: sql_synthesizer/openai_adapter.py

7. **Optimize caching layer performance** (Impact: Medium, Effort: Medium)
   - Development plan requirement (DEVELOPMENT_PLAN.md:55)
   - Cache miss on sql_synthesizer/cache.py:29

8. **Add configuration for cache backend** (Impact: Low, Effort: Medium)
   - Development plan requirement (DEVELOPMENT_PLAN.md:56)
   - Currently hardcoded TTLCache

### 🔍 Observability
9. **Enhance error messages and CLI UX** (Impact: Medium, Effort: Medium)
   - Development plan requirement (DEVELOPMENT_PLAN.md:58)
   - Improve user experience

10. **Add JSON log option** (Impact: Low, Effort: Low)
    - Development plan requirement (DEVELOPMENT_PLAN.md:63)
    - Enables structured log analysis

## Low Priority / Future Iterations

### 🌐 Features & Expansion
11. **Add responsive HTML templates** (Impact: Low, Effort: Medium)
    - Development plan requirement (DEVELOPMENT_PLAN.md:59)
    - Web UI improvements

12. **Deploy demo environment** (Impact: Low, Effort: High)
    - Development plan requirement (DEVELOPMENT_PLAN.md:41)
    - Marketing/demo value

## Technical Debt Log

### Code Quality Issues
- Missing type hints in some functions
- Hardcoded timeouts and retry logic
- No connection pooling for database operations
- No graceful error handling for LLM API failures

### Security Concerns
- SQL injection risk if user input not properly sanitized
- API keys might be logged in debug mode
- No rate limiting on API endpoints

### Performance Issues
- No async support for I/O operations
- Cache invalidation strategy needs improvement
- No query result pagination

## Next Actions (Iteration 1 Focus)
1. Fix python-dotenv dependency issue
2. Install and configure pre-commit hooks
3. Add input validation for SQL injection prevention
4. Improve test coverage to 85%
5. Add structured logging

## Success Metrics
- [ ] Test coverage >85%
- [ ] CI pipeline <15 min
- [ ] Error rate <5%
- [ ] 100% secrets from environment
- [ ] Pre-commit hooks active
- [ ] Zero high/critical security vulnerabilities