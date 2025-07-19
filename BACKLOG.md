# Project Backlog - Impact-Ranked (WSJF)

## Current Status (Updated 2025-07-19)
- Test Coverage: 86% ✅ (Target: >85% ACHIEVED)
- CI Pipeline: ✅ Configured and stable
- Code Quality: All tests passing, ruff clean
- Security: ✅ Pre-commit hooks active, input validation implemented
- Dependencies: ✅ All declared correctly

## Completed in Previous Iteration ✅
1. ✅ **Fixed python-dotenv dependency** - Added to setup.py
2. ✅ **Installed pre-commit hooks** - Secret scanning active
3. ✅ **Enhanced input validation** - SQL injection prevention implemented
4. ✅ **Improved test coverage to 86%** - Exceeded 85% target
5. ✅ **Comprehensive security test suite** - 9 new security tests

## High Impact / Low Effort (WSJF Score: High)

### 🔍 Observability & Logging
1. **Add structured logging with trace IDs** (Impact: Medium, Effort: Medium)
   - Development plan requirement (DEVELOPMENT_PLAN.md:64)
   - Improves observability and debugging
   - Current logging is basic, lacks correlation IDs

2. **Add JSON log option** (Impact: Low, Effort: Low)
   - Development plan requirement (DEVELOPMENT_PLAN.md:63)
   - Enables structured log analysis
   - Easy configuration flag

### 🚀 Performance & UX  
3. **Enhance error messages and CLI UX** (Impact: Medium, Effort: Medium)
   - Development plan requirement (DEVELOPMENT_PLAN.md:58)
   - Current error messages are technical, not user-friendly
   - Improve user experience

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