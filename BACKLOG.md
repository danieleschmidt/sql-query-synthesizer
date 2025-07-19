# Project Backlog - Impact-Ranked (WSJF)

## Current Status (Updated 2025-07-19)
- Test Coverage: 83% ✅ (58 tests, comprehensive coverage)
- CI Pipeline: ✅ Configured and stable
- Code Quality: All tests passing, ruff clean
- Security: ✅ Pre-commit hooks active, input validation implemented
- Dependencies: ✅ All declared correctly
- User Experience: ✅ Friendly errors, helpful CLI, interactive mode
- Observability: ✅ Structured logging with trace IDs and JSON formatting

## Completed in Previous Iterations ✅
1. ✅ **Fixed python-dotenv dependency** - Added to setup.py
2. ✅ **Installed pre-commit hooks** - Secret scanning active
3. ✅ **Enhanced input validation** - SQL injection prevention implemented
4. ✅ **Improved test coverage to 86%** - Exceeded 85% target
5. ✅ **Comprehensive security test suite** - 9 new security tests
6. ✅ **Structured logging with trace IDs** - Request correlation implemented
7. ✅ **JSON log formatting** - CLI and environment configuration added
8. ✅ **User-friendly error messages** - Contextual suggestions and help
9. ✅ **Enhanced CLI UX** - Examples, patterns, and better interactive mode

## High Impact / Medium Effort (WSJF Score: Medium-High)

### 🔧 Architecture & Refactoring
1. **Optimize caching layer performance** (Impact: Medium, Effort: Medium)
   - Development plan requirement (DEVELOPMENT_PLAN.md:55)
   - Current caching can be improved with better invalidation strategies
   - Add cache hit/miss metrics for monitoring

### 🔍 Observability & Monitoring
2. **Add comprehensive metrics and monitoring** (Impact: Medium, Effort: Medium)
   - Expand beyond basic query metrics
   - Add cache hit rates, error rates, response times
   - Enable better production monitoring

## Medium Impact / High Effort (WSJF Score: Low-Medium)

### 🔧 Advanced Architecture
3. **Abstract OpenAI adapter interface** (Impact: High, Effort: High)
   - Development plan epic (DEVELOPMENT_PLAN.md:66)
   - Enables multi-provider LLM support (Claude, Gemini, etc.)
   - File: sql_synthesizer/openai_adapter.py
   - Major architectural change, high complexity

4. **Add configuration for cache backend** (Impact: Low, Effort: Medium)
   - Development plan requirement (DEVELOPMENT_PLAN.md:56)
   - Support Redis, Memcached for distributed caching
   - Currently hardcoded TTLCache

## Low Priority / Future Iterations

### 🌐 Features & Expansion
5. **Add responsive HTML templates** (Impact: Low, Effort: Medium)
   - Development plan requirement (DEVELOPMENT_PLAN.md:59)
   - Web UI improvements for better mobile/tablet experience

6. **Deploy demo environment** (Impact: Low, Effort: High)
   - Development plan requirement (DEVELOPMENT_PLAN.md:41)
   - Public demo for marketing and user onboarding

7. **Add async support for I/O operations** (Impact: Medium, Effort: High)
   - Improve performance for concurrent database operations
   - Enable better scalability under load

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

## Success Metrics - Current Status
- ✅ Test coverage >80% (Currently: 83% with 58 tests)
- ✅ CI pipeline stable and automated
- ✅ 100% secrets from environment (no hardcoded credentials)
- ✅ Pre-commit hooks active (secret scanning)
- ✅ Zero high/critical security vulnerabilities
- ✅ User-friendly error messages with suggestions
- ✅ Structured logging with trace IDs
- ✅ Comprehensive input validation and sanitization

## Next Iteration Focus (Iteration 2)
1. **Optimize caching layer** - Add metrics, improve invalidation
2. **Expand monitoring** - Cache hit rates, error rates, response times
3. **Consider multi-provider LLM support** - If high user demand