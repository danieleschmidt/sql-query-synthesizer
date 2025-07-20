# Project Backlog - Impact-Ranked (WSJF)

## Current Status (Updated 2025-07-20)
- Test Coverage: 83% ✅ (58 tests, comprehensive coverage)
- CI Pipeline: ✅ Configured and stable
- Code Quality: All tests passing, ruff clean
- Security: ✅ Pre-commit hooks active, input validation implemented
- Dependencies: ✅ All declared correctly
- User Experience: ✅ Friendly errors, helpful CLI, interactive mode
- Observability: ✅ Structured logging with trace IDs and JSON formatting
- Architecture: ⚠️ Some technical debt identified (tight coupling, hardcoded values)

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

## Completed in Current Iteration ✅
1. ✅ **Optimize caching layer performance** - Enhanced TTLCache with LRU eviction, automatic cleanup, and comprehensive metrics
2. ✅ **Add comprehensive metrics and monitoring** - Added error rates, database metrics, OpenAI tracking, and production monitoring

## High Impact / Medium Effort (WSJF Score: High)

### 🏗️ Configuration Management & Hardcoded Values
1. **Create centralized Configuration class** (Impact: High, Effort: Medium)
   - Extract all hardcoded values (timeouts, ports, limits, buckets) to configurable settings
   - Support environment variable overrides with validation
   - Files: sql_synthesizer/query_agent.py, webapp.py, metrics.py
   - WSJF Score: 8/10 (reduces maintenance burden, improves deployment flexibility)

2. **Extract HTML templates from Python code** (Impact: Medium, Effort: Low)
   - Move embedded HTML PAGE variable to separate template files
   - Add proper template escaping and security headers
   - File: sql_synthesizer/webapp.py
   - Security improvement with low effort

### 🔧 Architecture Improvements
3. **Implement Service Layer pattern** (Impact: High, Effort: Medium)
   - Split QueryAgent into focused services (QueryService, SQLGenerator, QueryValidator)
   - Reduce tight coupling and improve testability
   - File: sql_synthesizer/query_agent.py (currently 200+ lines with multiple responsibilities)
   - WSJF Score: 7/10

## Medium Impact / High Effort (WSJF Score: Low-Medium)

### 🔧 Advanced Architecture
1. **Abstract OpenAI adapter interface** (Impact: High, Effort: High)
   - Development plan epic (DEVELOPMENT_PLAN.md:66)
   - Enables multi-provider LLM support (Claude, Gemini, etc.)
   - File: sql_synthesizer/openai_adapter.py
   - Major architectural change, high complexity

2. **Add configuration for cache backend** (Impact: Low, Effort: Medium)
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