"""SQL Query Synthesizer - Natural language to SQL conversion with enterprise security.

The SQL Query Synthesizer provides a secure, scalable solution for converting natural
language queries into safe SQL statements with comprehensive validation, caching,
and monitoring capabilities.

Key Features:
- Multi-database support (PostgreSQL, MySQL, SQLite)
- Advanced SQL injection prevention
- High-performance async operations
- Enterprise-grade security and audit logging
- Prometheus metrics and health monitoring
"""

from .query_agent import QueryAgent
from .async_query_agent import AsyncQueryAgent
from .types import QueryResult
from .cache import TTLCache
from .openai_adapter import OpenAIAdapter
from .llm_interface import LLMProvider, ProviderError, ProviderTimeoutError, ProviderAuthenticationError
from .generator import naive_generate_sql
from . import metrics
from .core import (
    SystemInfo, QueryMetadata, ResultFormatter, QueryTracker, 
    TraceIDGenerator, ErrorHandler, get_system_info, create_query_metadata
)

# Version information
__version__ = "0.2.2"
__author__ = "SQL Synthesizer Team"
__license__ = "MIT"

# Optional webapp import (requires Flask)
try:
    from .webapp import create_app
    _webapp_available = True
except ImportError:
    _webapp_available = False
    create_app = None

# Optional quantum optimization (requires numpy)
try:
    from .quantum import QuantumSQLSynthesizer, QuantumQueryOptimizer, QuantumTaskScheduler
    _quantum_available = True
except ImportError:
    _quantum_available = False
    QuantumSQLSynthesizer = None
    QuantumQueryOptimizer = None
    QuantumTaskScheduler = None

# Optional security audit logging
try:
    from .security_audit import SecurityAuditLogger, security_audit_logger
    _security_audit_available = True
except ImportError:
    _security_audit_available = False
    SecurityAuditLogger = None
    security_audit_logger = None

__all__ = [
    "QueryAgent",
    "AsyncQueryAgent",
    "QueryResult",
    "TTLCache",
    "OpenAIAdapter",
    "LLMProvider",
    "ProviderError",
    "ProviderTimeoutError", 
    "ProviderAuthenticationError",
    "naive_generate_sql",
    "metrics",
    "SystemInfo",
    "QueryMetadata", 
    "ResultFormatter",
    "QueryTracker",
    "TraceIDGenerator",
    "ErrorHandler",
    "get_system_info",
    "create_query_metadata",
    "__version__",
    "__author__",
    "__license__",
]

if _webapp_available:
    __all__.append("create_app")

if _quantum_available:
    __all__.extend(["QuantumSQLSynthesizer", "QuantumQueryOptimizer", "QuantumTaskScheduler"])

if _security_audit_available:
    __all__.extend(["SecurityAuditLogger", "security_audit_logger"])
