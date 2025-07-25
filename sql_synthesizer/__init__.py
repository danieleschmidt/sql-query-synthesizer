"""Main package for SQL Query Synthesizer."""

from .query_agent import QueryAgent
from .types import QueryResult
from .cache import TTLCache
from .openai_adapter import OpenAIAdapter
from .llm_interface import LLMProvider, ProviderError, ProviderTimeoutError, ProviderAuthenticationError
from .generator import naive_generate_sql
from . import metrics

# Optional webapp import (requires Flask)
try:
    from .webapp import create_app
    _webapp_available = True
except ImportError:
    _webapp_available = False
    create_app = None

__all__ = [
    "QueryAgent",
    "QueryResult",
    "TTLCache",
    "OpenAIAdapter",
    "LLMProvider",
    "ProviderError",
    "ProviderTimeoutError", 
    "ProviderAuthenticationError",
    "naive_generate_sql",
    "metrics",
]

if _webapp_available:
    __all__.append("create_app")
