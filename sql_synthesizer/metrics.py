"""Prometheus metrics for query execution, caching, and system health."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge
from .config import config

# Query metrics
QUERIES_TOTAL = Counter("queries_total", "Number of queries executed", ["type"])
QUERY_ERRORS_TOTAL = Counter("query_errors_total", "Number of query errors", ["error_type"])
QUERY_DURATION = Histogram(
    "query_duration_seconds",
    "Query execution latency",
    buckets=config.openai_request_buckets,
)

# Input validation metrics
INPUT_VALIDATION_ERRORS_TOTAL = Counter(
    "input_validation_errors_total", 
    "Number of input validation errors", 
    ["validation_error_type"]
)

# OpenAI API metrics
OPENAI_REQUESTS_TOTAL = Counter("openai_requests_total", "Number of OpenAI API requests", ["status"])
OPENAI_REQUEST_DURATION = Histogram(
    "openai_request_duration_seconds",
    "OpenAI API request latency",
    buckets=config.database_query_buckets,
)

# Database metrics
DATABASE_CONNECTIONS_TOTAL = Counter("database_connections_total", "Number of database connections", ["status"])
DATABASE_QUERY_DURATION = Histogram(
    "database_query_duration_seconds",
    "Database query execution latency",
    buckets=config.cache_operation_buckets,
)

# Cache metrics
CACHE_HITS_TOTAL = Counter("cache_hits_total", "Number of cache hits", ["cache_type"])
CACHE_MISSES_TOTAL = Counter("cache_misses_total", "Number of cache misses", ["cache_type"])
CACHE_EVICTIONS_TOTAL = Counter("cache_evictions_total", "Number of cache evictions", ["cache_type"])
CACHE_SIZE = Gauge("cache_size", "Current cache size", ["cache_type"])
CACHE_HIT_RATE = Gauge("cache_hit_rate", "Cache hit rate", ["cache_type"])

# SQL Generation metrics
SQL_GENERATION_FALLBACKS_TOTAL = Counter(
    "sql_generation_fallbacks_total", 
    "Number of SQL generation fallbacks to naive mode", 
    ["provider"]
)
SQL_GENERATION_ERRORS_TOTAL = Counter(
    "sql_generation_errors_total", 
    "Number of SQL generation errors", 
    ["error_type"]
)


def record_query(duration: float, qtype: str) -> None:
    """Record a completed query with *duration* and *qtype*."""
    QUERIES_TOTAL.labels(type=qtype).inc()
    QUERY_DURATION.observe(duration)


def record_query_error(error_type: str) -> None:
    """Record a query error by type."""
    QUERY_ERRORS_TOTAL.labels(error_type=error_type).inc()


def record_input_validation_error(validation_error_type: str) -> None:
    """Record an input validation error by type."""
    INPUT_VALIDATION_ERRORS_TOTAL.labels(validation_error_type=validation_error_type).inc()


def record_openai_request(duration: float, status: str) -> None:
    """Record an OpenAI API request with duration and status."""
    OPENAI_REQUESTS_TOTAL.labels(status=status).inc()
    OPENAI_REQUEST_DURATION.observe(duration)


def record_database_connection(status: str) -> None:
    """Record a database connection attempt with status."""
    DATABASE_CONNECTIONS_TOTAL.labels(status=status).inc()


def record_database_query(duration: float) -> None:
    """Record database query execution time."""
    DATABASE_QUERY_DURATION.observe(duration)


def record_cache_hit(cache_type: str) -> None:
    """Record a cache hit for the specified cache type."""
    CACHE_HITS_TOTAL.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    """Record a cache miss for the specified cache type."""
    CACHE_MISSES_TOTAL.labels(cache_type=cache_type).inc()


def record_cache_eviction(cache_type: str) -> None:
    """Record a cache eviction for the specified cache type."""
    CACHE_EVICTIONS_TOTAL.labels(cache_type=cache_type).inc()


def update_cache_metrics(cache_type: str, cache_stats: dict) -> None:
    """Update all cache metrics from cache statistics."""
    CACHE_SIZE.labels(cache_type=cache_type).set(cache_stats["size"])
    CACHE_HIT_RATE.labels(cache_type=cache_type).set(cache_stats["hit_rate"])


def record_sql_generation_fallback(provider: str) -> None:
    """Record SQL generation fallback to naive mode."""
    SQL_GENERATION_FALLBACKS_TOTAL.labels(provider=provider).inc()


def record_sql_generation_error(error_type: str) -> None:
    """Record SQL generation error."""
    SQL_GENERATION_ERRORS_TOTAL.labels(error_type=error_type).inc()
