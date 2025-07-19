"""Prometheus metrics for query execution and caching."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

# Total number of queries executed by type (generated or raw SQL)
QUERIES_TOTAL = Counter("queries_total", "Number of queries executed", ["type"])

# Execution duration of queries in seconds
QUERY_DURATION = Histogram(
    "query_duration_seconds",
    "Query execution latency",
    buckets=(0.01, 0.1, 0.5, 1, 2, 5),
)

# Cache metrics
CACHE_HITS_TOTAL = Counter("cache_hits_total", "Number of cache hits", ["cache_type"])
CACHE_MISSES_TOTAL = Counter("cache_misses_total", "Number of cache misses", ["cache_type"])
CACHE_EVICTIONS_TOTAL = Counter("cache_evictions_total", "Number of cache evictions", ["cache_type"])
CACHE_SIZE = Gauge("cache_size", "Current cache size", ["cache_type"])
CACHE_HIT_RATE = Gauge("cache_hit_rate", "Cache hit rate", ["cache_type"])


def record_query(duration: float, qtype: str) -> None:
    """Record a completed query with *duration* and *qtype*."""
    QUERIES_TOTAL.labels(type=qtype).inc()
    QUERY_DURATION.observe(duration)


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
