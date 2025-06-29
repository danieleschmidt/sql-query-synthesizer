"""Prometheus metrics for query execution."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# Total number of queries executed by type (generated or raw SQL)
QUERIES_TOTAL = Counter("queries_total", "Number of queries executed", ["type"])

# Execution duration of queries in seconds
QUERY_DURATION = Histogram(
    "query_duration_seconds",
    "Query execution latency",
    buckets=(0.01, 0.1, 0.5, 1, 2, 5),
)


def record_query(duration: float, qtype: str) -> None:
    """Record a completed query with *duration* and *qtype*."""
    QUERIES_TOTAL.labels(type=qtype).inc()
    QUERY_DURATION.observe(duration)
