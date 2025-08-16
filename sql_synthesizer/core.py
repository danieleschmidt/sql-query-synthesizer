"""Core functionality and utilities for SQL Query Synthesizer.

This module provides core functionality that supports the main query processing
pipeline including result formatting, error handling, and system utilities.
"""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types import QueryResult


@dataclass
class SystemInfo:
    """System information and status."""

    version: str
    uptime: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    cache_hit_rate: float
    memory_usage_mb: Optional[float] = None


@dataclass
class QueryMetadata:
    """Metadata for query execution tracking."""

    query_id: str
    timestamp: datetime
    duration_ms: float
    cache_hit: bool
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None


class ResultFormatter:
    """Formats query results for different output types."""

    @staticmethod
    def to_dict(result: QueryResult) -> Dict[str, Any]:
        """Convert QueryResult to dictionary."""
        # Extract columns from first row of data if available
        columns = []
        if result.data and isinstance(result.data[0], dict):
            columns = list(result.data[0].keys())

        return {
            "sql": result.sql,
            "data": result.data,
            "explanation": result.explanation,
            "columns": columns,
            "row_count": len(result.data) if result.data else 0,
            "query_time_ms": getattr(result, "query_time_ms", None),
        }

    @staticmethod
    def to_csv_rows(result: QueryResult) -> List[List[str]]:
        """Convert QueryResult to CSV-compatible rows."""
        if not result.data:
            return []

        # Extract columns from first row of data if available
        columns = []
        if result.data and isinstance(result.data[0], dict):
            columns = list(result.data[0].keys())

        if not columns:
            return []

        rows = [columns]  # Header row
        for row in result.data:
            csv_row = []
            for col in columns:
                value = row.get(col, "")
                # Convert to string and escape commas/quotes
                str_value = str(value) if value is not None else ""
                if "," in str_value or '"' in str_value:
                    str_value = f'"{str_value.replace('"', '""')}"'
                csv_row.append(str_value)
            rows.append(csv_row)

        return rows

    @staticmethod
    def format_explanation(sql: str, explanation: str) -> str:
        """Format query explanation with SQL for display."""
        return f"""
SQL Query:
{sql}

Explanation:
{explanation}
""".strip()


class QueryTracker:
    """Tracks query execution statistics and performance metrics."""

    def __init__(self):
        self._start_time = time.time()
        self._query_count = 0
        self._success_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._total_duration = 0.0

    def record_query(self, duration_ms: float, success: bool, cache_hit: bool = False):
        """Record query execution metrics."""
        self._query_count += 1
        self._total_duration += duration_ms

        if success:
            self._success_count += 1
        else:
            self._error_count += 1

        if cache_hit:
            self._cache_hits += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get current query execution statistics."""
        uptime = time.time() - self._start_time
        avg_duration = self._total_duration / max(self._query_count, 1)
        cache_hit_rate = self._cache_hits / max(self._query_count, 1) * 100

        return {
            "uptime_seconds": uptime,
            "total_queries": self._query_count,
            "successful_queries": self._success_count,
            "failed_queries": self._error_count,
            "success_rate": self._success_count / max(self._query_count, 1) * 100,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "average_duration_ms": avg_duration,
            "queries_per_second": self._query_count / max(uptime, 1),
        }

    def reset_statistics(self):
        """Reset all statistics counters."""
        self._start_time = time.time()
        self._query_count = 0
        self._success_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._total_duration = 0.0


class TraceIDGenerator:
    """Generates unique trace IDs for request correlation."""

    @staticmethod
    def generate(prefix: str = "trace") -> str:
        """Generate a unique trace ID with optional prefix."""
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def generate_query_id() -> str:
        """Generate a unique query ID."""
        return f"query-{uuid.uuid4().hex[:12]}"


class ErrorHandler:
    """Centralized error handling and logging."""

    @staticmethod
    def format_database_error(error: Exception) -> str:
        """Format database errors for user display."""
        error_msg = str(error).lower()

        if "connection" in error_msg:
            return "Database connection error. Please check your connection settings."
        elif "timeout" in error_msg:
            return "Database query timeout. Please try a simpler query."
        elif "permission" in error_msg or "access" in error_msg:
            return "Database access error. Please check your permissions."
        elif "syntax" in error_msg:
            return "SQL syntax error in generated query. Please rephrase your question."
        else:
            return "Database error occurred. Please try again or contact support."

    @staticmethod
    def format_llm_error(error: Exception) -> str:
        """Format LLM provider errors for user display."""
        error_msg = str(error).lower()

        if "timeout" in error_msg:
            return "Query generation timeout. Please try again."
        elif "authentication" in error_msg or "api key" in error_msg:
            return "LLM service authentication error. Please check configuration."
        elif "rate limit" in error_msg:
            return "Rate limit exceeded. Please try again later."
        else:
            return (
                "Query generation service error. Falling back to basic SQL generation."
            )


# Global instance for application-wide query tracking
query_tracker = QueryTracker()


def get_system_info() -> SystemInfo:
    """Get current system information and statistics."""
    stats = query_tracker.get_statistics()

    return SystemInfo(
        version="0.2.2",
        uptime=stats["uptime_seconds"],
        total_queries=stats["total_queries"],
        successful_queries=stats["successful_queries"],
        failed_queries=stats["failed_queries"],
        cache_hit_rate=stats["cache_hit_rate"],
    )


def create_query_metadata(
    query_id: str,
    duration_ms: float,
    cache_hit: bool = False,
    user_agent: Optional[str] = None,
    client_ip: Optional[str] = None,
) -> QueryMetadata:
    """Create query metadata for logging and tracking."""
    return QueryMetadata(
        query_id=query_id,
        timestamp=datetime.utcnow(),
        duration_ms=duration_ms,
        cache_hit=cache_hit,
        user_agent=user_agent,
        client_ip=client_ip,
    )
