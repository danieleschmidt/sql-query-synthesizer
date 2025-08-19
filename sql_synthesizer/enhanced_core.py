"""Enhanced Core Functionality with Advanced Reliability Patterns.

This module extends the core functionality with enterprise-grade reliability
patterns including circuit breakers, bulkheads, adaptive timeouts, and
self-healing mechanisms for production environments.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Union

from .core import (
    ErrorHandler,
    QueryMetadata,
    QueryTracker,
    ResultFormatter,
    SystemInfo,
)
from .robust_error_handling import (
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    RobustErrorHandler,
)
from .types import QueryResult

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class AdvancedSystemMetrics:
    """Advanced system metrics for enhanced monitoring."""

    # Performance metrics
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput_per_second: float = 0.0

    # Reliability metrics
    error_rate: float = 0.0
    circuit_breaker_trips: int = 0
    fallback_activations: int = 0
    timeout_count: int = 0

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    queue_depth: int = 0

    # Business metrics
    sql_generation_success_rate: float = 0.0
    cache_efficiency: float = 0.0
    user_satisfaction_score: float = 0.0


@dataclass
class ReliabilityConfiguration:
    """Configuration for reliability features."""

    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    circuit_breaker_half_open_max_calls: int = 3

    # Timeout settings
    default_timeout: float = 30.0
    adaptive_timeout_enabled: bool = True
    max_timeout: float = 120.0
    min_timeout: float = 5.0

    # Retry settings
    max_retries: int = 3
    retry_backoff_multiplier: float = 2.0
    retry_max_delay: float = 30.0

    # Bulkhead settings
    max_concurrent_queries: int = 100
    high_priority_queue_size: int = 20
    normal_priority_queue_size: int = 50
    low_priority_queue_size: int = 30

    # Health check settings
    health_check_interval: float = 60.0
    dependency_timeout: float = 10.0


@dataclass
class EnhancedSystemInfo(SystemInfo):
    """Extended system information with performance metrics."""

    active_connections: int = 0
    peak_connections: int = 0
    avg_response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    disk_usage_mb: Optional[float] = None
    system_load: Optional[float] = None


@dataclass
class EnhancedQueryMetadata(QueryMetadata):
    """Extended query metadata with additional tracking."""

    complexity_score: float = 0.0
    estimated_cost: float = 0.0
    optimization_applied: bool = False
    security_level: str = "standard"
    data_source: str = "primary"


class EnhancedQueryTracker(QueryTracker):
    """Enhanced query tracker with advanced metrics and reliability features."""

    def __init__(self, config: ReliabilityConfiguration = None):
        super().__init__()
        self.config = config or ReliabilityConfiguration()
        self._response_times: List[float] = []
        self._error_counts = defaultdict(int)
        self._circuit_breaker_trips = 0
        self._fallback_activations = 0
        self._timeout_count = 0
        self._last_cleanup = time.time()
        self._metrics_lock = threading.Lock()

    def record_advanced_query(
        self,
        duration_ms: float,
        success: bool,
        cache_hit: bool = False,
        error_category: Optional[ErrorCategory] = None,
        circuit_breaker_tripped: bool = False,
        fallback_used: bool = False,
        timed_out: bool = False,
    ):
        """Record advanced query metrics with reliability information."""
        with self._metrics_lock:
            # Record basic metrics
            self.record_query(duration_ms, success, cache_hit)

            # Record response time
            self._response_times.append(duration_ms)

            # Track reliability metrics
            if error_category:
                self._error_counts[error_category] += 1

            if circuit_breaker_tripped:
                self._circuit_breaker_trips += 1

            if fallback_used:
                self._fallback_activations += 1

            if timed_out:
                self._timeout_count += 1

            # Periodic cleanup to prevent memory growth
            if time.time() - self._last_cleanup > 300:  # 5 minutes
                self._cleanup_old_metrics()

    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory growth."""
        # Keep only last 1000 response times
        if len(self._response_times) > 1000:
            self._response_times = self._response_times[-1000:]

        self._last_cleanup = time.time()

    def get_advanced_statistics(self) -> AdvancedSystemMetrics:
        """Get comprehensive system metrics."""
        stats = self.get_statistics()

        # Calculate percentiles
        response_times = sorted(self._response_times) if self._response_times else [0]
        p95_index = int(len(response_times) * 0.95)
        p99_index = int(len(response_times) * 0.99)

        p95_response = response_times[min(p95_index, len(response_times) - 1)]
        p99_response = response_times[min(p99_index, len(response_times) - 1)]

        # Calculate error rate
        total_queries = self._query_count
        error_rate = (self._error_count / max(total_queries, 1)) * 100

        return AdvancedSystemMetrics(
            avg_response_time=stats["average_duration_ms"],
            p95_response_time=p95_response,
            p99_response_time=p99_response,
            throughput_per_second=stats["queries_per_second"],
            error_rate=error_rate,
            circuit_breaker_trips=self._circuit_breaker_trips,
            fallback_activations=self._fallback_activations,
            timeout_count=self._timeout_count,
            sql_generation_success_rate=stats["success_rate"],
            cache_efficiency=stats["cache_hit_rate"],
        )


class AdaptiveTimeoutManager:
    """Manages adaptive timeouts based on historical performance."""

    def __init__(self, config: ReliabilityConfiguration):
        self.config = config
        self._response_times: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = time.time()

    def record_response_time(self, operation: str, duration_ms: float):
        """Record response time for adaptive timeout calculation."""
        self._response_times[operation].append(duration_ms)

        # Cleanup old data
        if time.time() - self._last_cleanup > 300:  # 5 minutes
            self._cleanup_old_data()

    def _cleanup_old_data(self):
        """Clean up old response time data."""
        for operation in self._response_times:
            if len(self._response_times[operation]) > 100:
                self._response_times[operation] = self._response_times[operation][-100:]
        self._last_cleanup = time.time()

    def get_adaptive_timeout(self, operation: str) -> float:
        """Calculate adaptive timeout based on historical performance."""
        if not self.config.adaptive_timeout_enabled:
            return self.config.default_timeout

        response_times = self._response_times.get(operation, [])
        if len(response_times) < 5:  # Not enough data
            return self.config.default_timeout

        # Use 95th percentile + buffer
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_time = sorted_times[min(p95_index, len(sorted_times) - 1)]

        # Add 50% buffer and convert to seconds
        adaptive_timeout = (p95_time * 1.5) / 1000.0

        # Apply bounds
        return max(
            self.config.min_timeout, min(adaptive_timeout, self.config.max_timeout)
        )


class EnhancedErrorHandler(ErrorHandler):
    """Enhanced error handler with advanced recovery strategies."""

    def __init__(self, config: ReliabilityConfiguration = None):
        self.config = config or ReliabilityConfiguration()
        self.robust_handler = RobustErrorHandler()

    @staticmethod
    def format_enhanced_error(
        error: Exception, context: Optional[ErrorContext] = None
    ) -> str:
        """Format errors with enhanced context information."""
        base_message = ErrorHandler.format_database_error(error)

        if context:
            severity_indicator = {
                ErrorSeverity.LOW: "ℹ️",
                ErrorSeverity.MEDIUM: "⚠️",
                ErrorSeverity.HIGH: "🚨",
                ErrorSeverity.CRITICAL: "🔥",
            }.get(context.severity, "❓")

            return f"{severity_indicator} {base_message} (Error ID: {context.error_id})"

        return base_message

    def should_retry_error(
        self, error: Exception, attempt_count: int
    ) -> tuple[bool, float]:
        """Determine if error should be retried and calculate delay."""
        if attempt_count >= self.config.max_retries:
            return False, 0.0

        # Get error context
        error_context = self.robust_handler.classify_error(error)

        # Don't retry validation or authentication errors
        no_retry_categories = {ErrorCategory.VALIDATION, ErrorCategory.AUTHENTICATION}
        if error_context.category in no_retry_categories:
            return False, 0.0

        # Calculate exponential backoff delay
        base_delay = 1.0
        delay = min(
            base_delay * (self.config.retry_backoff_multiplier**attempt_count),
            self.config.retry_max_delay,
        )

        return True, delay


class BulkheadManager:
    """Manages resource isolation using the bulkhead pattern."""

    def __init__(self, config: ReliabilityConfiguration):
        self.config = config
        self._high_priority_semaphore = asyncio.Semaphore(
            config.high_priority_queue_size
        )
        self._normal_priority_semaphore = asyncio.Semaphore(
            config.normal_priority_queue_size
        )
        self._low_priority_semaphore = asyncio.Semaphore(config.low_priority_queue_size)
        self._global_semaphore = asyncio.Semaphore(config.max_concurrent_queries)

    @asynccontextmanager
    async def acquire_slot(self, priority: str = "normal"):
        """Acquire a bulkhead slot based on priority."""
        semaphore_map = {
            "high": self._high_priority_semaphore,
            "normal": self._normal_priority_semaphore,
            "low": self._low_priority_semaphore,
        }

        priority_semaphore = semaphore_map.get(
            priority, self._normal_priority_semaphore
        )

        async with self._global_semaphore:
            async with priority_semaphore:
                yield


class PerformanceTracker:
    """Legacy performance tracker for backward compatibility."""

    def __init__(self):
        self.start_time = time.time()
        self.query_count = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.response_times = []
        self.peak_connections = 0
        self.active_connections = 0
        self.error_count = 0
        self._lock = threading.Lock()

    def record_query_start(self) -> str:
        """Record the start of a query and return tracking ID."""
        with self._lock:
            self.query_count += 1
            self.active_connections += 1
            self.peak_connections = max(self.peak_connections, self.active_connections)
        return f"query_{self.query_count}_{int(time.time() * 1000)}"

    def record_query_end(self, duration_ms: float, success: bool = True):
        """Record the end of a query with its duration."""
        with self._lock:
            self.active_connections = max(0, self.active_connections - 1)
            self.response_times.append(duration_ms)

            if success:
                self.successful_queries += 1
            else:
                self.failed_queries += 1
                self.error_count += 1

            # Keep only recent response times for average calculation
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-500:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            avg_response_time = (
                sum(self.response_times) / len(self.response_times)
                if self.response_times
                else 0.0
            )
            error_rate = (
                (self.failed_queries / self.query_count * 100)
                if self.query_count > 0
                else 0.0
            )
            uptime = time.time() - self.start_time

            return {
                "uptime_seconds": uptime,
                "total_queries": self.query_count,
                "successful_queries": self.successful_queries,
                "failed_queries": self.failed_queries,
                "active_connections": self.active_connections,
                "peak_connections": self.peak_connections,
                "avg_response_time_ms": avg_response_time,
                "error_rate_percent": error_rate,
                "queries_per_second": self.query_count / uptime if uptime > 0 else 0.0,
            }


class EnhancedResultFormatter(ResultFormatter):
    """Enhanced result formatter with additional output formats."""

    @staticmethod
    def to_json_lines(results: List[QueryResult]) -> str:
        """Convert multiple query results to JSON Lines format."""
        lines = []
        for result in results:
            result_dict = EnhancedResultFormatter.to_dict(result)
            lines.append(json.dumps(result_dict, default=str))
        return "\n".join(lines)

    @staticmethod
    def to_csv_string(result: QueryResult) -> str:
        """Convert query result to CSV format."""
        if not result.data:
            return ""

        import csv
        import io

        output = io.StringIO()
        if isinstance(result.data[0], dict):
            # Dict format - extract headers
            headers = list(result.data[0].keys())
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            for row in result.data:
                writer.writerow(row)
        else:
            # List format
            writer = csv.writer(output)
            for row in result.data:
                writer.writerow(row)

        return output.getvalue()

    @staticmethod
    def to_markdown_table(result: QueryResult) -> str:
        """Convert query result to Markdown table format."""
        if not result.data:
            return "| (No data) |\n|---|"

        if isinstance(result.data[0], dict):
            headers = list(result.data[0].keys())
            header_row = "| " + " | ".join(headers) + " |"
            separator_row = "|" + "|".join([" --- " for _ in headers]) + "|"

            rows = []
            for row in result.data:
                row_values = [str(row.get(h, "")) for h in headers]
                rows.append("| " + " | ".join(row_values) + " |")

            return "\n".join([header_row, separator_row] + rows)
        else:
            # Simple list format
            rows = []
            for row in result.data:
                if isinstance(row, (list, tuple)):
                    row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
                else:
                    row_str = f"| {str(row)} |"
                rows.append(row_str)
            return "\n".join(rows)


# Global enhanced tracker instance
enhanced_query_tracker = EnhancedQueryTracker()


def create_enhanced_query_metadata(
    query_id: str,
    duration_ms: float,
    cache_hit: bool = False,
    user_agent: Optional[str] = None,
    client_ip: Optional[str] = None,
    priority: str = "normal",
    error_context: Optional[ErrorContext] = None,
) -> QueryMetadata:
    """Create enhanced query metadata with additional tracking information."""
    metadata = QueryMetadata(
        query_id=query_id,
        timestamp=datetime.utcnow(),
        duration_ms=duration_ms,
        cache_hit=cache_hit,
        user_agent=user_agent,
        client_ip=client_ip,
    )

    # Add enhanced attributes
    metadata.priority = priority  # type: ignore
    metadata.error_context = error_context  # type: ignore

    return metadata


def get_enhanced_system_info(
    include_advanced_metrics: bool = True,
) -> Union[SystemInfo, AdvancedSystemMetrics]:
    """Get enhanced system information with advanced metrics."""
    if include_advanced_metrics:
        return enhanced_query_tracker.get_advanced_statistics()
    else:
        stats = enhanced_query_tracker.get_statistics()
        return SystemInfo(
            version="0.2.2",
            uptime=stats["uptime_seconds"],
            total_queries=stats["total_queries"],
            successful_queries=stats["successful_queries"],
            failed_queries=stats["failed_queries"],
            cache_hit_rate=stats["cache_hit_rate"],
        )


class GlobalEventBus:
    """Global event bus for system-wide event handling."""

    def __init__(self):
        self.subscribers = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to an event type."""
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)

    def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event to all subscribers."""
        handlers = []
        with self._lock:
            handlers = self.subscribers.get(event_type, []).copy()

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # Handle async handlers
                    asyncio.create_task(handler(data))
                else:
                    handler(data)
            except Exception as e:
                logger.warning(f"Event handler error for {event_type}: {e}")

    def unsubscribe(self, event_type: str, handler: callable):
        """Unsubscribe from an event type."""
        with self._lock:
            if event_type in self.subscribers:
                try:
                    self.subscribers[event_type].remove(handler)
                except ValueError:
                    pass


class AdaptiveQueryOptimizer:
    """Adaptive query optimizer that learns from execution patterns."""

    def __init__(self):
        self.query_patterns = {}
        self.optimization_history = {}
        self._lock = threading.Lock()

    def analyze_query(self, sql: str, execution_time: float) -> Dict[str, Any]:
        """Analyze a query and provide optimization suggestions."""
        with self._lock:
            # Simple pattern recognition based on SQL structure
            patterns = []

            sql_upper = sql.upper()
            if "JOIN" in sql_upper:
                patterns.append("joins")
            if "GROUP BY" in sql_upper:
                patterns.append("aggregation")
            if "ORDER BY" in sql_upper:
                patterns.append("sorting")
            if "LIMIT" in sql_upper:
                patterns.append("pagination")

            # Track execution patterns
            query_hash = hash(sql)
            if query_hash not in self.query_patterns:
                self.query_patterns[query_hash] = {
                    "count": 0,
                    "avg_time": 0.0,
                    "patterns": patterns,
                    "sql": sql[:200],  # Store snippet for debugging
                }

            pattern_data = self.query_patterns[query_hash]
            pattern_data["count"] += 1
            pattern_data["avg_time"] = (
                pattern_data["avg_time"] * (pattern_data["count"] - 1) + execution_time
            ) / pattern_data["count"]

            # Generate suggestions based on patterns and performance
            suggestions = []
            if execution_time > 1000:  # > 1 second
                if "joins" in patterns:
                    suggestions.append("Consider adding indexes on join columns")
                if "sorting" in patterns:
                    suggestions.append("Consider adding index on ORDER BY columns")
                if "aggregation" in patterns:
                    suggestions.append(
                        "Consider materialized views for complex aggregations"
                    )

            return {
                "execution_count": pattern_data["count"],
                "average_time_ms": pattern_data["avg_time"],
                "patterns": patterns,
                "suggestions": suggestions,
                "is_frequent": pattern_data["count"] > 5,
                "is_slow": execution_time > pattern_data["avg_time"] * 1.5,
            }

    def get_top_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the top slow queries for optimization."""
        with self._lock:
            sorted_queries = sorted(
                self.query_patterns.items(),
                key=lambda x: x[1]["avg_time"],
                reverse=True,
            )

            return [
                {
                    "sql_snippet": data["sql"],
                    "execution_count": data["count"],
                    "avg_time_ms": data["avg_time"],
                    "patterns": data["patterns"],
                }
                for _, data in sorted_queries[:limit]
            ]


# Global instances
performance_tracker = PerformanceTracker()
event_bus = GlobalEventBus()
query_optimizer = AdaptiveQueryOptimizer()


def get_enhanced_system_info() -> EnhancedSystemInfo:
    """Get enhanced system information with performance metrics."""
    base_metrics = performance_tracker.get_metrics()

    return EnhancedSystemInfo(
        version="0.2.2-enhanced",
        uptime=base_metrics["uptime_seconds"],
        total_queries=base_metrics["total_queries"],
        successful_queries=base_metrics["successful_queries"],
        failed_queries=base_metrics["failed_queries"],
        cache_hit_rate=0.0,  # Will be updated by cache components
        active_connections=base_metrics["active_connections"],
        peak_connections=base_metrics["peak_connections"],
        avg_response_time_ms=base_metrics["avg_response_time_ms"],
        error_rate_percent=base_metrics["error_rate_percent"],
    )


def create_enhanced_query_metadata(
    query_id: str = None, duration_ms: float = 0.0, cache_hit: bool = False, **kwargs
) -> EnhancedQueryMetadata:
    """Create enhanced query metadata with additional tracking."""
    return EnhancedQueryMetadata(
        query_id=query_id or str(uuid.uuid4()),
        timestamp=datetime.now(),
        duration_ms=duration_ms,
        cache_hit=cache_hit,
        complexity_score=kwargs.get("complexity_score", 0.0),
        estimated_cost=kwargs.get("estimated_cost", 0.0),
        optimization_applied=kwargs.get("optimization_applied", False),
        security_level=kwargs.get("security_level", "standard"),
        data_source=kwargs.get("data_source", "primary"),
    )
