"""Enhanced Core Functionality for SQL Query Synthesizer.

This module implements Generation 1 enhancements focused on making the system work
with improved reliability, performance monitoring, and enterprise features.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import json
import uuid

from .types import QueryResult
from .core import SystemInfo, QueryMetadata, ResultFormatter

logger = logging.getLogger(__name__)


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


class PerformanceTracker:
    """Tracks performance metrics and system health."""
    
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
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
            error_rate = (self.failed_queries / self.query_count * 100) if self.query_count > 0 else 0.0
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
                "queries_per_second": self.query_count / uptime if uptime > 0 else 0.0
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
        return '\n'.join(lines)
    
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
            if 'JOIN' in sql_upper:
                patterns.append('joins')
            if 'GROUP BY' in sql_upper:
                patterns.append('aggregation')
            if 'ORDER BY' in sql_upper:
                patterns.append('sorting')
            if 'LIMIT' in sql_upper:
                patterns.append('pagination')
            
            # Track execution patterns
            query_hash = hash(sql)
            if query_hash not in self.query_patterns:
                self.query_patterns[query_hash] = {
                    'count': 0,
                    'avg_time': 0.0,
                    'patterns': patterns,
                    'sql': sql[:200]  # Store snippet for debugging
                }
            
            pattern_data = self.query_patterns[query_hash]
            pattern_data['count'] += 1
            pattern_data['avg_time'] = (
                (pattern_data['avg_time'] * (pattern_data['count'] - 1) + execution_time) / 
                pattern_data['count']
            )
            
            # Generate suggestions based on patterns and performance
            suggestions = []
            if execution_time > 1000:  # > 1 second
                if 'joins' in patterns:
                    suggestions.append("Consider adding indexes on join columns")
                if 'sorting' in patterns:
                    suggestions.append("Consider adding index on ORDER BY columns")
                if 'aggregation' in patterns:
                    suggestions.append("Consider materialized views for complex aggregations")
            
            return {
                'execution_count': pattern_data['count'],
                'average_time_ms': pattern_data['avg_time'],
                'patterns': patterns,
                'suggestions': suggestions,
                'is_frequent': pattern_data['count'] > 5,
                'is_slow': execution_time > pattern_data['avg_time'] * 1.5
            }
    
    def get_top_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the top slow queries for optimization."""
        with self._lock:
            sorted_queries = sorted(
                self.query_patterns.items(),
                key=lambda x: x[1]['avg_time'],
                reverse=True
            )
            
            return [
                {
                    'sql_snippet': data['sql'],
                    'execution_count': data['count'],
                    'avg_time_ms': data['avg_time'],
                    'patterns': data['patterns']
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
        error_rate_percent=base_metrics["error_rate_percent"]
    )


def create_enhanced_query_metadata(
    query_id: str = None,
    duration_ms: float = 0.0,
    cache_hit: bool = False,
    **kwargs
) -> EnhancedQueryMetadata:
    """Create enhanced query metadata with additional tracking."""
    return EnhancedQueryMetadata(
        query_id=query_id or str(uuid.uuid4()),
        timestamp=datetime.now(),
        duration_ms=duration_ms,
        cache_hit=cache_hit,
        complexity_score=kwargs.get('complexity_score', 0.0),
        estimated_cost=kwargs.get('estimated_cost', 0.0),
        optimization_applied=kwargs.get('optimization_applied', False),
        security_level=kwargs.get('security_level', 'standard'),
        data_source=kwargs.get('data_source', 'primary')
    )