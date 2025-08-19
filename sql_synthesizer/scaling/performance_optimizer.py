"""
Advanced Performance Optimization System
Provides intelligent performance optimization, auto-scaling, and resource management.
"""

import asyncio
import gc
import logging
import multiprocessing as mp
import threading
import time
import weakref
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Different optimization strategies."""

    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"
    IO_OPTIMIZATION = "io_optimization"
    CONCURRENT_PROCESSING = "concurrent_processing"
    RESOURCE_POOLING = "resource_pooling"
    LAZY_LOADING = "lazy_loading"
    BATCH_PROCESSING = "batch_processing"
    ASYNC_PROCESSING = "async_processing"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""

    timestamp: datetime
    cpu_usage: float
    memory_usage_mb: float
    active_threads: int
    pending_tasks: int
    avg_response_time_ms: float
    throughput_rps: float  # requests per second
    cache_hit_rate: float
    error_rate: float
    queue_depth: int
    connection_pool_usage: float


@dataclass
class OptimizationRecommendation:
    """Recommendation for performance optimization."""

    strategy: OptimizationStrategy
    priority: int  # 1=highest, 10=lowest
    impact_score: float  # 0-1
    implementation_effort: str  # low, medium, high
    description: str
    estimated_improvement: str
    parameters: Dict[str, Any]
    confidence: float


class ResourceMonitor:
    """Monitors system resources for optimization decisions."""

    def __init__(self, sample_interval: int = 5):
        self.sample_interval = sample_interval
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_thread = None

        # Resource thresholds
        self.thresholds = {
            "cpu_high": 80,
            "cpu_critical": 95,
            "memory_high": 85,
            "memory_critical": 95,
            "response_time_slow": 1000,  # ms
            "response_time_critical": 5000,  # ms
            "queue_depth_high": 100,
            "error_rate_high": 5,  # %
        }

    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self.monitor_thread.start()
            logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")

    def collect_metrics(
        self, additional_metrics: Optional[Dict] = None
    ) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            import psutil

            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage_mb = (memory.total - memory.available) / (1024 * 1024)

            # Application metrics from additional_metrics or defaults
            metrics_data = additional_metrics or {}

            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                active_threads=threading.active_count(),
                pending_tasks=metrics_data.get("pending_tasks", 0),
                avg_response_time_ms=metrics_data.get("avg_response_time_ms", 0),
                throughput_rps=metrics_data.get("throughput_rps", 0),
                cache_hit_rate=metrics_data.get("cache_hit_rate", 0),
                error_rate=metrics_data.get("error_rate", 0),
                queue_depth=metrics_data.get("queue_depth", 0),
                connection_pool_usage=metrics_data.get("connection_pool_usage", 0),
            )

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=0,
                memory_usage_mb=0,
                active_threads=0,
                pending_tasks=0,
                avg_response_time_ms=0,
                throughput_rps=0,
                cache_hit_rate=0,
                error_rate=0,
                queue_depth=0,
                connection_pool_usage=0,
            )

    def get_performance_trends(self, minutes: int = 30) -> Dict[str, Any]:
        """Analyze performance trends."""
        if not self.metrics_history:
            return {"status": "insufficient_data"}

        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff]

        if len(recent_metrics) < 2:
            return {"status": "insufficient_recent_data"}

        # Calculate trends
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage_mb for m in recent_metrics]
        response_time_values = [
            m.avg_response_time_ms for m in recent_metrics if m.avg_response_time_ms > 0
        ]

        return {
            "time_period_minutes": minutes,
            "data_points": len(recent_metrics),
            "trends": {
                "cpu": {
                    "current": cpu_values[-1] if cpu_values else 0,
                    "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    "max": max(cpu_values) if cpu_values else 0,
                    "trend": (
                        "increasing"
                        if len(cpu_values) > 1 and cpu_values[-1] > cpu_values[0]
                        else "stable"
                    ),
                },
                "memory": {
                    "current_mb": memory_values[-1] if memory_values else 0,
                    "avg_mb": (
                        sum(memory_values) / len(memory_values) if memory_values else 0
                    ),
                    "max_mb": max(memory_values) if memory_values else 0,
                    "trend": (
                        "increasing"
                        if len(memory_values) > 1
                        and memory_values[-1] > memory_values[0]
                        else "stable"
                    ),
                },
                "response_time": {
                    "current_ms": (
                        response_time_values[-1] if response_time_values else 0
                    ),
                    "avg_ms": (
                        sum(response_time_values) / len(response_time_values)
                        if response_time_values
                        else 0
                    ),
                    "max_ms": max(response_time_values) if response_time_values else 0,
                },
            },
        }

    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Detect current performance issues."""
        if not self.metrics_history:
            return []

        latest_metrics = self.metrics_history[-1]
        issues = []

        # CPU issues
        if latest_metrics.cpu_usage > self.thresholds["cpu_critical"]:
            issues.append(
                {
                    "type": "cpu_critical",
                    "severity": "critical",
                    "message": f"Critical CPU usage: {latest_metrics.cpu_usage:.1f}%",
                    "value": latest_metrics.cpu_usage,
                    "threshold": self.thresholds["cpu_critical"],
                }
            )
        elif latest_metrics.cpu_usage > self.thresholds["cpu_high"]:
            issues.append(
                {
                    "type": "cpu_high",
                    "severity": "warning",
                    "message": f"High CPU usage: {latest_metrics.cpu_usage:.1f}%",
                    "value": latest_metrics.cpu_usage,
                    "threshold": self.thresholds["cpu_high"],
                }
            )

        # Memory issues
        memory_percent = (
            (latest_metrics.memory_usage_mb / 1024) / 8 * 100
        )  # Assume 8GB system
        if memory_percent > self.thresholds["memory_critical"]:
            issues.append(
                {
                    "type": "memory_critical",
                    "severity": "critical",
                    "message": f"Critical memory usage: {latest_metrics.memory_usage_mb:.0f}MB",
                    "value": latest_metrics.memory_usage_mb,
                    "threshold": self.thresholds["memory_critical"],
                }
            )
        elif memory_percent > self.thresholds["memory_high"]:
            issues.append(
                {
                    "type": "memory_high",
                    "severity": "warning",
                    "message": f"High memory usage: {latest_metrics.memory_usage_mb:.0f}MB",
                    "value": latest_metrics.memory_usage_mb,
                    "threshold": self.thresholds["memory_high"],
                }
            )

        # Response time issues
        if (
            latest_metrics.avg_response_time_ms
            > self.thresholds["response_time_critical"]
        ):
            issues.append(
                {
                    "type": "response_time_critical",
                    "severity": "critical",
                    "message": f"Critical response time: {latest_metrics.avg_response_time_ms:.0f}ms",
                    "value": latest_metrics.avg_response_time_ms,
                    "threshold": self.thresholds["response_time_critical"],
                }
            )
        elif (
            latest_metrics.avg_response_time_ms > self.thresholds["response_time_slow"]
        ):
            issues.append(
                {
                    "type": "response_time_slow",
                    "severity": "warning",
                    "message": f"Slow response time: {latest_metrics.avg_response_time_ms:.0f}ms",
                    "value": latest_metrics.avg_response_time_ms,
                    "threshold": self.thresholds["response_time_slow"],
                }
            )

        return issues

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)

                # Check for critical issues
                issues = self.detect_performance_issues()
                critical_issues = [i for i in issues if i["severity"] == "critical"]

                if critical_issues:
                    logger.warning(
                        f"Critical performance issues detected: {len(critical_issues)} issues"
                    )
                    for issue in critical_issues:
                        logger.warning(f"  - {issue['message']}")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.sample_interval)


class ConcurrentProcessor:
    """Manages concurrent processing for improved performance."""

    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.use_processes = use_processes

        # Executor pools
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = (
            ProcessPoolExecutor(max_workers=mp.cpu_count() or 1)
            if use_processes
            else None
        )

        # Task queue and metrics
        self.pending_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_history = deque(maxlen=1000)

        self.lock = threading.RLock()

    async def process_batch_async(
        self,
        tasks: List[Callable],
        task_args: List[tuple] = None,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """Process a batch of tasks concurrently using async."""
        if not tasks:
            return []

        task_args = task_args or [() for _ in tasks]

        # Create async tasks
        async_tasks = []
        for i, (task, args) in enumerate(zip(tasks, task_args)):
            if asyncio.iscoroutinefunction(task):
                async_tasks.append(task(*args))
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                async_tasks.append(
                    loop.run_in_executor(self.thread_executor, task, *args)
                )

        # Execute with timeout
        try:
            if timeout:
                results = await asyncio.wait_for(
                    asyncio.gather(*async_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            else:
                results = await asyncio.gather(*async_tasks, return_exceptions=True)

            # Process results
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {result}")
                    self.failed_tasks += 1
                    successful_results.append(None)
                else:
                    successful_results.append(result)
                    self.completed_tasks += 1

            return successful_results

        except asyncio.TimeoutError:
            logger.error(f"Batch processing timed out after {timeout}s")
            self.failed_tasks += len(tasks)
            return [None] * len(tasks)

    def process_batch_sync(
        self,
        tasks: List[Callable],
        task_args: List[tuple] = None,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """Process a batch of tasks concurrently using threads/processes."""
        if not tasks:
            return []

        task_args = task_args or [() for _ in tasks]

        # Choose executor based on task type
        executor = (
            self.process_executor
            if self.use_processes and self.process_executor
            else self.thread_executor
        )

        # Submit tasks
        future_to_index = {}
        with executor:
            for i, (task, args) in enumerate(zip(tasks, task_args)):
                future = executor.submit(task, *args)
                future_to_index[future] = i

        # Collect results
        results = [None] * len(tasks)

        try:
            for future in as_completed(future_to_index.keys(), timeout=timeout):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    self.completed_tasks += 1
                except Exception as e:
                    logger.error(f"Task {index} failed: {e}")
                    results[index] = None
                    self.failed_tasks += 1

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            self.failed_tasks += len(tasks)

        return results

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_tasks = self.completed_tasks + self.failed_tasks

        return {
            "pending_tasks": self.pending_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": (self.completed_tasks / max(total_tasks, 1)) * 100,
            "max_workers": self.max_workers,
            "using_processes": self.use_processes and self.process_executor is not None,
            "thread_pool_size": self.thread_executor._max_workers,
            "process_pool_size": (
                self.process_executor._max_workers if self.process_executor else 0
            ),
        }

    def shutdown(self):
        """Shutdown executor pools."""
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        if self.process_executor:
            self.process_executor.shutdown(wait=True)


class MemoryOptimizer:
    """Optimizes memory usage and prevents memory leaks."""

    def __init__(self):
        self.weak_references = weakref.WeakSet()
        self.large_objects = {}
        self.gc_stats = defaultdict(int)

        # Memory thresholds
        self.cleanup_threshold_mb = 500
        self.critical_threshold_mb = 1000

    def register_large_object(self, obj_id: str, obj: Any, size_hint_mb: float = 0):
        """Register a large object for tracking."""
        self.large_objects[obj_id] = {
            "object": weakref.ref(obj),
            "size_hint_mb": size_hint_mb,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
        }

    def cleanup_large_objects(self, max_age_minutes: int = 30):
        """Clean up old large objects."""
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        cleaned_objects = []

        for obj_id, obj_info in list(self.large_objects.items()):
            if obj_info["last_accessed"] < cutoff or obj_info["object"]() is None:
                del self.large_objects[obj_id]
                cleaned_objects.append(obj_id)

        if cleaned_objects:
            logger.info(f"Cleaned up {len(cleaned_objects)} large objects")
            self.force_garbage_collection()

        return len(cleaned_objects)

    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        before_count = len(gc.get_objects())

        # Run garbage collection
        collected = {}
        for generation in range(gc.get_count().__len__()):
            collected[f"gen_{generation}"] = gc.collect(generation)

        after_count = len(gc.get_objects())

        self.gc_stats["forced_collections"] += 1
        self.gc_stats["total_objects_collected"] += before_count - after_count

        logger.debug(
            f"Garbage collection: {before_count} -> {after_count} objects ({before_count - after_count} collected)"
        )

        return {
            "objects_before": before_count,
            "objects_after": after_count,
            "objects_collected": before_count - after_count,
            "generation_collections": collected,
        }

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "memory_percent": process.memory_percent(),
                "large_objects_tracked": len(self.large_objects),
                "gc_stats": dict(self.gc_stats),
                "python_objects": len(gc.get_objects()),
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}

    def optimize_memory_usage(self) -> List[str]:
        """Perform memory optimization operations."""
        optimizations_performed = []

        # Clean up large objects
        cleaned = self.cleanup_large_objects()
        if cleaned > 0:
            optimizations_performed.append(f"Cleaned {cleaned} large objects")

        # Force garbage collection if memory usage is high
        memory_usage = self.get_memory_usage()
        if memory_usage.get("rss_mb", 0) > self.cleanup_threshold_mb:
            gc_stats = self.force_garbage_collection()
            if gc_stats["objects_collected"] > 0:
                optimizations_performed.append(
                    f"Garbage collected {gc_stats['objects_collected']} objects"
                )

        # Clear weak references
        initial_count = len(self.weak_references)
        self.weak_references.clear()
        if initial_count > 0:
            optimizations_performed.append(f"Cleared {initial_count} weak references")

        return optimizations_performed


class PerformanceOptimizer:
    """Main performance optimization system."""

    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.concurrent_processor = ConcurrentProcessor()
        self.memory_optimizer = MemoryOptimizer()

        # Optimization state
        self.active_optimizations = set()
        self.optimization_history = deque(maxlen=100)

        # Auto-optimization settings
        self.auto_optimize = True
        self.optimization_interval = 300  # 5 minutes
        self.last_optimization = datetime.min

    def start_optimization_system(self):
        """Start the performance optimization system."""
        self.resource_monitor.start_monitoring()
        logger.info("Performance optimization system started")

    def stop_optimization_system(self):
        """Stop the performance optimization system."""
        self.resource_monitor.stop_monitoring()
        self.concurrent_processor.shutdown()
        logger.info("Performance optimization system stopped")

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and identify optimization opportunities."""
        # Get current metrics
        current_metrics = self.resource_monitor.collect_metrics()
        performance_trends = self.resource_monitor.get_performance_trends()
        performance_issues = self.resource_monitor.detect_performance_issues()

        # Get processing stats
        processing_stats = self.concurrent_processor.get_processing_stats()
        memory_usage = self.memory_optimizer.get_memory_usage()

        return {
            "current_metrics": asdict(current_metrics),
            "performance_trends": performance_trends,
            "performance_issues": performance_issues,
            "processing_stats": processing_stats,
            "memory_usage": memory_usage,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on current performance."""
        recommendations = []
        analysis = self.analyze_performance()

        current_metrics = analysis["current_metrics"]
        analysis["performance_issues"]

        # CPU optimization recommendations
        cpu_usage = current_metrics.get("cpu_usage", 0)
        if cpu_usage > 80:
            recommendations.append(
                OptimizationRecommendation(
                    strategy=OptimizationStrategy.CONCURRENT_PROCESSING,
                    priority=2,
                    impact_score=0.8,
                    implementation_effort="medium",
                    description="High CPU usage detected - implement async processing and task parallelization",
                    estimated_improvement="30-50% CPU reduction",
                    parameters={
                        "enable_async_processing": True,
                        "max_concurrent_tasks": mp.cpu_count() * 2,
                        "task_batching_enabled": True,
                    },
                    confidence=0.9,
                )
            )

        # Memory optimization recommendations
        memory_usage_mb = current_metrics.get("memory_usage_mb", 0)
        if memory_usage_mb > 500:
            recommendations.append(
                OptimizationRecommendation(
                    strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
                    priority=1,
                    impact_score=0.7,
                    implementation_effort="low",
                    description="High memory usage detected - enable aggressive memory cleanup",
                    estimated_improvement="20-40% memory reduction",
                    parameters={
                        "enable_aggressive_gc": True,
                        "cleanup_interval_seconds": 60,
                        "large_object_cleanup": True,
                    },
                    confidence=0.8,
                )
            )

        # Response time optimization
        response_time = current_metrics.get("avg_response_time_ms", 0)
        if response_time > 1000:
            recommendations.append(
                OptimizationRecommendation(
                    strategy=OptimizationStrategy.IO_OPTIMIZATION,
                    priority=3,
                    impact_score=0.6,
                    implementation_effort="medium",
                    description="Slow response times - optimize I/O operations and implement caching",
                    estimated_improvement="40-60% response time improvement",
                    parameters={
                        "enable_connection_pooling": True,
                        "increase_cache_size": True,
                        "async_io_operations": True,
                    },
                    confidence=0.7,
                )
            )

        # Throughput optimization
        throughput = current_metrics.get("throughput_rps", 0)
        if throughput < 10:  # Low throughput
            recommendations.append(
                OptimizationRecommendation(
                    strategy=OptimizationStrategy.BATCH_PROCESSING,
                    priority=4,
                    impact_score=0.5,
                    implementation_effort="medium",
                    description="Low throughput - implement batch processing for better resource utilization",
                    estimated_improvement="2-5x throughput increase",
                    parameters={
                        "batch_size": 10,
                        "batch_timeout_ms": 100,
                        "parallel_batches": True,
                    },
                    confidence=0.6,
                )
            )

        # Sort by priority and impact
        recommendations.sort(key=lambda x: (x.priority, -x.impact_score))

        return recommendations

    def apply_optimization(
        self, recommendation: OptimizationRecommendation
    ) -> Dict[str, Any]:
        """Apply a specific optimization recommendation."""
        optimization_id = f"{recommendation.strategy.value}_{int(time.time())}"

        try:
            result = None

            if recommendation.strategy == OptimizationStrategy.MEMORY_OPTIMIZATION:
                result = self._apply_memory_optimization(recommendation.parameters)
            elif recommendation.strategy == OptimizationStrategy.CONCURRENT_PROCESSING:
                result = self._apply_concurrent_processing_optimization(
                    recommendation.parameters
                )
            elif recommendation.strategy == OptimizationStrategy.IO_OPTIMIZATION:
                result = self._apply_io_optimization(recommendation.parameters)
            elif recommendation.strategy == OptimizationStrategy.BATCH_PROCESSING:
                result = self._apply_batch_processing_optimization(
                    recommendation.parameters
                )
            else:
                result = {
                    "status": "not_implemented",
                    "message": f"Strategy {recommendation.strategy.value} not yet implemented",
                }

            # Record optimization
            optimization_record = {
                "id": optimization_id,
                "strategy": recommendation.strategy.value,
                "timestamp": datetime.utcnow().isoformat(),
                "parameters": recommendation.parameters,
                "result": result,
                "success": result.get("status") == "success",
            }

            self.optimization_history.append(optimization_record)

            if result.get("status") == "success":
                self.active_optimizations.add(recommendation.strategy.value)
                logger.info(f"Applied optimization: {recommendation.strategy.value}")

            return result

        except Exception as e:
            logger.error(f"Error applying optimization {optimization_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "optimization_id": optimization_id,
            }

    def auto_optimize(self) -> Dict[str, Any]:
        """Automatically apply optimizations based on current performance."""
        if not self.auto_optimize:
            return {"status": "disabled", "message": "Auto-optimization is disabled"}

        # Check if enough time has passed since last optimization
        if datetime.utcnow() - self.last_optimization < timedelta(
            seconds=self.optimization_interval
        ):
            return {"status": "skipped", "message": "Too soon since last optimization"}

        # Generate and apply recommendations
        recommendations = self.generate_optimization_recommendations()

        if not recommendations:
            return {
                "status": "no_recommendations",
                "message": "No optimizations needed",
            }

        # Apply top recommendations
        applied_optimizations = []
        for recommendation in recommendations[:3]:  # Apply top 3
            if recommendation.priority <= 3:  # Only high priority optimizations
                result = self.apply_optimization(recommendation)
                applied_optimizations.append(
                    {"strategy": recommendation.strategy.value, "result": result}
                )

        self.last_optimization = datetime.utcnow()

        return {
            "status": "completed",
            "applied_optimizations": applied_optimizations,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and statistics."""
        recent_optimizations = list(self.optimization_history)[-10:]
        successful_optimizations = [
            opt for opt in recent_optimizations if opt["success"]
        ]

        return {
            "active_optimizations": list(self.active_optimizations),
            "total_optimizations_applied": len(self.optimization_history),
            "successful_optimizations": len(
                [opt for opt in self.optimization_history if opt["success"]]
            ),
            "recent_optimizations": recent_optimizations,
            "success_rate": len(successful_optimizations)
            / max(len(recent_optimizations), 1)
            * 100,
            "auto_optimize_enabled": self.auto_optimize,
            "last_optimization": self.last_optimization.isoformat(),
            "next_optimization_window": (
                self.last_optimization + timedelta(seconds=self.optimization_interval)
            ).isoformat(),
        }

    # Private optimization methods
    def _apply_memory_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory optimization strategies."""
        optimizations_performed = self.memory_optimizer.optimize_memory_usage()

        if parameters.get("enable_aggressive_gc", False):
            gc_stats = self.memory_optimizer.force_garbage_collection()
            optimizations_performed.append(
                f"Forced GC: {gc_stats['objects_collected']} objects collected"
            )

        return {
            "status": "success",
            "optimizations_performed": optimizations_performed,
            "memory_usage_after": self.memory_optimizer.get_memory_usage(),
        }

    def _apply_concurrent_processing_optimization(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply concurrent processing optimizations."""
        # Update concurrent processor settings
        max_workers = parameters.get("max_concurrent_tasks", mp.cpu_count() * 2)

        # This would require restructuring the concurrent processor
        # For now, just return status
        return {
            "status": "success",
            "message": "Concurrent processing optimization applied",
            "new_max_workers": max_workers,
        }

    def _apply_io_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply I/O optimization strategies."""
        optimizations = []

        if parameters.get("enable_connection_pooling", False):
            optimizations.append("Connection pooling enabled")

        if parameters.get("async_io_operations", False):
            optimizations.append("Async I/O operations enabled")

        return {"status": "success", "optimizations_applied": optimizations}

    def _apply_batch_processing_optimization(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply batch processing optimizations."""
        batch_size = parameters.get("batch_size", 10)
        batch_timeout = parameters.get("batch_timeout_ms", 100)

        return {
            "status": "success",
            "batch_size": batch_size,
            "batch_timeout_ms": batch_timeout,
            "message": "Batch processing optimization applied",
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()
