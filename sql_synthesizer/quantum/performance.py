"""
Adaptive performance optimization for quantum components
"""

import time
import statistics
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import collections

from .exceptions import QuantumOptimizationError


class OptimizationMode(Enum):
    """Performance optimization modes"""
    CONSERVATIVE = "conservative"  # Focus on reliability
    BALANCED = "balanced"         # Balance speed and quality
    AGGRESSIVE = "aggressive"     # Focus on speed
    ADAPTIVE = "adaptive"         # Auto-adjust based on load


@dataclass 
class PerformanceMetrics:
    """Performance metrics for quantum operations"""
    operation_type: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_type: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_type": self.operation_type,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "success": self.success,
            "error_type": self.error_type,
            "timestamp": self.timestamp
        }


class QuantumPerformanceMonitor:
    """
    Real-time performance monitoring and adaptive optimization
    """
    
    def __init__(self, window_size: int = 1000, adaptation_interval: float = 60.0):
        self.window_size = window_size
        self.adaptation_interval = adaptation_interval
        
        # Metrics storage
        self._metrics: Dict[str, collections.deque] = {}
        self._lock = threading.RLock()
        
        # Performance thresholds
        self._thresholds = {
            "max_execution_time": 30.0,
            "max_memory_usage": 0.8,
            "min_success_rate": 0.95,
            "max_error_rate": 0.05
        }
        
        # Adaptive parameters
        self._current_mode = OptimizationMode.BALANCED
        self._last_adaptation = time.time()
        self._performance_score = 1.0
        
        # Performance baselines
        self._baselines: Dict[str, float] = {}
        
        # Optimization callbacks
        self._optimization_callbacks: List[Callable] = []
    
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric"""
        with self._lock:
            operation_type = metric.operation_type
            
            if operation_type not in self._metrics:
                self._metrics[operation_type] = collections.deque(maxlen=self.window_size)
            
            self._metrics[operation_type].append(metric)
            
            # Check if adaptation is needed
            if time.time() - self._last_adaptation > self.adaptation_interval:
                self._adapt_performance()
    
    def get_performance_stats(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Args:
            operation_type: Specific operation type or None for all
            
        Returns:
            Performance statistics dictionary
        """
        with self._lock:
            if operation_type:
                return self._calculate_stats_for_operation(operation_type)
            
            # Calculate stats for all operations
            all_stats = {}
            for op_type in self._metrics:
                all_stats[op_type] = self._calculate_stats_for_operation(op_type)
            
            # Calculate overall stats
            all_metrics = []
            for metrics in self._metrics.values():
                all_metrics.extend(metrics)
            
            if all_metrics:
                all_stats["overall"] = self._calculate_stats_from_metrics(all_metrics)
                all_stats["overall"]["current_mode"] = self._current_mode.value
                all_stats["overall"]["performance_score"] = self._performance_score
            
            return all_stats
    
    def _calculate_stats_for_operation(self, operation_type: str) -> Dict[str, Any]:
        """Calculate statistics for a specific operation type"""
        if operation_type not in self._metrics:
            return {}
        
        metrics = list(self._metrics[operation_type])
        return self._calculate_stats_from_metrics(metrics)
    
    def _calculate_stats_from_metrics(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate statistics from a list of metrics"""
        if not metrics:
            return {}
        
        # Execution time statistics
        execution_times = [m.execution_time for m in metrics]
        
        # Success rate
        successes = sum(1 for m in metrics if m.success)
        success_rate = successes / len(metrics)
        
        # Error analysis
        errors = [m.error_type for m in metrics if not m.success and m.error_type]
        error_counts = collections.Counter(errors)
        
        # Memory usage statistics
        memory_usage = [m.memory_usage for m in metrics if m.memory_usage > 0]
        
        stats = {
            "count": len(metrics),
            "success_rate": success_rate,
            "error_rate": 1.0 - success_rate,
            "execution_time": {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "p95": statistics.quantiles(execution_times, n=20)[18] if len(execution_times) > 20 else max(execution_times),
                "p99": statistics.quantiles(execution_times, n=100)[98] if len(execution_times) > 100 else max(execution_times)
            },
            "error_breakdown": dict(error_counts.most_common(10))
        }
        
        if memory_usage:
            stats["memory_usage"] = {
                "mean": statistics.mean(memory_usage),
                "max": max(memory_usage),
                "min": min(memory_usage)
            }
        
        return stats
    
    def _adapt_performance(self):
        """Adapt performance parameters based on current metrics"""
        with self._lock:
            current_time = time.time()
            
            # Calculate overall performance score
            overall_stats = self.get_performance_stats().get("overall", {})
            
            if not overall_stats:
                return
            
            # Performance scoring
            success_rate = overall_stats.get("success_rate", 1.0)
            avg_exec_time = overall_stats.get("execution_time", {}).get("mean", 0.0)
            p95_exec_time = overall_stats.get("execution_time", {}).get("p95", 0.0)
            
            # Score components (0-1, higher is better)
            success_score = success_rate
            speed_score = max(0, 1.0 - avg_exec_time / self._thresholds["max_execution_time"])
            reliability_score = max(0, 1.0 - p95_exec_time / (self._thresholds["max_execution_time"] * 2))
            
            self._performance_score = (success_score * 0.5 + speed_score * 0.3 + reliability_score * 0.2)
            
            # Adapt optimization mode
            old_mode = self._current_mode
            
            if success_rate < self._thresholds["min_success_rate"]:
                # Poor success rate - be more conservative
                self._current_mode = OptimizationMode.CONSERVATIVE
            elif avg_exec_time > self._thresholds["max_execution_time"]:
                # Slow execution - be more aggressive
                self._current_mode = OptimizationMode.AGGRESSIVE
            elif self._performance_score > 0.8:
                # Good performance - maintain balance
                self._current_mode = OptimizationMode.BALANCED
            else:
                # Mediocre performance - use adaptive mode
                self._current_mode = OptimizationMode.ADAPTIVE
            
            # Notify callbacks if mode changed
            if old_mode != self._current_mode:
                for callback in self._optimization_callbacks:
                    try:
                        callback(old_mode, self._current_mode, overall_stats)
                    except Exception:
                        pass  # Don't let callback errors break adaptation
            
            self._last_adaptation = current_time
    
    def get_optimization_parameters(self, operation_type: str) -> Dict[str, Any]:
        """
        Get optimized parameters for an operation type
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Optimized parameters dictionary
        """
        with self._lock:
            base_params = self._get_base_parameters(operation_type)
            
            # Adjust based on current mode
            if self._current_mode == OptimizationMode.CONSERVATIVE:
                base_params.update({
                    "timeout_multiplier": 2.0,
                    "retry_count": 5,
                    "batch_size": min(base_params.get("batch_size", 10), 5),
                    "parallel_workers": min(base_params.get("parallel_workers", 4), 2),
                    "cache_ttl_multiplier": 2.0
                })
            elif self._current_mode == OptimizationMode.AGGRESSIVE:
                base_params.update({
                    "timeout_multiplier": 0.5,
                    "retry_count": 1,
                    "batch_size": base_params.get("batch_size", 10) * 2,
                    "parallel_workers": base_params.get("parallel_workers", 4) * 2,
                    "cache_ttl_multiplier": 0.5
                })
            elif self._current_mode == OptimizationMode.ADAPTIVE:
                # Adjust based on performance score
                score = self._performance_score
                base_params.update({
                    "timeout_multiplier": 0.5 + score,
                    "retry_count": max(1, int(3 * score)),
                    "batch_size": max(5, int(base_params.get("batch_size", 10) * score)),
                    "parallel_workers": max(2, int(base_params.get("parallel_workers", 4) * score))
                })
            
            return base_params
    
    def _get_base_parameters(self, operation_type: str) -> Dict[str, Any]:
        """Get base parameters for operation type"""
        base_params = {
            "timeout_multiplier": 1.0,
            "retry_count": 3,
            "batch_size": 10,
            "parallel_workers": 4,
            "cache_ttl_multiplier": 1.0
        }
        
        # Operation-specific adjustments
        if operation_type == "quantum_optimization":
            base_params.update({
                "batch_size": 20,
                "parallel_workers": 8,
                "iterations": 1000
            })
        elif operation_type == "quantum_scheduling":
            base_params.update({
                "batch_size": 50,
                "parallel_workers": 6,
                "scheduling_window": 300
            })
        elif operation_type == "plan_generation":
            base_params.update({
                "batch_size": 100,
                "parallel_workers": 12,
                "max_plans": 50
            })
        
        return base_params
    
    def add_optimization_callback(self, callback: Callable):
        """Add callback for optimization mode changes"""
        self._optimization_callbacks.append(callback)
    
    def set_threshold(self, name: str, value: float):
        """Set performance threshold"""
        if name in self._thresholds:
            self._thresholds[name] = value
    
    def get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        overall_stats = self.get_performance_stats().get("overall", {})
        
        if not overall_stats:
            return recommendations
        
        success_rate = overall_stats.get("success_rate", 1.0)
        avg_exec_time = overall_stats.get("execution_time", {}).get("mean", 0.0)
        p95_exec_time = overall_stats.get("execution_time", {}).get("p95", 0.0)
        
        # Success rate recommendations
        if success_rate < 0.95:
            recommendations.append("Consider switching to conservative mode for better reliability")
            
        if success_rate < 0.9:
            recommendations.append("Review error patterns and implement additional error handling")
        
        # Performance recommendations
        if avg_exec_time > 20.0:
            recommendations.append("Consider increasing parallel workers or reducing batch sizes")
            
        if p95_exec_time > 45.0:
            recommendations.append("Implement timeout optimizations and circuit breakers")
        
        # Memory recommendations
        memory_stats = overall_stats.get("memory_usage", {})
        if memory_stats and memory_stats.get("max", 0) > 0.8:
            recommendations.append("Memory usage is high - consider implementing memory pooling")
        
        # Mode-specific recommendations
        if self._current_mode == OptimizationMode.CONSERVATIVE and self._performance_score > 0.9:
            recommendations.append("Performance is good - consider switching to balanced mode")
            
        elif self._current_mode == OptimizationMode.AGGRESSIVE and success_rate < 0.95:
            recommendations.append("Aggressive mode causing failures - consider balanced mode")
        
        return recommendations


class QuantumResourcePool:
    """
    Resource pooling for quantum components to improve performance
    """
    
    def __init__(self, max_optimizers: int = 10, max_schedulers: int = 5):
        self.max_optimizers = max_optimizers
        self.max_schedulers = max_schedulers
        
        # Resource pools
        self._optimizer_pool: List[Any] = []  # Actual optimizers
        self._scheduler_pool: List[Any] = []  # Actual schedulers
        
        # Pool locks
        self._optimizer_lock = threading.Semaphore(max_optimizers)
        self._scheduler_lock = threading.Semaphore(max_schedulers)
        
        # Usage tracking
        self._optimizer_usage = collections.defaultdict(int)
        self._scheduler_usage = collections.defaultdict(int)
        
        # Pool statistics
        self._pool_stats = {
            "optimizers_created": 0,
            "schedulers_created": 0,
            "optimizers_reused": 0,
            "schedulers_reused": 0
        }
    
    def get_optimizer(self, **kwargs):
        """Get optimizer from pool or create new one"""
        self._optimizer_lock.acquire()
        
        try:
            # Try to reuse existing optimizer
            if self._optimizer_pool:
                optimizer = self._optimizer_pool.pop()
                self._pool_stats["optimizers_reused"] += 1
                return optimizer
            
            # Create new optimizer
            from .core import QuantumQueryOptimizer
            optimizer = QuantumQueryOptimizer(**kwargs)
            self._pool_stats["optimizers_created"] += 1
            return optimizer
            
        except Exception:
            self._optimizer_lock.release()
            raise
    
    def return_optimizer(self, optimizer):
        """Return optimizer to pool"""
        try:
            # Reset optimizer state
            optimizer.reset_quantum_state()
            
            # Add back to pool if there's space
            if len(self._optimizer_pool) < self.max_optimizers:
                self._optimizer_pool.append(optimizer)
            
        finally:
            self._optimizer_lock.release()
    
    def get_scheduler(self, **kwargs):
        """Get scheduler from pool or create new one"""
        self._scheduler_lock.acquire()
        
        try:
            # Try to reuse existing scheduler
            if self._scheduler_pool:
                scheduler = self._scheduler_pool.pop()
                self._pool_stats["schedulers_reused"] += 1
                return scheduler
            
            # Create new scheduler
            from .scheduler import QuantumTaskScheduler
            scheduler = QuantumTaskScheduler(**kwargs)
            self._pool_stats["schedulers_created"] += 1
            return scheduler
            
        except Exception:
            self._scheduler_lock.release()
            raise
    
    def return_scheduler(self, scheduler):
        """Return scheduler to pool"""
        try:
            # Clear scheduler state if needed
            # (Implementation depends on scheduler reset capabilities)
            
            # Add back to pool if there's space
            if len(self._scheduler_pool) < self.max_schedulers:
                self._scheduler_pool.append(scheduler)
            
        finally:
            self._scheduler_lock.release()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics"""
        return {
            **self._pool_stats,
            "optimizer_pool_size": len(self._optimizer_pool),
            "scheduler_pool_size": len(self._scheduler_pool),
            "max_optimizers": self.max_optimizers,
            "max_schedulers": self.max_schedulers,
            "optimizer_reuse_rate": (
                self._pool_stats["optimizers_reused"] / 
                max(1, self._pool_stats["optimizers_created"] + self._pool_stats["optimizers_reused"])
            ),
            "scheduler_reuse_rate": (
                self._pool_stats["schedulers_reused"] / 
                max(1, self._pool_stats["schedulers_created"] + self._pool_stats["schedulers_reused"])
            )
        }


# Global instances
quantum_performance_monitor = QuantumPerformanceMonitor()
quantum_resource_pool = QuantumResourcePool()