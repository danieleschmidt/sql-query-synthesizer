"""
Scaling Module - Performance Optimization and Auto-Scaling

Provides advanced performance optimization, resource management,
and auto-scaling capabilities for high-load production environments.
"""

from .performance_optimizer import (
    OptimizationStrategy,
    PerformanceMetrics,
    OptimizationRecommendation,
    ResourceMonitor,
    ConcurrentProcessor,
    MemoryOptimizer,
    PerformanceOptimizer,
    performance_optimizer
)

from .auto_scaler import (
    ScalingStrategy,
    ScalingMetric,
    ScalingRule,
    AutoScaler,
    LoadBalancer,
    ResourcePool,
    auto_scaler
)

from .connection_pool import (
    PooledConnection,
    ConnectionPool,
    ConnectionPoolManager,
    PoolConfiguration,
    HealthChecker
)

__all__ = [
    "OptimizationStrategy",
    "PerformanceMetrics",
    "OptimizationRecommendation",
    "ResourceMonitor",
    "ConcurrentProcessor",
    "MemoryOptimizer",
    "PerformanceOptimizer",
    "performance_optimizer",
    "ScalingStrategy",
    "ScalingMetric",
    "ScalingRule",
    "AutoScaler",
    "LoadBalancer",
    "ResourcePool", 
    "auto_scaler",
    "PooledConnection",
    "ConnectionPool",
    "ConnectionPoolManager",
    "PoolConfiguration",
    "HealthChecker"
]