"""
Scaling Module - Performance Optimization and Auto-Scaling

Provides advanced performance optimization, resource management,
and auto-scaling capabilities for high-load production environments.
"""

from .auto_scaler import (
    AutoScaler,
    LoadBalancer,
    ResourcePool,
    ScalingMetric,
    ScalingRule,
    ScalingStrategy,
    auto_scaler,
)
from .connection_pool import (
    ConnectionPool,
    ConnectionPoolManager,
    HealthChecker,
    PoolConfiguration,
    PooledConnection,
)
from .performance_optimizer import (
    ConcurrentProcessor,
    MemoryOptimizer,
    OptimizationRecommendation,
    OptimizationStrategy,
    PerformanceMetrics,
    PerformanceOptimizer,
    ResourceMonitor,
    performance_optimizer,
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
    "HealthChecker",
]
