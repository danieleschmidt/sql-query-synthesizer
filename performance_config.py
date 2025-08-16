"""
Advanced Performance Configuration for SQL Query Synthesizer.

This module provides comprehensive performance optimization settings including:
- Connection pooling with adaptive sizing
- Advanced caching strategies with TTL optimization
- Query optimization hints and statistics
- Resource monitoring and auto-scaling triggers
- Performance profiling and benchmarking integration
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Available caching strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"


class ConnectionPoolStrategy(Enum):
    """Connection pool sizing strategies."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    ELASTIC = "elastic"


@dataclass
class DatabasePerformanceConfig:
    """Database connection and query performance configuration."""

    # Connection Pool Settings
    pool_size: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_DB_POOL_SIZE", "20")))
    max_overflow: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_DB_MAX_OVERFLOW", "40")))
    pool_pre_ping: bool = field(default_factory=lambda: os.environ.get("QUERY_AGENT_DB_POOL_PRE_PING", "true").lower() == "true")
    pool_recycle: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_DB_POOL_RECYCLE", "3600")))

    # Advanced Pool Settings
    pool_strategy: ConnectionPoolStrategy = field(default=ConnectionPoolStrategy.ADAPTIVE)
    pool_scale_up_threshold: float = field(default=0.8)  # Scale up when 80% utilized
    pool_scale_down_threshold: float = field(default=0.3)  # Scale down when 30% utilized
    pool_scale_interval: int = field(default=60)  # Check every 60 seconds

    # Query Performance
    query_timeout: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_DATABASE_TIMEOUT", "45")))
    slow_query_threshold: float = field(default=5.0)  # Log queries slower than 5 seconds
    query_cache_enabled: bool = field(default=True)
    query_cache_ttl: int = field(default=300)  # 5 minutes

    # Connection Health
    health_check_interval: int = field(default=30)
    max_retries: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_DB_CONNECT_RETRIES", "3")))
    retry_delay: float = field(default_factory=lambda: float(os.environ.get("QUERY_AGENT_DB_RETRY_DELAY", "1.0")))

    # Query Optimization
    enable_query_hints: bool = field(default=True)
    enable_execution_plan_analysis: bool = field(default=False)
    auto_explain_threshold: float = field(default=10.0)  # Auto-explain queries > 10s


@dataclass
class CachePerformanceConfig:
    """Advanced caching configuration for optimal performance."""

    # Primary Cache Settings
    backend: str = field(default_factory=lambda: os.environ.get("QUERY_AGENT_CACHE_BACKEND", "redis"))
    strategy: CacheStrategy = field(default=CacheStrategy.ADAPTIVE)
    default_ttl: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_CACHE_TTL", "3600")))

    # Cache Sizing and Limits
    max_memory_mb: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_CACHE_MAX_MEMORY_MB", "1024")))
    max_items: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_CACHE_MAX_ITEMS", "10000")))
    eviction_policy: str = field(default="allkeys-lru")

    # Multi-tier Caching
    enable_l1_cache: bool = field(default=True)  # In-memory L1 cache
    l1_cache_size: int = field(default=1000)
    l1_cache_ttl: int = field(default=60)

    enable_l2_cache: bool = field(default=True)  # Redis/Memcached L2 cache
    l2_cache_ttl: int = field(default=3600)

    # Cache Warming and Preloading
    enable_cache_warming: bool = field(default=True)
    warm_cache_on_startup: bool = field(default=False)
    preload_popular_queries: bool = field(default=True)

    # Advanced Features
    enable_cache_compression: bool = field(default=True)
    compression_threshold: int = field(default=1024)  # Compress items > 1KB
    enable_cache_encryption: bool = field(default=False)

    # Performance Monitoring
    enable_hit_rate_monitoring: bool = field(default=True)
    hit_rate_alert_threshold: float = field(default=0.7)  # Alert if hit rate < 70%

    # Redis-specific settings
    redis_host: str = field(default_factory=lambda: os.environ.get("QUERY_AGENT_REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_REDIS_PORT", "6379")))
    redis_db: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_REDIS_DB", "0")))
    redis_connection_pool_size: int = field(default=20)
    redis_socket_timeout: float = field(default=5.0)
    redis_socket_connect_timeout: float = field(default=5.0)


@dataclass
class ApplicationPerformanceConfig:
    """Application-level performance configuration."""

    # Async Processing
    enable_async_processing: bool = field(default=True)
    async_pool_size: int = field(default=10)
    async_queue_size: int = field(default=1000)

    # Request Processing
    max_concurrent_requests: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_MAX_CONCURRENT", "100")))
    request_timeout: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_REQUEST_TIMEOUT", "30")))

    # Memory Management
    memory_limit_mb: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_MEMORY_LIMIT_MB", "2048")))
    gc_threshold: float = field(default=0.8)  # Trigger GC at 80% memory usage

    # Performance Monitoring
    enable_profiling: bool = field(default=False)
    profile_sample_rate: float = field(default=0.01)  # 1% sampling
    enable_metrics_collection: bool = field(default=True)
    metrics_export_interval: int = field(default=30)

    # Auto-scaling Triggers
    cpu_scale_up_threshold: float = field(default=0.7)
    memory_scale_up_threshold: float = field(default=0.8)
    response_time_threshold: float = field(default=2.0)  # Scale up if avg response > 2s


@dataclass
class LLMPerformanceConfig:
    """LLM API performance and optimization settings."""

    # API Settings
    timeout: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_OPENAI_TIMEOUT", "30")))
    max_retries: int = field(default=3)
    backoff_factor: float = field(default=2.0)

    # Request Optimization
    enable_request_batching: bool = field(default=True)
    batch_size: int = field(default=5)
    batch_timeout: float = field(default=1.0)  # Max wait time for batching

    # Response Caching
    enable_response_caching: bool = field(default=True)
    response_cache_ttl: int = field(default=86400)  # 24 hours
    cache_key_strategy: str = field(default="hash")  # "hash" or "semantic"

    # Circuit Breaker
    circuit_breaker_enabled: bool = field(default=True)
    failure_threshold: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")))
    recovery_timeout: float = field(default_factory=lambda: float(os.environ.get("QUERY_AGENT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60.0")))

    # Rate Limiting
    requests_per_minute: int = field(default=60)
    burst_limit: int = field(default=10)

    # Cost Optimization
    enable_cost_tracking: bool = field(default=True)
    cost_alert_threshold: float = field(default=100.0)  # Alert if monthly cost > $100
    preferred_model: str = field(default_factory=lambda: os.environ.get("QUERY_AGENT_OPENAI_MODEL", "gpt-3.5-turbo"))


@dataclass
class PerformanceConfig:
    """Comprehensive performance configuration."""

    database: DatabasePerformanceConfig = field(default_factory=DatabasePerformanceConfig)
    cache: CachePerformanceConfig = field(default_factory=CachePerformanceConfig)
    application: ApplicationPerformanceConfig = field(default_factory=ApplicationPerformanceConfig)
    llm: LLMPerformanceConfig = field(default_factory=LLMPerformanceConfig)

    # Global Performance Settings
    environment: str = field(default_factory=lambda: os.environ.get("QUERY_AGENT_ENV", "development"))
    enable_performance_logging: bool = field(default=True)
    performance_log_level: str = field(default="INFO")

    # Benchmarking and Testing
    enable_benchmarking: bool = field(default=False)
    benchmark_sample_rate: float = field(default=0.1)  # 10% of requests
    performance_test_mode: bool = field(default=False)


class PerformanceOptimizer:
    """Dynamic performance optimization based on runtime metrics."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.metrics = {}
        self.optimization_callbacks: List[Callable] = []

    async def start_monitoring(self):
        """Start performance monitoring and optimization loop."""

        if not self.config.application.enable_metrics_collection:
            return

        logger.info("Starting performance monitoring...")

        while True:
            try:
                await self._collect_metrics()
                await self._analyze_performance()
                await self._apply_optimizations()
                await asyncio.sleep(self.config.application.metrics_export_interval)
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _collect_metrics(self):
        """Collect current performance metrics."""

        import psutil

        # System metrics
        self.metrics.update({
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        })

        # Application-specific metrics would be collected here
        # e.g., from prometheus metrics, database stats, cache stats

    async def _analyze_performance(self):
        """Analyze metrics and identify optimization opportunities."""

        # CPU analysis
        if self.metrics.get("cpu_percent", 0) > self.config.application.cpu_scale_up_threshold * 100:
            await self._trigger_optimization("high_cpu")

        # Memory analysis
        if self.metrics.get("memory_percent", 0) > self.config.application.memory_scale_up_threshold * 100:
            await self._trigger_optimization("high_memory")

    async def _apply_optimizations(self):
        """Apply dynamic optimizations based on current conditions."""

        # Example optimizations
        if "high_cpu" in self.metrics.get("optimization_triggers", []):
            await self._optimize_for_cpu()

        if "high_memory" in self.metrics.get("optimization_triggers", []):
            await self._optimize_for_memory()

    async def _trigger_optimization(self, trigger: str):
        """Trigger specific optimization."""

        if "optimization_triggers" not in self.metrics:
            self.metrics["optimization_triggers"] = []

        if trigger not in self.metrics["optimization_triggers"]:
            self.metrics["optimization_triggers"].append(trigger)
            logger.info(f"Performance optimization triggered: {trigger}")

    async def _optimize_for_cpu(self):
        """Apply CPU-focused optimizations."""

        logger.info("Applying CPU optimizations...")

        # Reduce concurrent operations
        # Increase caching
        # Optimize query patterns

    async def _optimize_for_memory(self):
        """Apply memory-focused optimizations."""

        logger.info("Applying memory optimizations...")

        # Trigger garbage collection
        # Reduce cache sizes
        # Clear unnecessary buffers


def load_performance_config() -> PerformanceConfig:
    """Load performance configuration based on environment."""

    config = PerformanceConfig()

    # Environment-specific adjustments
    if config.environment == "production":
        config.database.pool_size = max(config.database.pool_size, 30)
        config.cache.enable_cache_warming = True
        config.application.enable_profiling = False
        config.application.enable_metrics_collection = True

    elif config.environment == "development":
        config.database.pool_size = min(config.database.pool_size, 5)
        config.cache.enable_cache_warming = False
        config.application.enable_profiling = True
        config.llm.enable_response_caching = False  # For development testing

    elif config.environment == "testing":
        config.database.pool_size = 2
        config.cache.backend = "memory"
        config.application.enable_metrics_collection = False

    logger.info(f"Performance configuration loaded for {config.environment} environment")
    return config


async def initialize_performance_monitoring(config: PerformanceConfig):
    """Initialize performance monitoring and optimization."""

    if not config.application.enable_metrics_collection:
        logger.info("Performance monitoring disabled")
        return None

    optimizer = PerformanceOptimizer(config)

    # Start monitoring in background
    asyncio.create_task(optimizer.start_monitoring())

    logger.info("Performance monitoring initialized")
    return optimizer


# Example usage and testing
if __name__ == "__main__":

    # Load configuration
    perf_config = load_performance_config()

    print(f"Performance Configuration for {perf_config.environment}:")
    print(f"  Database Pool Size: {perf_config.database.pool_size}")
    print(f"  Cache Backend: {perf_config.cache.backend}")
    print(f"  Max Concurrent Requests: {perf_config.application.max_concurrent_requests}")
    print(f"  LLM Timeout: {perf_config.llm.timeout}s")
    print(f"  Metrics Collection: {perf_config.application.enable_metrics_collection}")

    # Example of starting performance monitoring
    async def demo():
        optimizer = await initialize_performance_monitoring(perf_config)
        if optimizer:
            print("Performance monitoring started...")
            await asyncio.sleep(5)  # Demo run for 5 seconds
            print("Demo completed")

    # Run demo
    asyncio.run(demo())
