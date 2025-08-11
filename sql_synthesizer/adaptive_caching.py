"""Adaptive Caching System for SQL Query Synthesizer.

This module implements intelligent caching strategies that adapt to query patterns,
data freshness requirements, and system load to optimize performance.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Different caching strategies based on query patterns."""
    IMMEDIATE = "immediate"           # Cache immediately, short TTL
    DELAYED = "delayed"               # Cache after multiple hits
    PREDICTIVE = "predictive"        # Pre-cache based on patterns
    CONDITIONAL = "conditional"       # Cache based on execution time
    ADAPTIVE = "adaptive"            # Dynamically adjust based on usage


class DataFreshness(Enum):
    """Data freshness requirements."""
    REAL_TIME = "real_time"          # Always fresh data
    NEAR_REAL_TIME = "near_real_time"  # < 1 minute old
    RECENT = "recent"                # < 5 minutes old
    STABLE = "stable"                # < 1 hour old
    ARCHIVAL = "archival"            # Can be hours/days old


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    execution_time_ms: float
    data_size_bytes: int
    freshness_requirement: DataFreshness
    tags: List[str]
    cost_to_regenerate: float
    hit_probability: float = 0.0


@dataclass
class QueryPattern:
    """Query execution pattern for predictive caching."""
    query_hash: str
    frequency_per_hour: float
    typical_execution_time_ms: float
    peak_hours: List[int]
    seasonal_pattern: Dict[str, float]
    user_groups: List[str]
    data_dependencies: List[str]


class AdaptiveCacheManager:
    """Intelligent cache manager that adapts to usage patterns."""
    
    def __init__(
        self, 
        max_memory_mb: int = 1000,
        default_ttl: int = 3600,
        cleanup_interval: int = 300
    ):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
        # Analytics
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)  # key -> [timestamps]
        
        # Threading
        self._lock = threading.RLock()
        self._cleanup_task = None
        
        # Adaptive parameters
        self.hit_threshold_for_caching = 2  # Cache after N hits
        self.cost_threshold_for_caching = 100.0  # Cache queries > 100ms
        self.memory_pressure_threshold = 0.8  # Start aggressive eviction at 80%
        
        # Start background cleanup
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        def cleanup_loop():
            while True:
                try:
                    self._perform_cleanup()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        self._cleanup_task = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_task.start()
    
    def should_cache(
        self, 
        query_hash: str, 
        execution_time_ms: float, 
        data_size_bytes: int,
        freshness: DataFreshness = DataFreshness.STABLE
    ) -> Tuple[bool, CacheStrategy]:
        """Determine if and how to cache a query result."""
        
        with self._lock:
            # Always cache if execution time is high
            if execution_time_ms > self.cost_threshold_for_caching:
                return True, CacheStrategy.IMMEDIATE
            
            # Check access patterns
            access_history = self.access_patterns.get(query_hash, [])
            
            # Cache frequently accessed queries
            if len(access_history) >= self.hit_threshold_for_caching:
                # Calculate frequency
                now = time.time()
                recent_accesses = [t for t in access_history if now - t < 3600]  # Last hour
                
                if len(recent_accesses) >= 3:  # 3+ times in last hour
                    return True, CacheStrategy.IMMEDIATE
                elif len(access_history) >= 5:  # 5+ times total
                    return True, CacheStrategy.DELAYED
            
            # Don't cache if memory pressure is high and query is cheap
            if self._get_memory_pressure() > self.memory_pressure_threshold:
                if execution_time_ms < 50:  # Less than 50ms
                    return False, CacheStrategy.IMMEDIATE
            
            # Cache based on data freshness requirements
            if freshness in [DataFreshness.STABLE, DataFreshness.ARCHIVAL]:
                return True, CacheStrategy.CONDITIONAL
            
            return False, CacheStrategy.IMMEDIATE
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with analytics tracking."""
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                self._record_access(key)
                return default
            
            entry = self.cache[key]
            
            # Check expiration
            if self._is_expired(entry):
                del self.cache[key]
                self.current_memory_bytes -= entry.data_size_bytes
                self.miss_count += 1
                self._record_access(key)
                return default
            
            # Update access statistics
            entry.last_accessed = time.time()
            entry.access_count += 1
            self.hit_count += 1
            self._record_access(key)
            
            return entry.value
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        execution_time_ms: float = 0,
        freshness: DataFreshness = DataFreshness.STABLE,
        tags: List[str] = None
    ):
        """Set item in cache with intelligent management."""
        
        # Estimate data size
        data_size = self._estimate_size(value)
        
        # Check if we should cache this item
        should_cache, strategy = self.should_cache(
            key, execution_time_ms, data_size, freshness
        )
        
        if not should_cache:
            logger.debug(f"Skipping cache for key {key[:12]}... (strategy: {strategy.value})")
            return
        
        with self._lock:
            now = time.time()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                execution_time_ms=execution_time_ms,
                data_size_bytes=data_size,
                freshness_requirement=freshness,
                tags=tags or [],
                cost_to_regenerate=execution_time_ms
            )
            
            # Check memory limits and evict if necessary
            self._ensure_memory_capacity(data_size)
            
            # Store entry
            self.cache[key] = entry
            self.current_memory_bytes += data_size
            
            logger.debug(
                f"Cached key {key[:12]}... (size: {data_size} bytes, "
                f"execution: {execution_time_ms:.1f}ms, strategy: {strategy.value})"
            )
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate cache entries by tags."""
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            logger.info(f"Invalidated {len(keys_to_remove)} entries by tags: {tags}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate size distribution
            size_buckets = {"small": 0, "medium": 0, "large": 0}
            for entry in self.cache.values():
                if entry.data_size_bytes < 1024:  # < 1KB
                    size_buckets["small"] += 1
                elif entry.data_size_bytes < 1024 * 100:  # < 100KB
                    size_buckets["medium"] += 1
                else:
                    size_buckets["large"] += 1
            
            # Calculate freshness distribution
            freshness_dist = defaultdict(int)
            for entry in self.cache.values():
                freshness_dist[entry.freshness_requirement.value] += 1
            
            # Memory utilization
            memory_utilization = (self.current_memory_bytes / self.max_memory_bytes * 100)
            
            return {
                "cache_size": len(self.cache),
                "memory_usage_mb": self.current_memory_bytes / (1024 * 1024),
                "memory_utilization_percent": memory_utilization,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate_percent": hit_rate,
                "eviction_count": self.eviction_count,
                "size_distribution": dict(size_buckets),
                "freshness_distribution": dict(freshness_dist),
                "avg_entry_size_kb": (
                    (self.current_memory_bytes / len(self.cache) / 1024) 
                    if self.cache else 0
                ),
                "top_accessed_keys": self._get_top_accessed_keys(10)
            }
    
    def _record_access(self, key: str):
        """Record access for pattern analysis."""
        now = time.time()
        self.access_patterns[key].append(now)
        
        # Keep only recent access history
        cutoff = now - 24 * 3600  # 24 hours
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        now = time.time()
        age = now - entry.created_at
        
        # Adaptive TTL based on freshness requirement
        if entry.freshness_requirement == DataFreshness.REAL_TIME:
            return age > 10  # 10 seconds
        elif entry.freshness_requirement == DataFreshness.NEAR_REAL_TIME:
            return age > 60  # 1 minute
        elif entry.freshness_requirement == DataFreshness.RECENT:
            return age > 300  # 5 minutes
        elif entry.freshness_requirement == DataFreshness.STABLE:
            return age > 3600  # 1 hour
        else:  # ARCHIVAL
            return age > 24 * 3600  # 24 hours
    
    def _get_memory_pressure(self) -> float:
        """Calculate current memory pressure (0.0 to 1.0)."""
        return self.current_memory_bytes / self.max_memory_bytes
    
    def _ensure_memory_capacity(self, needed_bytes: int):
        """Ensure sufficient memory capacity by evicting entries."""
        if self.current_memory_bytes + needed_bytes <= self.max_memory_bytes:
            return
        
        # Calculate how much memory to free
        target_memory = self.max_memory_bytes * 0.7  # Target 70% utilization
        memory_to_free = self.current_memory_bytes + needed_bytes - target_memory
        
        # Get eviction candidates sorted by priority (lower = evict first)
        candidates = []
        now = time.time()
        
        for key, entry in self.cache.items():
            priority = self._calculate_eviction_priority(entry, now)
            candidates.append((priority, key, entry))
        
        candidates.sort(key=lambda x: x[0])  # Sort by priority
        
        # Evict entries until we have enough space
        freed_memory = 0
        for priority, key, entry in candidates:
            if freed_memory >= memory_to_free:
                break
            
            self._remove_entry(key)
            freed_memory += entry.data_size_bytes
            self.eviction_count += 1
        
        logger.info(f"Evicted {len(candidates[:self.eviction_count])} entries, freed {freed_memory} bytes")
    
    def _calculate_eviction_priority(self, entry: CacheEntry, now: float) -> float:
        """Calculate eviction priority (lower = more likely to evict)."""
        
        # Base priority factors
        age_factor = now - entry.last_accessed
        access_frequency = entry.access_count / ((now - entry.created_at) / 3600 + 1)  # per hour
        cost_factor = entry.cost_to_regenerate
        size_penalty = entry.data_size_bytes / (1024 * 1024)  # MB penalty
        
        # Calculate priority (lower = evict first)
        priority = (
            cost_factor * 0.3 +           # Higher cost = keep longer
            access_frequency * 0.4 +      # More frequent = keep longer  
            (1 / (age_factor + 1)) * 0.2 + # Recently used = keep longer
            (-size_penalty * 0.1)         # Larger size = evict sooner
        )
        
        return priority
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and update memory tracking."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory_bytes -= entry.data_size_bytes
            del self.cache[key]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            if hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            else:
                # Fallback estimation
                return len(str(value).encode('utf-8'))
        except:
            return 1024  # Default estimate
    
    def _get_top_accessed_keys(self, limit: int) -> List[Dict[str, Any]]:
        """Get the most frequently accessed cache keys."""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        
        return [
            {
                "key": key[:20] + "..." if len(key) > 20 else key,
                "access_count": entry.access_count,
                "size_kb": entry.data_size_bytes / 1024,
                "age_minutes": (time.time() - entry.created_at) / 60
            }
            for key, entry in sorted_entries[:limit]
        ]
    
    def _perform_cleanup(self):
        """Perform periodic cleanup of expired entries."""
        if time.time() - self.last_cleanup < self.cleanup_interval:
            return
        
        with self._lock:
            expired_keys = []
            now = time.time()
            
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            self.last_cleanup = now
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class PredictiveCacheWarmer:
    """Predictive cache warmer that pre-loads frequently accessed queries."""
    
    def __init__(self, cache_manager: AdaptiveCacheManager):
        self.cache_manager = cache_manager
        self.prediction_history: Dict[str, List[float]] = defaultdict(list)
        self.warmup_tasks: List[asyncio.Task] = []
    
    def record_query_execution(self, query_hash: str, execution_time: float):
        """Record query execution for pattern learning."""
        now = time.time()
        self.prediction_history[query_hash].append(now)
        
        # Keep only recent history
        cutoff = now - 7 * 24 * 3600  # 7 days
        self.prediction_history[query_hash] = [
            t for t in self.prediction_history[query_hash] if t > cutoff
        ]
    
    def predict_next_queries(self, lookahead_minutes: int = 30) -> List[str]:
        """Predict queries likely to be executed in the near future."""
        now = time.time()
        predictions = []
        
        for query_hash, execution_times in self.prediction_history.items():
            if len(execution_times) < 3:  # Need at least 3 executions for prediction
                continue
            
            # Simple time-based prediction: look for patterns in execution times
            recent_executions = [t for t in execution_times if now - t < 24 * 3600]  # Last 24 hours
            
            if len(recent_executions) < 2:
                continue
            
            # Calculate average interval between executions
            intervals = []
            for i in range(1, len(recent_executions)):
                intervals.append(recent_executions[i] - recent_executions[i-1])
            
            if not intervals:
                continue
            
            avg_interval = sum(intervals) / len(intervals)
            last_execution = recent_executions[-1]
            
            # Predict if query should execute soon
            time_since_last = now - last_execution
            if time_since_last >= avg_interval * 0.8:  # 80% of average interval
                predicted_time = last_execution + avg_interval
                if predicted_time <= now + lookahead_minutes * 60:
                    predictions.append(query_hash)
        
        return predictions
    
    async def warm_cache_proactively(self, query_executor_func):
        """Proactively warm cache for predicted queries."""
        predictions = self.predict_next_queries()
        
        if not predictions:
            return
        
        logger.info(f"Warming cache for {len(predictions)} predicted queries")
        
        # Execute predicted queries asynchronously
        tasks = []
        for query_hash in predictions[:5]:  # Limit to top 5 predictions
            # Only create warmup task if query executor is provided
            if query_executor_func and callable(query_executor_func):
                task = asyncio.create_task(
                    self._warm_single_query(query_hash, query_executor_func)
                )
                tasks.append(task)
        
        if tasks:
            # Wait for warmup tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _warm_single_query(self, query_hash: str, query_executor_func):
        """Warm cache for a single predicted query."""
        try:
            # This would need to be implemented based on how queries are stored/retrieved
            # For now, this is a placeholder
            logger.debug(f"Warming cache for query {query_hash}")
            # await query_executor_func(query_hash)
        except Exception as e:
            logger.warning(f"Failed to warm cache for query {query_hash}: {e}")


# Global adaptive cache instance
adaptive_cache = AdaptiveCacheManager()
predictive_warmer = PredictiveCacheWarmer(adaptive_cache)