"""
Intelligent caching system for quantum optimization plans
"""

import collections
import hashlib
import pickle
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .core import QueryPlan


class CacheStrategy(Enum):
    """Cache eviction strategies"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class CacheEntry:
    """Represents a cached quantum plan"""

    key: str
    plan: QueryPlan
    created_at: float
    last_accessed: float
    access_count: int = 0
    cost_reduction: float = 0.0
    success_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Age of cache entry in seconds"""
        return time.time() - self.created_at

    @property
    def idle_time(self) -> float:
        """Time since last access in seconds"""
        return time.time() - self.last_accessed

    def access(self):
        """Mark entry as accessed"""
        self.access_count += 1
        self.last_accessed = time.time()

    def calculate_value_score(self) -> float:
        """Calculate value score for adaptive caching"""
        # Higher score = more valuable to keep in cache
        recency_score = 1.0 / (1.0 + self.idle_time / 3600.0)  # Decay over hours
        frequency_score = min(self.access_count / 10.0, 1.0)  # Max score at 10 accesses
        performance_score = self.cost_reduction * self.success_rate

        return recency_score * 0.4 + frequency_score * 0.3 + performance_score * 0.3


class QuantumPlanCache:
    """
    High-performance intelligent cache for quantum optimization plans
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float = 3600.0,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_persistence: bool = False,
        persistence_path: Optional[str] = None,
    ):

        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_persistence = persistence_path is not None
        self.persistence_path = persistence_path

        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = collections.OrderedDict()  # For LRU
        self._frequency_counter = collections.Counter()  # For LFU

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_requests = 0

        # Thread safety
        self._lock = threading.RLock()

        # Performance tracking
        self._performance_history: List[Tuple[str, float, float]] = (
            []
        )  # (key, cost_reduction, success)

        # Load from persistence if enabled
        if self.enable_persistence:
            self._load_from_disk()

    def get_cache_key(self, query_components: Dict[str, Any]) -> str:
        """
        Generate cache key from query components

        Args:
            query_components: Dictionary containing tables, joins, filters, etc.

        Returns:
            Unique cache key string
        """
        # Normalize components for consistent caching
        normalized = {
            "tables": sorted(query_components.get("tables", [])),
            "joins": sorted(query_components.get("joins", [])),
            "filters": sorted([str(f) for f in query_components.get("filters", [])]),
            "aggregations": sorted(query_components.get("aggregations", [])),
            "limit": query_components.get("limit"),
            "order_by": query_components.get("order_by"),
        }

        # Create hash from normalized components
        cache_data = str(sorted(normalized.items()))
        return hashlib.sha256(cache_data.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[QueryPlan]:
        """
        Retrieve cached plan

        Args:
            key: Cache key

        Returns:
            Cached query plan or None if not found
        """
        with self._lock:
            self._total_requests += 1

            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL expiration
            if self._is_expired(entry):
                self._remove_entry(key)
                self._misses += 1
                return None

            # Update access information
            entry.access()
            self._update_access_order(key)

            self._hits += 1
            return entry.plan

    def put(
        self,
        key: str,
        plan: QueryPlan,
        ttl: Optional[float] = None,
        cost_reduction: float = 0.0,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Store plan in cache

        Args:
            key: Cache key
            plan: Query plan to cache
            ttl: Time to live (optional)
            cost_reduction: Performance improvement this plan provides
            metadata: Additional metadata

        Returns:
            True if stored, False if cache is full and entry couldn't be evicted
        """
        with self._lock:
            # Check if we need to make space
            if len(self._cache) >= self.max_size and key not in self._cache:
                if not self._evict_entry():
                    return False  # Could not evict any entry

            current_time = time.time()

            # Create or update entry
            if key in self._cache:
                entry = self._cache[key]
                entry.plan = plan
                entry.last_accessed = current_time
                entry.cost_reduction = cost_reduction
                if metadata:
                    entry.metadata.update(metadata)
            else:
                entry = CacheEntry(
                    key=key,
                    plan=plan,
                    created_at=current_time,
                    last_accessed=current_time,
                    cost_reduction=cost_reduction,
                    metadata=metadata or {},
                )
                self._cache[key] = entry

            # Update tracking structures
            self._update_access_order(key)
            self._frequency_counter[key] += 1

            # Update performance history
            self._performance_history.append((key, cost_reduction, 1.0))
            if len(self._performance_history) > 1000:
                self._performance_history = self._performance_history[-1000:]

            # Persist if enabled
            if self.enable_persistence:
                self._persist_to_disk()

            return True

    def invalidate(self, key: str) -> bool:
        """
        Remove entry from cache

        Args:
            key: Cache key to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_counter.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._total_requests = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            hit_rate = (
                self._hits / self._total_requests if self._total_requests > 0 else 0.0
            )

            # Calculate average cost reduction
            if self._performance_history:
                avg_cost_reduction = sum(
                    cr for _, cr, _ in self._performance_history
                ) / len(self._performance_history)
            else:
                avg_cost_reduction = 0.0

            # Memory usage estimation (rough)
            estimated_memory = len(self._cache) * 1024  # 1KB per entry estimate

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "total_requests": self._total_requests,
                "hit_rate": hit_rate,
                "average_cost_reduction": avg_cost_reduction,
                "estimated_memory_bytes": estimated_memory,
                "strategy": self.strategy.value,
                "default_ttl": self.default_ttl,
            }

    def get_top_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top cache entries by value score"""
        with self._lock:
            entries_with_scores = [
                (entry, entry.calculate_value_score()) for entry in self._cache.values()
            ]

            # Sort by score descending
            entries_with_scores.sort(key=lambda x: x[1], reverse=True)

            return [
                {
                    "key": entry.key,
                    "age": entry.age,
                    "access_count": entry.access_count,
                    "cost_reduction": entry.cost_reduction,
                    "value_score": score,
                    "plan_cost": entry.plan.cost,
                }
                for entry, score in entries_with_scores[:limit]
            ]

    def optimize_cache(self):
        """Optimize cache by removing low-value entries"""
        with self._lock:
            if len(self._cache) < self.max_size * 0.8:
                return  # Cache not full enough to optimize

            # Calculate value scores for all entries
            entries_with_scores = [
                (key, entry, entry.calculate_value_score())
                for key, entry in self._cache.items()
            ]

            # Sort by score ascending (lowest first)
            entries_with_scores.sort(key=lambda x: x[2])

            # Remove bottom 20% of entries
            remove_count = max(1, len(entries_with_scores) // 5)

            for key, _, _ in entries_with_scores[:remove_count]:
                self._remove_entry(key)
                self._evictions += 1

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if self.default_ttl <= 0:
            return False
        return entry.age > self.default_ttl

    def _evict_entry(self) -> bool:
        """Evict an entry based on the configured strategy"""
        if not self._cache:
            return False

        if self.strategy == CacheStrategy.LRU:
            key = next(iter(self._access_order))  # Oldest in access order
        elif self.strategy == CacheStrategy.LFU:
            key = self._frequency_counter.most_common()[-1][0]  # Least frequent
        elif self.strategy == CacheStrategy.TTL:
            # Find oldest entry
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k].created_at
            )
            key = oldest_key
        else:  # ADAPTIVE
            # Find entry with lowest value score
            key = min(
                self._cache.keys(), key=lambda k: self._cache[k].calculate_value_score()
            )

        self._remove_entry(key)
        self._evictions += 1
        return True

    def _remove_entry(self, key: str):
        """Remove entry and update tracking structures"""
        if key in self._cache:
            del self._cache[key]

        if key in self._access_order:
            del self._access_order[key]

        if key in self._frequency_counter:
            del self._frequency_counter[key]

    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self._access_order:
            del self._access_order[key]
        self._access_order[key] = True

    def _persist_to_disk(self):
        """Persist cache to disk (simplified implementation)"""
        if not self.persistence_path:
            return

        try:
            # Only persist top entries to avoid large files
            top_entries = [(k, v) for k, v in self._cache.items() if v.access_count > 1]

            with open(self.persistence_path, "wb") as f:
                pickle.dump(top_entries, f)

        except Exception:
            # Silently fail persistence - not critical
            pass

    def _load_from_disk(self):
        """Load cache from disk"""
        if not self.persistence_path:
            return

        try:
            with open(self.persistence_path, "rb") as f:
                entries = pickle.load(f)

            # Restore entries (with validation)
            for key, entry in entries:
                if isinstance(entry, CacheEntry) and len(self._cache) < self.max_size:
                    # Reset timestamps to avoid immediate expiration
                    entry.created_at = time.time()
                    entry.last_accessed = time.time()
                    self._cache[key] = entry

        except (FileNotFoundError, pickle.PickleError, EOFError):
            # File doesn't exist or is corrupted - start fresh
            pass


class QuantumCacheManager:
    """
    Multi-level cache manager for quantum optimization
    """

    def __init__(self):
        # L1 Cache: Small, fast, in-memory cache for recent plans
        self.l1_cache = QuantumPlanCache(
            max_size=100, default_ttl=300.0, strategy=CacheStrategy.LRU  # 5 minutes
        )

        # L2 Cache: Larger cache for frequently used plans
        self.l2_cache = QuantumPlanCache(
            max_size=5000, default_ttl=3600.0, strategy=CacheStrategy.ADAPTIVE  # 1 hour
        )

        # L3 Cache: Long-term cache for high-value plans
        self.l3_cache = QuantumPlanCache(
            max_size=50000,
            default_ttl=86400.0,  # 24 hours
            strategy=CacheStrategy.ADAPTIVE,
            enable_persistence=True,
            persistence_path="/tmp/quantum_cache_l3.pkl",
        )

        self._cache_levels = [self.l1_cache, self.l2_cache, self.l3_cache]

    def get(self, key: str) -> Optional[QueryPlan]:
        """Get plan from multi-level cache"""
        # Try each cache level
        for i, cache in enumerate(self._cache_levels):
            plan = cache.get(key)
            if plan:
                # Cache hit - promote to higher levels
                for j in range(i):
                    self._cache_levels[j].put(key, plan)
                return plan

        return None

    def put(
        self,
        key: str,
        plan: QueryPlan,
        cost_reduction: float = 0.0,
        metadata: Dict[str, Any] = None,
    ):
        """Store plan in appropriate cache levels"""
        # Always store in L1
        self.l1_cache.put(key, plan, cost_reduction=cost_reduction, metadata=metadata)

        # Store in L2 if high value
        if cost_reduction > 0.1:  # 10% improvement
            self.l2_cache.put(
                key, plan, cost_reduction=cost_reduction, metadata=metadata
            )

        # Store in L3 if very high value
        if cost_reduction > 0.3:  # 30% improvement
            self.l3_cache.put(
                key, plan, cost_reduction=cost_reduction, metadata=metadata
            )

    def invalidate(self, key: str):
        """Invalidate key from all cache levels"""
        for cache in self._cache_levels:
            cache.invalidate(key)

    def get_combined_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache levels"""
        stats = {}
        for i, cache in enumerate(self._cache_levels, 1):
            stats[f"l{i}_cache"] = cache.get_stats()

        # Calculate combined hit rate
        total_hits = sum(stats[f"l{i}_cache"]["hits"] for i in range(1, 4))
        total_requests = sum(
            stats[f"l{i}_cache"]["total_requests"] for i in range(1, 4)
        )
        combined_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

        stats["combined"] = {
            "total_hits": total_hits,
            "total_requests": total_requests,
            "combined_hit_rate": combined_hit_rate,
        }

        return stats

    def optimize_all_caches(self):
        """Optimize all cache levels"""
        for cache in self._cache_levels:
            cache.optimize_cache()


# Global cache manager instance
quantum_cache_manager = QuantumCacheManager()
