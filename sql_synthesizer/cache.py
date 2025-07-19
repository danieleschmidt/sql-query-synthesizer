"""Enhanced in-memory TTL cache with performance metrics and optimizations."""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional
from collections import OrderedDict


class TTLCache:
    """A time-based cache with performance metrics and memory management."""

    def __init__(self, ttl: int = 0, max_size: Optional[int] = None) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        
        # Performance metrics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
        # Thread safety
        self._lock = threading.RLock()

    def get(self, key: str) -> Any:
        """Get value from cache, updating access time for LRU."""
        with self._lock:
            now = time.time()
            ts_val = self._items.get(key)
            
            if ts_val and (self.ttl <= 0 or now - ts_val[0] <= self.ttl):
                # Cache hit - move to end for LRU
                self._items.move_to_end(key)
                self._hit_count += 1
                return ts_val[1]
            
            # Cache miss or expired
            if key in self._items:
                del self._items[key]
                self._eviction_count += 1
            
            self._miss_count += 1
            raise KeyError(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with automatic size management."""
        with self._lock:
            now = time.time()
            
            # Check if key already exists
            if key in self._items:
                # Update existing key - move to end
                self._items[key] = (now, value)
                self._items.move_to_end(key)
            else:
                # New key - check size limit
                if self.max_size and len(self._items) >= self.max_size:
                    # Evict oldest item (LRU)
                    self._items.popitem(last=False)
                    self._eviction_count += 1
                
                self._items[key] = (now, value)

    def clear(self) -> None:
        """Clear all cache entries and reset metrics."""
        with self._lock:
            self._items.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of items removed."""
        if self.ttl <= 0:
            return 0
        
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, (timestamp, _) in self._items.items()
                if now - timestamp > self.ttl
            ]
            
            for key in expired_keys:
                del self._items[key]
                self._eviction_count += 1
            
            return len(expired_keys)

    @property
    def size(self) -> int:
        """Current number of items in cache."""
        return len(self._items)
    
    def __len__(self) -> int:
        """Support len() function."""
        return self.size

    @property
    def hit_count(self) -> int:
        """Number of cache hits."""
        return self._hit_count

    @property
    def miss_count(self) -> int:
        """Number of cache misses."""
        return self._miss_count

    @property
    def eviction_count(self) -> int:
        """Number of cache evictions."""
        return self._eviction_count

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        with self._lock:
            self._hit_count = 0
            self._miss_count = 0
            self._eviction_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_ops = self._hit_count + self._miss_count
            return {
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "eviction_count": self._eviction_count,
                "size": self.size,
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hit_rate": self.hit_rate,
                "total_operations": total_ops,
            }
