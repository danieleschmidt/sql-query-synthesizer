"""Enhanced cache system with multiple backend support (TTL, Redis, Memcached)."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

# Optional imports for cache backends
try:
    import redis
except ImportError:
    redis = None

try:
    import pymemcache.client.base as pymemcache
except ImportError:
    pymemcache = None


class CacheError(Exception):
    """Base exception for cache-related errors."""

    pass


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Any:
        """Get value from cache. Raises KeyError if not found."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def cleanup_expired(self) -> dict[str, int]:
        """Remove expired entries and return cleanup statistics."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        pass


class TTLCache:
    """A time-based cache with performance metrics and memory management."""

    def __init__(self, ttl: int = 0, max_size: int | None = None) -> None:
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
                key
                for key, (timestamp, _) in self._items.items()
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

    def get_stats(self) -> dict[str, Any]:
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


class TTLCacheBackend(CacheBackend):
    """TTL cache backend wrapper implementing the CacheBackend interface."""

    def __init__(self, ttl: int = 0, max_size: int | None = None):
        self._cache = TTLCache(ttl=ttl, max_size=max_size)

    @property
    def ttl(self) -> int:
        """Get the TTL value for this cache backend."""
        return self._cache.ttl

    def get(self, key: str) -> Any:
        """Get value from cache. Raises KeyError if not found."""
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache.set(key, value)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def cleanup_expired(self) -> dict[str, int]:
        """Remove expired entries and return cleanup statistics."""
        cleaned_count = self._cache.cleanup_expired()
        return {"total_cleaned": cleaned_count, "remaining_size": self._cache.size}

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


class RedisCacheBackend(CacheBackend):
    """Redis cache backend for distributed caching."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,
        password: str | None = None,
    ):
        if redis is None:
            raise CacheError(
                "Redis library not installed. Install with: pip install redis"
            )

        self.ttl = ttl
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.RLock()

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.client.ping()
        except (ConnectionError, redis.RedisError) as e:
            raise CacheError(f"Failed to connect to Redis: {e}")

    def get(self, key: str) -> Any:
        """Get value from cache. Raises KeyError if not found."""
        try:
            with self._lock:
                value = self.client.get(key)
                if value is None:
                    self._miss_count += 1
                    raise KeyError(key)

                self._hit_count += 1
                return value
        except redis.RedisError as e:
            raise CacheError(f"Redis get operation failed: {e}")

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with TTL."""
        try:
            self.client.setex(key, self.ttl, value)
        except redis.RedisError as e:
            raise CacheError(f"Redis set operation failed: {e}")

    def clear(self) -> None:
        """Clear all cache entries (flushes current database)."""
        try:
            self.client.flushdb()
        except redis.RedisError as e:
            raise CacheError(f"Redis clear operation failed: {e}")

    def cleanup_expired(self) -> dict[str, int]:
        """Redis automatically handles TTL, so this is a no-op."""
        return {"total_cleaned": 0, "remaining_size": -1}  # Size unknown in Redis

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_ops = self._hit_count + self._miss_count
            try:
                info = self.client.info("memory")
                memory_usage = info.get("used_memory", 0)
            except redis.RedisError:
                memory_usage = -1

            return {
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "eviction_count": -1,  # Not tracked
                "size": -1,  # Not easily available
                "max_size": None,
                "ttl": self.ttl,
                "hit_rate": self._hit_count / total_ops if total_ops > 0 else 0.0,
                "total_operations": total_ops,
                "memory_usage_bytes": memory_usage,
                "backend": "redis",
            }


class MemcachedCacheBackend(CacheBackend):
    """Memcached cache backend for distributed caching."""

    def __init__(self, servers: list[str], ttl: int = 3600):
        if pymemcache is None:
            raise CacheError(
                "pymemcache library not installed. Install with: pip install pymemcache"
            )

        self.ttl = ttl
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.RLock()

        try:
            self.client = pymemcache.Client(servers, timeout=5, connect_timeout=5)
            # Test connection by setting a test key
            self.client.set("_connection_test", "ok", expire=1)
        except (ConnectionError, OSError) as e:
            raise CacheError(f"Failed to connect to Memcached: {e}")
        except ImportError as e:
            raise CacheError(f"Memcached dependencies not available: {e}")

    def get(self, key: str) -> Any:
        """Get value from cache. Raises KeyError if not found."""
        try:
            with self._lock:
                value = self.client.get(key)
                if value is None:
                    self._miss_count += 1
                    raise KeyError(key)

                self._hit_count += 1
                return value
        except (ConnectionError, OSError) as e:
            raise CacheError(f"Memcached get operation failed: {e}")
        except (ValueError, TypeError) as e:
            raise CacheError(f"Invalid key or serialization error: {e}")

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with TTL."""
        try:
            self.client.set(key, value, expire=self.ttl)
        except (ConnectionError, OSError) as e:
            raise CacheError(f"Memcached set operation failed: {e}")
        except (ValueError, TypeError) as e:
            raise CacheError(f"Cannot serialize value for caching: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self.client.flush_all()
        except (ConnectionError, OSError) as e:
            raise CacheError(f"Memcached clear operation failed: {e}")

    def cleanup_expired(self) -> dict[str, int]:
        """Memcached automatically handles TTL, so this is a no-op."""
        return {"total_cleaned": 0, "remaining_size": -1}  # Size unknown in Memcached

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_ops = self._hit_count + self._miss_count
            try:
                stats = self.client.stats()
                memory_usage = stats.get(b"bytes", 0) if stats else 0
            except Exception:
                memory_usage = -1

            return {
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "eviction_count": -1,  # Not tracked locally
                "size": -1,  # Not easily available
                "max_size": None,
                "ttl": self.ttl,
                "hit_rate": self._hit_count / total_ops if total_ops > 0 else 0.0,
                "total_operations": total_ops,
                "memory_usage_bytes": memory_usage,
                "backend": "memcached",
            }


def create_cache_backend(backend_type: str, ttl: int = 3600, **kwargs) -> CacheBackend:
    """Factory function to create cache backends based on configuration.

    Args:
        backend_type: Type of cache backend ("memory", "redis", "memcached")
        ttl: Time-to-live for cache entries in seconds
        **kwargs: Backend-specific configuration options

    Returns:
        CacheBackend: Configured cache backend instance

    Raises:
        ValueError: If backend_type is not supported
        CacheError: If backend initialization fails
    """
    backend_type = backend_type.lower().strip()

    if backend_type == "memory":
        max_size = kwargs.get("max_size")
        return TTLCacheBackend(ttl=ttl, max_size=max_size)

    elif backend_type == "redis":
        return RedisCacheBackend(
            host=kwargs.get("redis_host", "localhost"),
            port=kwargs.get("redis_port", 6379),
            db=kwargs.get("redis_db", 0),
            password=kwargs.get("redis_password"),
            ttl=ttl,
        )

    elif backend_type == "memcached":
        servers = kwargs.get("memcached_servers", ["localhost:11211"])
        return MemcachedCacheBackend(servers=servers, ttl=ttl)

    else:
        raise ValueError(
            f"Unknown cache backend type: {backend_type}. "
            f"Supported types: memory, redis, memcached"
        )
