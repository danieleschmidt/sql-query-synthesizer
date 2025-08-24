"""Tests for cache performance metrics and optimizations."""

import time

import pytest

from sql_synthesizer.cache import TTLCache


def test_cache_hit_miss_metrics():
    """Test that cache tracks hit and miss statistics."""
    cache = TTLCache(ttl=10)

    # Initial state
    assert cache.hit_count == 0
    assert cache.miss_count == 0
    assert cache.hit_rate == 0.0

    # Cache miss
    with pytest.raises(KeyError):
        cache.get("missing")
    assert cache.miss_count == 1
    assert cache.hit_rate == 0.0

    # Cache set and hit
    cache.set("key1", "value1")
    value = cache.get("key1")
    assert value == "value1"
    assert cache.hit_count == 1
    assert cache.miss_count == 1
    assert cache.hit_rate == 0.5

    # Another hit
    cache.get("key1")
    assert cache.hit_count == 2
    assert cache.miss_count == 1
    assert cache.hit_rate == 2 / 3


def test_cache_size_tracking():
    """Test that cache tracks its current size."""
    cache = TTLCache(ttl=10)

    assert cache.size == 0
    assert len(cache) == 0

    cache.set("key1", "value1")
    assert cache.size == 1
    assert len(cache) == 1

    cache.set("key2", "value2")
    assert cache.size == 2
    assert len(cache) == 2

    # Overwrite existing key shouldn't change size
    cache.set("key1", "new_value")
    assert cache.size == 2

    # Clear should reset size
    cache.clear()
    assert cache.size == 0


def test_cache_automatic_cleanup():
    """Test that expired entries are automatically cleaned up."""
    cache = TTLCache(ttl=0.1)  # Very short TTL

    cache.set("temp1", "value1")
    cache.set("temp2", "value2")
    assert cache.size == 2

    # Wait for expiration
    time.sleep(0.15)

    # Access should trigger cleanup
    cache.cleanup_expired()
    assert cache.size == 0


def test_cache_memory_limit():
    """Test cache with memory-based eviction."""
    cache = TTLCache(ttl=10, max_size=2)

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    assert cache.size == 2

    # Adding third item should evict oldest
    cache.set("key3", "value3")
    assert cache.size == 2

    # key1 should be evicted (LRU)
    with pytest.raises(KeyError):
        cache.get("key1")

    # key2 and key3 should still exist
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_cache_statistics_reset():
    """Test that cache statistics can be reset."""
    cache = TTLCache(ttl=10)

    # Generate some stats
    cache.set("key", "value")
    cache.get("key")
    with pytest.raises(KeyError):
        cache.get("missing")

    assert cache.hit_count > 0
    assert cache.miss_count > 0

    # Reset stats
    cache.reset_stats()
    assert cache.hit_count == 0
    assert cache.miss_count == 0
    assert cache.hit_rate == 0.0


def test_cache_eviction_metrics():
    """Test that cache tracks eviction statistics."""
    cache = TTLCache(ttl=0.1, max_size=2)

    # Initial eviction count
    assert cache.eviction_count == 0

    # Fill cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    # Trigger size-based eviction
    cache.set("key3", "value3")
    assert cache.eviction_count == 1

    # Wait for expiration and trigger TTL-based eviction
    time.sleep(0.15)
    cache.cleanup_expired()
    # Should have evicted expired entries
    assert cache.eviction_count > 1


def test_cache_performance_monitoring():
    """Test cache performance monitoring capabilities."""
    cache = TTLCache(ttl=10)

    # Set up some cache operations
    for i in range(10):
        cache.set(f"key{i}", f"value{i}")

    # Generate hits and misses
    for i in range(5):
        cache.get(f"key{i}")  # hits

    for i in range(10, 15):
        try:
            cache.get(f"key{i}")  # misses
        except KeyError:
            pass

    stats = cache.get_stats()

    assert stats["hit_count"] == 5
    assert stats["miss_count"] == 5
    assert stats["size"] == 10
    assert stats["hit_rate"] == 0.5
    assert "eviction_count" in stats
    assert "total_operations" in stats


def test_cache_thread_safety_metrics():
    """Test that cache metrics are thread-safe."""
    import threading
    import time

    cache = TTLCache(ttl=10)

    def worker():
        """TODO: Add docstring"""
        for i in range(100):
            cache.set(f"key{i}", f"value{i}")
            try:
                cache.get(f"key{i}")
            except KeyError:
                pass

    threads = [threading.Thread(target=worker) for _ in range(3)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Should have consistent metrics despite concurrent access
    stats = cache.get_stats()
    assert stats["hit_count"] + stats["miss_count"] == stats["total_operations"]
    assert stats["hit_rate"] >= 0.0
    assert stats["hit_rate"] <= 1.0
