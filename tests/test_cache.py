import time

from sql_synthesizer.cache import TTLCache


def test_ttl_cache_basic():
    cache = TTLCache(ttl=1)
    cache.set("a", 1)
    assert cache.get("a") == 1
    time.sleep(1.1)
    try:
        cache.get("a")
    except KeyError:
        pass
    else:
        assert False, "expected KeyError"


def test_ttl_cache_clear():
    """Test cache clear functionality."""
    cache = TTLCache(ttl=10)
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    # Verify items are in cache
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"

    # Clear cache
    cache.clear()

    # Verify items are gone
    try:
        cache.get("key1")
        assert False, "expected KeyError after clear"
    except KeyError:
        pass

    try:
        cache.get("key2")
        assert False, "expected KeyError after clear"
    except KeyError:
        pass


def test_ttl_cache_no_ttl():
    """Test cache with ttl=0 (disabled TTL)."""
    cache = TTLCache(ttl=0)
    cache.set("key", "value")

    # Should always return value when TTL is disabled
    assert cache.get("key") == "value"
    time.sleep(0.1)  # Small delay to ensure time passes
    assert cache.get("key") == "value"
