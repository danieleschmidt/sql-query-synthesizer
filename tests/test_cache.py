from sql_synthesizer.cache import TTLCache
import time


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
