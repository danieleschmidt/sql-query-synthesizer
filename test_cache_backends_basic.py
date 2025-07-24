#!/usr/bin/env python3
"""Basic test for cache backend functionality."""

import sys
sys.path.insert(0, '/root/repo')

def test_memory_cache_backend():
    """Test memory cache backend creation and basic operations."""
    try:
        from sql_synthesizer.cache import create_cache_backend, CacheError
        
        # Test memory backend creation
        cache = create_cache_backend("memory", ttl=60, max_size=100)
        print("✅ Memory cache backend created successfully")
        
        # Test basic operations
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got '{value}'"
        print("✅ Memory cache get/set operations work")
        
        # Test stats
        stats = cache.get_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert "hit_count" in stats, "Stats should include hit_count"
        print("✅ Memory cache stats work")
        
        # Test cleanup
        cleanup_result = cache.cleanup_expired()
        assert isinstance(cleanup_result, dict), "Cleanup should return a dictionary"
        print("✅ Memory cache cleanup works")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory cache backend test failed: {e}")
        return False

def test_cache_backend_factory():
    """Test cache backend factory function."""
    try:
        from sql_synthesizer.cache import create_cache_backend
        
        # Test invalid backend type
        try:
            create_cache_backend("invalid_backend", ttl=60)
            print("❌ Should have raised ValueError for invalid backend")
            return False
        except ValueError as e:
            if "Unknown cache backend type" in str(e):
                print("✅ Factory correctly rejects invalid backend types")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        
        # Test memory backend
        memory_cache = create_cache_backend("memory", ttl=60)
        assert memory_cache is not None
        print("✅ Factory creates memory backend")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache backend factory test failed: {e}")
        return False

def test_config_cache_settings():
    """Test cache configuration settings."""
    try:
        import os
        from sql_synthesizer.config import Config
        
        # Test default settings
        config = Config(force_reload=True)
        assert config.cache_backend == "memory"
        assert config.cache_ttl == 3600
        assert config.cache_max_size == 1000
        print("✅ Default cache configuration loaded")
        
        # Test environment override
        os.environ["QUERY_AGENT_CACHE_BACKEND"] = "redis"
        os.environ["QUERY_AGENT_CACHE_TTL"] = "7200"
        os.environ["QUERY_AGENT_REDIS_HOST"] = "test-redis"
        
        config = Config(force_reload=True)
        assert config.cache_backend == "redis"
        assert config.cache_ttl == 7200
        assert config.redis_host == "test-redis"
        print("✅ Environment cache configuration override works")
        
        # Clean up environment
        del os.environ["QUERY_AGENT_CACHE_BACKEND"]
        del os.environ["QUERY_AGENT_CACHE_TTL"]
        del os.environ["QUERY_AGENT_REDIS_HOST"]
        
        return True
        
    except Exception as e:
        print(f"❌ Config cache settings test failed: {e}")
        return False

def test_redis_cache_backend_mock():
    """Test Redis cache backend with mock (no actual Redis needed)."""
    try:
        from sql_synthesizer.cache import RedisCacheBackend, CacheError
        
        # Test without redis library
        try:
            # This should fail gracefully if redis is not installed
            cache = RedisCacheBackend(host="nonexistent", port=6379, db=0, ttl=60)
            print("⚠️  Redis backend created (Redis may be installed and running)")
        except CacheError as e:
            if "Redis library not installed" in str(e) or "Failed to connect to Redis" in str(e):
                print("✅ Redis backend correctly handles missing Redis or connection failure")
            else:
                print(f"❌ Unexpected Redis error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Redis cache backend test failed: {e}")
        return False

def main():
    """Run all cache backend tests."""
    print("🧪 Testing cache backend implementation...\n")
    
    tests = [
        ("Memory cache backend", test_memory_cache_backend),
        ("Cache backend factory", test_cache_backend_factory),
        ("Config cache settings", test_config_cache_settings),
        ("Redis cache backend mock", test_redis_cache_backend_mock)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All cache backend tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())