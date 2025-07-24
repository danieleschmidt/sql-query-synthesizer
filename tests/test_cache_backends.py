"""Tests for cache backend abstraction and Redis/Memcached support."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, Optional

from sql_synthesizer.cache import CacheBackend, TTLCacheBackend, RedisCacheBackend, MemcachedCacheBackend
from sql_synthesizer.cache import create_cache_backend, CacheError


class TestCacheBackendInterface:
    """Test the abstract cache backend interface."""
    
    def test_cache_backend_is_abstract(self):
        """Test that CacheBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CacheBackend()


class TestTTLCacheBackend:
    """Test TTL cache backend wrapper."""
    
    def test_ttl_cache_backend_get_set(self):
        """Test basic get/set operations."""
        cache = TTLCacheBackend(ttl=60)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test get non-existent key
        with pytest.raises(KeyError):
            cache.get("nonexistent")
    
    def test_ttl_cache_backend_stats(self):
        """Test statistics functionality."""
        cache = TTLCacheBackend(ttl=60)
        
        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Get stats
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert "hit_count" in stats
        assert "miss_count" in stats
    
    def test_ttl_cache_backend_cleanup(self):
        """Test cleanup functionality."""
        cache = TTLCacheBackend(ttl=60)
        
        cache.set("key1", "value1")
        result = cache.cleanup_expired()
        
        # Should return cleanup stats
        assert isinstance(result, dict)
        assert "total_cleaned" in result


class TestRedisCacheBackend:
    """Test Redis cache backend."""
    
    @patch('sql_synthesizer.cache.redis')
    def test_redis_backend_creation(self, mock_redis):
        """Test Redis backend creation."""
        mock_client = Mock()
        mock_redis.Redis.return_value = mock_client
        
        cache = RedisCacheBackend(host="localhost", port=6379, db=0, ttl=60)
        
        # Verify Redis client was created
        mock_redis.Redis.assert_called_once_with(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
    
    @patch('sql_synthesizer.cache.redis')
    def test_redis_backend_get_set(self, mock_redis):
        """Test Redis get/set operations."""
        mock_client = Mock()
        mock_redis.Redis.return_value = mock_client
        
        cache = RedisCacheBackend(host="localhost", port=6379, db=0, ttl=60)
        
        # Test set
        cache.set("key1", "value1")
        mock_client.setex.assert_called_once_with("key1", 60, "value1")
        
        # Test get - success
        mock_client.get.return_value = "value1"
        result = cache.get("key1")
        assert result == "value1"
        
        # Test get - miss
        mock_client.get.return_value = None
        with pytest.raises(KeyError):
            cache.get("nonexistent")
    
    @patch('sql_synthesizer.cache.redis')
    def test_redis_backend_connection_error(self, mock_redis):
        """Test Redis connection error handling."""
        mock_redis.Redis.side_effect = ConnectionError("Redis unavailable")
        
        with pytest.raises(CacheError):
            RedisCacheBackend(host="localhost", port=6379, db=0, ttl=60)
    
    @patch('sql_synthesizer.cache.redis')
    def test_redis_backend_operation_error(self, mock_redis):
        """Test Redis operation error handling."""
        mock_client = Mock()
        mock_redis.Redis.return_value = mock_client
        mock_client.setex.side_effect = ConnectionError("Connection lost")
        
        cache = RedisCacheBackend(host="localhost", port=6379, db=0, ttl=60)
        
        with pytest.raises(CacheError):
            cache.set("key1", "value1")


class TestMemcachedCacheBackend:
    """Test Memcached cache backend."""
    
    @patch('sql_synthesizer.cache.pymemcache')
    def test_memcached_backend_creation(self, mock_pymemcache):
        """Test Memcached backend creation."""
        mock_client = Mock()
        mock_pymemcache.Client.return_value = mock_client
        
        cache = MemcachedCacheBackend(servers=["localhost:11211"], ttl=60)
        
        # Verify Memcached client was created
        mock_pymemcache.Client.assert_called_once_with(
            ["localhost:11211"],
            timeout=5,
            connect_timeout=5
        )
    
    @patch('sql_synthesizer.cache.pymemcache')
    def test_memcached_backend_get_set(self, mock_pymemcache):
        """Test Memcached get/set operations."""
        mock_client = Mock()
        mock_pymemcache.Client.return_value = mock_client
        
        cache = MemcachedCacheBackend(servers=["localhost:11211"], ttl=60)
        
        # Test set
        cache.set("key1", "value1")
        mock_client.set.assert_called_once_with("key1", "value1", expire=60)
        
        # Test get - success
        mock_client.get.return_value = "value1"
        result = cache.get("key1")
        assert result == "value1"
        
        # Test get - miss
        mock_client.get.return_value = None
        with pytest.raises(KeyError):
            cache.get("nonexistent")


class TestCacheBackendFactory:
    """Test cache backend factory function."""
    
    def test_create_ttl_cache_backend(self):
        """Test creating TTL cache backend."""
        cache = create_cache_backend(
            backend_type="memory",
            ttl=60,
            max_size=100
        )
        
        assert isinstance(cache, TTLCacheBackend)
    
    @patch('sql_synthesizer.cache.redis')
    def test_create_redis_cache_backend(self, mock_redis):
        """Test creating Redis cache backend."""
        mock_redis.Redis.return_value = Mock()
        
        cache = create_cache_backend(
            backend_type="redis",
            ttl=60,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0
        )
        
        assert isinstance(cache, RedisCacheBackend)
    
    @patch('sql_synthesizer.cache.pymemcache')
    def test_create_memcached_cache_backend(self, mock_pymemcache):
        """Test creating Memcached cache backend."""
        mock_pymemcache.Client.return_value = Mock()
        
        cache = create_cache_backend(
            backend_type="memcached",
            ttl=60,
            memcached_servers=["localhost:11211"]
        )
        
        assert isinstance(cache, MemcachedCacheBackend)
    
    def test_create_invalid_backend_type(self):
        """Test creating invalid backend type."""
        with pytest.raises(ValueError, match="Unknown cache backend type"):
            create_cache_backend(backend_type="invalid", ttl=60)


class TestCacheBackendIntegration:
    """Integration tests for cache backends."""
    
    def test_cache_backend_consistency(self):
        """Test that all backends have consistent interface."""
        # Test TTL backend
        ttl_cache = create_cache_backend("memory", ttl=60)
        
        # Basic operations should work on all backends
        ttl_cache.set("test_key", "test_value")
        assert ttl_cache.get("test_key") == "test_value"
        
        stats = ttl_cache.get_stats()
        assert isinstance(stats, dict)
        assert "size" in stats
        
        ttl_cache.clear()
        with pytest.raises(KeyError):
            ttl_cache.get("test_key")
    
    def test_cache_backend_error_handling(self):
        """Test error handling across backends."""
        cache = create_cache_backend("memory", ttl=60)
        
        # Test get non-existent key
        with pytest.raises(KeyError):
            cache.get("nonexistent_key")
        
        # Clear should not raise errors
        cache.clear()
        cache.clear()  # Should be safe to call multiple times