"""Database connection management with pooling and health monitoring."""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncContextManager
from contextlib import asynccontextmanager
from dataclasses import dataclass

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
    from sqlalchemy.pool import QueuePool, StaticPool
    from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
    from sqlalchemy import text, inspect
except ImportError:
    # Graceful fallback for testing without SQLAlchemy
    create_async_engine = None
    AsyncEngine = None
    AsyncSession = None
    async_sessionmaker = None
    QueuePool = None
    StaticPool = None
    SQLAlchemyError = Exception
    DisconnectionError = Exception
    text = None
    inspect = None

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    connect_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30


@dataclass
class ConnectionStats:
    """Database connection pool statistics."""
    pool_size: int
    checked_out: int
    overflow: int
    checked_in: int
    total_connections: int
    invalid_connections: int
    health_score: float  # 0.0 to 1.0


class DatabaseManager:
    """Manages database connections with health monitoring and retry logic."""
    
    def __init__(self, config: ConnectionConfig):
        """Initialize database manager with configuration."""
        if create_async_engine is None:
            raise ImportError("SQLAlchemy async support not available")
        
        self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._health_check_cache = {}
        self._last_health_check = 0
        self._connection_errors = 0
        
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            # Determine pool class based on database type
            if 'sqlite' in self.config.url.lower():
                # SQLite uses StaticPool for better concurrency
                pool_class = StaticPool
                pool_size = 1
                max_overflow = 0
            else:
                pool_class = QueuePool
                pool_size = self.config.pool_size
                max_overflow = self.config.max_overflow
            
            self.engine = create_async_engine(
                self.config.url,
                poolclass=pool_class,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=False,  # Set to True for SQL debugging
                future=True
            )
            
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test the connection
            await self._test_connection()
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    async def _test_connection(self) -> None:
        """Test database connection with retry logic."""
        for attempt in range(self.config.connect_retries):
            try:
                async with self.engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")
                self._connection_errors = 0
                return
            except Exception as e:
                self._connection_errors += 1
                if attempt < self.config.connect_retries - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Connection test failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Database connection test failed after {self.config.connect_retries} attempts: {e}")
                    raise
    
    @asynccontextmanager
    async def session(self) -> AsyncContextManager[AsyncSession]:
        """Get database session with automatic cleanup."""
        if not self.session_factory:
            raise RuntimeError("Database manager not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a raw SQL query with parameters."""
        try:
            async with self.session() as session:
                result = await session.execute(text(query), params or {})
                if query.strip().upper().startswith('SELECT'):
                    return result.fetchall()
                else:
                    return result.rowcount
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def get_connection_stats(self) -> ConnectionStats:
        """Get current connection pool statistics."""
        if not self.engine:
            return ConnectionStats(0, 0, 0, 0, 0, 0, 0.0)
        
        pool = self.engine.pool
        
        try:
            # Calculate health score based on various factors
            total_connections = pool.checkedout() + pool.checkedin()
            max_connections = pool.size() + pool.overflow()
            
            # Health factors
            connection_utilization = total_connections / max(max_connections, 1)
            error_rate = min(self._connection_errors / 10.0, 1.0)  # Last 10 operations
            
            health_score = max(0.0, 1.0 - connection_utilization * 0.5 - error_rate * 0.5)
            
            return ConnectionStats(
                pool_size=pool.size(),
                checked_out=pool.checkedout(),
                overflow=pool.overflow(),
                checked_in=pool.checkedin(),
                total_connections=total_connections,
                invalid_connections=pool.invalidated(),
                health_score=health_score
            )
        except Exception as e:
            logger.warning(f"Failed to get connection stats: {e}")
            return ConnectionStats(0, 0, 0, 0, 0, 0, 0.0)
    
    async def health_check(self, use_cache: bool = True) -> Dict[str, Any]:
        """Perform comprehensive database health check."""
        current_time = time.time()
        cache_ttl = 30  # Cache health check for 30 seconds
        
        if use_cache and current_time - self._last_health_check < cache_ttl:
            return self._health_check_cache
        
        health_info = {
            'healthy': False,
            'response_time_ms': None,
            'connection_stats': None,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Test basic connectivity
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            
            response_time = (time.time() - start_time) * 1000
            stats = await self.get_connection_stats()
            
            health_info.update({
                'healthy': True,
                'response_time_ms': round(response_time, 2),
                'connection_stats': stats
            })
            
        except Exception as e:
            health_info['error'] = str(e)
            logger.error(f"Database health check failed: {e}")
        
        self._health_check_cache = health_info
        self._last_health_check = current_time
        
        return health_info
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific table."""
        try:
            async with self.engine.begin() as conn:
                # Get table schema information
                inspector = inspect(conn.sync_connection)
                
                # Check if table exists
                table_exists = await conn.run_sync(
                    lambda sync_conn: inspector.has_table(table_name)
                )
                
                if not table_exists:
                    return {'exists': False}
                
                # Get column information
                columns = await conn.run_sync(
                    lambda sync_conn: inspector.get_columns(table_name)
                )
                
                # Get row count (approximate)
                count_result = await conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                )
                row_count = count_result.scalar()
                
                return {
                    'exists': True,
                    'columns': [
                        {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col.get('nullable', True),
                            'primary_key': col.get('primary_key', False)
                        }
                        for col in columns
                    ],
                    'row_count': row_count
                }
                
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            return {'exists': False, 'error': str(e)}
    
    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> Optional[DatabaseManager]:
    """Get the global database manager instance."""
    return _db_manager


async def initialize_database(config: ConnectionConfig) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    
    _db_manager = DatabaseManager(config)
    await _db_manager.initialize()
    return _db_manager


async def close_database() -> None:
    """Close the global database manager."""
    global _db_manager
    
    if _db_manager:
        await _db_manager.close()
        _db_manager = None