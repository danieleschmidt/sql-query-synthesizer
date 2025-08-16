"""
Advanced Connection Pool Management
High-performance connection pooling with health monitoring and auto-scaling.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncContextManager, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states."""

    CREATING = "creating"
    AVAILABLE = "available"
    IN_USE = "in_use"
    VALIDATING = "validating"
    INVALID = "invalid"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class PoolConfiguration:
    """Connection pool configuration."""

    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    pool_recycle_seconds: int = 3600  # 1 hour
    pool_pre_ping: bool = True
    pool_timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    health_check_interval_seconds: int = 60
    connection_timeout_seconds: float = 10.0
    query_timeout_seconds: float = 30.0


@dataclass
class PooledConnection:
    """Wrapper for a pooled connection."""

    connection_id: str
    connection: Any
    created_at: datetime
    last_used: datetime
    use_count: int
    state: ConnectionState
    pool_name: str
    health_check_passed: bool = True
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class HealthChecker:
    """Health checker for database connections."""

    def __init__(self, check_function: Optional[Callable] = None):
        self.check_function = check_function or self._default_health_check
        self.health_history = defaultdict(lambda: deque(maxlen=10))

    async def check_connection_health(self, connection: Any) -> bool:
        """Check if a connection is healthy."""
        try:
            result = await self.check_function(connection)
            return bool(result)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def record_health_check(self, connection_id: str, healthy: bool):
        """Record health check result."""
        self.health_history[connection_id].append(
            {"timestamp": datetime.utcnow(), "healthy": healthy}
        )

    def get_health_score(self, connection_id: str) -> float:
        """Get health score for a connection (0.0 - 1.0)."""
        history = self.health_history[connection_id]
        if not history:
            return 1.0

        healthy_count = sum(1 for record in history if record["healthy"])
        return healthy_count / len(history)

    async def _default_health_check(self, connection: Any) -> bool:
        """Default health check - attempts a simple query."""
        if hasattr(connection, "execute"):
            try:
                await connection.execute("SELECT 1")
                return True
            except:
                return False
        return True


class ConnectionPool:
    """Advanced connection pool with health monitoring."""

    def __init__(
        self,
        pool_name: str,
        connection_factory: Callable,
        config: PoolConfiguration,
        health_checker: Optional[HealthChecker] = None,
        async_factory: bool = True,
    ):

        self.pool_name = pool_name
        self.connection_factory = connection_factory
        self.config = config
        self.async_factory = async_factory
        self.health_checker = health_checker or HealthChecker()

        # Connection tracking
        self.connections: Dict[str, PooledConnection] = {}
        self.available_connections = deque()
        self.in_use_connections = set()

        # Pool state
        self.total_created = 0
        self.total_closed = 0
        self.current_overflow = 0
        self.creation_in_progress = 0

        # Statistics
        self.stats = {
            "connections_requested": 0,
            "connections_served": 0,
            "connections_created": 0,
            "connections_recycled": 0,
            "health_checks_performed": 0,
            "failed_health_checks": 0,
            "connection_errors": 0,
            "pool_exhausted_events": 0,
            "avg_wait_time_ms": 0.0,
        }

        # Synchronization
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)

        # Background tasks
        self.maintenance_task = None
        self.health_check_task = None
        self._running = False

    async def start(self):
        """Start the connection pool."""
        async with self.lock:
            if self._running:
                return

            self._running = True

            # Create initial connections
            await self._ensure_min_connections()

            # Start background tasks
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
            self.health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info(
                f"Connection pool '{self.pool_name}' started with {len(self.connections)} connections"
            )

    async def stop(self):
        """Stop the connection pool."""
        async with self.lock:
            if not self._running:
                return

            self._running = False

            # Cancel background tasks
            if self.maintenance_task:
                self.maintenance_task.cancel()
            if self.health_check_task:
                self.health_check_task.cancel()

            # Close all connections
            for connection_id in list(self.connections.keys()):
                await self._close_connection(connection_id)

            logger.info(f"Connection pool '{self.pool_name}' stopped")

    @asynccontextmanager
    async def acquire(
        self, timeout: Optional[float] = None
    ) -> AsyncContextManager[Any]:
        """Acquire a connection from the pool."""
        timeout = timeout or self.config.pool_timeout_seconds
        start_time = time.time()

        try:
            connection = await self._get_connection(timeout)
            self.stats["connections_served"] += 1

            yield connection.connection

        except Exception as e:
            self.stats["connection_errors"] += 1
            logger.error(f"Error with pooled connection: {e}")
            raise
        finally:
            # Connection is automatically returned in _get_connection context
            wait_time_ms = (time.time() - start_time) * 1000
            self._update_avg_wait_time(wait_time_ms)

    async def _get_connection(self, timeout: float) -> PooledConnection:
        """Get a connection from the pool."""
        deadline = time.time() + timeout

        async with self.condition:
            while True:
                # Try to get an available connection
                connection = await self._try_get_available_connection()
                if connection:
                    return connection

                # Check if we can create a new connection
                if await self._can_create_connection():
                    connection = await self._create_new_connection()
                    if connection:
                        return connection

                # Wait for a connection to become available
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    self.stats["pool_exhausted_events"] += 1
                    raise asyncio.TimeoutError(
                        f"Could not acquire connection from pool '{self.pool_name}' within {timeout}s"
                    )

                try:
                    await asyncio.wait_for(
                        self.condition.wait(), timeout=remaining_time
                    )
                except asyncio.TimeoutError:
                    self.stats["pool_exhausted_events"] += 1
                    raise asyncio.TimeoutError(
                        f"Could not acquire connection from pool '{self.pool_name}' within {timeout}s"
                    )

    async def _try_get_available_connection(self) -> Optional[PooledConnection]:
        """Try to get an available connection."""
        self.stats["connections_requested"] += 1

        while self.available_connections:
            connection_id = self.available_connections.popleft()
            connection = self.connections.get(connection_id)

            if not connection or connection.state != ConnectionState.AVAILABLE:
                continue

            # Check if connection needs to be recycled
            if self._should_recycle_connection(connection):
                await self._recycle_connection(connection_id)
                continue

            # Validate connection if pre-ping is enabled
            if self.config.pool_pre_ping:
                if not await self._validate_connection(connection):
                    await self._close_connection(connection_id)
                    continue

            # Connection is good to use
            connection.state = ConnectionState.IN_USE
            connection.last_used = datetime.utcnow()
            connection.use_count += 1
            self.in_use_connections.add(connection_id)

            return connection

        return None

    async def _can_create_connection(self) -> bool:
        """Check if we can create a new connection."""
        total_connections = len(self.connections) + self.creation_in_progress
        return total_connections < (self.config.max_size + self.config.max_overflow)

    async def _create_new_connection(self) -> Optional[PooledConnection]:
        """Create a new connection."""
        if self.creation_in_progress >= 5:  # Limit concurrent creation
            return None

        self.creation_in_progress += 1

        try:
            connection_id = f"{self.pool_name}_{uuid.uuid4().hex[:8]}"

            # Create the actual connection
            if self.async_factory:
                raw_connection = await self.connection_factory()
            else:
                loop = asyncio.get_event_loop()
                raw_connection = await loop.run_in_executor(
                    None, self.connection_factory
                )

            # Wrap in PooledConnection
            pooled_connection = PooledConnection(
                connection_id=connection_id,
                connection=raw_connection,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                use_count=1,
                state=ConnectionState.IN_USE,
                pool_name=self.pool_name,
                metadata={},
            )

            self.connections[connection_id] = pooled_connection
            self.in_use_connections.add(connection_id)

            self.total_created += 1
            self.stats["connections_created"] += 1

            if len(self.connections) > self.config.max_size:
                self.current_overflow += 1

            logger.debug(
                f"Created new connection {connection_id} for pool {self.pool_name}"
            )
            return pooled_connection

        except Exception as e:
            logger.error(f"Failed to create connection for pool {self.pool_name}: {e}")
            self.stats["connection_errors"] += 1
            return None
        finally:
            self.creation_in_progress -= 1

    async def return_connection(self, connection: Any):
        """Return a connection to the pool."""
        async with self.condition:
            # Find the pooled connection
            pooled_connection = None
            for conn in self.connections.values():
                if conn.connection is connection:
                    pooled_connection = conn
                    break

            if not pooled_connection:
                logger.warning("Attempted to return unknown connection to pool")
                return

            connection_id = pooled_connection.connection_id

            # Remove from in-use set
            self.in_use_connections.discard(connection_id)

            # Check if connection should be closed
            if (
                not self._running
                or pooled_connection.error_count > 5
                or not pooled_connection.health_check_passed
            ):
                await self._close_connection(connection_id)
            else:
                # Return to available pool
                pooled_connection.state = ConnectionState.AVAILABLE
                self.available_connections.append(connection_id)

            # Notify waiters
            self.condition.notify()

    async def _validate_connection(self, connection: PooledConnection) -> bool:
        """Validate a connection's health."""
        try:
            connection.state = ConnectionState.VALIDATING
            healthy = await self.health_checker.check_connection_health(
                connection.connection
            )

            self.health_checker.record_health_check(connection.connection_id, healthy)
            connection.health_check_passed = healthy
            connection.last_health_check = datetime.utcnow()

            self.stats["health_checks_performed"] += 1
            if not healthy:
                self.stats["failed_health_checks"] += 1
                connection.error_count += 1

            return healthy

        except Exception as e:
            logger.error(f"Error validating connection {connection.connection_id}: {e}")
            self.stats["failed_health_checks"] += 1
            connection.error_count += 1
            return False

    def _should_recycle_connection(self, connection: PooledConnection) -> bool:
        """Check if a connection should be recycled."""
        age = datetime.utcnow() - connection.created_at
        return age.total_seconds() > self.config.pool_recycle_seconds

    async def _recycle_connection(self, connection_id: str):
        """Recycle an old connection."""
        await self._close_connection(connection_id)
        self.stats["connections_recycled"] += 1
        logger.debug(f"Recycled connection {connection_id}")

    async def _close_connection(self, connection_id: str):
        """Close and remove a connection from the pool."""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        try:
            connection.state = ConnectionState.CLOSING

            # Close the actual connection
            if hasattr(connection.connection, "close"):
                if asyncio.iscoroutinefunction(connection.connection.close):
                    await connection.connection.close()
                else:
                    connection.connection.close()

            connection.state = ConnectionState.CLOSED

        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {e}")
        finally:
            # Remove from tracking
            self.connections.pop(connection_id, None)
            self.in_use_connections.discard(connection_id)

            try:
                self.available_connections.remove(connection_id)
            except ValueError:
                pass

            self.total_closed += 1

            if self.current_overflow > 0:
                self.current_overflow -= 1

    async def _ensure_min_connections(self):
        """Ensure minimum number of connections are available."""
        while len(self.connections) < self.config.min_size and self._running:
            connection = await self._create_new_connection()
            if connection:
                # Return to available pool
                connection.state = ConnectionState.AVAILABLE
                connection.use_count = 0
                self.in_use_connections.discard(connection.connection_id)
                self.available_connections.append(connection.connection_id)
            else:
                break

    async def _maintenance_loop(self):
        """Background maintenance loop."""
        while self._running:
            try:
                async with self.lock:
                    # Ensure minimum connections
                    await self._ensure_min_connections()

                    # Clean up old connections
                    await self._cleanup_old_connections()

            except Exception as e:
                logger.error(f"Error in pool maintenance: {e}")

            await asyncio.sleep(self.config.health_check_interval_seconds)

    async def _health_check_loop(self):
        """Background health check loop."""
        while self._running:
            try:
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            await asyncio.sleep(self.config.health_check_interval_seconds)

    async def _perform_health_checks(self):
        """Perform health checks on idle connections."""
        async with self.lock:
            # Check a subset of available connections
            connections_to_check = list(self.available_connections)[:5]

            for connection_id in connections_to_check:
                connection = self.connections.get(connection_id)
                if connection and connection.state == ConnectionState.AVAILABLE:
                    if not await self._validate_connection(connection):
                        await self._close_connection(connection_id)

    async def _cleanup_old_connections(self):
        """Clean up old and unused connections."""
        now = datetime.utcnow()
        connections_to_close = []

        for connection_id, connection in self.connections.items():
            # Don't close in-use connections
            if connection_id in self.in_use_connections:
                continue

            # Close connections that are too old or have too many errors
            age = now - connection.created_at
            if (
                age.total_seconds() > self.config.pool_recycle_seconds
                or connection.error_count > 10
            ):
                connections_to_close.append(connection_id)

        for connection_id in connections_to_close:
            await self._close_connection(connection_id)

    def _update_avg_wait_time(self, wait_time_ms: float):
        """Update average wait time."""
        current_avg = self.stats["avg_wait_time_ms"]
        total_requests = self.stats["connections_requested"]

        if total_requests > 1:
            self.stats["avg_wait_time_ms"] = (
                current_avg * (total_requests - 1) + wait_time_ms
            ) / total_requests
        else:
            self.stats["avg_wait_time_ms"] = wait_time_ms

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_name": self.pool_name,
            "configuration": asdict(self.config),
            "connections": {
                "total": len(self.connections),
                "available": len(self.available_connections),
                "in_use": len(self.in_use_connections),
                "creating": self.creation_in_progress,
                "overflow": self.current_overflow,
            },
            "lifetime_stats": {
                "total_created": self.total_created,
                "total_closed": self.total_closed,
            },
            "performance_stats": self.stats.copy(),
            "health_scores": {
                conn_id: self.health_checker.get_health_score(conn_id)
                for conn_id in self.connections.keys()
            },
        }


class ConnectionPoolManager:
    """Manages multiple connection pools."""

    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self.default_config = PoolConfiguration()

    def create_pool(
        self,
        pool_name: str,
        connection_factory: Callable,
        config: Optional[PoolConfiguration] = None,
        **kwargs,
    ) -> ConnectionPool:
        """Create a new connection pool."""
        config = config or self.default_config

        pool = ConnectionPool(
            pool_name=pool_name,
            connection_factory=connection_factory,
            config=config,
            **kwargs,
        )

        self.pools[pool_name] = pool
        logger.info(f"Created connection pool: {pool_name}")

        return pool

    async def start_all_pools(self):
        """Start all registered pools."""
        for pool in self.pools.values():
            await pool.start()

    async def stop_all_pools(self):
        """Stop all registered pools."""
        for pool in self.pools.values():
            await pool.stop()

    def get_pool(self, pool_name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name."""
        return self.pools.get(pool_name)

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        return {
            pool_name: pool.get_statistics() for pool_name, pool in self.pools.items()
        }
