"""
Database connection management with connection pooling and error handling.

This module provides a robust database connection manager that handles:
- Connection pooling with configurable parameters
- Automatic retry logic with exponential backoff
- Connection health checks and recovery
- Proper error handling and logging
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy import exc as sqlalchemy_exc

from .config import config

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""

    pass


class DatabaseConnectionManager:
    """
    Manages database connections with pooling, retries, and health checks.

    Features:
    - Connection pooling with configurable size and overflow
    - Automatic connection health checks (pre_ping)
    - Retry logic with exponential backoff
    - Connection lifecycle event logging
    - Graceful error handling and recovery
    """

    def __init__(self, database_url: str):
        """
        Initialize the database connection manager.

        Args:
            database_url: SQLAlchemy-compatible database URL

        Raises:
            DatabaseConnectionError: If initial connection validation fails
        """
        self.database_url = database_url
        self._engine: Optional[Engine] = None
        self._connection_stats = {
            "total_connections": 0,
            "failed_connections": 0,
            "retries_attempted": 0,
            "last_error": None,
        }

        # Initialize engine with retry logic
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine with connection pooling and error handling."""
        engine_kwargs = {
            # Connection pool configuration
            "pool_pre_ping": config.db_pool_pre_ping,
            "pool_recycle": config.db_pool_recycle,
            # Engine configuration
            "echo": False,  # Set to True for SQL debugging
            "future": True,  # Use SQLAlchemy 2.0 style
        }

        # Only add pool parameters for databases that support connection pooling
        if not self.database_url.startswith("sqlite"):
            engine_kwargs["pool_size"] = config.db_pool_size
            engine_kwargs["max_overflow"] = config.db_max_overflow

        # Handle database-specific connection arguments
        connect_args = {}

        if "postgresql" in self.database_url:
            connect_args["connect_timeout"] = config.database_timeout
            connect_args["application_name"] = "sql_synthesizer"
        elif "mysql" in self.database_url:
            connect_args["connect_timeout"] = config.database_timeout
            connect_args["charset"] = "utf8mb4"
        # SQLite doesn't support connect_timeout, so we skip it

        if connect_args:
            engine_kwargs["connect_args"] = connect_args

        for attempt in range(config.db_connect_retries + 1):
            try:
                self._engine = create_engine(self.database_url, **engine_kwargs)

                # Set up connection pool event listeners
                self._setup_pool_events()

                # Validate connection
                self._validate_connection()

                logger.info(
                    "Database engine initialized successfully",
                    extra={
                        "pool_size": config.db_pool_size,
                        "max_overflow": config.db_max_overflow,
                        "pool_recycle": config.db_pool_recycle,
                        "attempt": attempt + 1,
                    },
                )
                return

            except (sqlalchemy_exc.SQLAlchemyError, ConnectionError, OSError) as e:
                self._connection_stats["failed_connections"] += 1
                self._connection_stats["last_error"] = str(e)

                if attempt < config.db_connect_retries:
                    self._connection_stats["retries_attempted"] += 1
                    delay = config.db_retry_delay * (2**attempt)  # Exponential backoff

                    logger.warning(
                        "Database connection failed, retrying",
                        extra={
                            "attempt": attempt + 1,
                            "max_retries": config.db_connect_retries,
                            "retry_delay": delay,
                            "error": str(e),
                        },
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Failed to initialize database connection after all retries",
                        extra={"total_attempts": attempt + 1, "error": str(e)},
                    )
                    raise DatabaseConnectionError(
                        f"Failed to connect to database after {attempt + 1} attempts: {e}"
                    )

    def _setup_pool_events(self) -> None:
        """Set up SQLAlchemy pool event listeners for monitoring and logging."""
        if not self._engine:
            return

        @event.listens_for(self._engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Log successful database connections."""
            self._connection_stats["total_connections"] += 1
            logger.debug("New database connection established")

        @event.listens_for(self._engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log connection checkout from pool."""
            logger.debug("Connection checked out from pool")

        @event.listens_for(self._engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log connection checkin to pool."""
            logger.debug("Connection checked back into pool")

        @event.listens_for(self._engine, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            """Log connection invalidation."""
            logger.warning(
                "Database connection invalidated",
                extra={"error": str(exception) if exception else "Unknown"},
            )

    def _validate_connection(self) -> None:
        """Validate that the database connection is working."""
        if not self._engine:
            raise DatabaseConnectionError("Engine not initialized")

        try:
            with self._engine.connect() as conn:
                # Execute a simple query to validate connection
                conn.execute(text("SELECT 1"))
                logger.debug("Database connection validation successful")
        except (sqlalchemy_exc.SQLAlchemyError, ConnectionError, OSError) as e:
            raise DatabaseConnectionError(f"Connection validation failed: {e}")

    @property
    def engine(self) -> Engine:
        """
        Get the SQLAlchemy engine.

        Returns:
            Engine: The configured SQLAlchemy engine

        Raises:
            DatabaseConnectionError: If engine is not initialized
        """
        if not self._engine:
            raise DatabaseConnectionError("Database engine not initialized")
        return self._engine

    @contextmanager
    def get_connection(self):
        """
        Get a database connection with automatic cleanup.

        This context manager provides a database connection that is automatically
        returned to the pool when the context exits, even if an exception occurs.

        Yields:
            Connection: SQLAlchemy database connection

        Raises:
            DatabaseConnectionError: If connection cannot be obtained
        """
        if not self._engine:
            raise DatabaseConnectionError("Database engine not initialized")

        connection = None
        try:
            connection = self._engine.connect()
            logger.debug("Database connection obtained from pool")
            yield connection
        except sqlalchemy_exc.DisconnectionError as e:
            logger.error("Database disconnection error", extra={"error": str(e)})
            # Attempt to reinitialize engine on disconnection
            self._initialize_engine()
            raise DatabaseConnectionError(f"Database disconnection: {e}")
        except sqlalchemy_exc.OperationalError as e:
            logger.error("Database operational error", extra={"error": str(e)})
            raise DatabaseConnectionError(f"Database operational error: {e}")
        except (sqlalchemy_exc.SQLAlchemyError, ConnectionError, OSError) as e:
            logger.error("Unexpected database error", extra={"error": str(e)})
            raise DatabaseConnectionError(f"Unexpected database error: {e}")
        finally:
            if connection:
                connection.close()
                logger.debug("Database connection returned to pool")

    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics for monitoring.

        Returns:
            Dict containing connection statistics
        """
        pool_stats = {}
        if self._engine and hasattr(self._engine.pool, "size"):
            pool_stats = {
                "pool_size": self._engine.pool.size(),
                "checked_in": self._engine.pool.checkedin(),
                "checked_out": self._engine.pool.checkedout(),
                "overflow": getattr(self._engine.pool, "overflow", lambda: 0)(),
            }

        return {
            **self._connection_stats,
            **pool_stats,
            "engine_disposed": self._engine is None
            or hasattr(self._engine, "closed")
            and self._engine.closed,
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.

        Returns:
            Dict containing health check results
        """
        health_status = {"healthy": False, "timestamp": time.time(), "error": None}

        try:
            with self.get_connection() as conn:
                # Execute a simple query
                result = conn.execute(text("SELECT 1")).scalar()
                if result == 1:
                    health_status["healthy"] = True
                    logger.debug("Database health check passed")
                else:
                    health_status["error"] = "Unexpected health check result"
        except (sqlalchemy_exc.SQLAlchemyError, ConnectionError, OSError) as e:
            health_status["error"] = str(e)
            logger.warning("Database health check failed", extra={"error": str(e)})

        return health_status

    def dispose(self) -> None:
        """Dispose of the database engine and close all connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine disposed")
            self._engine = None
