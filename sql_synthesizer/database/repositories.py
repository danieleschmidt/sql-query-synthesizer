"""Data access layer with repository pattern implementation."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Base repository with common CRUD operations."""

    def __init__(self, db_manager):
        """Initialize repository with database manager."""
        self.db_manager = db_manager

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: Any) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass

    @abstractmethod
    async def delete(self, entity_id: Any) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def list_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """List all entities with pagination."""
        pass


class QueryHistoryRepository(BaseRepository):
    """Repository for managing query history records."""

    async def create(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new query history record."""
        try:
            query = """
            INSERT INTO query_history 
            (query_id, user_question, generated_sql, execution_time_ms, 
             success, error_message, cache_hit, user_agent, client_ip, created_at)
            VALUES 
            (:query_id, :user_question, :generated_sql, :execution_time_ms,
             :success, :error_message, :cache_hit, :user_agent, :client_ip, :created_at)
            """

            params = {
                "query_id": query_data.get("query_id"),
                "user_question": query_data.get("user_question"),
                "generated_sql": query_data.get("generated_sql"),
                "execution_time_ms": query_data.get("execution_time_ms"),
                "success": query_data.get("success", True),
                "error_message": query_data.get("error_message"),
                "cache_hit": query_data.get("cache_hit", False),
                "user_agent": query_data.get("user_agent"),
                "client_ip": query_data.get("client_ip"),
                "created_at": query_data.get("created_at", datetime.utcnow()),
            }

            await self.db_manager.execute_query(query, params)
            return query_data

        except Exception as e:
            logger.error(f"Failed to create query history record: {e}")
            raise

    async def get_by_id(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get query history by query ID."""
        try:
            query = "SELECT * FROM query_history WHERE query_id = :query_id"
            result = await self.db_manager.execute_query(query, {"query_id": query_id})

            if result:
                return self._row_to_dict(result[0])
            return None

        except Exception as e:
            logger.error(f"Failed to get query history {query_id}: {e}")
            return None

    async def update(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update query history record."""
        try:
            query = """
            UPDATE query_history 
            SET user_question = :user_question,
                generated_sql = :generated_sql,
                execution_time_ms = :execution_time_ms,
                success = :success,
                error_message = :error_message,
                cache_hit = :cache_hit,
                updated_at = :updated_at
            WHERE query_id = :query_id
            """

            params = query_data.copy()
            params["updated_at"] = datetime.utcnow()

            await self.db_manager.execute_query(query, params)
            return query_data

        except Exception as e:
            logger.error(f"Failed to update query history: {e}")
            raise

    async def delete(self, query_id: str) -> bool:
        """Delete query history record."""
        try:
            query = "DELETE FROM query_history WHERE query_id = :query_id"
            rows_affected = await self.db_manager.execute_query(
                query, {"query_id": query_id}
            )
            return rows_affected > 0

        except Exception as e:
            logger.error(f"Failed to delete query history {query_id}: {e}")
            return False

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List all query history records with pagination."""
        try:
            query = """
            SELECT * FROM query_history 
            ORDER BY created_at DESC 
            LIMIT :limit OFFSET :offset
            """

            result = await self.db_manager.execute_query(
                query, {"limit": limit, "offset": offset}
            )

            return [self._row_to_dict(row) for row in result]

        except Exception as e:
            logger.error(f"Failed to list query history: {e}")
            return []

    async def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent successful queries."""
        try:
            query = """
            SELECT * FROM query_history 
            WHERE success = true 
            ORDER BY created_at DESC 
            LIMIT :limit
            """

            result = await self.db_manager.execute_query(query, {"limit": limit})
            return [self._row_to_dict(row) for row in result]

        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            return []

    async def get_query_statistics(self) -> Dict[str, Any]:
        """Get query execution statistics."""
        try:
            stats_query = """
            SELECT 
                COUNT(*) as total_queries,
                COUNT(CASE WHEN success = true THEN 1 END) as successful_queries,
                COUNT(CASE WHEN success = false THEN 1 END) as failed_queries,
                COUNT(CASE WHEN cache_hit = true THEN 1 END) as cache_hits,
                AVG(execution_time_ms) as avg_execution_time,
                MAX(execution_time_ms) as max_execution_time,
                MIN(execution_time_ms) as min_execution_time
            FROM query_history
            """

            result = await self.db_manager.execute_query(stats_query)

            if result:
                row = result[0]
                return {
                    "total_queries": row[0] or 0,
                    "successful_queries": row[1] or 0,
                    "failed_queries": row[2] or 0,
                    "cache_hits": row[3] or 0,
                    "success_rate": (row[1] or 0) / max(row[0] or 1, 1) * 100,
                    "cache_hit_rate": (row[3] or 0) / max(row[0] or 1, 1) * 100,
                    "avg_execution_time_ms": round(row[4] or 0, 2),
                    "max_execution_time_ms": row[5] or 0,
                    "min_execution_time_ms": row[6] or 0,
                }

            return {}

        except Exception as e:
            logger.error(f"Failed to get query statistics: {e}")
            return {}

    async def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """Clean up old query history records."""
        try:
            query = f"""
            DELETE FROM query_history 
            WHERE created_at < datetime('now', '-{days_to_keep} days')
            """

            rows_deleted = await self.db_manager.execute_query(query)
            logger.info(f"Cleaned up {rows_deleted} old query history records")
            return rows_deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        # Assuming row is a tuple with specific column order
        columns = [
            "id",
            "query_id",
            "user_question",
            "generated_sql",
            "execution_time_ms",
            "success",
            "error_message",
            "cache_hit",
            "user_agent",
            "client_ip",
            "created_at",
            "updated_at",
        ]

        return {col: row[i] if i < len(row) else None for i, col in enumerate(columns)}


class SystemMetricsRepository(BaseRepository):
    """Repository for system performance metrics."""

    async def create(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record system metrics."""
        try:
            query = """
            INSERT INTO system_metrics 
            (metric_name, metric_value, metric_type, tags, recorded_at)
            VALUES 
            (:metric_name, :metric_value, :metric_type, :tags, :recorded_at)
            """

            params = {
                "metric_name": metrics_data.get("metric_name"),
                "metric_value": metrics_data.get("metric_value"),
                "metric_type": metrics_data.get("metric_type", "gauge"),
                "tags": metrics_data.get("tags", "{}"),
                "recorded_at": metrics_data.get("recorded_at", datetime.utcnow()),
            }

            await self.db_manager.execute_query(query, params)
            return metrics_data

        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
            raise

    async def get_by_id(self, metric_id: int) -> Optional[Dict[str, Any]]:
        """Get metric by ID."""
        try:
            query = "SELECT * FROM system_metrics WHERE id = :metric_id"
            result = await self.db_manager.execute_query(
                query, {"metric_id": metric_id}
            )

            if result:
                return self._row_to_dict(result[0])
            return None

        except Exception as e:
            logger.error(f"Failed to get metric {metric_id}: {e}")
            return None

    async def update(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update metrics record (typically not needed for time-series data)."""
        # Metrics are typically immutable time-series data
        raise NotImplementedError("Metrics are immutable time-series data")

    async def delete(self, metric_id: int) -> bool:
        """Delete metric record."""
        try:
            query = "DELETE FROM system_metrics WHERE id = :metric_id"
            rows_affected = await self.db_manager.execute_query(
                query, {"metric_id": metric_id}
            )
            return rows_affected > 0

        except Exception as e:
            logger.error(f"Failed to delete metric {metric_id}: {e}")
            return False

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List all metrics with pagination."""
        try:
            query = """
            SELECT * FROM system_metrics 
            ORDER BY recorded_at DESC 
            LIMIT :limit OFFSET :offset
            """

            result = await self.db_manager.execute_query(
                query, {"limit": limit, "offset": offset}
            )

            return [self._row_to_dict(row) for row in result]

        except Exception as e:
            logger.error(f"Failed to list metrics: {e}")
            return []

    async def get_metrics_by_name(
        self, metric_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get metrics by name."""
        try:
            query = """
            SELECT * FROM system_metrics 
            WHERE metric_name = :metric_name 
            ORDER BY recorded_at DESC 
            LIMIT :limit
            """

            result = await self.db_manager.execute_query(
                query, {"metric_name": metric_name, "limit": limit}
            )

            return [self._row_to_dict(row) for row in result]

        except Exception as e:
            logger.error(f"Failed to get metrics for {metric_name}: {e}")
            return []

    async def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest value for each metric."""
        try:
            query = """
            SELECT metric_name, metric_value, metric_type, recorded_at
            FROM system_metrics m1
            WHERE recorded_at = (
                SELECT MAX(recorded_at) 
                FROM system_metrics m2 
                WHERE m2.metric_name = m1.metric_name
            )
            ORDER BY metric_name
            """

            result = await self.db_manager.execute_query(query)

            metrics = {}
            for row in result:
                metrics[row[0]] = {
                    "value": row[1],
                    "type": row[2],
                    "recorded_at": row[3],
                }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get latest metrics: {e}")
            return {}

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        columns = [
            "id",
            "metric_name",
            "metric_value",
            "metric_type",
            "tags",
            "recorded_at",
        ]
        return {col: row[i] if i < len(row) else None for i, col in enumerate(columns)}
