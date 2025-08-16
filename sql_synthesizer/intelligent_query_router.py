"""Intelligent Query Router for SQL Query Synthesizer.

This module implements smart query routing, load balancing, and connection
optimization based on query patterns and system load.
"""

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels for routing decisions."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ANALYTICAL = "analytical"


class DatabaseRole(Enum):
    """Database role for read/write separation."""

    PRIMARY = "primary"
    REPLICA = "replica"
    ANALYTICS = "analytics"
    CACHE = "cache"


@dataclass
class DatabaseEndpoint:
    """Database endpoint configuration."""

    name: str
    url: str
    role: DatabaseRole
    priority: int = 1
    max_connections: int = 10
    current_load: float = 0.0
    healthy: bool = True
    response_time_ms: float = 0.0
    error_count: int = 0
    last_health_check: float = 0.0


@dataclass
class QueryRoute:
    """Query routing decision."""

    endpoint: DatabaseEndpoint
    reason: str
    estimated_performance: float
    cache_strategy: str = "default"


class QueryAnalyzer:
    """Analyzes SQL queries to determine routing strategy."""

    @staticmethod
    def analyze_query_complexity(sql: str) -> QueryComplexity:
        """Analyze SQL query to determine complexity level."""
        sql_upper = sql.upper().strip()

        # Count complexity indicators
        complexity_indicators = 0

        # Join complexity
        join_count = sql_upper.count("JOIN")
        if join_count > 0:
            complexity_indicators += min(join_count, 3)

        # Subquery complexity
        subquery_count = sql_upper.count("SELECT") - 1  # Subtract main SELECT
        if subquery_count > 0:
            complexity_indicators += min(subquery_count * 2, 4)

        # Aggregation complexity
        if any(func in sql_upper for func in ["GROUP BY", "HAVING", "DISTINCT"]):
            complexity_indicators += 1

        # Window functions
        if "OVER(" in sql_upper:
            complexity_indicators += 2

        # CTEs and recursive queries
        if "WITH" in sql_upper:
            complexity_indicators += 1
            if "RECURSIVE" in sql_upper:
                complexity_indicators += 2

        # Classify based on indicators
        if complexity_indicators == 0:
            return QueryComplexity.SIMPLE
        elif complexity_indicators <= 2:
            return QueryComplexity.MEDIUM
        elif complexity_indicators <= 5:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.ANALYTICAL

    @staticmethod
    def is_read_only(sql: str) -> bool:
        """Determine if query is read-only."""
        sql_upper = sql.upper().strip()

        # Obvious read-only patterns
        if sql_upper.startswith("SELECT"):
            return True
        if sql_upper.startswith("WITH") and "SELECT" in sql_upper:
            return True
        if sql_upper.startswith("SHOW"):
            return True
        if sql_upper.startswith("DESCRIBE") or sql_upper.startswith("DESC"):
            return True
        if sql_upper.startswith("EXPLAIN"):
            return True

        # Write operations
        write_operations = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "TRUNCATE",
        ]
        return not any(sql_upper.startswith(op) for op in write_operations)

    @staticmethod
    def estimate_cost(sql: str, table_sizes: Dict[str, int] = None) -> float:
        """Estimate query execution cost (0-100 scale)."""
        sql_upper = sql.upper()
        cost = 1.0  # Base cost

        # Join cost
        join_count = sql_upper.count("JOIN")
        cost += join_count * 5

        # Subquery cost
        subquery_count = sql_upper.count("SELECT") - 1
        cost += subquery_count * 3

        # Aggregation cost
        if "GROUP BY" in sql_upper:
            cost += 3
        if "ORDER BY" in sql_upper:
            cost += 2
        if "DISTINCT" in sql_upper:
            cost += 2

        # Full table scan indicators
        if "WHERE" not in sql_upper and "SELECT *" in sql_upper:
            cost += 10

        # Window functions are expensive
        if "OVER(" in sql_upper:
            cost += 8

        # Normalize to 0-100 scale
        return min(cost * 2, 100.0)


class LoadBalancer:
    """Intelligent load balancer for database connections."""

    def __init__(self):
        self.endpoints: List[DatabaseEndpoint] = []
        self.health_check_interval = 30  # seconds
        self.last_health_check = 0
        self._endpoint_stats: Dict[str, Dict[str, float]] = {}

    def add_endpoint(self, endpoint: DatabaseEndpoint):
        """Add a database endpoint to the pool."""
        self.endpoints.append(endpoint)
        self._endpoint_stats[endpoint.name] = {
            "total_requests": 0,
            "successful_requests": 0,
            "total_response_time": 0.0,
            "last_request_time": 0.0,
        }
        logger.info(f"Added database endpoint: {endpoint.name} ({endpoint.role.value})")

    def select_endpoint(
        self, complexity: QueryComplexity, is_read_only: bool, estimated_cost: float
    ) -> Optional[DatabaseEndpoint]:
        """Select the best endpoint for a query."""

        # Filter endpoints based on query requirements
        candidates = self._filter_candidates(complexity, is_read_only, estimated_cost)

        if not candidates:
            logger.warning("No suitable database endpoints available")
            return None

        # Apply load balancing algorithm
        selected = self._apply_load_balancing(candidates, estimated_cost)

        if selected:
            logger.debug(
                f"Selected endpoint {selected.name} for {complexity.value} "
                f"{'read-only' if is_read_only else 'write'} query (cost: {estimated_cost:.1f})"
            )

        return selected

    def _filter_candidates(
        self, complexity: QueryComplexity, is_read_only: bool, estimated_cost: float
    ) -> List[DatabaseEndpoint]:
        """Filter endpoints based on query requirements."""
        candidates = []

        for endpoint in self.endpoints:
            if not endpoint.healthy:
                continue

            # Role-based filtering
            if is_read_only:
                # Read queries can go to any role
                if endpoint.role in [
                    DatabaseRole.PRIMARY,
                    DatabaseRole.REPLICA,
                    DatabaseRole.ANALYTICS,
                ]:
                    candidates.append(endpoint)
            else:
                # Write queries must go to primary
                if endpoint.role == DatabaseRole.PRIMARY:
                    candidates.append(endpoint)

            # Load filtering - reject overloaded endpoints
            if endpoint.current_load > 0.9:  # 90% load threshold
                candidates = [c for c in candidates if c != endpoint]

            # Cost-based filtering for expensive queries
            if estimated_cost > 50 and endpoint.role == DatabaseRole.ANALYTICS:
                # Prefer analytics endpoints for expensive queries
                candidates.insert(0, endpoint)

        return candidates

    def _apply_load_balancing(
        self, candidates: List[DatabaseEndpoint], estimated_cost: float
    ) -> Optional[DatabaseEndpoint]:
        """Apply load balancing algorithm to select best endpoint."""

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Weighted round-robin based on multiple factors
        scores = []

        for endpoint in candidates:
            stats = self._endpoint_stats[endpoint.name]

            # Base score from priority
            score = endpoint.priority * 10

            # Load factor (lower load = higher score)
            load_factor = 1.0 - endpoint.current_load
            score += load_factor * 20

            # Response time factor (faster = higher score)
            if endpoint.response_time_ms > 0:
                response_factor = min(1000 / endpoint.response_time_ms, 10)
                score += response_factor * 5

            # Success rate factor
            if stats["total_requests"] > 0:
                success_rate = stats["successful_requests"] / stats["total_requests"]
                score += success_rate * 15

            # Recent activity bonus
            time_since_last_request = time.time() - stats["last_request_time"]
            if time_since_last_request < 300:  # 5 minutes
                score += 5

            scores.append((score, endpoint))

        # Select endpoint using weighted random selection
        total_score = sum(score for score, _ in scores)
        if total_score <= 0:
            return random.choice(candidates)

        # Weighted selection
        rand_value = random.uniform(0, total_score)
        cumulative = 0

        for score, endpoint in scores:
            cumulative += score
            if rand_value <= cumulative:
                return endpoint

        return candidates[0]  # Fallback

    def update_endpoint_stats(
        self, endpoint_name: str, response_time: float, success: bool
    ):
        """Update endpoint statistics after query execution."""
        if endpoint_name not in self._endpoint_stats:
            return

        stats = self._endpoint_stats[endpoint_name]
        stats["total_requests"] += 1
        stats["total_response_time"] += response_time
        stats["last_request_time"] = time.time()

        if success:
            stats["successful_requests"] += 1

        # Update endpoint object
        for endpoint in self.endpoints:
            if endpoint.name == endpoint_name:
                endpoint.response_time_ms = (
                    stats["total_response_time"] / stats["total_requests"]
                )
                if not success:
                    endpoint.error_count += 1
                break

    async def health_check_endpoints(self):
        """Perform health checks on all endpoints."""
        current_time = time.time()

        if current_time - self.last_health_check < self.health_check_interval:
            return

        self.last_health_check = current_time

        for endpoint in self.endpoints:
            try:
                # Simple health check - could be enhanced with actual connection test
                endpoint.healthy = True
                endpoint.last_health_check = current_time

                # Update load estimation based on recent activity
                stats = self._endpoint_stats[endpoint.name]
                recent_requests = max(
                    1, stats["total_requests"] - endpoint.current_load * 100
                )
                endpoint.current_load = min(
                    recent_requests / endpoint.max_connections, 1.0
                )

            except Exception as e:
                logger.warning(f"Health check failed for endpoint {endpoint.name}: {e}")
                endpoint.healthy = False
                endpoint.error_count += 1


class IntelligentQueryRouter:
    """Main query router that combines analysis and load balancing."""

    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.load_balancer = LoadBalancer()
        self.routing_history: List[Tuple[str, QueryRoute, float]] = (
            []
        )  # (query_hash, route, execution_time)
        self._lock = asyncio.Lock()

    def add_database_endpoint(
        self,
        name: str,
        url: str,
        role: DatabaseRole,
        priority: int = 1,
        max_connections: int = 10,
    ):
        """Add a database endpoint for routing."""
        endpoint = DatabaseEndpoint(
            name=name,
            url=url,
            role=role,
            priority=priority,
            max_connections=max_connections,
        )
        self.load_balancer.add_endpoint(endpoint)

    async def route_query(self, sql: str, context: Dict[str, Any] = None) -> QueryRoute:
        """Route a query to the best available endpoint."""
        context = context or {}

        # Analyze query
        complexity = self.analyzer.analyze_query_complexity(sql)
        is_read_only = self.analyzer.is_read_only(sql)
        estimated_cost = self.analyzer.estimate_cost(sql)

        # Perform health checks
        await self.load_balancer.health_check_endpoints()

        # Select endpoint
        endpoint = self.load_balancer.select_endpoint(
            complexity, is_read_only, estimated_cost
        )

        if not endpoint:
            raise Exception("No available database endpoints for query routing")

        # Determine caching strategy
        cache_strategy = self._determine_cache_strategy(
            complexity, estimated_cost, context
        )

        # Create routing decision
        route = QueryRoute(
            endpoint=endpoint,
            reason=f"Selected {endpoint.role.value} for {complexity.value} query (cost: {estimated_cost:.1f})",
            estimated_performance=100
            - estimated_cost,  # Higher performance = lower cost
            cache_strategy=cache_strategy,
        )

        # Record routing decision
        query_hash = hashlib.md5(sql.encode()).hexdigest()[:12]
        async with self._lock:
            self.routing_history.append(
                (query_hash, route, 0)
            )  # execution_time will be updated later

            # Keep history limited
            if len(self.routing_history) > 1000:
                self.routing_history = self.routing_history[-500:]

        logger.info(f"Routed query {query_hash} to {endpoint.name}: {route.reason}")

        return route

    def _determine_cache_strategy(
        self,
        complexity: QueryComplexity,
        estimated_cost: float,
        context: Dict[str, Any],
    ) -> str:
        """Determine appropriate caching strategy for the query."""

        # High-cost queries benefit from longer caching
        if estimated_cost > 50:
            return "long_term"  # Cache for hours

        # Medium complexity queries get standard caching
        if complexity in [QueryComplexity.MEDIUM, QueryComplexity.COMPLEX]:
            return "standard"  # Cache for minutes

        # Simple queries get short-term caching
        if complexity == QueryComplexity.SIMPLE:
            return "short_term"  # Cache for seconds

        # Analytical queries get specialized caching
        if complexity == QueryComplexity.ANALYTICAL:
            return "analytical"  # Cache with special eviction rules

        return "default"

    async def update_execution_stats(
        self, query_hash: str, execution_time: float, success: bool
    ):
        """Update execution statistics for a routed query."""
        async with self._lock:
            # Find and update the routing history entry
            for i, (hash_val, route, _) in enumerate(self.routing_history):
                if hash_val == query_hash:
                    self.routing_history[i] = (hash_val, route, execution_time)
                    break

        # Update load balancer stats
        route = None
        for hash_val, r, _ in self.routing_history:
            if hash_val == query_hash:
                route = r
                break

        if route:
            self.load_balancer.update_endpoint_stats(
                route.endpoint.name, execution_time, success
            )

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        if not self.routing_history:
            return {"total_queries": 0}

        endpoint_stats = {}
        complexity_stats = {}
        total_queries = len(self.routing_history)
        total_execution_time = 0

        for query_hash, route, execution_time in self.routing_history:
            # Endpoint statistics
            endpoint_name = route.endpoint.name
            if endpoint_name not in endpoint_stats:
                endpoint_stats[endpoint_name] = {
                    "query_count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                }

            endpoint_stats[endpoint_name]["query_count"] += 1
            endpoint_stats[endpoint_name]["total_time"] += execution_time

            if endpoint_stats[endpoint_name]["query_count"] > 0:
                endpoint_stats[endpoint_name]["avg_time"] = (
                    endpoint_stats[endpoint_name]["total_time"]
                    / endpoint_stats[endpoint_name]["query_count"]
                )

            total_execution_time += execution_time

        avg_execution_time = (
            total_execution_time / total_queries if total_queries > 0 else 0
        )

        return {
            "total_queries": total_queries,
            "avg_execution_time_ms": avg_execution_time,
            "endpoint_distribution": endpoint_stats,
            "active_endpoints": len(self.load_balancer.endpoints),
            "healthy_endpoints": sum(
                1 for e in self.load_balancer.endpoints if e.healthy
            ),
        }


# Global router instance
query_router = IntelligentQueryRouter()
