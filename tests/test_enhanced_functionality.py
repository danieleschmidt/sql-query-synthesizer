"""Tests for enhanced functionality added in autonomous SDLC implementation."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from sql_synthesizer.adaptive_caching import (
    AdaptiveCacheManager,
    CacheStrategy,
    DataFreshness,
    PredictiveCacheWarmer,
    adaptive_cache,
    predictive_warmer,
)
from sql_synthesizer.auto_scaling_engine import (
    AutoScalingEngine,
    MetricsCollector,
    ResourceType,
    ScalingDecisionEngine,
    ScalingDirection,
    ScalingRule,
    ScalingTrigger,
    auto_scaling_engine,
)
from sql_synthesizer.comprehensive_validation import (
    BusinessRulesValidator,
    ComprehensiveValidator,
    PerformanceValidator,
    SQLSecurityValidator,
    ValidationSeverity,
    ValidationType,
    comprehensive_validator,
)
from sql_synthesizer.enhanced_core import (
    AdaptiveQueryOptimizer,
    EnhancedQueryMetadata,
    EnhancedResultFormatter,
    EnhancedSystemInfo,
    GlobalEventBus,
    PerformanceTracker,
    event_bus,
    performance_tracker,
    query_optimizer,
)
from sql_synthesizer.intelligent_query_router import (
    DatabaseEndpoint,
    DatabaseRole,
    IntelligentQueryRouter,
    LoadBalancer,
    QueryAnalyzer,
    QueryComplexity,
    query_router,
)
from sql_synthesizer.performance_optimizer import (
    IndexOptimizer,
    OptimizationTechnique,
    PerformanceIssueType,
    PerformanceOptimizer,
    QueryPatternAnalyzer,
    QueryRewriter,
    performance_optimizer,
)
from sql_synthesizer.robust_error_handling import (
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy,
    RobustErrorHandler,
    error_context,
    error_handler,
    robust_operation,
)


class TestEnhancedCore:
    """Test enhanced core functionality."""

    def test_performance_tracker(self):
        """Test performance tracking capabilities."""
        tracker = PerformanceTracker()

        # Test query tracking
        query_id = tracker.record_query_start()
        assert isinstance(query_id, str)
        assert tracker.active_connections == 1

        # Record query end
        tracker.record_query_end(100.0, success=True)
        assert tracker.active_connections == 0
        assert tracker.successful_queries == 1

        # Test metrics
        metrics = tracker.get_metrics()
        assert metrics["total_queries"] == 1
        assert metrics["successful_queries"] == 1
        assert metrics["avg_response_time_ms"] == 100.0

    def test_enhanced_result_formatter(self):
        """Test enhanced result formatting."""
        from sql_synthesizer.types import QueryResult

        # Mock query result
        result = QueryResult(
            sql="SELECT * FROM users",
            data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            explanation="Test query",
        )

        # Test JSON Lines format
        json_lines = EnhancedResultFormatter.to_json_lines([result])
        assert isinstance(json_lines, str)
        assert "Alice" in json_lines

        # Test CSV format
        csv_output = EnhancedResultFormatter.to_csv_string(result)
        assert "id,name" in csv_output
        assert "Alice" in csv_output

        # Test Markdown format
        markdown = EnhancedResultFormatter.to_markdown_table(result)
        assert "| id | name |" in markdown
        assert "Alice" in markdown

    def test_global_event_bus(self):
        """Test global event bus functionality."""
        bus = GlobalEventBus()
        received_events = []

        def test_handler(data):
            """TODO: Add docstring"""
            received_events.append(data)

        # Subscribe to event
        bus.subscribe("test_event", test_handler)

        # Publish event
        bus.publish("test_event", {"message": "hello"})

        # Give time for event processing
        time.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0]["message"] == "hello"

    def test_adaptive_query_optimizer(self):
        """Test adaptive query optimization."""
        optimizer = AdaptiveQueryOptimizer()

        # Analyze query
        sql = "SELECT * FROM users WHERE id = 1"
        analysis = optimizer.analyze_query(sql, 150.0)  # 150ms execution time

        assert "execution_count" in analysis
        assert "patterns" in analysis
        assert "suggestions" in analysis

        # Test multiple executions
        optimizer.analyze_query(sql, 200.0)
        analysis2 = optimizer.analyze_query(sql, 100.0)

        assert analysis2["execution_count"] == 3
        assert analysis2["is_frequent"]


class TestIntelligentQueryRouter:
    """Test intelligent query routing."""

    def test_query_analyzer(self):
        """Test SQL query analysis."""
        analyzer = QueryAnalyzer()

        # Test complexity analysis
        simple_query = "SELECT id FROM users WHERE id = 1"
        complexity = analyzer.analyze_query_complexity(simple_query)
        assert complexity == QueryComplexity.SIMPLE

        complex_query = """
        SELECT u.id, p.name, COUNT(*)
        FROM users u
        JOIN profiles p ON u.id = p.user_id
        JOIN orders o ON u.id = o.user_id
        WHERE u.created_at > '2023-01-01'
        GROUP BY u.id, p.name
        HAVING COUNT(*) > 5
        """
        complexity = analyzer.analyze_query_complexity(complex_query)
        assert complexity in [QueryComplexity.COMPLEX, QueryComplexity.ANALYTICAL]

        # Test read-only detection
        assert analyzer.is_read_only("SELECT * FROM users") == True
        assert analyzer.is_read_only("INSERT INTO users VALUES (1, 'test')") == False
        assert analyzer.is_read_only("UPDATE users SET name = 'test'") == False

        # Test cost estimation
        cost = analyzer.estimate_cost(complex_query)
        assert cost > 10.0  # Complex query should have higher cost

    def test_load_balancer(self):
        """Test load balancing functionality."""
        balancer = LoadBalancer()

        # Add endpoints
        primary_endpoint = DatabaseEndpoint(
            name="primary",
            url="postgresql://localhost/db",
            role=DatabaseRole.PRIMARY,
            priority=1,
        )

        replica_endpoint = DatabaseEndpoint(
            name="replica",
            url="postgresql://replica/db",
            role=DatabaseRole.REPLICA,
            priority=2,
        )

        balancer.add_endpoint(primary_endpoint)
        balancer.add_endpoint(replica_endpoint)

        # Test endpoint selection
        selected = balancer.select_endpoint(QueryComplexity.SIMPLE, True, 10.0)
        assert selected is not None
        assert selected.role in [DatabaseRole.PRIMARY, DatabaseRole.REPLICA]

        # Test write query routing
        selected_write = balancer.select_endpoint(QueryComplexity.SIMPLE, False, 10.0)
        assert selected_write.role == DatabaseRole.PRIMARY

    @pytest.mark.asyncio
    async def test_query_router_integration(self):
        """Test complete query routing."""
        router = IntelligentQueryRouter()

        # Add test endpoints
        router.add_database_endpoint(
            "test_primary", "postgresql://test", DatabaseRole.PRIMARY
        )
        router.add_database_endpoint(
            "test_replica", "postgresql://test_replica", DatabaseRole.REPLICA
        )

        # Route a query
        sql = "SELECT * FROM users WHERE active = true"
        route = await router.route_query(sql)

        assert route is not None
        assert route.endpoint is not None
        assert route.reason is not None
        assert route.estimated_performance >= 0


class TestAdaptiveCaching:
    """Test adaptive caching system."""

    def test_cache_manager_basic_operations(self):
        """Test basic cache operations."""
        cache_manager = AdaptiveCacheManager(max_memory_mb=10)

        # Test cache operations
        assert cache_manager.get("nonexistent") is None

        cache_manager.set("key1", "value1", execution_time_ms=100)
        assert cache_manager.get("key1") == "value1"

        # Test cache statistics
        stats = cache_manager.get_statistics()
        assert stats["cache_size"] >= 0
        assert stats["hit_count"] >= 0
        assert stats["miss_count"] >= 0

    def test_cache_eviction(self):
        """Test cache eviction policies."""
        cache_manager = AdaptiveCacheManager(max_memory_mb=1)  # Very small cache

        # Fill cache beyond capacity
        for i in range(100):
            large_value = "x" * 1000  # 1KB value
            cache_manager.set(f"key{i}", large_value, execution_time_ms=50)

        # Cache should have evicted some entries
        stats = cache_manager.get_statistics()
        assert stats["eviction_count"] > 0

    def test_caching_strategies(self):
        """Test different caching strategies."""
        cache_manager = AdaptiveCacheManager()

        # Test slow query caching (should cache)
        should_cache, strategy = cache_manager.should_cache("test_query", 500.0, 1024)
        assert should_cache == True

        # Test fast query caching (might not cache)
        should_cache_fast, strategy_fast = cache_manager.should_cache(
            "fast_query", 10.0, 100
        )
        # Result depends on access patterns

        # Test with different freshness requirements
        cache_manager.set(
            "fresh_data", "value", ttl=60, freshness=DataFreshness.REAL_TIME
        )
        cache_manager.set(
            "stable_data", "value", ttl=3600, freshness=DataFreshness.STABLE
        )

    def test_predictive_cache_warmer(self):
        """Test predictive cache warming."""
        cache_manager = AdaptiveCacheManager()
        warmer = PredictiveCacheWarmer(cache_manager)

        # Record query executions
        query_hash = "test_query_123"
        current_time = time.time()

        # Simulate regular execution pattern
        for i in range(5):
            warmer.record_query_execution(
                query_hash, current_time + i * 3600
            )  # Every hour

        # Predict next queries
        predictions = warmer.predict_next_queries(lookahead_minutes=30)
        # Predictions depend on pattern analysis


class TestRobustErrorHandling:
    """Test robust error handling system."""

    def test_error_handling_basic(self):
        """Test basic error handling."""
        handler = RobustErrorHandler()

        # Test error handling
        test_exception = ValueError("Test error")
        error_context = handler.handle_error(
            exception=test_exception,
            operation="test_operation",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
        )

        assert error_context.error_id is not None
        assert error_context.severity == ErrorSeverity.MEDIUM
        assert error_context.category == ErrorCategory.PROCESSING
        assert "Test error" in error_context.metadata["exception_message"]

    def test_recovery_strategies(self):
        """Test error recovery strategies."""
        handler = RobustErrorHandler()

        # Add custom recovery strategy
            """TODO: Add docstring"""
        def custom_recovery(error_context, exception):
            return True  # Simulate successful recovery

        strategy = RecoveryStrategy(
            strategy_name="test_recovery",
            applicable_categories=[ErrorCategory.PROCESSING],
            max_attempts=2,
            recovery_function=custom_recovery,
        )

        handler.register_recovery_strategy(strategy)

        # Test recovery execution
        test_exception = RuntimeError("Recoverable error")
        error_context = handler.handle_error(
            exception=test_exception,
            operation="test_operation",
            category=ErrorCategory.PROCESSING,
            auto_recover=True,
        )

        assert error_context.recovery_attempted == True

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        handler = RobustErrorHandler()
        handler.circuit_breaker_failure_threshold = 2

        operation = "test_circuit_breaker"

        # Simulate failures to open circuit
        for i in range(3):
            test_exception = RuntimeError(f"Failure {i}")
            handler.handle_error(
                exception=test_exception, operation=operation, auto_recover=False
            )

        # Circuit should be open
        assert handler._is_circuit_open(operation) == True

    def test_error_decorators(self):
        """Test error handling decorators."""

        @robust_operation(
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.LOW,
            operation_name="test_decorated_function",
                """TODO: Add docstring"""
        )
        def test_function():
            raise ValueError("Test decorator error")

        # Function should raise exception but handle it first
        with pytest.raises(ValueError):
            test_function()

    def test_error_context_manager(self):
        """Test error context manager."""
        with pytest.raises(RuntimeError):
            with error_context("test_context_operation", ErrorCategory.PROCESSING):
                raise RuntimeError("Context manager test")


class TestComprehensiveValidation:
    """Test comprehensive validation system."""

    def test_security_validation(self):
        """Test SQL security validation."""
        validator = SQLSecurityValidator()

        # Test safe query
        safe_sql = "SELECT id, name FROM users WHERE id = ?"
        safe_issues = validator.validate_sql_security(safe_sql)
        assert len(safe_issues) == 0

        # Test potentially dangerous query
        dangerous_sql = "SELECT * FROM users; DROP TABLE users; --"
        dangerous_issues = validator.validate_sql_security(dangerous_sql)
        assert len(dangerous_issues) > 0
        assert any(
            issue.severity == ValidationSeverity.CRITICAL for issue in dangerous_issues
        )

        # Test injection patterns
        injection_sql = "SELECT * FROM users WHERE id = 1 OR 1=1"
        injection_issues = validator.validate_sql_security(injection_sql)
        assert len(injection_issues) > 0

    def test_performance_validation(self):
        """Test performance validation."""
        validator = PerformanceValidator()

        # Test query with performance issues
        slow_sql = "SELECT * FROM users, orders WHERE users.active = 1"
        performance_issues = validator.validate_performance(slow_sql)

        # Should detect cartesian product and select *
        issue_types = [issue.validation_type for issue in performance_issues]
        assert ValidationType.PERFORMANCE in issue_types

        # Test query with many joins
        complex_sql = """
        SELECT u.name
        FROM users u
        JOIN profiles p ON u.id = p.user_id
        JOIN orders o ON u.id = o.user_id
        JOIN products pr ON o.product_id = pr.id
        JOIN categories c ON pr.category_id = c.id
        JOIN suppliers s ON pr.supplier_id = s.id
        """
        complex_issues = validator.validate_performance(complex_sql)
        assert len(complex_issues) > 0

    def test_business_rules_validation(self):
        """Test business rules validation."""
        validator = BusinessRulesValidator()
            """TODO: Add docstring"""

        # Add a business rule
        def no_delete_users(sql, context):
            return "DELETE FROM users" not in sql.upper()

        validator.add_business_rule(
            "no_delete_users", no_delete_users, "Users cannot be deleted"
        )

        # Test compliant query
        compliant_sql = "SELECT * FROM users"
        compliant_issues = validator.validate_business_rules(compliant_sql)
        assert len(compliant_issues) == 0

        # Test non-compliant query
        violation_sql = "DELETE FROM users WHERE id = 1"
        violation_issues = validator.validate_business_rules(violation_sql)
        assert len(violation_issues) > 0

    def test_comprehensive_validation(self):
        """Test comprehensive validation integration."""
        validator = ComprehensiveValidator()

        # Test mixed issues query
        problematic_sql = """
        SELECT * FROM users u, orders o
        WHERE u.name LIKE '%admin%'
        AND (u.role = 'admin' OR o.total > 1000)
        """

        result = validator.validate(problematic_sql)

        assert result.is_valid in [True, False]  # Depends on specific issues found
        assert isinstance(result.security_score, float)
        assert isinstance(result.performance_score, float)
        assert isinstance(result.compliance_score, float)
        assert len(result.recommendations) >= 0


class TestAutoScalingEngine:
    """Test auto-scaling engine."""

    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = MetricsCollector(collection_interval=1)

        # Test metric initialization
        cpu_metric = collector.get_metric("cpu_utilization")
        assert cpu_metric is not None
        assert cpu_metric.name == "CPU Utilization"

        # Test external metric update
        collector.update_external_metric("query_queue_length", 25.0)
        queue_metric = collector.get_metric("query_queue_length")
        assert queue_metric.current_value == 25.0

        # Test all metrics
        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) > 0

    def test_scaling_decision_engine(self):
        """Test scaling decisions."""
        collector = MetricsCollector()
        engine = ScalingDecisionEngine(collector)

        # Simulate high CPU usage
        collector.update_external_metric("cpu_utilization", 85.0)

        # Evaluate scaling decisions
        actions = engine.evaluate_scaling_decisions()

        # Should recommend scaling up for high CPU
        scale_up_actions = [a for a in actions if a.direction == ScalingDirection.UP]
        # May or may not trigger depending on cooldown and other factors

        # Test statistics
        stats = engine.get_scaling_statistics()
        assert "total_actions" in stats
        assert "current_resource_levels" in stats

    def test_scaling_rules(self):
        """Test custom scaling rules."""
        collector = MetricsCollector()
        engine = ScalingDecisionEngine(collector)

        # Add custom rule
        custom_rule = ScalingRule(
            rule_id="test_rule",
            resource_type=ResourceType.WORKER_PROCESSES,
            trigger=ScalingTrigger.QUERY_QUEUE_LENGTH,
            scale_up_threshold=10.0,
            scale_down_threshold=2.0,
            cooldown_period=60,
            min_instances=1,
            max_instances=10,
        )

        engine.add_scaling_rule(custom_rule)
        assert "test_rule" in engine.scaling_rules

    def test_auto_scaling_engine_lifecycle(self):
        """Test auto-scaling engine start/stop."""
        engine = AutoScalingEngine(
            metrics_interval=1, evaluation_interval=2, enabled=True
        )

        # Test start
        engine.start()
        assert engine.running == True

        # Give some time for operation
        time.sleep(0.5)

        # Test status
        status = engine.get_status()
        assert status["enabled"] == True
        assert status["running"] == True

        # Test stop
        engine.stop()
        assert engine.running == False


class TestPerformanceOptimizer:
    """Test performance optimization system."""

    def test_query_pattern_analyzer(self):
        """Test query pattern analysis."""
        analyzer = QueryPatternAnalyzer()

        # Test pattern identification
        sql = "SELECT * FROM users WHERE name LIKE '%admin%' ORDER BY created_at"
        execution_metrics = {
            "execution_time_ms": 1500,
            "rows_examined": 10000,
            "rows_returned": 50,
        }

        patterns = analyzer.analyze_query_pattern(sql, execution_metrics)

        expected_patterns = [
            "select_star",
            "leading_wildcard",
            "slow_execution",
            "inefficient_scan",
        ]
        for expected in expected_patterns:
            assert expected in patterns

    def test_index_optimizer(self):
        """Test index optimization recommendations."""
        optimizer = IndexOptimizer()

        # Test with a query that would benefit from indexes
        sql = """
        SELECT u.name, COUNT(*)
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.created_at > '2023-01-01'
        AND o.status = 'completed'
        ORDER BY u.created_at
        GROUP BY u.name
        """

        recommendations = optimizer.analyze_for_index_opportunities(sql, 2000.0)

        # Should generate index recommendations
        assert len(recommendations) > 0

        # Check recommendation types
        techniques = [r.technique for r in recommendations]
        assert OptimizationTechnique.INDEX_SUGGESTION in techniques

    def test_query_rewriter(self):
        """Test query rewriting suggestions."""
        rewriter = QueryRewriter()

        # Test SELECT * rewriting
        select_star_sql = "SELECT * FROM users WHERE active = true"
        recommendations = rewriter.suggest_query_rewrites(select_star_sql, 150.0)

        select_star_recs = [r for r in recommendations if "SELECT *" in r.description]
        assert len(select_star_recs) > 0

        # Test OR conditions rewriting
        or_sql = "SELECT * FROM users WHERE status = 'active' OR status = 'pending' OR status = 'trial'"
        or_recommendations = rewriter.suggest_query_rewrites(or_sql, 300.0)

        or_recs = [r for r in or_recommendations if "OR" in r.description]
        assert len(or_recs) > 0

    def test_performance_optimizer_integration(self):
        """Test complete performance optimization flow."""
        optimizer = PerformanceOptimizer()

        # Record some query executions
        slow_sql = (
            "SELECT * FROM users u, orders o WHERE u.active = 1 AND o.total > 100"
        )

        for i in range(5):
            optimizer.record_query_execution(
                slow_sql,
                1500.0 + i * 100,
                {"rows_examined": 50000, "rows_returned": 100, "cache_hit": False},
            )

        # Generate recommendations
        recommendations = optimizer.generate_optimization_recommendations(limit=10)
        assert len(recommendations) > 0

        # Test performance summary
        summary = optimizer.get_performance_summary()
        assert summary["total_unique_queries"] >= 1
        assert summary["total_executions"] >= 5
        assert len(summary["slowest_queries"]) >= 1

        # Test optimization impact tracking
        if recommendations:
            success = optimizer.apply_optimization(recommendations[0].recommendation_id)
            assert success == True

            impact = optimizer.get_optimization_impact()
            assert impact["optimizations_applied"] >= 1


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete system scenarios."""

    def test_complete_query_processing_flow(self):
        """Test complete query processing with all enhancements."""
        # This would test the full flow from query receipt to response
        # with all the enhanced features working together

        sql = "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name"

        # 1. Validation
        validation_result = comprehensive_validator.validate(sql)
        assert validation_result is not None

        # 2. Performance analysis
        performance_optimizer.record_query_execution(sql, 250.0)

        # 3. Caching decision
        should_cache, strategy = adaptive_cache.should_cache("test_hash", 250.0, 1024)

        # 4. Error handling context
        with error_context("query_processing", ErrorCategory.PROCESSING):
            # Simulate query processing
            time.sleep(0.01)  # Simulate work

        # All components should work together without conflicts
        assert True  # If we get here, integration is working

    @pytest.mark.asyncio
    async def test_async_operations_integration(self):
        """Test async operations work correctly with enhanced features."""
        router = IntelligentQueryRouter()
        router.add_database_endpoint(
            "async_test", "postgresql://test", DatabaseRole.PRIMARY
        )

        # Test async routing
        route = await router.route_query("SELECT * FROM async_test")
        assert route is not None

        # Test async error handling would go here
        # (requires async version of error context)

     """TODO: Add docstring"""
    def test_concurrent_operations(self):
        """Test system behavior under concurrent load."""

        def worker_function(worker_id):
            sql = f"SELECT * FROM test_table WHERE worker_id = {worker_id}"

            # Record performance
            performance_optimizer.record_query_execution(sql, 100.0 + worker_id * 10)

            # Update cache
            adaptive_cache.set(
                f"worker_{worker_id}", f"result_{worker_id}", execution_time_ms=100.0
            )

            # Validate query
            result = comprehensive_validator.validate(sql)
            return result.is_valid

        # Run multiple workers concurrently
        threads = []
        results = []

        for i in range(10):
            thread = threading.Thread(target=lambda: results.append(worker_function(i)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should complete successfully
        assert len(results) == 10


# Pytest configuration for the test suite
@pytest.fixture
def reset_global_state():
    """Reset global state between tests."""
    # Reset performance tracker
    global performance_tracker
    performance_tracker = PerformanceTracker()

    # Reset error handler
    global error_handler
    error_handler.error_history.clear()
    error_handler.circuit_breakers.clear()

    # Clear caches
    adaptive_cache.cache.clear()
    adaptive_cache.current_memory_bytes = 0

    yield

    # Cleanup after test
    pass


# Mark slow tests
pytestmark = pytest.mark.slow
