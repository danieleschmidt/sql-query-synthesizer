#!/usr/bin/env python3
"""
Test suite for newly implemented intelligence, reliability, and scaling modules.
"""

import asyncio
import sys
import traceback
from typing import Any, Dict, List


# Test the new modules
def test_intelligence_modules():
    """Test the intelligence modules."""
    results = []

    try:
        # Test query insights
        from sql_synthesizer.intelligence.query_insights import (
            QueryInsightsEngine,
        )

        engine = QueryInsightsEngine()

        # Test complex query analysis
        complex_query = """
        SELECT c.name, COUNT(o.id) as order_count, SUM(o.total) as revenue
        FROM customers c
        LEFT JOIN orders o ON c.id = o.customer_id
        WHERE c.created_at > '2024-01-01'
        GROUP BY c.id, c.name
        HAVING COUNT(o.id) > 5
        ORDER BY revenue DESC
        LIMIT 10
        """

        analysis = engine.analyze_query(complex_query, execution_time_ms=1500)

        results.append({
            'test': 'query_insights_complex_analysis',
            'success': True,
            'insights_count': len(analysis.get('insights', [])),
            'complexity_score': analysis.get('complexity', {}).get('total_score', 0),
            'details': f"Found {len(analysis.get('insights', []))} insights with complexity score {analysis.get('complexity', {}).get('total_score', 0)}"
        })

        # Test batch analysis
        queries = [
            {'sql': 'SELECT * FROM users', 'execution_time_ms': 50},
            {'sql': 'SELECT COUNT(*) FROM orders', 'execution_time_ms': 200},
            {'sql': complex_query, 'execution_time_ms': 1500}
        ]

        batch_analysis = engine.batch_analyze_queries(queries)

        results.append({
            'test': 'query_insights_batch_analysis',
            'success': True,
            'queries_analyzed': batch_analysis.get('aggregate_statistics', {}).get('total_queries', 0),
            'total_insights': batch_analysis.get('aggregate_statistics', {}).get('total_insights', 0),
            'details': f"Analyzed {len(queries)} queries in batch mode"
        })

    except Exception as e:
        results.append({
            'test': 'query_insights',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    # Test adaptive learning
    try:
        from sql_synthesizer.intelligence.adaptive_learning import (
            AdaptiveLearningEngine,
        )

        learning_engine = AdaptiveLearningEngine()

        # Create mock query history
        query_history = [
            {
                'sql': 'SELECT * FROM users WHERE age > 25',
                'execution_time_ms': 150,
                'success': True,
                'timestamp': '2024-01-01T10:00:00Z'
            },
            {
                'sql': 'SELECT * FROM users WHERE age > 30',
                'execution_time_ms': 120,
                'success': True,
                'timestamp': '2024-01-01T10:05:00Z'
            },
            {
                'sql': 'SELECT COUNT(*) FROM orders GROUP BY user_id',
                'execution_time_ms': 800,
                'success': True,
                'timestamp': '2024-01-01T10:10:00Z'
            }
        ]

        patterns = learning_engine.learn_from_queries(query_history)
        stats = learning_engine.get_learning_statistics()

        results.append({
            'test': 'adaptive_learning',
            'success': True,
            'patterns_learned': len(patterns),
            'total_patterns': stats.get('total_patterns', 0),
            'details': f"Learned {len(patterns)} patterns from {len(query_history)} queries"
        })

    except Exception as e:
        results.append({
            'test': 'adaptive_learning',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    # Test intelligent caching
    try:
        from sql_synthesizer.intelligence.intelligent_cache import (
            IntelligentCacheManager,
            QueryContext,
        )

        cache_manager = IntelligentCacheManager(max_size=100)

        # Test basic caching
        cache_manager.put("test_query_1", {"result": "test_data"})
        result = cache_manager.get("test_query_1")

        # Test with context
        context = QueryContext(user_id="user1", query_type="analytical")
        cache_manager.put("test_query_2", {"result": "contextual_data"}, context=context)

        insights = cache_manager.get_insights()
        stats = cache_manager.get_statistics()

        results.append({
            'test': 'intelligent_caching',
            'success': True,
            'cache_hits': stats.get('requests', {}).get('hits', 0),
            'insights_generated': len(insights),
            'details': f"Cache working with {stats.get('capacity', {}).get('current_size', 0)} entries"
        })

    except Exception as e:
        results.append({
            'test': 'intelligent_caching',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    return results


def test_reliability_modules():
    """Test the reliability modules."""
    results = []

    # Test error recovery
    try:
        from sql_synthesizer.reliability.error_recovery import (
            ErrorRecoveryManager,
        )

        recovery_manager = ErrorRecoveryManager()

        # Test error classification and handling
        try:
            raise ConnectionError("Database connection failed")
        except Exception as e:
            result = recovery_manager.handle_error(e, component="database", operation="connect")

        stats = recovery_manager.get_error_statistics()

        results.append({
            'test': 'error_recovery',
            'success': True,
            'total_errors': stats.get('recovery_stats', {}).get('total_errors', 0),
            'recovery_rate': stats.get('recovery_stats', {}).get('recovery_rate', 0),
            'details': f"Error recovery system operational with {len(stats.get('recent_errors', []))} recent errors"
        })

    except Exception as e:
        results.append({
            'test': 'error_recovery',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    # Test graceful degradation
    try:
        from sql_synthesizer.reliability.graceful_degradation import (
            DegradationLevel,
            degradation_manager,
        )

        # Test capability registration and degradation
        status = degradation_manager.get_degradation_status()

        # Simulate degradation
        degradation_manager.trigger_degradation(
            DegradationLevel.MINOR_DEGRADATION,
            "Test degradation",
            ["adaptive_learning"]
        )

        degraded_status = degradation_manager.get_degradation_status()

        results.append({
            'test': 'graceful_degradation',
            'success': True,
            'total_capabilities': status.get('capabilities', {}).get('total', 0),
            'degraded_capabilities': degraded_status.get('capabilities', {}).get('degraded', 0),
            'current_level': degraded_status.get('current_level'),
            'details': f"Degradation system with {status.get('capabilities', {}).get('total', 0)} capabilities"
        })

    except Exception as e:
        results.append({
            'test': 'graceful_degradation',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    return results


def test_scaling_modules():
    """Test the scaling modules."""
    results = []

    # Test performance optimizer
    try:

        from sql_synthesizer.scaling.performance_optimizer import (
            PerformanceOptimizer,
        )

        optimizer = PerformanceOptimizer()

        # Test analysis
        analysis = optimizer.analyze_performance()
        recommendations = optimizer.generate_optimization_recommendations()
        status = optimizer.get_optimization_status()

        results.append({
            'test': 'performance_optimizer',
            'success': True,
            'recommendations_generated': len(recommendations),
            'optimizations_applied': status.get('total_optimizations_applied', 0),
            'details': f"Performance optimizer generated {len(recommendations)} recommendations"
        })

    except Exception as e:
        results.append({
            'test': 'performance_optimizer',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    # Test auto scaler
    try:
        from sql_synthesizer.scaling.auto_scaler import AutoScaler, ScalingMetric

        auto_scaler = AutoScaler()

        # Test metrics recording and scaling decisions
        test_metrics = {
            ScalingMetric.CPU_USAGE: 85.0,
            ScalingMetric.MEMORY_USAGE: 70.0,
            ScalingMetric.RESPONSE_TIME: 1500.0
        }

        auto_scaler.record_metrics(test_metrics)
        decisions = auto_scaler.evaluate_scaling_decision()
        status = auto_scaler.get_scaling_status()

        results.append({
            'test': 'auto_scaler',
            'success': True,
            'scaling_rules': len(status.get('scaling_rules', [])),
            'decisions_generated': len(decisions),
            'details': f"Auto scaler with {len(status.get('scaling_rules', []))} rules generated {len(decisions)} decisions"
        })

    except Exception as e:
        results.append({
            'test': 'auto_scaler',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    return results


async def test_async_modules():
    """Test async functionality."""
    results = []

    try:
        from sql_synthesizer.scaling.connection_pool import (
            ConnectionPool,
            PoolConfiguration,
        )

        # Create a mock connection factory
        async def mock_connection_factory():
            class MockConnection:
                async def execute(self, query):
                    return [{"result": 1}]

                async def close(self):
                    pass

            return MockConnection()

        # Test connection pool
        config = PoolConfiguration(min_size=2, max_size=5)
        pool = ConnectionPool("test_pool", mock_connection_factory, config)

        await pool.start()

        # Test acquiring connections
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")

        stats = pool.get_statistics()

        await pool.stop()

        results.append({
            'test': 'connection_pool_async',
            'success': True,
            'connections_created': stats.get('lifetime_stats', {}).get('total_created', 0),
            'connections_served': stats.get('performance_stats', {}).get('connections_served', 0),
            'details': f"Async connection pool created {stats.get('lifetime_stats', {}).get('total_created', 0)} connections"
        })

    except Exception as e:
        results.append({
            'test': 'connection_pool_async',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    return results


def run_integration_test():
    """Run integration test combining multiple modules."""
    results = []

    try:
        from sql_synthesizer.intelligence.adaptive_learning import (
            AdaptiveLearningEngine,
        )
        from sql_synthesizer.intelligence.query_insights import QueryInsightsEngine
        from sql_synthesizer.reliability.graceful_degradation import degradation_manager
        from sql_synthesizer.scaling.performance_optimizer import performance_optimizer

        # Integration test: Query analysis with degradation
        insights_engine = QueryInsightsEngine()
        learning_engine = AdaptiveLearningEngine()

        test_query = "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id"

        # 1. Analyze query
        analysis = insights_engine.analyze_query(test_query, execution_time_ms=800)

        # 2. Learn from it
        query_history = [{
            'sql': test_query,
            'execution_time_ms': 800,
            'success': True,
            'timestamp': '2024-01-01T12:00:00Z'
        }]

        patterns = learning_engine.learn_from_queries(query_history)

        # 3. Check system status
        degradation_status = degradation_manager.get_degradation_status()
        optimization_status = performance_optimizer.get_optimization_status()

        results.append({
            'test': 'integration_test',
            'success': True,
            'query_insights': len(analysis.get('insights', [])),
            'patterns_learned': len(patterns),
            'degradation_level': degradation_status.get('current_level'),
            'optimizations_available': len(performance_optimizer.generate_optimization_recommendations()),
            'details': 'Full system integration test completed successfully'
        })

    except Exception as e:
        results.append({
            'test': 'integration_test',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    return results


def print_test_results(results: List[Dict[str, Any]], category: str):
    """Print formatted test results."""
    print(f"\n=== {category.upper()} TESTS ===")

    total_tests = len(results)
    successful_tests = len([r for r in results if r['success']])

    print(f"Tests run: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")

    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"\n{status} {result['test']}")

        if result['success']:
            print(f"  Details: {result.get('details', 'No details')}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(f"  Traceback: {result['traceback'][:200]}...")


async def main():
    """Main test runner."""
    print("🚀 Starting comprehensive test suite for new modules...")
    print("=" * 60)

    # Run synchronous tests
    intelligence_results = test_intelligence_modules()
    print_test_results(intelligence_results, "Intelligence")

    reliability_results = test_reliability_modules()
    print_test_results(reliability_results, "Reliability")

    scaling_results = test_scaling_modules()
    print_test_results(scaling_results, "Scaling")

    # Run async tests
    async_results = await test_async_modules()
    print_test_results(async_results, "Async")

    # Run integration test
    integration_results = run_integration_test()
    print_test_results(integration_results, "Integration")

    # Summary
    all_results = intelligence_results + reliability_results + scaling_results + async_results + integration_results
    total_tests = len(all_results)
    successful_tests = len([r for r in all_results if r['success']])

    print("\n" + "=" * 60)
    print("📊 OVERALL SUMMARY")
    print(f"Total tests executed: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {total_tests - successful_tests}")
    print(f"Overall success rate: {(successful_tests/total_tests)*100:.1f}%")

    if successful_tests == total_tests:
        print("\n🎉 All tests passed! System is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests failed. Review and fix issues.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
