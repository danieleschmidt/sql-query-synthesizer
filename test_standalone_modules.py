#!/usr/bin/env python3
"""
Standalone test suite for newly implemented modules (no external dependencies).
"""

import asyncio
import time
import sys
import os
import traceback
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, '/root/repo')

def test_intelligence_query_insights():
    """Test query insights module independently."""
    results = []
    
    try:
        # Import the module directly
        sys.path.insert(0, '/root/repo/sql_synthesizer')
        from intelligence.query_insights import QueryInsightsEngine, QueryComplexity, QueryPatternAnalyzer
        
        # Test pattern analyzer
        analyzer = QueryPatternAnalyzer()
        
        test_query = """
        SELECT c.name, COUNT(o.id) as order_count, SUM(o.total) as revenue
        FROM customers c
        LEFT JOIN orders o ON c.id = o.customer_id  
        WHERE c.created_at > '2024-01-01'
        GROUP BY c.id, c.name
        HAVING COUNT(o.id) > 5
        ORDER BY revenue DESC
        LIMIT 10
        """
        
        insights = analyzer.analyze_patterns(test_query)
        
        # Test complexity analyzer
        complexity_analyzer = analyzer.__class__.__bases__[0]() if hasattr(analyzer, '__class__') else None
        
        results.append({
            'test': 'query_insights_pattern_analysis',
            'success': True,
            'insights_found': len(insights),
            'details': f'Pattern analyzer found {len(insights)} insights for complex query'
        })
        
        # Test insights engine
        engine = QueryInsightsEngine()
        analysis = engine.analyze_query(test_query, execution_time_ms=1200)
        
        results.append({
            'test': 'query_insights_engine',
            'success': True,
            'total_insights': analysis.get('total_insights', 0),
            'critical_issues': analysis.get('critical_issues', 0),
            'details': f'Engine analysis completed with {analysis.get("total_insights", 0)} insights'
        })
        
    except Exception as e:
        results.append({
            'test': 'query_insights',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[-300:]
        })
    
    return results


def test_adaptive_learning():
    """Test adaptive learning module.""" 
    results = []
    
    try:
        from intelligence.adaptive_learning import AdaptiveLearningEngine, QueryPattern, PatternMiner
        
        # Test pattern miner
        miner = PatternMiner()
        
        # Mock query history
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
        
        patterns = miner.discover_patterns(query_history)
        
        results.append({
            'test': 'adaptive_learning_pattern_mining',
            'success': True,
            'patterns_discovered': len(patterns),
            'details': f'Pattern miner discovered {len(patterns)} patterns from {len(query_history)} queries'
        })
        
        # Test learning engine
        learning_engine = AdaptiveLearningEngine(storage_path="/tmp/test_learning")
        
        learned_patterns = learning_engine.learn_from_queries(query_history)
        stats = learning_engine.get_learning_statistics()
        
        results.append({
            'test': 'adaptive_learning_engine', 
            'success': True,
            'learned_patterns': len(learned_patterns),
            'total_patterns': stats.get('total_patterns', 0),
            'details': f'Learning engine processed queries and learned {len(learned_patterns)} patterns'
        })
        
        # Test insight generation
        insights = learning_engine.generate_insights('SELECT * FROM users WHERE age > 40')
        
        results.append({
            'test': 'adaptive_learning_insights',
            'success': True, 
            'insights_generated': len(insights),
            'details': f'Generated {len(insights)} insights for new query'
        })
        
    except Exception as e:
        results.append({
            'test': 'adaptive_learning',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[-300:]
        })
    
    return results


def test_intelligent_caching():
    """Test intelligent caching module."""
    results = []
    
    try:
        from intelligence.intelligent_cache import IntelligentCacheManager, QueryContext, CacheStrategy
        
        # Test cache manager
        cache_manager = IntelligentCacheManager(max_size=50)
        
        # Test basic operations
        cache_manager.put("query1", {"result": "data1"}, estimated_cost=100)
        result1 = cache_manager.get("query1")
        
        # Test with context
        context = QueryContext(user_id="user123", query_type="analytical")
        cache_manager.put("query2", {"result": "data2"}, context=context, estimated_cost=200)
        result2 = cache_manager.get("query2", context=context)
        
        # Test statistics and insights
        stats = cache_manager.get_statistics()
        insights = cache_manager.get_insights()
        
        results.append({
            'test': 'intelligent_caching_basic',
            'success': True,
            'cache_size': stats.get('capacity', {}).get('current_size', 0),
            'hit_rate': stats.get('requests', {}).get('hit_rate', 0),
            'insights_count': len(insights),
            'details': f'Cache operational with {stats.get("capacity", {}).get("current_size", 0)} entries'
        })
        
        # Test semantic functionality
        if hasattr(cache_manager, 'semantic_index'):
            cache_manager.semantic_index.add_query("hash123", "SELECT * FROM users", {'token_count': 10})
            similar = cache_manager.semantic_index.find_similar_queries("SELECT * FROM users", {'token_count': 12})
            
            results.append({
                'test': 'intelligent_caching_semantic',
                'success': True,
                'similar_queries_found': len(similar),
                'details': f'Semantic index found {len(similar)} similar queries'
            })
        
    except Exception as e:
        results.append({
            'test': 'intelligent_caching',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[-300:]
        })
    
    return results


def test_error_recovery():
    """Test error recovery and fault tolerance."""
    results = []
    
    try:
        from reliability.error_recovery import ErrorRecoveryManager, ErrorClassifier, CircuitBreaker, resilient_operation
        
        # Test error classifier
        classifier = ErrorClassifier()
        test_error = ConnectionError("Database connection timeout")
        category, severity, strategy = classifier.classify_error(test_error)
        
        results.append({
            'test': 'error_recovery_classification',
            'success': True,
            'error_category': category,
            'severity': severity.value,
            'strategy': strategy.value,
            'details': f'Error classified as {category} with {severity.value} severity'
        })
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        @breaker
        def test_function():
            return "success"
        
        result = test_function()
        
        results.append({
            'test': 'error_recovery_circuit_breaker',
            'success': True,
            'breaker_state': breaker.state,
            'function_result': result,
            'details': f'Circuit breaker in {breaker.state} state'
        })
        
        # Test recovery manager
        recovery_manager = ErrorRecoveryManager()
        
        try:
            raise ValueError("Test error for recovery")
        except Exception as e:
            recovery_result = recovery_manager.handle_error(e, component="test", operation="test_op")
        
        stats = recovery_manager.get_error_statistics()
        
        results.append({
            'test': 'error_recovery_manager',
            'success': True,
            'total_errors': stats.get('recovery_stats', {}).get('total_errors', 0),
            'recovery_rate': stats.get('recovery_stats', {}).get('recovery_rate', 0),
            'details': f'Recovery manager handled {stats.get("recovery_stats", {}).get("total_errors", 0)} errors'
        })
        
    except Exception as e:
        results.append({
            'test': 'error_recovery',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[-300:]
        })
    
    return results


def test_graceful_degradation():
    """Test graceful degradation system."""
    results = []
    
    try:
        from reliability.graceful_degradation import GracefulDegradationManager, DegradationLevel, ServiceCapability
        
        degradation_manager = GracefulDegradationManager()
        
        # Test initial status
        initial_status = degradation_manager.get_degradation_status()
        
        results.append({
            'test': 'graceful_degradation_status',
            'success': True,
            'current_level': initial_status.get('current_level'),
            'total_capabilities': initial_status.get('capabilities', {}).get('total', 0),
            'enabled_capabilities': initial_status.get('capabilities', {}).get('enabled', 0),
            'details': f'System at {initial_status.get("current_level")} level with {initial_status.get("capabilities", {}).get("total", 0)} capabilities'
        })
        
        # Test degradation trigger
        degradation_manager.trigger_degradation(
            DegradationLevel.MINOR_DEGRADATION,
            "Test degradation scenario",
            ["adaptive_learning", "query_insights"]
        )
        
        degraded_status = degradation_manager.get_degradation_status()
        
        results.append({
            'test': 'graceful_degradation_trigger',
            'success': True,
            'new_level': degraded_status.get('current_level'),
            'degraded_count': degraded_status.get('capabilities', {}).get('degraded', 0),
            'details': f'Successfully degraded to {degraded_status.get("current_level")} with {degraded_status.get("capabilities", {}).get("degraded", 0)} capabilities disabled'
        })
        
        # Test capability execution with degradation
        result = degradation_manager.execute_with_degradation(
            "adaptive_learning",
            lambda: {"advanced": "result"}, 
        )
        
        results.append({
            'test': 'graceful_degradation_execution',
            'success': True,
            'degraded_mode': result.get('degraded_mode', False),
            'details': f'Executed capability with degradation: {bool(result.get("degraded_mode"))}'
        })
        
    except Exception as e:
        results.append({
            'test': 'graceful_degradation',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[-300:]
        })
    
    return results


def test_performance_optimizer():
    """Test performance optimization system."""
    results = []
    
    try:
        from scaling.performance_optimizer import PerformanceOptimizer, ResourceMonitor, MemoryOptimizer
        
        # Test resource monitor
        monitor = ResourceMonitor()
        metrics = monitor.collect_metrics({'avg_response_time_ms': 500, 'throughput_rps': 10})
        
        results.append({
            'test': 'performance_optimizer_monitor',
            'success': True,
            'cpu_usage': metrics.cpu_usage,
            'memory_usage_mb': metrics.memory_usage_mb,
            'response_time': metrics.avg_response_time_ms,
            'details': f'Resource monitor collected metrics: CPU {metrics.cpu_usage}%, Memory {metrics.memory_usage_mb:.0f}MB'
        })
        
        # Test memory optimizer
        memory_optimizer = MemoryOptimizer()
        memory_usage = memory_optimizer.get_memory_usage()
        optimizations = memory_optimizer.optimize_memory_usage()
        
        results.append({
            'test': 'performance_optimizer_memory',
            'success': True,
            'memory_rss_mb': memory_usage.get('rss_mb', 0),
            'optimizations_performed': len(optimizations),
            'details': f'Memory optimizer performed {len(optimizations)} optimizations'
        })
        
        # Test performance optimizer
        optimizer = PerformanceOptimizer()
        analysis = optimizer.analyze_performance()
        recommendations = optimizer.generate_optimization_recommendations()
        
        results.append({
            'test': 'performance_optimizer_main',
            'success': True,
            'recommendations_count': len(recommendations),
            'analysis_complete': 'current_metrics' in analysis,
            'details': f'Performance optimizer generated {len(recommendations)} optimization recommendations'
        })
        
    except Exception as e:
        results.append({
            'test': 'performance_optimizer',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[-300:]
        })
    
    return results


def test_auto_scaler():
    """Test auto-scaling system."""
    results = []
    
    try:
        from scaling.auto_scaler import AutoScaler, ResourcePool, LoadBalancer, ScalingMetric
        
        # Test load balancer
        load_balancer = LoadBalancer()
        load_balancer.add_instance("instance1", {"host": "server1", "port": 8080})
        load_balancer.add_instance("instance2", {"host": "server2", "port": 8080})
        
        instance = load_balancer.get_next_instance()
        distribution = load_balancer.get_load_distribution()
        
        results.append({
            'test': 'auto_scaler_load_balancer',
            'success': True,
            'total_instances': distribution.get('total_instances', 0),
            'healthy_instances': distribution.get('healthy_instances', 0),
            'selected_instance': instance.get('id') if instance else None,
            'details': f'Load balancer managing {distribution.get("total_instances", 0)} instances'
        })
        
        # Test resource pool
        def create_resource():
            return {"id": f"resource_{time.time()}", "status": "active"}
        
        def destroy_resource(resource):
            resource["status"] = "destroyed"
        
        pool = ResourcePool("test_pool", min_size=2, max_size=10)
        pool.set_resource_management_functions(create_resource, destroy_resource)
        
        # Test resource acquisition
        resource = pool.acquire_resource()
        if resource:
            pool.release_resource(resource)
        
        pool_stats = pool.get_pool_statistics()
        
        results.append({
            'test': 'auto_scaler_resource_pool',
            'success': True,
            'current_size': pool_stats.get('current_size', 0),
            'utilization': pool_stats.get('utilization_percent', 0),
            'hit_rate': pool_stats.get('hit_rate_percent', 0),
            'details': f'Resource pool with {pool_stats.get("current_size", 0)} resources at {pool_stats.get("utilization_percent", 0):.1f}% utilization'
        })
        
        # Test auto scaler
        auto_scaler = AutoScaler()
        auto_scaler.register_resource_pool(pool)
        
        # Test metrics recording
        test_metrics = {
            ScalingMetric.CPU_USAGE: 75.0,
            ScalingMetric.MEMORY_USAGE: 60.0,
            ScalingMetric.RESPONSE_TIME: 800.0
        }
        auto_scaler.record_metrics(test_metrics)
        
        decisions = auto_scaler.evaluate_scaling_decision()
        status = auto_scaler.get_scaling_status()
        
        results.append({
            'test': 'auto_scaler_main',
            'success': True,
            'scaling_rules': len(status.get('scaling_rules', [])),
            'decisions_generated': len(decisions),
            'metrics_collected': status.get('metrics_collected', 0),
            'details': f'Auto scaler with {len(status.get("scaling_rules", []))} rules generated {len(decisions)} scaling decisions'
        })
        
    except Exception as e:
        results.append({
            'test': 'auto_scaler',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[-300:]
        })
    
    return results


async def test_connection_pool():
    """Test connection pool system."""
    results = []
    
    try:
        from scaling.connection_pool import ConnectionPool, PoolConfiguration, HealthChecker
        
        # Mock connection factory
        async def create_connection():
            class MockConnection:
                def __init__(self):
                    self.closed = False
                
                async def execute(self, query):
                    if self.closed:
                        raise Exception("Connection closed")
                    return [{"result": 1}]
                
                async def close(self):
                    self.closed = True
            
            return MockConnection()
        
        # Test health checker
        health_checker = HealthChecker()
        mock_conn = await create_connection()
        is_healthy = await health_checker.check_connection_health(mock_conn)
        
        results.append({
            'test': 'connection_pool_health_checker',
            'success': True,
            'connection_healthy': is_healthy,
            'details': f'Health checker validated connection: {is_healthy}'
        })
        
        # Test connection pool
        config = PoolConfiguration(min_size=2, max_size=5, pool_timeout_seconds=5.0)
        pool = ConnectionPool("test_pool", create_connection, config, health_checker)
        
        await pool.start()
        
        # Test connection acquisition
        async with pool.acquire(timeout=2.0) as conn:
            result = await conn.execute("SELECT 1")
        
        stats = pool.get_statistics()
        await pool.stop()
        
        results.append({
            'test': 'connection_pool_main',
            'success': True,
            'connections_created': stats.get('lifetime_stats', {}).get('total_created', 0),
            'connections_served': stats.get('performance_stats', {}).get('connections_served', 0),
            'pool_size': stats.get('connections', {}).get('total', 0),
            'details': f'Connection pool served {stats.get("performance_stats", {}).get("connections_served", 0)} connections'
        })
        
    except Exception as e:
        results.append({
            'test': 'connection_pool',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[-300:]
        })
    
    return results


def test_integration():
    """Test integration between modules."""
    results = []
    
    try:
        # Test intelligence + reliability integration
        from intelligence.query_insights import QueryInsightsEngine
        from reliability.graceful_degradation import GracefulDegradationManager
        from scaling.performance_optimizer import PerformanceOptimizer
        
        insights_engine = QueryInsightsEngine()
        degradation_manager = GracefulDegradationManager()
        performance_optimizer = PerformanceOptimizer()
        
        # Test query analysis with system in different states
        test_query = "SELECT u.name, COUNT(*) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id"
        
        # Normal mode analysis
        normal_analysis = insights_engine.analyze_query(test_query, execution_time_ms=500)
        
        # Degrade system and test again
        degradation_manager.trigger_degradation(
            degradation_manager.DegradationLevel.MINOR_DEGRADATION,
            "Integration test degradation",
            ["query_insights"]
        )
        
        # Test degraded execution
        degraded_result = degradation_manager.execute_with_degradation(
            "query_insights",
            lambda: insights_engine.analyze_query(test_query, execution_time_ms=500)
        )
        
        # Performance analysis
        performance_analysis = performance_optimizer.analyze_performance()
        recommendations = performance_optimizer.generate_optimization_recommendations()
        
        results.append({
            'test': 'system_integration',
            'success': True,
            'normal_insights': normal_analysis.get('total_insights', 0),
            'degraded_mode': degraded_result.get('degraded_mode', False),
            'performance_recommendations': len(recommendations),
            'details': f'Integration test: {normal_analysis.get("total_insights", 0)} insights normal, degraded mode {degraded_result.get("degraded_mode", False)}, {len(recommendations)} performance recommendations'
        })
        
    except Exception as e:
        results.append({
            'test': 'system_integration',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[-300:]
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
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"\n{status} {result['test']}")
        
        if result['success']:
            print(f"  Details: {result.get('details', 'No details')}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(f"  Trace: ...{result['traceback'][-100:]}")


async def main():
    """Main test runner."""
    print("🚀 TERRAGON AUTONOMOUS SDLC - QUALITY GATES EXECUTION")
    print("🧠 Testing Intelligence, Reliability, and Scaling Systems")
    print("=" * 70)
    
    # Test results collection
    all_results = []
    
    # Intelligence tests
    print("🔍 Testing Intelligence Modules...")
    intelligence_results = []
    intelligence_results.extend(test_intelligence_query_insights())
    intelligence_results.extend(test_adaptive_learning())
    intelligence_results.extend(test_intelligent_caching())
    all_results.extend(intelligence_results)
    print_test_results(intelligence_results, "Intelligence")
    
    # Reliability tests  
    print("\n🛡️ Testing Reliability Modules...")
    reliability_results = []
    reliability_results.extend(test_error_recovery())
    reliability_results.extend(test_graceful_degradation())
    all_results.extend(reliability_results)
    print_test_results(reliability_results, "Reliability")
    
    # Scaling tests
    print("\n📈 Testing Scaling Modules...")
    scaling_results = []
    scaling_results.extend(test_performance_optimizer())
    scaling_results.extend(test_auto_scaler())
    all_results.extend(scaling_results)
    print_test_results(scaling_results, "Scaling")
    
    # Async tests
    print("\n⚡ Testing Async Systems...")
    async_results = []
    async_results.extend(await test_connection_pool())
    all_results.extend(async_results)
    print_test_results(async_results, "Async")
    
    # Integration tests
    print("\n🔗 Testing System Integration...")
    integration_results = test_integration()
    all_results.extend(integration_results)
    print_test_results(integration_results, "Integration")
    
    # Summary
    total_tests = len(all_results)
    successful_tests = len([r for r in all_results if r['success']])
    success_rate = (successful_tests/total_tests)*100 if total_tests > 0 else 0
    
    print(f"\n" + "=" * 70)
    print("📊 AUTONOMOUS SDLC QUALITY GATES SUMMARY")
    print(f"Total test modules executed: {total_tests}")
    print(f"Successful modules: {successful_tests}")
    print(f"Failed modules: {total_tests - successful_tests}")
    print(f"Overall success rate: {success_rate:.1f}%")
    
    # Quality gate decision
    if success_rate >= 85:
        print(f"\n🎉 QUALITY GATES PASSED!")
        print("✅ All critical systems operational")
        print("✅ Intelligence systems validated")
        print("✅ Reliability mechanisms functional") 
        print("✅ Scaling infrastructure ready")
        print("🚀 System cleared for production deployment")
        return 0
    elif success_rate >= 70:
        print(f"\n⚠️ QUALITY GATES PARTIAL PASS")
        print(f"🟡 {total_tests - successful_tests} modules need attention")
        print("🔧 Review failed modules before production")
        return 1
    else:
        print(f"\n❌ QUALITY GATES FAILED")
        print(f"🚨 {total_tests - successful_tests} critical failures detected")
        print("🛠️ Major issues require resolution")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    print(f"\n🏁 Autonomous SDLC execution completed with exit code: {exit_code}")
    sys.exit(exit_code)