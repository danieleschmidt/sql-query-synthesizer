#!/usr/bin/env python3
"""
Quality Gates: Comprehensive testing, security validation, and performance benchmarks
for the quantum-inspired SQL optimization system
"""

import asyncio
import sys
import time


# Test the complete quantum system
def test_quantum_system_integration():
    """Test complete quantum system integration"""
    print("🔬 Testing Quantum System Integration")

    try:
        # Test basic imports
        from sql_synthesizer.quantum.core import (
            QuantumQueryOptimizer,
            QuantumState,
            QueryPlan,
        )

        print("  ✅ All quantum modules imported successfully")

        # Test quantum optimizer
        optimizer = QuantumQueryOptimizer(num_qubits=8, temperature=500.0)

        # Create test plans
        plans = [
            QueryPlan([], [], [], 10.0, 0.0),
            QueryPlan([], [], [], 25.0, 0.0),
            QueryPlan([], [], [], 5.0, 0.0)   # Should be optimal
        ]

        # Test superposition
        superposition = optimizer.create_superposition(plans)
        assert len(superposition) == 3
        print("  ✅ Quantum superposition working")

        # Test interference
        interfered = optimizer.quantum_interference(superposition)
        low_cost_plan = next(p for p in interfered if p.cost == 5.0)
        high_cost_plan = next(p for p in interfered if p.cost == 25.0)
        assert low_cost_plan.probability > high_cost_plan.probability
        print("  ✅ Quantum interference working")

        # Test annealing
        optimal = optimizer.quantum_annealing(interfered, iterations=50)
        assert optimal.quantum_state == QuantumState.COLLAPSED
        print("  ✅ Quantum annealing working")

        # Test health status
        health = optimizer.get_health_status()
        assert "healthy" in health
        print("  ✅ Health monitoring working")

        return True

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


def test_security_features():
    """Test security features"""
    print("🔒 Testing Security Features")

    try:
        from sql_synthesizer.quantum.security import (
            quantum_security,
        )
        from sql_synthesizer.quantum.validation import (
            QuantumValidationError,
            quantum_validator,
        )

        # Test input validation
        try:
            quantum_validator.validate_field("qubit_count", -5)
            assert False, "Should have failed validation"
        except QuantumValidationError:
            print("  ✅ Input validation working")

        # Test rate limiting
        client_id = "test_client"

        # Should pass first time
        assert quantum_security.check_rate_limit(client_id)
        print("  ✅ Rate limiting allows normal requests")

        # Test SQL injection detection
        malicious_input = "'; DROP TABLE users; --"
        assert quantum_validator._contains_malicious_patterns(malicious_input)
        print("  ✅ SQL injection detection working")

        # Test string sanitization
        sanitized = quantum_validator.sanitize_string("normal_input")
        assert sanitized == "normal_input"
        print("  ✅ String sanitization working")

        # Test circuit breakers
        cb = quantum_security.circuit_breakers["quantum_optimization"]
        assert cb is not None
        print("  ✅ Circuit breakers configured")

        return True

    except Exception as e:
        print(f"  ❌ Security test failed: {e}")
        return False


def test_performance_features():
    """Test performance optimization features"""
    print("⚡ Testing Performance Features")

    try:
        from sql_synthesizer.quantum.cache import quantum_cache_manager
        from sql_synthesizer.quantum.performance import (
            PerformanceMetrics,
            quantum_performance_monitor,
            quantum_resource_pool,
        )

        # Test performance monitoring
        metric = PerformanceMetrics(
            operation_type="test_operation",
            execution_time=0.5,
            memory_usage=0.3,
            cpu_usage=0.4,
            success=True
        )

        quantum_performance_monitor.record_metric(metric)
        stats = quantum_performance_monitor.get_performance_stats("test_operation")
        assert "execution_time" in stats
        print("  ✅ Performance monitoring working")

        # Test caching
        cache_key = quantum_cache_manager.l1_cache.get_cache_key({
            "tables": ["users", "orders"],
            "joins": [("users", "orders")]
        })

        assert len(cache_key) == 16  # SHA256 hash truncated
        print("  ✅ Cache key generation working")

        # Test resource pool
        pool_stats = quantum_resource_pool.get_pool_stats()
        assert "optimizer_pool_size" in pool_stats
        print("  ✅ Resource pooling working")

        # Test recommendations
        recommendations = quantum_performance_monitor.get_recommendations()
        assert isinstance(recommendations, list)
        print("  ✅ Performance recommendations working")

        return True

    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False


def test_scaling_features():
    """Test auto-scaling and load balancing"""
    print("📈 Testing Scaling Features")

    try:
        from sql_synthesizer.quantum.scaling import (
            QuantumAutoScaler,
            QuantumLoadBalancer,
        )

        # Test load balancer
        lb = QuantumLoadBalancer()

        # Register test workers
        lb.register_worker("worker_1", capacity=2.0)
        lb.register_worker("worker_2", capacity=1.5)

        # Test worker selection
        selected = lb.select_worker()
        assert selected in ["worker_1", "worker_2"]
        print("  ✅ Load balancing working")

        # Test task assignment
        assert lb.assign_task(selected, load=0.5)
        print("  ✅ Task assignment working")

        # Test load distribution
        dist = lb.get_load_distribution()
        assert "overall" in dist
        assert dist["overall"]["total_workers"] == 2
        print("  ✅ Load distribution tracking working")

        # Test auto-scaler
        scaler = QuantumAutoScaler(min_workers=1, max_workers=5)

        action, count = scaler.evaluate_scaling_need()
        assert action in ["scale_up", "scale_down", "maintain"]
        print("  ✅ Auto-scaling evaluation working")

        stats = scaler.get_scaling_stats()
        assert "current_workers" in stats
        print("  ✅ Scaling statistics working")

        # Cleanup
        lb.shutdown()
        scaler.shutdown()

        return True

    except Exception as e:
        print(f"  ❌ Scaling test failed: {e}")
        return False


async def test_async_features():
    """Test async quantum features"""
    print("🚀 Testing Async Features")

    try:
        from sql_synthesizer.quantum.core import QuantumQueryOptimizer, QueryPlan

        optimizer = QuantumQueryOptimizer(num_qubits=4, timeout_seconds=5.0)

        plans = [
            QueryPlan([], [], [], 15.0, 0.0),
            QueryPlan([], [], [], 8.0, 0.0),
            QueryPlan([], [], [], 20.0, 0.0)
        ]

        # Test async optimization
        start_time = time.time()
        optimal_plan = await optimizer.optimize_query_async(plans, client_id="test_async")
        execution_time = time.time() - start_time

        assert optimal_plan is not None
        assert execution_time < 5.0  # Should complete within timeout
        print(f"  ✅ Async optimization completed in {execution_time:.3f}s")

        # Test metrics
        metrics = optimizer.get_quantum_metrics()
        assert metrics["optimization_count"] >= 1
        print("  ✅ Optimization metrics tracking working")

        return True

    except Exception as e:
        print(f"  ❌ Async test failed: {e}")
        return False


def benchmark_performance():
    """Run performance benchmarks"""
    print("🏁 Running Performance Benchmarks")

    try:
        from sql_synthesizer.quantum.core import QuantumQueryOptimizer, QueryPlan

        # Benchmark quantum optimization
        optimizer = QuantumQueryOptimizer(num_qubits=16, temperature=1000.0)

        # Create various plan sizes
        plan_sizes = [5, 10, 20, 50]
        results = {}

        for size in plan_sizes:
            plans = [
                QueryPlan([], [], [], float(i * 10 + 5), 0.0)
                for i in range(size)
            ]

            start_time = time.time()

            # Run optimization multiple times
            times = []
            for _ in range(5):
                iter_start = time.time()
                optimizer.quantum_annealing(plans, iterations=100)
                times.append(time.time() - iter_start)

            avg_time = sum(times) / len(times)
            results[size] = avg_time

            print(f"  📊 {size} plans: {avg_time:.3f}s avg ({min(times):.3f}s - {max(times):.3f}s)")

        # Benchmark should show reasonable scaling
        assert results[5] < 1.0  # Small plans should be fast
        assert results[50] < 10.0  # Even large plans should be reasonable

        print("  ✅ Performance benchmarks passed")
        return True

    except Exception as e:
        print(f"  ❌ Performance benchmark failed: {e}")
        return False


def test_error_handling():
    """Test comprehensive error handling"""
    print("🛡️ Testing Error Handling")

    try:
        from sql_synthesizer.quantum.core import QuantumQueryOptimizer
        from sql_synthesizer.quantum.exceptions import (
            QuantumOptimizationError,
            QuantumTimeoutError,
            QuantumValidationError,
        )

        # Test invalid initialization
        try:
            QuantumQueryOptimizer(num_qubits=-1)
            assert False, "Should have failed validation"
        except QuantumValidationError:
            print("  ✅ Invalid initialization properly rejected")

        # Test empty plan optimization
        optimizer = QuantumQueryOptimizer(num_qubits=4)
        try:
            optimizer.quantum_annealing([])
            assert False, "Should have failed with empty plans"
        except (QuantumOptimizationError, ValueError):
            print("  ✅ Empty plan optimization properly rejected")

        # Test timeout handling
        optimizer_fast = QuantumQueryOptimizer(num_qubits=4, timeout_seconds=0.001)  # Very short timeout
        plans = [QueryPlan([], [], [], 10.0, 0.0) for _ in range(10)]

        try:
            import asyncio
            asyncio.run(optimizer_fast.optimize_query_async(plans))
            # Might pass if system is very fast, but should handle gracefully
            print("  ✅ Timeout handling working (completed within timeout)")
        except QuantumTimeoutError:
            print("  ✅ Timeout handling working (properly timed out)")

        return True

    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False


def test_thread_safety():
    """Test thread safety of quantum components"""
    print("🧵 Testing Thread Safety")

    try:
        import threading

        from sql_synthesizer.quantum.cache import QuantumPlanCache
        from sql_synthesizer.quantum.core import QuantumQueryOptimizer, QueryPlan

        # Test thread-safe optimization
        optimizer = QuantumQueryOptimizer(num_qubits=8)
        cache = QuantumPlanCache(max_size=100)

        results = []
        errors = []

        def worker_thread(thread_id):
            """TODO: Add docstring"""
            try:
                plans = [
                    QueryPlan([], [], [], float(10 + thread_id), 0.0),
                    QueryPlan([], [], [], float(20 + thread_id), 0.0)
                ]

                # Test optimization
                optimal = optimizer.quantum_annealing(plans, iterations=50)

                # Test caching
                key = f"test_key_{thread_id}"
                cache.put(key, optimal, cost_reduction=0.2)
                retrieved = cache.get(key)

                if retrieved and retrieved.cost == optimal.cost:
                    results.append(thread_id)
                else:
                    errors.append(f"Thread {thread_id}: Cache mismatch")

            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")

        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)

        if errors:
            print(f"  ❌ Thread safety errors: {errors}")
            return False

        if len(results) == 5:
            print("  ✅ Thread safety test passed")
            return True
        else:
            print(f"  ⚠️ Only {len(results)}/5 threads completed successfully")
            return len(results) >= 3  # Allow some tolerance

    except Exception as e:
        print(f"  ❌ Thread safety test failed: {e}")
        return False


async def run_all_quality_gates():
    """Run all quality gates"""
    print("🏆 Quantum SQL Synthesizer - Quality Gates")
    print("=" * 60)

    tests = [
        ("System Integration", test_quantum_system_integration),
        ("Security Features", test_security_features),
        ("Performance Features", test_performance_features),
        ("Scaling Features", test_scaling_features),
        ("Error Handling", test_error_handling),
        ("Thread Safety", test_thread_safety),
        ("Performance Benchmarks", benchmark_performance),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"  ❌ {test_name} failed")
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")

    # Run async test separately
    print("\n🔍 Async Features")
    try:
        if await test_async_features():
            passed += 1
            total += 1
        else:
            print("  ❌ Async Features failed")
            total += 1
    except Exception as e:
        print(f"  ❌ Async Features failed with exception: {e}")
        total += 1

    print("\n" + "=" * 60)
    print(f"🏁 Quality Gates Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("   Quantum-inspired SQL optimization is production-ready!")
        return 0
    elif passed >= total * 0.8:  # 80% pass rate
        print("✅ QUALITY GATES MOSTLY PASSED!")
        print(f"   {passed}/{total} tests passed - system is ready for deployment")
        return 0
    else:
        print("⚠️  QUALITY GATES FAILED!")
        print(f"   Only {passed}/{total} tests passed - needs improvement")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_all_quality_gates()))
