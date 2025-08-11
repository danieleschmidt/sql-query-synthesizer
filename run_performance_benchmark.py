#!/usr/bin/env python3
"""Performance benchmark runner for SQL Query Synthesizer."""

import time
import asyncio
import threading
import statistics
import json
import sys
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our enhanced modules
from sql_synthesizer.enhanced_core import performance_tracker, query_optimizer
from sql_synthesizer.adaptive_caching import adaptive_cache
from sql_synthesizer.intelligent_query_router import query_router, QueryComplexity, DatabaseRole
from sql_synthesizer.auto_scaling_engine import auto_scaling_engine
from sql_synthesizer.performance_optimizer import performance_optimizer


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_performance_tracking(self, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark performance tracking overhead."""
        print(f"🏃‍♂️ Benchmarking performance tracking ({iterations} iterations)...")
        
        times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            
            # Track a query
            query_id = performance_tracker.record_query_start()
            performance_tracker.record_query_end(100.0 + i % 50, success=True)
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
        
        return {
            'operation': 'performance_tracking',
            'iterations': iterations,
            'avg_time_ms': statistics.mean(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0,
            'total_time_ms': sum(times),
            'ops_per_second': iterations / (sum(times) / 1000)
        }
    
    def benchmark_caching_operations(self, iterations: int = 5000) -> Dict[str, Any]:
        """Benchmark caching operations."""
        print(f"💾 Benchmarking caching operations ({iterations} iterations)...")
        
        # Benchmark cache writes
        write_times = []
        for i in range(iterations):
            key = f"test_key_{i}"
            value = f"test_value_{i}" * 10  # Make values larger
            
            start = time.perf_counter()
            adaptive_cache.set(key, value, execution_time_ms=50.0)
            end = time.perf_counter()
            
            write_times.append((end - start) * 1000)
        
        # Benchmark cache reads
        read_times = []
        for i in range(min(iterations, 1000)):  # Read subset of keys
            key = f"test_key_{i}"
            
            start = time.perf_counter()
            value = adaptive_cache.get(key)
            end = time.perf_counter()
            
            read_times.append((end - start) * 1000)
        
        return {
            'operation': 'caching_operations',
            'write_iterations': iterations,
            'read_iterations': len(read_times),
            'avg_write_time_ms': statistics.mean(write_times),
            'avg_read_time_ms': statistics.mean(read_times),
            'write_ops_per_second': iterations / (sum(write_times) / 1000),
            'read_ops_per_second': len(read_times) / (sum(read_times) / 1000),
            'cache_stats': adaptive_cache.get_statistics()
        }
    
    def benchmark_query_routing(self, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark query routing decisions."""
        print(f"🎯 Benchmarking query routing ({iterations} iterations)...")
        
        # Add test endpoints
        query_router.add_database_endpoint("bench_primary", "postgresql://test", DatabaseRole.PRIMARY)
        query_router.add_database_endpoint("bench_replica", "postgresql://replica", DatabaseRole.REPLICA)
        
        test_queries = [
            "SELECT id FROM users WHERE active = true",
            "SELECT COUNT(*) FROM orders WHERE created_at > '2024-01-01'",
            "INSERT INTO logs (message) VALUES ('test')",
            "UPDATE users SET last_login = NOW() WHERE id = 1",
            """
            SELECT u.name, COUNT(o.id) 
            FROM users u 
            LEFT JOIN orders o ON u.id = o.user_id 
            WHERE u.created_at > '2023-01-01' 
            GROUP BY u.name 
            HAVING COUNT(o.id) > 5
            """
        ]
        
        times = []
        
        async def route_queries():
            for i in range(iterations):
                query = test_queries[i % len(test_queries)]
                
                start = time.perf_counter()
                route = await query_router.route_query(query)
                end = time.perf_counter()
                
                times.append((end - start) * 1000)
        
        # Run async benchmark
        asyncio.run(route_queries())
        
        return {
            'operation': 'query_routing',
            'iterations': iterations,
            'avg_time_ms': statistics.mean(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'ops_per_second': iterations / (sum(times) / 1000),
            'routing_stats': query_router.get_routing_stats()
        }
    
    def benchmark_concurrent_operations(self, threads: int = 10, ops_per_thread: int = 100) -> Dict[str, Any]:
        """Benchmark concurrent operations."""
        print(f"🔀 Benchmarking concurrent operations ({threads} threads, {ops_per_thread} ops each)...")
        
        def worker_task(worker_id: int) -> Dict[str, Any]:
            """Worker task for concurrent benchmark."""
            worker_times = []
            
            for i in range(ops_per_thread):
                start = time.perf_counter()
                
                # Mix of operations
                query_id = performance_tracker.record_query_start()
                adaptive_cache.set(f"worker_{worker_id}_key_{i}", f"value_{i}", execution_time_ms=75.0)
                cached_value = adaptive_cache.get(f"worker_{worker_id}_key_{i}")
                performance_tracker.record_query_end(75.0 + i % 25, success=True)
                
                end = time.perf_counter()
                worker_times.append((end - start) * 1000)
            
            return {
                'worker_id': worker_id,
                'avg_time_ms': statistics.mean(worker_times),
                'total_time_ms': sum(worker_times),
                'operations': len(worker_times)
            }
        
        start_time = time.perf_counter()
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(threads)]
            worker_results = [future.result() for future in as_completed(futures)]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Aggregate results
        total_operations = sum(result['operations'] for result in worker_results)
        avg_worker_time = statistics.mean([result['avg_time_ms'] for result in worker_results])
        
        return {
            'operation': 'concurrent_operations',
            'threads': threads,
            'ops_per_thread': ops_per_thread,
            'total_operations': total_operations,
            'total_time_seconds': total_time,
            'avg_worker_time_ms': avg_worker_time,
            'overall_ops_per_second': total_operations / total_time,
            'worker_results': worker_results[:5]  # Include first 5 worker results
        }
    
    def benchmark_optimization_analysis(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark query optimization analysis."""
        print(f"🔧 Benchmarking optimization analysis ({iterations} iterations)...")
        
        test_queries = [
            "SELECT * FROM users WHERE name LIKE '%admin%'",
            "SELECT u.id, COUNT(o.id) FROM users u, orders o WHERE u.active = 1 GROUP BY u.id",
            "SELECT DISTINCT name FROM users ORDER BY created_at",
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100)",
            """
            SELECT u.name, p.title, COUNT(*) 
            FROM users u 
            JOIN profiles p ON u.id = p.user_id 
            JOIN orders o ON u.id = o.user_id 
            WHERE u.created_at > '2023-01-01' 
            AND p.visibility = 'public' 
            GROUP BY u.name, p.title 
            ORDER BY COUNT(*) DESC
            """
        ]
        
        times = []
        
        for i in range(iterations):
            query = test_queries[i % len(test_queries)]
            
            # Record execution for analysis
            performance_optimizer.record_query_execution(
                query, 
                250.0 + i % 200, 
                {
                    'rows_examined': 1000 + i % 500,
                    'rows_returned': 50 + i % 25,
                    'cache_hit': i % 3 == 0
                }
            )
            
            start = time.perf_counter()
            recommendations = performance_optimizer.generate_optimization_recommendations(limit=5)
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
        
        return {
            'operation': 'optimization_analysis',
            'iterations': iterations,
            'avg_time_ms': statistics.mean(times),
            'max_time_ms': max(times),
            'ops_per_second': iterations / (sum(times) / 1000),
            'performance_summary': performance_optimizer.get_performance_summary()
        }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage of key components."""
        print("🧠 Benchmarking memory usage...")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load cache with data
            for i in range(1000):
                adaptive_cache.set(f"mem_test_{i}", "x" * 1000, execution_time_ms=100.0)
            
            cache_loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load performance data
            for i in range(500):
                performance_optimizer.record_query_execution(
                    f"SELECT * FROM table_{i % 10} WHERE id = {i}",
                    150.0 + i % 100
                )
            
            perf_loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                'operation': 'memory_usage',
                'baseline_memory_mb': baseline_memory,
                'cache_loaded_memory_mb': cache_loaded_memory,
                'perf_loaded_memory_mb': perf_loaded_memory,
                'cache_overhead_mb': cache_loaded_memory - baseline_memory,
                'perf_overhead_mb': perf_loaded_memory - cache_loaded_memory,
                'total_overhead_mb': perf_loaded_memory - baseline_memory
            }
            
        except ImportError:
            return {
                'operation': 'memory_usage',
                'error': 'psutil not available for memory measurement'
            }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites."""
        print("🚀 Starting comprehensive performance benchmark...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run individual benchmarks
        benchmarks = [
            ('performance_tracking', lambda: self.benchmark_performance_tracking(1000)),
            ('caching_operations', lambda: self.benchmark_caching_operations(2000)),
            ('query_routing', lambda: self.benchmark_query_routing(500)),
            ('concurrent_operations', lambda: self.benchmark_concurrent_operations(8, 50)),
            ('optimization_analysis', lambda: self.benchmark_optimization_analysis(50)),
            ('memory_usage', lambda: self.benchmark_memory_usage()),
        ]
        
        results = {}
        
        for name, benchmark_func in benchmarks:
            try:
                print(f"\n📊 Running {name} benchmark...")
                result = benchmark_func()
                results[name] = result
                
                # Print summary
                if 'avg_time_ms' in result:
                    print(f"   ⏱️  Average time: {result['avg_time_ms']:.2f}ms")
                if 'ops_per_second' in result:
                    print(f"   🏃  Operations/sec: {result['ops_per_second']:.0f}")
                    
            except Exception as e:
                print(f"   ❌ Benchmark {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        end_time = time.time()
        
        # Overall summary
        results['summary'] = {
            'total_benchmark_time_seconds': end_time - start_time,
            'benchmarks_completed': len([r for r in results.values() if 'error' not in r]),
            'benchmarks_failed': len([r for r in results.values() if 'error' in r]),
        }
        
        return results
    
    def print_benchmark_report(self, results: Dict[str, Any]):
        """Print a comprehensive benchmark report."""
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE BENCHMARK REPORT")
        print("=" * 60)
        
        summary = results.get('summary', {})
        print(f"🕒 Total benchmark time: {summary.get('total_benchmark_time_seconds', 0):.2f} seconds")
        print(f"✅ Completed benchmarks: {summary.get('benchmarks_completed', 0)}")
        print(f"❌ Failed benchmarks: {summary.get('benchmarks_failed', 0)}")
        
        print("\n🏆 Performance Highlights:")
        
        # Extract key metrics
        highlights = []
        
        if 'performance_tracking' in results:
            pt = results['performance_tracking']
            if 'ops_per_second' in pt:
                highlights.append(f"   📊 Performance tracking: {pt['ops_per_second']:.0f} ops/sec")
        
        if 'caching_operations' in results:
            cache = results['caching_operations']
            if 'write_ops_per_second' in cache:
                highlights.append(f"   💾 Cache writes: {cache['write_ops_per_second']:.0f} ops/sec")
            if 'read_ops_per_second' in cache:
                highlights.append(f"   💾 Cache reads: {cache['read_ops_per_second']:.0f} ops/sec")
        
        if 'concurrent_operations' in results:
            concurrent = results['concurrent_operations']
            if 'overall_ops_per_second' in concurrent:
                highlights.append(f"   🔀 Concurrent ops: {concurrent['overall_ops_per_second']:.0f} ops/sec")
        
        if 'memory_usage' in results:
            memory = results['memory_usage']
            if 'total_overhead_mb' in memory:
                highlights.append(f"   🧠 Memory overhead: {memory['total_overhead_mb']:.1f} MB")
        
        for highlight in highlights:
            print(highlight)
        
        # Performance grades
        print("\n📊 Performance Grades:")
        
        grades = []
        
        # Grade caching performance
        if 'caching_operations' in results and 'write_ops_per_second' in results['caching_operations']:
            write_ops = results['caching_operations']['write_ops_per_second']
            if write_ops > 10000:
                grades.append("   💾 Caching: A+ (Excellent)")
            elif write_ops > 5000:
                grades.append("   💾 Caching: A (Very Good)")
            elif write_ops > 1000:
                grades.append("   💾 Caching: B (Good)")
            else:
                grades.append("   💾 Caching: C (Needs Improvement)")
        
        # Grade concurrent performance
        if 'concurrent_operations' in results and 'overall_ops_per_second' in results['concurrent_operations']:
            concurrent_ops = results['concurrent_operations']['overall_ops_per_second']
            if concurrent_ops > 1000:
                grades.append("   🔀 Concurrency: A+ (Excellent)")
            elif concurrent_ops > 500:
                grades.append("   🔀 Concurrency: A (Very Good)")
            elif concurrent_ops > 100:
                grades.append("   🔀 Concurrency: B (Good)")
            else:
                grades.append("   🔀 Concurrency: C (Needs Improvement)")
        
        for grade in grades:
            print(grade)
        
        print("\n" + "=" * 60)


def main():
    """Main benchmark runner."""
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    results = benchmark.run_all_benchmarks()
    
    # Print report
    benchmark.print_benchmark_report(results)
    
    # Save detailed results
    with open('/root/repo/benchmark-results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📁 Detailed results saved to: benchmark-results.json")
    
    # Determine if benchmarks passed
    failed_count = results.get('summary', {}).get('benchmarks_failed', 0)
    if failed_count == 0:
        print("🎉 All benchmarks completed successfully!")
        return True
    else:
        print(f"⚠️  {failed_count} benchmarks failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)