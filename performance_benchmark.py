"""
Comprehensive Performance Benchmarking for Quantum SDLC Systems

Advanced performance testing, profiling, and optimization recommendations
for the quantum-inspired autonomous SDLC framework.
"""

import asyncio
import time
import gc
import logging
import statistics
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sys
import tracemalloc
import psutil
import concurrent.futures
from contextlib import asynccontextmanager, contextmanager
import cProfile
import pstats
import tempfile


class BenchmarkType(Enum):
    """Types of performance benchmarks"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    CONCURRENCY = "concurrency"
    SCALABILITY = "scalability"
    STRESS = "stress"


class PerformanceLevel(Enum):
    """Performance assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    benchmark_name: str
    benchmark_type: BenchmarkType
    duration: float
    throughput: Optional[float] = None
    memory_used: Optional[float] = None
    cpu_percent: Optional[float] = None
    success_rate: float = 1.0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    benchmark_timestamp: float
    total_duration: float
    system_info: Dict[str, Any]
    benchmark_results: List[BenchmarkResult]
    performance_summary: Dict[str, Any]
    recommendations: List[str]
    comparison_baseline: Optional[Dict[str, Any]] = None


class SystemProfiler:
    """System resource profiler"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.samples: List[Dict[str, Any]] = []
        self._monitor_task: Optional[asyncio.Task] = None
    
    @asynccontextmanager
    async def profile(self):
        """Context manager for profiling system resources"""
        await self.start_monitoring()
        try:
            yield self
        finally:
            await self.stop_monitoring()
    
    async def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.samples.clear()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU and memory info
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # System info
                system_cpu = psutil.cpu_percent(interval=None)
                system_memory = psutil.virtual_memory()
                
                sample = {
                    'timestamp': time.time(),
                    'process_cpu_percent': cpu_percent,
                    'process_memory_mb': memory_info.rss / 1024 / 1024,
                    'process_memory_percent': memory_percent,
                    'system_cpu_percent': system_cpu,
                    'system_memory_percent': system_memory.percent,
                    'system_memory_available_gb': system_memory.available / 1024 / 1024 / 1024
                }
                
                self.samples.append(sample)
                
                await asyncio.sleep(self.sample_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Continue monitoring despite errors
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        if not self.samples:
            return {}
        
        # Extract metrics
        cpu_values = [s['process_cpu_percent'] for s in self.samples]
        memory_values = [s['process_memory_mb'] for s in self.samples]
        
        return {
            'duration': self.samples[-1]['timestamp'] - self.samples[0]['timestamp'] if self.samples else 0,
            'sample_count': len(self.samples),
            'cpu_stats': {
                'mean': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory_stats': {
                'mean_mb': statistics.mean(memory_values),
                'max_mb': max(memory_values),
                'min_mb': min(memory_values),
                'std_mb': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            },
            'peak_usage': {
                'cpu_percent': max(cpu_values),
                'memory_mb': max(memory_values)
            }
        }


class QuantumPerformanceBenchmark:
    """
    Comprehensive performance benchmark suite for quantum SDLC systems
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.profiler = SystemProfiler()
        
        # Benchmark configuration
        self.warm_up_iterations = 3
        self.benchmark_iterations = 10
        self.stress_duration = 30.0  # seconds
        
        # Performance thresholds
        self.thresholds = {
            'max_response_time_ms': 1000,
            'min_throughput_ops_sec': 10,
            'max_memory_mb': 512,
            'max_cpu_percent': 80,
            'min_success_rate': 0.95
        }
    
    async def run_comprehensive_benchmark(self, project_path: Path = None) -> PerformanceReport:
        """Run comprehensive performance benchmark"""
        
        start_time = time.time()
        project_path = project_path or Path.cwd()
        
        self.logger.info("🚀 Starting comprehensive performance benchmark")
        
        # Get system information
        system_info = self._get_system_info()
        
        # Run benchmarks
        results = []
        
        # 1. Basic latency tests
        self.logger.info("⏱️ Running latency benchmarks")
        latency_results = await self._run_latency_benchmarks()
        results.extend(latency_results)
        
        # 2. Throughput tests
        self.logger.info("📊 Running throughput benchmarks")
        throughput_results = await self._run_throughput_benchmarks()
        results.extend(throughput_results)
        
        # 3. Memory tests
        self.logger.info("💾 Running memory benchmarks")
        memory_results = await self._run_memory_benchmarks()
        results.extend(memory_results)
        
        # 4. Concurrency tests
        self.logger.info("🔄 Running concurrency benchmarks")
        concurrency_results = await self._run_concurrency_benchmarks()
        results.extend(concurrency_results)
        
        # 5. Scalability tests
        self.logger.info("📈 Running scalability benchmarks")
        scalability_results = await self._run_scalability_benchmarks()
        results.extend(scalability_results)
        
        # 6. Stress tests
        self.logger.info("💪 Running stress benchmarks")
        stress_results = await self._run_stress_benchmarks()
        results.extend(stress_results)
        
        total_duration = time.time() - start_time
        
        # Generate summary and recommendations
        performance_summary = self._generate_performance_summary(results)
        recommendations = self._generate_recommendations(results, performance_summary)
        
        report = PerformanceReport(
            benchmark_timestamp=start_time,
            total_duration=total_duration,
            system_info=system_info,
            benchmark_results=results,
            performance_summary=performance_summary,
            recommendations=recommendations
        )
        
        self.logger.info(f"✅ Performance benchmark completed in {total_duration:.2f}s")
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        
        try:
            return {
                'python_version': sys.version,
                'cpu_count': multiprocessing.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'disk_usage': {
                    'total_gb': psutil.disk_usage('.').total / 1024 / 1024 / 1024,
                    'free_gb': psutil.disk_usage('.').free / 1024 / 1024 / 1024
                },
                'platform': sys.platform
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system info: {str(e)}")
            return {}
    
    async def _run_latency_benchmarks(self) -> List[BenchmarkResult]:
        """Run latency benchmarks"""
        
        results = []
        
        # Test 1: Simple function call latency
        result = await self._benchmark_simple_function_call()
        results.append(result)
        
        # Test 2: Async function call latency
        result = await self._benchmark_async_function_call()
        results.append(result)
        
        # Test 3: Quantum optimizer latency
        result = await self._benchmark_quantum_optimizer_latency()
        results.append(result)
        
        return results
    
    async def _benchmark_simple_function_call(self) -> BenchmarkResult:
        """Benchmark simple function call latency"""
        
        def simple_function(x: int) -> int:
            return x * 2 + 1
        
        return await self._run_latency_benchmark(
            name="simple_function_call",
            func=lambda: simple_function(42),
            iterations=10000
        )
    
    async def _benchmark_async_function_call(self) -> BenchmarkResult:
        """Benchmark async function call latency"""
        
        async def async_function(x: int) -> int:
            await asyncio.sleep(0)  # Yield control
            return x * 2 + 1
        
        return await self._run_latency_benchmark(
            name="async_function_call",
            func=lambda: async_function(42),
            iterations=1000,
            is_async=True
        )
    
    async def _benchmark_quantum_optimizer_latency(self) -> BenchmarkResult:
        """Benchmark quantum optimizer latency"""
        
        try:
            from sql_synthesizer.quantum.autonomous_optimizer import (
                AutonomousQuantumOptimizer, SDLCTask, SDLCPhase
            )
            
            optimizer = AutonomousQuantumOptimizer(
                max_parallel_tasks=2,
                optimization_timeout=5.0
            )
            
            # Simple task for benchmarking
            task = SDLCTask(
                task_id="bench_task",
                phase=SDLCPhase.ANALYSIS,
                description="Benchmark task",
                priority=5.0,
                estimated_effort=1.0
            )
            
            async def optimize_task():
                return await optimizer.optimize_sdlc_tasks([task])
            
            return await self._run_latency_benchmark(
                name="quantum_optimizer_latency",
                func=optimize_task,
                iterations=10,
                is_async=True
            )
            
        except ImportError:
            return BenchmarkResult(
                benchmark_name="quantum_optimizer_latency",
                benchmark_type=BenchmarkType.LATENCY,
                duration=0.0,
                error_count=1,
                success_rate=0.0,
                metadata={"error": "quantum optimizer not available"}
            )
    
    async def _run_throughput_benchmarks(self) -> List[BenchmarkResult]:
        """Run throughput benchmarks"""
        
        results = []
        
        # Test 1: Sequential processing throughput
        result = await self._benchmark_sequential_throughput()
        results.append(result)
        
        # Test 2: Concurrent processing throughput
        result = await self._benchmark_concurrent_throughput()
        results.append(result)
        
        # Test 3: Quantum system throughput
        result = await self._benchmark_quantum_system_throughput()
        results.append(result)
        
        return results
    
    async def _benchmark_sequential_throughput(self) -> BenchmarkResult:
        """Benchmark sequential processing throughput"""
        
        def process_item(item: int) -> int:
            # Simulate some work
            result = 0
            for i in range(100):
                result += item * i
            return result
        
        start_time = time.time()
        processed = 0
        test_duration = 5.0  # seconds
        
        async with self.profiler.profile():
            end_time = start_time + test_duration
            while time.time() < end_time:
                process_item(processed)
                processed += 1
                
                # Yield control occasionally
                if processed % 100 == 0:
                    await asyncio.sleep(0)
        
        duration = time.time() - start_time
        throughput = processed / duration
        
        profile_stats = self.profiler.get_statistics()
        
        return BenchmarkResult(
            benchmark_name="sequential_throughput",
            benchmark_type=BenchmarkType.THROUGHPUT,
            duration=duration,
            throughput=throughput,
            memory_used=profile_stats.get('peak_usage', {}).get('memory_mb', 0),
            cpu_percent=profile_stats.get('peak_usage', {}).get('cpu_percent', 0),
            metadata={
                'items_processed': processed,
                'profile_stats': profile_stats
            }
        )
    
    async def _benchmark_concurrent_throughput(self) -> BenchmarkResult:
        """Benchmark concurrent processing throughput"""
        
        async def process_item_async(item: int) -> int:
            # Simulate async work
            await asyncio.sleep(0.001)  # 1ms simulated I/O
            result = 0
            for i in range(50):  # Less CPU work since we have concurrency
                result += item * i
            return result
        
        start_time = time.time()
        test_duration = 5.0  # seconds
        concurrency_limit = 20
        
        async with self.profiler.profile():
            semaphore = asyncio.Semaphore(concurrency_limit)
            tasks = []
            processed = 0
            
            async def bounded_process(item: int) -> int:
                async with semaphore:
                    return await process_item_async(item)
            
            end_time = start_time + test_duration
            task_counter = 0
            
            while time.time() < end_time:
                # Start new tasks
                for _ in range(min(10, concurrency_limit - len(tasks))):
                    if time.time() >= end_time:
                        break
                    task = asyncio.create_task(bounded_process(task_counter))
                    tasks.append(task)
                    task_counter += 1
                
                # Check completed tasks
                done_tasks = [t for t in tasks if t.done()]
                processed += len(done_tasks)
                tasks = [t for t in tasks if not t.done()]
                
                await asyncio.sleep(0.01)  # Brief pause
            
            # Wait for remaining tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                processed += len(tasks)
        
        duration = time.time() - start_time
        throughput = processed / duration
        
        profile_stats = self.profiler.get_statistics()
        
        return BenchmarkResult(
            benchmark_name="concurrent_throughput",
            benchmark_type=BenchmarkType.THROUGHPUT,
            duration=duration,
            throughput=throughput,
            memory_used=profile_stats.get('peak_usage', {}).get('memory_mb', 0),
            cpu_percent=profile_stats.get('peak_usage', {}).get('cpu_percent', 0),
            metadata={
                'items_processed': processed,
                'concurrency_limit': concurrency_limit,
                'profile_stats': profile_stats
            }
        )
    
    async def _benchmark_quantum_system_throughput(self) -> BenchmarkResult:
        """Benchmark quantum system throughput"""
        
        try:
            from sql_synthesizer.quantum.monitoring import QuantumMonitoringSystem, MetricType
            
            monitoring = QuantumMonitoringSystem()
            
            start_time = time.time()
            test_duration = 3.0  # seconds
            
            async with self.profiler.profile():
                processed = 0
                end_time = start_time + test_duration
                
                while time.time() < end_time:
                    # Record metrics to simulate load
                    monitoring.record_metric("test_metric", processed * 1.5)
                    monitoring.record_metric("cpu_usage", 50.0 + (processed % 20))
                    monitoring.record_metric("memory_usage", 100.0 + (processed % 50))
                    
                    processed += 1
                    
                    # Yield control occasionally
                    if processed % 50 == 0:
                        await asyncio.sleep(0.001)
            
            duration = time.time() - start_time
            throughput = processed / duration
            
            profile_stats = self.profiler.get_statistics()
            
            return BenchmarkResult(
                benchmark_name="quantum_system_throughput",
                benchmark_type=BenchmarkType.THROUGHPUT,
                duration=duration,
                throughput=throughput,
                memory_used=profile_stats.get('peak_usage', {}).get('memory_mb', 0),
                cpu_percent=profile_stats.get('peak_usage', {}).get('cpu_percent', 0),
                metadata={
                    'metrics_recorded': processed * 3,  # 3 metrics per iteration
                    'profile_stats': profile_stats
                }
            )
            
        except ImportError:
            return BenchmarkResult(
                benchmark_name="quantum_system_throughput",
                benchmark_type=BenchmarkType.THROUGHPUT,
                duration=0.0,
                error_count=1,
                success_rate=0.0,
                metadata={"error": "quantum monitoring not available"}
            )
    
    async def _run_memory_benchmarks(self) -> List[BenchmarkResult]:
        """Run memory benchmarks"""
        
        results = []
        
        # Test 1: Memory allocation benchmark
        result = await self._benchmark_memory_allocation()
        results.append(result)
        
        # Test 2: Memory leak detection
        result = await self._benchmark_memory_leak_detection()
        results.append(result)
        
        return results
    
    async def _benchmark_memory_allocation(self) -> BenchmarkResult:
        """Benchmark memory allocation patterns"""
        
        tracemalloc.start()
        
        start_time = time.time()
        
        # Allocate and deallocate memory in patterns
        data_structures = []
        
        try:
            for i in range(1000):
                # Create various data structures
                large_list = list(range(1000))
                large_dict = {j: j*2 for j in range(500)}
                large_str = "x" * 1000
                
                data_structures.append((large_list, large_dict, large_str))
                
                # Periodically clear old data
                if i % 100 == 0 and data_structures:
                    data_structures = data_structures[-50:]  # Keep only recent
                    gc.collect()  # Force garbage collection
                
                if i % 50 == 0:
                    await asyncio.sleep(0)  # Yield control
            
            # Final cleanup
            data_structures.clear()
            gc.collect()
            
            duration = time.time() - start_time
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return BenchmarkResult(
                benchmark_name="memory_allocation",
                benchmark_type=BenchmarkType.MEMORY,
                duration=duration,
                memory_used=peak / 1024 / 1024,  # Convert to MB
                success_rate=1.0,
                metadata={
                    'current_memory_mb': current / 1024 / 1024,
                    'peak_memory_mb': peak / 1024 / 1024,
                    'allocations_performed': 1000
                }
            )
            
        except Exception as e:
            tracemalloc.stop()
            return BenchmarkResult(
                benchmark_name="memory_allocation",
                benchmark_type=BenchmarkType.MEMORY,
                duration=time.time() - start_time,
                error_count=1,
                success_rate=0.0,
                metadata={"error": str(e)}
            )
    
    async def _benchmark_memory_leak_detection(self) -> BenchmarkResult:
        """Benchmark memory leak detection"""
        
        tracemalloc.start()
        initial_memory = psutil.Process().memory_info().rss
        
        start_time = time.time()
        
        # Simulate potential memory leaks
        cached_data = {}
        
        try:
            for i in range(500):
                # Add data to cache (potential leak source)
                key = f"cache_key_{i}"
                cached_data[key] = {
                    'data': list(range(100)),
                    'metadata': {'created': time.time(), 'id': i}
                }
                
                # Simulate some cache cleanup (not perfect, may leak)
                if i % 50 == 0 and len(cached_data) > 100:
                    # Remove oldest 20% of entries
                    keys_to_remove = list(cached_data.keys())[:len(cached_data) // 5]
                    for key in keys_to_remove:
                        del cached_data[key]
                
                if i % 25 == 0:
                    await asyncio.sleep(0)  # Yield control
            
            duration = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss
            memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
            
            # Get tracemalloc statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Check if memory growth is concerning
            success_rate = 1.0 if memory_growth < 50 else 0.5  # Flag if >50MB growth
            
            return BenchmarkResult(
                benchmark_name="memory_leak_detection",
                benchmark_type=BenchmarkType.MEMORY,
                duration=duration,
                memory_used=peak / 1024 / 1024,
                success_rate=success_rate,
                metadata={
                    'memory_growth_mb': memory_growth,
                    'cache_entries_final': len(cached_data),
                    'peak_traced_memory_mb': peak / 1024 / 1024,
                    'potential_leak': memory_growth > 50
                }
            )
            
        except Exception as e:
            tracemalloc.stop()
            return BenchmarkResult(
                benchmark_name="memory_leak_detection",
                benchmark_type=BenchmarkType.MEMORY,
                duration=time.time() - start_time,
                error_count=1,
                success_rate=0.0,
                metadata={"error": str(e)}
            )
    
    async def _run_concurrency_benchmarks(self) -> List[BenchmarkResult]:
        """Run concurrency benchmarks"""
        
        results = []
        
        # Test different concurrency levels
        for concurrency in [1, 5, 10, 20, 50]:
            result = await self._benchmark_concurrency_level(concurrency)
            results.append(result)
        
        return results
    
    async def _benchmark_concurrency_level(self, concurrency: int) -> BenchmarkResult:
        """Benchmark specific concurrency level"""
        
        async def worker_task(worker_id: int, duration: float) -> Dict[str, Any]:
            start = time.time()
            operations = 0
            
            while time.time() - start < duration:
                # Simulate work
                await asyncio.sleep(0.001)  # 1ms async work
                
                # Some CPU work
                result = sum(i * i for i in range(100))
                operations += 1
            
            return {
                'worker_id': worker_id,
                'operations': operations,
                'duration': time.time() - start
            }
        
        start_time = time.time()
        test_duration = 2.0  # seconds per concurrency test
        
        async with self.profiler.profile():
            # Start workers
            tasks = [
                asyncio.create_task(worker_task(i, test_duration))
                for i in range(concurrency)
            ]
            
            # Wait for completion
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        
        # Process results
        successful_results = [r for r in results if isinstance(r, dict)]
        total_operations = sum(r['operations'] for r in successful_results)
        throughput = total_operations / duration
        
        profile_stats = self.profiler.get_statistics()
        
        return BenchmarkResult(
            benchmark_name=f"concurrency_level_{concurrency}",
            benchmark_type=BenchmarkType.CONCURRENCY,
            duration=duration,
            throughput=throughput,
            memory_used=profile_stats.get('peak_usage', {}).get('memory_mb', 0),
            cpu_percent=profile_stats.get('peak_usage', {}).get('cpu_percent', 0),
            success_rate=len(successful_results) / concurrency,
            error_count=len(results) - len(successful_results),
            metadata={
                'concurrency_level': concurrency,
                'total_operations': total_operations,
                'profile_stats': profile_stats
            }
        )
    
    async def _run_scalability_benchmarks(self) -> List[BenchmarkResult]:
        """Run scalability benchmarks"""
        
        results = []
        
        # Test scalability with increasing load
        for load_factor in [1, 2, 4, 8]:
            result = await self._benchmark_scalability_load(load_factor)
            results.append(result)
        
        return results
    
    async def _benchmark_scalability_load(self, load_factor: int) -> BenchmarkResult:
        """Benchmark scalability at specific load level"""
        
        start_time = time.time()
        test_duration = 3.0
        
        # Scale up the work based on load factor
        work_items = 100 * load_factor
        concurrency = min(10 * load_factor, 50)  # Cap at 50
        
        async def process_work_batch(batch_items: List[int]) -> int:
            processed = 0
            for item in batch_items:
                # Simulate processing
                await asyncio.sleep(0.001 * load_factor)  # Scaling delay
                result = sum(i * item for i in range(50))
                processed += 1
            return processed
        
        async with self.profiler.profile():
            # Divide work into batches
            batch_size = max(1, work_items // concurrency)
            batches = [
                list(range(i, min(i + batch_size, work_items)))
                for i in range(0, work_items, batch_size)
            ]
            
            # Process batches concurrently
            tasks = [
                asyncio.create_task(process_work_batch(batch))
                for batch in batches[:concurrency]  # Limit concurrent batches
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        
        # Process results
        successful_results = [r for r in results if isinstance(r, int)]
        total_processed = sum(successful_results)
        throughput = total_processed / duration
        
        profile_stats = self.profiler.get_statistics()
        
        # Calculate scalability efficiency
        baseline_throughput = 50  # Assumed baseline at load_factor=1
        expected_throughput = baseline_throughput * load_factor
        efficiency = throughput / expected_throughput if expected_throughput > 0 else 0
        
        return BenchmarkResult(
            benchmark_name=f"scalability_load_{load_factor}x",
            benchmark_type=BenchmarkType.SCALABILITY,
            duration=duration,
            throughput=throughput,
            memory_used=profile_stats.get('peak_usage', {}).get('memory_mb', 0),
            cpu_percent=profile_stats.get('peak_usage', {}).get('cpu_percent', 0),
            success_rate=len(successful_results) / len(batches),
            metadata={
                'load_factor': load_factor,
                'work_items': work_items,
                'concurrency': concurrency,
                'scalability_efficiency': efficiency,
                'profile_stats': profile_stats
            }
        )
    
    async def _run_stress_benchmarks(self) -> List[BenchmarkResult]:
        """Run stress benchmarks"""
        
        results = []
        
        # CPU stress test
        result = await self._benchmark_cpu_stress()
        results.append(result)
        
        # Memory stress test
        result = await self._benchmark_memory_stress()
        results.append(result)
        
        # Combined stress test
        result = await self._benchmark_combined_stress()
        results.append(result)
        
        return results
    
    async def _benchmark_cpu_stress(self) -> BenchmarkResult:
        """CPU stress benchmark"""
        
        start_time = time.time()
        stress_duration = 5.0  # seconds
        
        async with self.profiler.profile():
            def cpu_intensive_work(n: int) -> int:
                # CPU-intensive calculation
                result = 0
                for i in range(n):
                    result += i * i
                    result = result % 1000000  # Prevent overflow
                return result
            
            operations = 0
            end_time = start_time + stress_duration
            
            while time.time() < end_time:
                # Perform CPU-intensive work
                cpu_intensive_work(10000)
                operations += 1
                
                # Yield control occasionally
                if operations % 10 == 0:
                    await asyncio.sleep(0)
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        profile_stats = self.profiler.get_statistics()
        
        return BenchmarkResult(
            benchmark_name="cpu_stress",
            benchmark_type=BenchmarkType.STRESS,
            duration=duration,
            throughput=throughput,
            memory_used=profile_stats.get('peak_usage', {}).get('memory_mb', 0),
            cpu_percent=profile_stats.get('peak_usage', {}).get('cpu_percent', 0),
            metadata={
                'operations': operations,
                'profile_stats': profile_stats
            }
        )
    
    async def _benchmark_memory_stress(self) -> BenchmarkResult:
        """Memory stress benchmark"""
        
        start_time = time.time()
        stress_duration = 5.0
        
        async with self.profiler.profile():
            memory_hogs = []
            allocations = 0
            
            end_time = start_time + stress_duration
            
            while time.time() < end_time:
                # Allocate large chunks of memory
                chunk = [i for i in range(10000)]  # ~40KB per chunk
                memory_hogs.append(chunk)
                allocations += 1
                
                # Occasionally free some memory
                if len(memory_hogs) > 100:
                    memory_hogs = memory_hogs[-50:]  # Keep only recent allocations
                    gc.collect()
                
                if allocations % 10 == 0:
                    await asyncio.sleep(0)
        
        duration = time.time() - start_time
        
        # Clean up
        memory_hogs.clear()
        gc.collect()
        
        profile_stats = self.profiler.get_statistics()
        
        return BenchmarkResult(
            benchmark_name="memory_stress",
            benchmark_type=BenchmarkType.STRESS,
            duration=duration,
            memory_used=profile_stats.get('peak_usage', {}).get('memory_mb', 0),
            cpu_percent=profile_stats.get('peak_usage', {}).get('cpu_percent', 0),
            metadata={
                'allocations': allocations,
                'peak_memory_chunks': len(memory_hogs),
                'profile_stats': profile_stats
            }
        )
    
    async def _benchmark_combined_stress(self) -> BenchmarkResult:
        """Combined CPU and memory stress benchmark"""
        
        start_time = time.time()
        stress_duration = 5.0
        
        async with self.profiler.profile():
            memory_hogs = []
            operations = 0
            
            end_time = start_time + stress_duration
            
            while time.time() < end_time:
                # CPU work
                cpu_result = sum(i * i for i in range(1000))
                
                # Memory work
                memory_chunk = [cpu_result + i for i in range(1000)]
                memory_hogs.append(memory_chunk)
                
                operations += 1
                
                # Memory management
                if len(memory_hogs) > 50:
                    memory_hogs = memory_hogs[-25:]
                    gc.collect()
                
                if operations % 5 == 0:
                    await asyncio.sleep(0)
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        # Cleanup
        memory_hogs.clear()
        gc.collect()
        
        profile_stats = self.profiler.get_statistics()
        
        return BenchmarkResult(
            benchmark_name="combined_stress",
            benchmark_type=BenchmarkType.STRESS,
            duration=duration,
            throughput=throughput,
            memory_used=profile_stats.get('peak_usage', {}).get('memory_mb', 0),
            cpu_percent=profile_stats.get('peak_usage', {}).get('cpu_percent', 0),
            metadata={
                'operations': operations,
                'profile_stats': profile_stats
            }
        )
    
    async def _run_latency_benchmark(self, name: str, func: Callable,
                                   iterations: int = 1000, 
                                   is_async: bool = False) -> BenchmarkResult:
        """Run a latency benchmark"""
        
        # Warm up
        for _ in range(min(self.warm_up_iterations, iterations // 10)):
            try:
                if is_async:
                    await func()
                else:
                    func()
            except:
                pass  # Ignore warm-up errors
        
        # Benchmark
        latencies = []
        errors = 0
        
        for _ in range(iterations):
            start = time.time()
            try:
                if is_async:
                    await func()
                else:
                    func()
                latencies.append((time.time() - start) * 1000)  # Convert to ms
            except Exception:
                errors += 1
        
        if latencies:
            avg_latency_ms = statistics.mean(latencies)
            throughput = 1000 / avg_latency_ms  # ops/second
            success_rate = (iterations - errors) / iterations
        else:
            avg_latency_ms = 0
            throughput = 0
            success_rate = 0
        
        return BenchmarkResult(
            benchmark_name=name,
            benchmark_type=BenchmarkType.LATENCY,
            duration=avg_latency_ms / 1000,  # Convert back to seconds
            throughput=throughput,
            success_rate=success_rate,
            error_count=errors,
            metadata={
                'avg_latency_ms': avg_latency_ms,
                'min_latency_ms': min(latencies) if latencies else 0,
                'max_latency_ms': max(latencies) if latencies else 0,
                'p95_latency_ms': (
                    statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0
                ),
                'iterations': iterations
            }
        )
    
    def _generate_performance_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate performance summary"""
        
        summary = {
            'total_benchmarks': len(results),
            'by_type': {},
            'overall_performance': PerformanceLevel.GOOD,
            'performance_scores': {},
            'key_metrics': {}
        }
        
        # Group by benchmark type
        by_type = {}
        for result in results:
            bench_type = result.benchmark_type.value
            if bench_type not in by_type:
                by_type[bench_type] = []
            by_type[bench_type].append(result)
        
        summary['by_type'] = by_type
        
        # Calculate performance scores
        scores = {}
        
        # Latency score (lower is better)
        latency_results = by_type.get('latency', [])
        if latency_results:
            avg_latency = statistics.mean(r.duration * 1000 for r in latency_results)  # ms
            if avg_latency < self.thresholds['max_response_time_ms'] / 10:
                scores['latency'] = 'excellent'
            elif avg_latency < self.thresholds['max_response_time_ms'] / 5:
                scores['latency'] = 'good'
            elif avg_latency < self.thresholds['max_response_time_ms']:
                scores['latency'] = 'acceptable'
            else:
                scores['latency'] = 'poor'
        
        # Throughput score (higher is better)
        throughput_results = by_type.get('throughput', [])
        if throughput_results:
            avg_throughput = statistics.mean(r.throughput for r in throughput_results if r.throughput)
            if avg_throughput > self.thresholds['min_throughput_ops_sec'] * 10:
                scores['throughput'] = 'excellent'
            elif avg_throughput > self.thresholds['min_throughput_ops_sec'] * 5:
                scores['throughput'] = 'good'
            elif avg_throughput > self.thresholds['min_throughput_ops_sec']:
                scores['throughput'] = 'acceptable'
            else:
                scores['throughput'] = 'poor'
        
        # Memory score (lower is better)
        memory_results = by_type.get('memory', [])
        if memory_results:
            max_memory = max(r.memory_used for r in memory_results if r.memory_used)
            if max_memory < self.thresholds['max_memory_mb'] / 4:
                scores['memory'] = 'excellent'
            elif max_memory < self.thresholds['max_memory_mb'] / 2:
                scores['memory'] = 'good'
            elif max_memory < self.thresholds['max_memory_mb']:
                scores['memory'] = 'acceptable'
            else:
                scores['memory'] = 'poor'
        
        # Success rate score
        success_rates = [r.success_rate for r in results if r.success_rate is not None]
        if success_rates:
            avg_success_rate = statistics.mean(success_rates)
            if avg_success_rate >= 0.99:
                scores['reliability'] = 'excellent'
            elif avg_success_rate >= self.thresholds['min_success_rate']:
                scores['reliability'] = 'good'
            elif avg_success_rate >= 0.9:
                scores['reliability'] = 'acceptable'
            else:
                scores['reliability'] = 'poor'
        
        summary['performance_scores'] = scores
        
        # Overall performance level
        score_values = {'excellent': 4, 'good': 3, 'acceptable': 2, 'poor': 1, 'critical': 0}
        if scores:
            avg_score = statistics.mean(score_values.get(s, 1) for s in scores.values())
            if avg_score >= 3.5:
                summary['overall_performance'] = PerformanceLevel.EXCELLENT
            elif avg_score >= 2.5:
                summary['overall_performance'] = PerformanceLevel.GOOD
            elif avg_score >= 1.5:
                summary['overall_performance'] = PerformanceLevel.ACCEPTABLE
            else:
                summary['overall_performance'] = PerformanceLevel.POOR
        
        # Key metrics
        summary['key_metrics'] = {
            'fastest_operation_ms': min(r.duration * 1000 for r in results if r.duration > 0),
            'highest_throughput_ops_sec': max(r.throughput for r in results if r.throughput),
            'peak_memory_usage_mb': max(r.memory_used for r in results if r.memory_used),
            'overall_success_rate': statistics.mean(success_rates) if success_rates else 0
        }
        
        return summary
    
    def _generate_recommendations(self, results: List[BenchmarkResult], 
                                summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        
        recommendations = []
        scores = summary.get('performance_scores', {})
        
        # Latency recommendations
        if scores.get('latency') in ['poor', 'critical']:
            recommendations.append(
                "⚡ LATENCY: High latency detected. Consider optimizing hot paths, "
                "reducing I/O operations, and implementing caching."
            )
        elif scores.get('latency') == 'acceptable':
            recommendations.append(
                "⏱️ LATENCY: Moderate latency. Review critical code paths for optimization opportunities."
            )
        
        # Throughput recommendations
        if scores.get('throughput') in ['poor', 'critical']:
            recommendations.append(
                "📊 THROUGHPUT: Low throughput detected. Consider increasing concurrency, "
                "optimizing algorithms, and reducing blocking operations."
            )
        
        # Memory recommendations
        if scores.get('memory') in ['poor', 'critical']:
            recommendations.append(
                "💾 MEMORY: High memory usage detected. Review for memory leaks, "
                "optimize data structures, and implement memory pooling."
            )
        
        # Reliability recommendations
        if scores.get('reliability') in ['poor', 'critical']:
            recommendations.append(
                "🛡️ RELIABILITY: Low success rate detected. Improve error handling, "
                "add retries, and review failure scenarios."
            )
        
        # Concurrency analysis
        concurrency_results = [r for r in results if 'concurrency' in r.benchmark_name]
        if concurrency_results:
            # Analyze concurrency scaling
            throughputs = [(r.metadata.get('concurrency_level', 1), r.throughput or 0) 
                          for r in concurrency_results if r.throughput]
            
            if len(throughputs) > 2:
                # Check if throughput scales well with concurrency
                max_throughput = max(t[1] for t in throughputs)
                single_thread_throughput = next((t[1] for t in throughputs if t[0] == 1), 0)
                
                if single_thread_throughput > 0:
                    scaling_factor = max_throughput / single_thread_throughput
                    optimal_concurrency = max(throughputs, key=lambda x: x[1])[0]
                    
                    if scaling_factor < 2:
                        recommendations.append(
                            "🔄 CONCURRENCY: Poor scaling detected. Review for locks, "
                            "shared resources, and GIL limitations."
                        )
                    
                    recommendations.append(
                        f"🎯 CONCURRENCY: Optimal concurrency level appears to be around "
                        f"{optimal_concurrency} for this workload."
                    )
        
        # Scalability analysis
        scalability_results = [r for r in results if 'scalability' in r.benchmark_name]
        if scalability_results:
            efficiencies = [r.metadata.get('scalability_efficiency', 0) for r in scalability_results]
            avg_efficiency = statistics.mean(efficiencies) if efficiencies else 0
            
            if avg_efficiency < 0.5:
                recommendations.append(
                    "📈 SCALABILITY: Poor scalability efficiency. Consider async patterns, "
                    "connection pooling, and resource optimization."
                )
            elif avg_efficiency < 0.8:
                recommendations.append(
                    "📈 SCALABILITY: Moderate scalability. Review bottlenecks at higher loads."
                )
        
        # System-specific recommendations
        stress_results = [r for r in results if r.benchmark_type == BenchmarkType.STRESS]
        if stress_results:
            max_cpu = max(r.cpu_percent for r in stress_results if r.cpu_percent)
            max_memory = max(r.memory_used for r in stress_results if r.memory_used)
            
            if max_cpu > self.thresholds['max_cpu_percent']:
                recommendations.append(
                    f"🔥 CPU STRESS: High CPU usage ({max_cpu:.1f}%) under stress. "
                    "Consider CPU optimization and load distribution."
                )
            
            if max_memory > self.thresholds['max_memory_mb']:
                recommendations.append(
                    f"💾 MEMORY STRESS: High memory usage ({max_memory:.1f}MB) under stress. "
                    "Implement memory management strategies."
                )
        
        # General recommendations based on overall performance
        overall = summary.get('overall_performance')
        if overall == PerformanceLevel.POOR:
            recommendations.extend([
                "🚨 OVERALL: Performance is below acceptable levels. Prioritize optimization efforts.",
                "📋 Consider comprehensive performance profiling and code review.",
                "⚡ Implement performance monitoring in production."
            ])
        elif overall == PerformanceLevel.ACCEPTABLE:
            recommendations.extend([
                "✅ OVERALL: Performance is acceptable but has room for improvement.",
                "🔍 Consider targeted optimizations for high-impact areas."
            ])
        
        # Add general best practices
        recommendations.extend([
            "📊 Implement continuous performance monitoring.",
            "🎯 Set up performance regression testing in CI/CD.",
            "📈 Consider load testing with realistic workloads.",
            "🔧 Profile code regularly to identify optimization opportunities."
        ])
        
        return recommendations[:15]  # Limit to top 15 recommendations
    
    def export_report(self, report: PerformanceReport, output_path: Path) -> Path:
        """Export performance report to JSON"""
        
        report_dict = {
            'benchmark_timestamp': report.benchmark_timestamp,
            'total_duration': report.total_duration,
            'system_info': report.system_info,
            'benchmark_results': [
                {
                    'benchmark_name': r.benchmark_name,
                    'benchmark_type': r.benchmark_type.value,
                    'duration': r.duration,
                    'throughput': r.throughput,
                    'memory_used': r.memory_used,
                    'cpu_percent': r.cpu_percent,
                    'success_rate': r.success_rate,
                    'error_count': r.error_count,
                    'metadata': r.metadata
                }
                for r in report.benchmark_results
            ],
            'performance_summary': report.performance_summary,
            'recommendations': report.recommendations,
            'metadata': {
                'benchmark_version': '1.0.0',
                'export_time': time.time()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"📊 Performance report exported to: {output_path}")
        return output_path
    
    def print_summary(self, report: PerformanceReport):
        """Print performance report summary"""
        
        print("\n" + "="*80)
        print("⚡ PERFORMANCE BENCHMARK REPORT")
        print("="*80)
        
        print(f"Total Duration: {report.total_duration:.2f}s")
        print(f"Benchmarks Run: {len(report.benchmark_results)}")
        print(f"Overall Performance: {report.performance_summary['overall_performance'].value.upper()}")
        
        print("\n🏆 PERFORMANCE SCORES:")
        scores = report.performance_summary.get('performance_scores', {})
        score_icons = {'excellent': '🟢', 'good': '🟡', 'acceptable': '🟠', 'poor': '🔴'}
        
        for category, score in scores.items():
            icon = score_icons.get(score, '⚪')
            print(f"  {icon} {category.title()}: {score.upper()}")
        
        print("\n📊 KEY METRICS:")
        metrics = report.performance_summary.get('key_metrics', {})
        for metric, value in metrics.items():
            if 'ms' in metric:
                print(f"  • {metric.replace('_', ' ').title()}: {value:.2f}")
            elif 'ops_sec' in metric:
                print(f"  • {metric.replace('_', ' ').title()}: {value:.1f}")
            elif 'mb' in metric:
                print(f"  • {metric.replace('_', ' ').title()}: {value:.1f}")
            else:
                print(f"  • {metric.replace('_', ' ').title()}: {value:.3f}")
        
        print("\n🔍 BENCHMARK RESULTS BY TYPE:")
        by_type = report.performance_summary.get('by_type', {})
        for bench_type, results in by_type.items():
            print(f"  📈 {bench_type.upper()}: {len(results)} tests")
            
            # Show best performing test in category
            if results:
                if bench_type == 'latency':
                    best = min(results, key=lambda r: r.duration)
                    print(f"     Best: {best.benchmark_name} ({best.duration*1000:.2f}ms)")
                elif bench_type == 'throughput':
                    best = max(results, key=lambda r: r.throughput or 0)
                    print(f"     Best: {best.benchmark_name} ({best.throughput:.1f} ops/sec)")
                elif bench_type == 'concurrency':
                    best = max(results, key=lambda r: r.throughput or 0)
                    concurrency = best.metadata.get('concurrency_level', 'N/A')
                    print(f"     Best: {best.benchmark_name} (concurrency: {concurrency})")
        
        if report.recommendations:
            print("\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:8], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


async def main():
    """Main performance benchmark function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("PerformanceBenchmark")
    
    # Initialize benchmark suite
    benchmark = QuantumPerformanceBenchmark(logger)
    
    # Run comprehensive benchmark
    report = await benchmark.run_comprehensive_benchmark()
    
    # Print summary
    benchmark.print_summary(report)
    
    # Export detailed report
    report_path = Path("performance_benchmark_report.json")
    benchmark.export_report(report, report_path)
    
    return report


if __name__ == "__main__":
    asyncio.run(main())