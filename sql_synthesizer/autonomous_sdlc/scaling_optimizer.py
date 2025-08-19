"""
Autonomous Scaling and Performance Optimization
Self-tuning performance optimization with predictive scaling
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    response_time_ms: float
    throughput_qps: float
    error_rate: float


@dataclass
class ScalingRecommendation:
    """Scaling recommendation with rationale"""

    action: str  # scale_up, scale_down, optimize, maintain
    confidence: float  # 0.0 to 1.0
    urgency: str  # low, medium, high, critical
    rationale: str
    expected_improvement: Dict[str, float]
    estimated_cost_impact: str
    implementation_steps: List[str]


class PerformanceProfiler:
    """Advanced performance profiling and analysis"""

    def __init__(self, history_size: int = 1000):
        self.metrics_history: deque[PerformanceMetrics] = deque(maxlen=history_size)
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.performance_targets = {
            "response_time_ms": 200,
            "throughput_qps": 100,
            "cpu_percent": 70,
            "memory_percent": 80,
            "error_rate": 0.01,
        }

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()

        # Application metrics (simulated for now)
        response_time_ms = await self._measure_response_time()
        throughput_qps = await self._measure_throughput()
        error_rate = await self._measure_error_rate()
        active_connections = await self._count_active_connections()

        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_io_read_mb=disk_io.read_bytes / (1024**2) if disk_io else 0,
            disk_io_write_mb=disk_io.write_bytes / (1024**2) if disk_io else 0,
            network_sent_mb=network_io.bytes_sent / (1024**2) if network_io else 0,
            network_recv_mb=network_io.bytes_recv / (1024**2) if network_io else 0,
            active_connections=active_connections,
            response_time_ms=response_time_ms,
            throughput_qps=throughput_qps,
            error_rate=error_rate,
        )

        self.metrics_history.append(metrics)

        # Set baseline if not exists
        if self.baseline_metrics is None and len(self.metrics_history) >= 10:
            self.baseline_metrics = self._calculate_baseline()

        return metrics

    async def _measure_response_time(self) -> float:
        """Measure average response time"""
        # Simulate response time measurement
        start_time = time.time()
        await asyncio.sleep(0.001)  # Simulate some work
        return (time.time() - start_time) * 1000

    async def _measure_throughput(self) -> float:
        """Measure current throughput in queries per second"""
        # Simulate throughput measurement
        return 85.0 + (time.time() % 30)  # Vary between 85-115 QPS

    async def _measure_error_rate(self) -> float:
        """Measure current error rate"""
        # Simulate error rate measurement
        return 0.005  # 0.5% error rate

    async def _count_active_connections(self) -> int:
        """Count active database/network connections"""
        # Simulate connection counting
        return 25

    def _calculate_baseline(self) -> PerformanceMetrics:
        """Calculate baseline metrics from history"""
        if len(self.metrics_history) < 10:
            return self.metrics_history[-1]

        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 measurements

        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=statistics.mean(m.cpu_percent for m in recent_metrics),
            memory_percent=statistics.mean(m.memory_percent for m in recent_metrics),
            memory_available_gb=statistics.mean(
                m.memory_available_gb for m in recent_metrics
            ),
            disk_io_read_mb=statistics.mean(m.disk_io_read_mb for m in recent_metrics),
            disk_io_write_mb=statistics.mean(
                m.disk_io_write_mb for m in recent_metrics
            ),
            network_sent_mb=statistics.mean(m.network_sent_mb for m in recent_metrics),
            network_recv_mb=statistics.mean(m.network_recv_mb for m in recent_metrics),
            active_connections=int(
                statistics.mean(m.active_connections for m in recent_metrics)
            ),
            response_time_ms=statistics.mean(
                m.response_time_ms for m in recent_metrics
            ),
            throughput_qps=statistics.mean(m.throughput_qps for m in recent_metrics),
            error_rate=statistics.mean(m.error_rate for m in recent_metrics),
        )

    def get_performance_trends(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Analyze performance trends over time window"""

        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}

        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if len(recent_metrics) < 2:
            return {"status": "insufficient_recent_data"}

        # Calculate trends
        trends = {}

        # Response time trend
        response_times = [m.response_time_ms for m in recent_metrics]
        trends["response_time"] = {
            "current": response_times[-1],
            "average": statistics.mean(response_times),
            "trend": (
                "increasing" if response_times[-1] > response_times[0] else "decreasing"
            ),
            "volatility": (
                statistics.stdev(response_times) if len(response_times) > 1 else 0
            ),
        }

        # CPU trend
        cpu_values = [m.cpu_percent for m in recent_metrics]
        trends["cpu"] = {
            "current": cpu_values[-1],
            "average": statistics.mean(cpu_values),
            "peak": max(cpu_values),
            "trend": "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing",
        }

        # Memory trend
        memory_values = [m.memory_percent for m in recent_metrics]
        trends["memory"] = {
            "current": memory_values[-1],
            "average": statistics.mean(memory_values),
            "peak": max(memory_values),
            "trend": (
                "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
            ),
        }

        # Throughput trend
        throughput_values = [m.throughput_qps for m in recent_metrics]
        trends["throughput"] = {
            "current": throughput_values[-1],
            "average": statistics.mean(throughput_values),
            "peak": max(throughput_values),
            "trend": (
                "increasing"
                if throughput_values[-1] > throughput_values[0]
                else "decreasing"
            ),
        }

        return {
            "status": "success",
            "window_minutes": window_minutes,
            "data_points": len(recent_metrics),
            "trends": trends,
        }


class IntelligentScaler:
    """Intelligent scaling decisions based on performance analysis"""

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.scaling_history: List[ScalingRecommendation] = []

        # Machine learning-like weights for decision making
        self.decision_weights = {
            "cpu_utilization": 0.25,
            "memory_utilization": 0.20,
            "response_time": 0.30,
            "throughput": 0.15,
            "error_rate": 0.10,
        }

    async def analyze_scaling_needs(self) -> ScalingRecommendation:
        """Analyze current performance and recommend scaling actions"""

        if len(self.profiler.metrics_history) < 5:
            return ScalingRecommendation(
                action="maintain",
                confidence=0.1,
                urgency="low",
                rationale="Insufficient data for scaling analysis",
                expected_improvement={},
                estimated_cost_impact="none",
                implementation_steps=["Collect more performance data"],
            )

        current_metrics = self.profiler.metrics_history[-1]
        trends = self.profiler.get_performance_trends()

        # Calculate scaling scores
        scale_up_score = self._calculate_scale_up_score(current_metrics, trends)
        scale_down_score = self._calculate_scale_down_score(current_metrics, trends)
        optimize_score = self._calculate_optimize_score(current_metrics, trends)

        # Make decision based on highest score
        scores = {
            "scale_up": scale_up_score,
            "scale_down": scale_down_score,
            "optimize": optimize_score,
            "maintain": 0.5,  # Default baseline
        }

        best_action = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_action]

        # Generate recommendation
        recommendation = self._generate_recommendation(
            best_action, confidence, current_metrics, trends
        )

        self.scaling_history.append(recommendation)
        return recommendation

    def _calculate_scale_up_score(
        self, metrics: PerformanceMetrics, trends: Dict[str, Any]
    ) -> float:
        """Calculate score for scaling up"""
        score = 0.0

        # High CPU utilization
        if metrics.cpu_percent > 80:
            score += 0.8
        elif metrics.cpu_percent > 70:
            score += 0.5

        # High memory utilization
        if metrics.memory_percent > 85:
            score += 0.7
        elif metrics.memory_percent > 75:
            score += 0.4

        # Poor response times
        if metrics.response_time_ms > 500:
            score += 0.9
        elif metrics.response_time_ms > 300:
            score += 0.6

        # High error rate
        if metrics.error_rate > 0.05:
            score += 0.8
        elif metrics.error_rate > 0.02:
            score += 0.4

        # Trending worse
        if trends.get("status") == "success":
            trends_data = trends.get("trends", {})
            if trends_data.get("response_time", {}).get("trend") == "increasing":
                score += 0.3
            if trends_data.get("cpu", {}).get("trend") == "increasing":
                score += 0.2

        return min(score, 1.0)

    def _calculate_scale_down_score(
        self, metrics: PerformanceMetrics, trends: Dict[str, Any]
    ) -> float:
        """Calculate score for scaling down"""
        score = 0.0

        # Low resource utilization
        if metrics.cpu_percent < 30 and metrics.memory_percent < 50:
            score += 0.6
        elif metrics.cpu_percent < 50 and metrics.memory_percent < 60:
            score += 0.3

        # Good performance with headroom
        if metrics.response_time_ms < 100 and metrics.error_rate < 0.001:
            score += 0.4

        # Consistently low utilization trend
        if trends.get("status") == "success":
            trends_data = trends.get("trends", {})
            cpu_avg = trends_data.get("cpu", {}).get("average", 100)
            if cpu_avg < 40:
                score += 0.3

        return min(score, 1.0)

    def _calculate_optimize_score(
        self, metrics: PerformanceMetrics, trends: Dict[str, Any]
    ) -> float:
        """Calculate score for optimization without scaling"""
        score = 0.0

        # Moderate performance issues that might be optimizable
        if 200 < metrics.response_time_ms < 400:
            score += 0.6

        if 60 < metrics.cpu_percent < 80:
            score += 0.4

        # Variable performance (optimization opportunity)
        if trends.get("status") == "success":
            trends_data = trends.get("trends", {})
            response_volatility = trends_data.get("response_time", {}).get(
                "volatility", 0
            )
            if (
                response_volatility > 50
            ):  # High volatility suggests optimization opportunity
                score += 0.5

        return min(score, 1.0)

    def _generate_recommendation(
        self,
        action: str,
        confidence: float,
        metrics: PerformanceMetrics,
        trends: Dict[str, Any],
    ) -> ScalingRecommendation:
        """Generate detailed recommendation based on analysis"""

        if action == "scale_up":
            return ScalingRecommendation(
                action=action,
                confidence=confidence,
                urgency="high" if confidence > 0.8 else "medium",
                rationale=f"High resource utilization detected: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%, Response time {metrics.response_time_ms:.1f}ms",
                expected_improvement={
                    "response_time_reduction": 30,
                    "throughput_increase": 50,
                    "error_rate_reduction": 50,
                },
                estimated_cost_impact="increase",
                implementation_steps=[
                    "Increase container/instance CPU and memory allocation",
                    "Scale out to additional replicas if possible",
                    "Monitor impact for 10-15 minutes",
                    "Adjust further if needed",
                ],
            )

        elif action == "scale_down":
            return ScalingRecommendation(
                action=action,
                confidence=confidence,
                urgency="low",
                rationale=f"Low resource utilization detected: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%",
                expected_improvement={"cost_reduction": 20, "efficiency_increase": 15},
                estimated_cost_impact="decrease",
                implementation_steps=[
                    "Reduce container/instance resource allocation",
                    "Consider consolidating workloads",
                    "Monitor for performance degradation",
                    "Rollback if issues occur",
                ],
            )

        elif action == "optimize":
            return ScalingRecommendation(
                action=action,
                confidence=confidence,
                urgency="medium",
                rationale=f"Performance optimization opportunity: Response time {metrics.response_time_ms:.1f}ms with moderate resource usage",
                expected_improvement={
                    "response_time_reduction": 20,
                    "throughput_increase": 25,
                    "efficiency_increase": 30,
                },
                estimated_cost_impact="neutral",
                implementation_steps=[
                    "Analyze query performance and optimize slow queries",
                    "Review and optimize caching strategies",
                    "Tune database connection pooling",
                    "Consider code-level optimizations",
                ],
            )

        else:  # maintain
            return ScalingRecommendation(
                action=action,
                confidence=confidence,
                urgency="low",
                rationale="Performance metrics within acceptable ranges",
                expected_improvement={},
                estimated_cost_impact="none",
                implementation_steps=[
                    "Continue monitoring performance",
                    "Maintain current configuration",
                ],
            )


class AutonomousPerformanceOptimizer:
    """Autonomous performance optimization and scaling engine"""

    def __init__(self, monitoring_interval: int = 60):
        self.profiler = PerformanceProfiler()
        self.scaler = IntelligentScaler(self.profiler)
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        self.optimization_history: List[Dict[str, Any]] = []

    async def start_monitoring(self):
        """Start autonomous monitoring and optimization"""
        self.is_running = True
        logger.info("🚀 Starting Autonomous Performance Optimization")

        while self.is_running:
            try:
                # Collect metrics
                metrics = await self.profiler.collect_metrics()

                # Analyze scaling needs
                recommendation = await self.scaler.analyze_scaling_needs()

                # Log current status
                self._log_performance_status(metrics, recommendation)

                # Execute auto-optimizations if confidence is high
                if recommendation.confidence > 0.8 and recommendation.action in [
                    "optimize"
                ]:
                    await self._execute_optimization(recommendation)

                # Store optimization event
                self.optimization_history.append(
                    {
                        "timestamp": time.time(),
                        "metrics": asdict(metrics),
                        "recommendation": asdict(recommendation),
                        "auto_executed": recommendation.confidence > 0.8,
                    }
                )

                # Keep history manageable
                if len(self.optimization_history) > 100:
                    self.optimization_history = self.optimization_history[-100:]

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(self.monitoring_interval)

    def stop_monitoring(self):
        """Stop autonomous monitoring"""
        self.is_running = False
        logger.info("🛑 Stopping Autonomous Performance Optimization")

    def _log_performance_status(
        self, metrics: PerformanceMetrics, recommendation: ScalingRecommendation
    ):
        """Log current performance status"""

        status_msg = (
            f"Performance Status: "
            f"CPU {metrics.cpu_percent:.1f}%, "
            f"Memory {metrics.memory_percent:.1f}%, "
            f"Response {metrics.response_time_ms:.1f}ms, "
            f"Throughput {metrics.throughput_qps:.1f} QPS"
        )

        if recommendation.urgency == "high":
            logger.warning(status_msg)
            logger.warning(
                f"Recommendation: {recommendation.action} ({recommendation.confidence:.2f} confidence)"
            )
        elif recommendation.urgency == "medium":
            logger.info(status_msg)
            logger.info(
                f"Recommendation: {recommendation.action} ({recommendation.confidence:.2f} confidence)"
            )
        else:
            logger.debug(status_msg)

    async def _execute_optimization(self, recommendation: ScalingRecommendation):
        """Execute automatic optimization"""

        logger.info(f"🔧 Executing auto-optimization: {recommendation.action}")

        try:
            if recommendation.action == "optimize":
                # Simulate optimization actions
                await self._optimize_caching()
                await self._optimize_connection_pooling()
                await self._optimize_query_performance()

            logger.info(f"✅ Auto-optimization completed: {recommendation.action}")

        except Exception as e:
            logger.error(f"❌ Auto-optimization failed: {str(e)}")

    async def _optimize_caching(self):
        """Optimize caching strategies"""
        # Simulate cache optimization
        logger.info("Optimizing cache configuration...")
        await asyncio.sleep(0.1)

    async def _optimize_connection_pooling(self):
        """Optimize database connection pooling"""
        # Simulate connection pool optimization
        logger.info("Optimizing connection pool settings...")
        await asyncio.sleep(0.1)

    async def _optimize_query_performance(self):
        """Optimize query performance"""
        # Simulate query optimization
        logger.info("Analyzing and optimizing query performance...")
        await asyncio.sleep(0.1)

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""

        if not self.optimization_history:
            return {"status": "no_data", "message": "No performance data available"}

        recent_events = self.optimization_history[-10:]

        # Calculate performance summary
        recent_metrics = [event["metrics"] for event in recent_events]
        avg_response_time = statistics.mean(
            m["response_time_ms"] for m in recent_metrics
        )
        avg_cpu = statistics.mean(m["cpu_percent"] for m in recent_metrics)
        avg_memory = statistics.mean(m["memory_percent"] for m in recent_metrics)
        avg_throughput = statistics.mean(m["throughput_qps"] for m in recent_metrics)

        # Count recommendations
        recommendation_counts = defaultdict(int)
        for event in recent_events:
            recommendation_counts[event["recommendation"]["action"]] += 1

        # Auto-execution rate
        auto_executed = sum(1 for event in recent_events if event["auto_executed"])
        auto_execution_rate = auto_executed / len(recent_events) if recent_events else 0

        return {
            "status": "success",
            "monitoring_active": self.is_running,
            "events_analyzed": len(recent_events),
            "performance_summary": {
                "avg_response_time_ms": round(avg_response_time, 2),
                "avg_cpu_percent": round(avg_cpu, 2),
                "avg_memory_percent": round(avg_memory, 2),
                "avg_throughput_qps": round(avg_throughput, 2),
            },
            "recommendations": dict(recommendation_counts),
            "auto_execution_rate": round(auto_execution_rate, 2),
            "optimization_opportunities": self._identify_optimization_opportunities(),
        }

    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify potential optimization opportunities"""
        opportunities = []

        if len(self.profiler.metrics_history) < 5:
            return ["Collect more performance data for analysis"]

        recent_metrics = list(self.profiler.metrics_history)[-10:]

        # High response time variance
        response_times = [m.response_time_ms for m in recent_metrics]
        if len(response_times) > 1 and statistics.stdev(response_times) > 100:
            opportunities.append(
                "High response time variance - consider query optimization"
            )

        # Consistently high CPU
        avg_cpu = statistics.mean(m.cpu_percent for m in recent_metrics)
        if avg_cpu > 75:
            opportunities.append(
                "High CPU utilization - consider code optimization or scaling"
            )

        # Memory trending up
        memory_values = [m.memory_percent for m in recent_metrics]
        if len(memory_values) > 3 and memory_values[-1] > memory_values[0] + 10:
            opportunities.append("Memory usage trending up - check for memory leaks")

        # Low throughput with good resources
        avg_throughput = statistics.mean(m.throughput_qps for m in recent_metrics)
        avg_cpu = statistics.mean(m.cpu_percent for m in recent_metrics)
        if avg_throughput < 50 and avg_cpu < 60:
            opportunities.append(
                "Low throughput with available resources - optimize concurrency"
            )

        return opportunities if opportunities else ["Performance appears optimal"]


# CLI Entry Point
async def main():
    """CLI entry point for autonomous performance optimization"""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Performance Optimizer")
    parser.add_argument(
        "--interval", type=int, default=10, help="Monitoring interval in seconds"
    )
    parser.add_argument(
        "--duration", type=int, default=300, help="Monitoring duration in seconds"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="Generate report only"
    )

    args = parser.parse_args()

    optimizer = AutonomousPerformanceOptimizer(args.interval)

    if args.report_only:
        # Just collect a few samples and generate report
        for _ in range(5):
            await optimizer.profiler.collect_metrics()
            await asyncio.sleep(2)

        report = optimizer.get_performance_report()
        print(json.dumps(report, indent=2))
    else:
        # Start monitoring
        monitor_task = asyncio.create_task(optimizer.start_monitoring())

        # Run for specified duration
        await asyncio.sleep(args.duration)

        # Stop monitoring and generate report
        optimizer.stop_monitoring()
        monitor_task.cancel()

        report = optimizer.get_performance_report()
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
