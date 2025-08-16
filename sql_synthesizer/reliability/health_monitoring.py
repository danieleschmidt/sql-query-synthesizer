"""
Advanced Health Monitoring System
Comprehensive health checks, dependency monitoring, and system observability.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""

    name: str
    check_function: Callable
    timeout_seconds: int = 30
    critical: bool = False
    enabled: bool = True
    interval_seconds: int = 60
    failure_threshold: int = 3
    recovery_threshold: int = 2


@dataclass
class ComponentHealth:
    """Health status of a system component."""

    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    metadata: Optional[Dict[str, Any]] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None


class HealthMonitor:
    """Monitors health of individual components."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, ComponentHealth] = {}
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.RLock()

    def register_check(self, check: HealthCheck):
        """Register a health check for this component."""
        with self.lock:
            self.checks[check.name] = check
            logger.info(
                f"Registered health check '{check.name}' for component '{self.component_name}'"
            )

    async def run_check(self, check_name: str) -> ComponentHealth:
        """Run a specific health check."""
        check = self.checks.get(check_name)
        if not check or not check.enabled:
            return ComponentHealth(
                component=self.component_name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{check_name}' not found or disabled",
                timestamp=datetime.utcnow(),
                response_time_ms=0,
            )

        start_time = time.time()

        try:
            # Run check with timeout
            if asyncio.iscoroutinefunction(check.check_function):
                result = await asyncio.wait_for(
                    check.check_function(), timeout=check.timeout_seconds
                )
            else:
                # Run sync function in executor with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, check.check_function),
                    timeout=check.timeout_seconds,
                )

            response_time_ms = (time.time() - start_time) * 1000

            # Parse result
            if isinstance(result, dict):
                status = HealthStatus(result.get("status", HealthStatus.HEALTHY.value))
                message = result.get("message", "Check passed")
                metadata = result.get("metadata", {})
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Check passed" if result else "Check failed"
                metadata = {}
            else:
                status = HealthStatus.HEALTHY
                message = str(result)
                metadata = {}

            health = ComponentHealth(
                component=self.component_name,
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms,
                metadata=metadata,
            )

            # Update tracking
            self._update_health_tracking(check_name, health, True)

            return health

        except asyncio.TimeoutError:
            health = ComponentHealth(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {check.timeout_seconds}s",
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000,
            )
            self._update_health_tracking(check_name, health, False)
            return health

        except Exception as e:
            health = ComponentHealth(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000,
            )
            self._update_health_tracking(check_name, health, False)
            return health

    async def run_all_checks(self) -> Dict[str, ComponentHealth]:
        """Run all registered health checks."""
        results = {}

        # Run checks concurrently
        check_tasks = []
        for check_name in self.checks.keys():
            task = asyncio.create_task(self.run_check(check_name))
            check_tasks.append((check_name, task))

        # Collect results
        for check_name, task in check_tasks:
            try:
                result = await task
                results[check_name] = result
            except Exception as e:
                logger.error(f"Error running health check '{check_name}': {e}")
                results[check_name] = ComponentHealth(
                    component=self.component_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution error: {str(e)}",
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                )

        return results

    def get_overall_health(self) -> ComponentHealth:
        """Get overall health status of the component."""
        with self.lock:
            if not self.results:
                return ComponentHealth(
                    component=self.component_name,
                    status=HealthStatus.UNKNOWN,
                    message="No health checks have been run",
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                )

            # Determine overall status
            critical_unhealthy = any(
                result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
                for check_name, result in self.results.items()
                if self.checks[check_name].critical
            )

            if critical_unhealthy:
                overall_status = HealthStatus.CRITICAL
                message = "Critical health check(s) failing"
            else:
                statuses = [result.status for result in self.results.values()]
                if HealthStatus.UNHEALTHY in statuses:
                    overall_status = HealthStatus.DEGRADED
                    message = "Some health checks failing"
                elif HealthStatus.DEGRADED in statuses:
                    overall_status = HealthStatus.DEGRADED
                    message = "Component operating in degraded mode"
                else:
                    overall_status = HealthStatus.HEALTHY
                    message = "All health checks passing"

            return ComponentHealth(
                component=self.component_name,
                status=overall_status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=sum(r.response_time_ms for r in self.results.values())
                / len(self.results),
            )

    def get_health_history(self, check_name: Optional[str] = None) -> List[Dict]:
        """Get health check history."""
        with self.lock:
            if check_name:
                return list(self.history.get(check_name, []))
            else:
                # Return combined history
                combined = []
                for check_name, history in self.history.items():
                    combined.extend(history)
                return sorted(combined, key=lambda x: x["timestamp"])

    def _update_health_tracking(
        self, check_name: str, health: ComponentHealth, success: bool
    ):
        """Update health tracking metrics."""
        with self.lock:
            self.results[check_name] = health

            # Update history
            self.history[check_name].append(
                {
                    "check_name": check_name,
                    "status": health.status.value,
                    "message": health.message,
                    "timestamp": health.timestamp.isoformat(),
                    "response_time_ms": health.response_time_ms,
                }
            )

            # Update consecutive counters
            if success:
                health.consecutive_successes += 1
                health.consecutive_failures = 0
                health.last_success = health.timestamp
            else:
                health.consecutive_failures += 1
                health.consecutive_successes = 0
                health.last_failure = health.timestamp


class DependencyChecker:
    """Checks health of external dependencies."""

    def __init__(self):
        self.dependency_checks = {}

    def register_dependency(self, name: str, check_function: Callable, **kwargs):
        """Register a dependency health check."""
        self.dependency_checks[name] = HealthCheck(
            name=name, check_function=check_function, **kwargs
        )

    async def check_database_connection(self, db_manager) -> Dict[str, Any]:
        """Check database connection health."""
        try:
            start_time = time.time()

            # Simple health query
            await db_manager.execute_query("SELECT 1")

            response_time = (time.time() - start_time) * 1000

            # Get connection pool stats
            pool_stats = getattr(db_manager, "get_connection_stats", lambda: {})()

            return {
                "status": HealthStatus.HEALTHY.value,
                "message": f"Database connection healthy (response: {response_time:.1f}ms)",
                "metadata": {
                    "response_time_ms": response_time,
                    "pool_stats": pool_stats,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Database connection failed: {str(e)}",
                "metadata": {"error": str(e)},
            }

    async def check_llm_service(self, llm_adapter) -> Dict[str, Any]:
        """Check LLM service health."""
        try:
            start_time = time.time()

            # Simple test query
            result = await llm_adapter.generate_sql("SELECT 1", max_tokens=10)

            response_time = (time.time() - start_time) * 1000

            if result and "sql" in result:
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "message": f"LLM service healthy (response: {response_time:.1f}ms)",
                    "metadata": {"response_time_ms": response_time},
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED.value,
                    "message": "LLM service responded but result quality is questionable",
                    "metadata": {"response_time_ms": response_time, "result": result},
                }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"LLM service check failed: {str(e)}",
                "metadata": {"error": str(e)},
            }

    async def check_cache_service(self, cache_manager) -> Dict[str, Any]:
        """Check cache service health."""
        try:
            start_time = time.time()

            # Test cache operations
            test_key = f"health_check_{int(time.time())}"
            test_value = "health_test"

            # Test put and get
            cache_manager.put(test_key, test_value)
            retrieved = cache_manager.get(test_key)

            response_time = (time.time() - start_time) * 1000

            if retrieved == test_value:
                stats = getattr(cache_manager, "get_statistics", lambda: {})()
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "message": f"Cache service healthy (response: {response_time:.1f}ms)",
                    "metadata": {
                        "response_time_ms": response_time,
                        "cache_stats": stats,
                    },
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED.value,
                    "message": "Cache service responding but data integrity issue detected",
                    "metadata": {"response_time_ms": response_time},
                }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Cache service check failed: {str(e)}",
                "metadata": {"error": str(e)},
            }


class PerformanceMonitor:
    """Monitors system performance metrics."""

    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90,
            "response_time_ms": 2000,
        }

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                "active_threads": threading.active_count(),
                "load_average": (
                    psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0
                ),
            }

            # Store in history
            self.metrics_history.append(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {"timestamp": datetime.utcnow().isoformat(), "error": str(e)}

    def check_performance_health(self) -> Dict[str, Any]:
        """Check if system performance is within acceptable thresholds."""
        metrics = self.collect_system_metrics()

        if "error" in metrics:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": f"Cannot collect performance metrics: {metrics['error']}",
                "metadata": metrics,
            }

        issues = []
        status = HealthStatus.HEALTHY

        # Check thresholds
        if metrics["cpu_percent"] > self.thresholds["cpu_percent"]:
            issues.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
            status = HealthStatus.DEGRADED

        if metrics["memory_percent"] > self.thresholds["memory_percent"]:
            issues.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
            status = HealthStatus.DEGRADED

        if metrics["disk_percent"] > self.thresholds["disk_percent"]:
            issues.append(f"High disk usage: {metrics['disk_percent']:.1f}%")
            if metrics["disk_percent"] > 95:
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.DEGRADED

        message = "Performance metrics within acceptable range"
        if issues:
            message = f"Performance issues detected: {'; '.join(issues)}"

        return {"status": status.value, "message": message, "metadata": metrics}

    def get_performance_trends(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance trends over the specified time period."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)

        recent_metrics = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]

        if not recent_metrics:
            return {"message": "Insufficient data for trend analysis"}

        # Calculate trends
        cpu_values = [m["cpu_percent"] for m in recent_metrics if "cpu_percent" in m]
        memory_values = [
            m["memory_percent"] for m in recent_metrics if "memory_percent" in m
        ]

        return {
            "time_period_minutes": minutes,
            "data_points": len(recent_metrics),
            "cpu_trend": {
                "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0,
            },
            "memory_trend": {
                "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0,
            },
        }


class SystemHealthManager:
    """Main health monitoring system coordinator."""

    def __init__(self):
        self.component_monitors: Dict[str, HealthMonitor] = {}
        self.dependency_checker = DependencyChecker()
        self.performance_monitor = PerformanceMonitor()

        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 60  # seconds

        self.health_history = deque(maxlen=1000)
        self.alerts_sent = defaultdict(lambda: datetime.min)
        self.alert_cooldown = timedelta(minutes=15)

    def register_component(self, component_name: str) -> HealthMonitor:
        """Register a component for health monitoring."""
        if component_name not in self.component_monitors:
            self.component_monitors[component_name] = HealthMonitor(component_name)
            logger.info(
                f"Registered component '{component_name}' for health monitoring"
            )

        return self.component_monitors[component_name]

    def start_monitoring(self):
        """Start continuous health monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitor_thread.start()
            logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        # Collect component health
        component_health = {}
        overall_status = HealthStatus.HEALTHY

        for component_name, monitor in self.component_monitors.items():
            component_results = await monitor.run_all_checks()
            component_overall = monitor.get_overall_health()

            component_health[component_name] = {
                "overall_status": component_overall.status.value,
                "overall_message": component_overall.message,
                "checks": {
                    name: asdict(result) for name, result in component_results.items()
                },
                "response_time_ms": component_overall.response_time_ms,
            }

            # Update overall status
            if component_overall.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif (
                component_overall.status == HealthStatus.UNHEALTHY
                and overall_status != HealthStatus.CRITICAL
            ):
                overall_status = HealthStatus.UNHEALTHY
            elif (
                component_overall.status == HealthStatus.DEGRADED
                and overall_status == HealthStatus.HEALTHY
            ):
                overall_status = HealthStatus.DEGRADED

        # Get performance metrics
        performance_health = self.performance_monitor.check_performance_health()

        # Overall system assessment
        if performance_health["status"] == HealthStatus.CRITICAL.value:
            overall_status = HealthStatus.CRITICAL
        elif (
            performance_health["status"] == HealthStatus.DEGRADED.value
            and overall_status == HealthStatus.HEALTHY
        ):
            overall_status = HealthStatus.DEGRADED

        health_report = {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": component_health,
            "performance": performance_health,
            "summary": self._generate_health_summary(
                component_health, performance_health
            ),
        }

        # Store in history
        self.health_history.append(health_report)

        return health_report

    def get_health_insights(self) -> List[Dict[str, Any]]:
        """Generate insights based on health monitoring data."""
        insights = []

        if len(self.health_history) < 5:
            return insights

        # Recent health reports
        recent_reports = list(self.health_history)[-10:]

        # Check for recurring issues
        status_counts = defaultdict(int)
        for report in recent_reports:
            status_counts[report["overall_status"]] += 1

        if status_counts.get("degraded", 0) >= 3:
            insights.append(
                {
                    "type": "recurring_degradation",
                    "message": "System has been in degraded state multiple times recently",
                    "recommendation": "Investigate underlying causes of performance degradation",
                    "priority": "medium",
                }
            )

        if status_counts.get("critical", 0) >= 1:
            insights.append(
                {
                    "type": "critical_issues",
                    "message": "Critical health issues detected recently",
                    "recommendation": "Immediate attention required for system stability",
                    "priority": "high",
                }
            )

        # Performance trends
        performance_trends = self.performance_monitor.get_performance_trends(60)

        if "cpu_trend" in performance_trends:
            cpu_avg = performance_trends["cpu_trend"]["avg"]
            if cpu_avg > 70:
                insights.append(
                    {
                        "type": "high_cpu_usage",
                        "message": f"Average CPU usage is high ({cpu_avg:.1f}%) over the last hour",
                        "recommendation": "Consider resource optimization or scaling",
                        "priority": "medium",
                    }
                )

        return insights

    def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                health_report = loop.run_until_complete(self.get_system_health())

                # Check for alerts
                self._check_and_send_alerts(health_report)

                loop.close()

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

            time.sleep(self.monitor_interval)

    def _generate_health_summary(
        self, component_health: Dict, performance_health: Dict
    ) -> str:
        """Generate human-readable health summary."""
        total_components = len(component_health)
        healthy_components = sum(
            1
            for comp in component_health.values()
            if comp["overall_status"] == "healthy"
        )

        if (
            healthy_components == total_components
            and performance_health["status"] == "healthy"
        ):
            return "All systems operational"
        elif healthy_components >= total_components * 0.8:
            return f"Mostly healthy - {healthy_components}/{total_components} components OK"
        else:
            return f"Multiple issues detected - {healthy_components}/{total_components} components healthy"

    def _check_and_send_alerts(self, health_report: Dict):
        """Check if alerts should be sent based on health report."""
        current_time = datetime.utcnow()
        overall_status = health_report["overall_status"]

        # Check if we should send alert
        should_alert = False
        alert_type = None

        if overall_status == "critical":
            alert_type = "critical"
            should_alert = True
        elif overall_status == "unhealthy":
            alert_type = "unhealthy"
            should_alert = True

        # Apply cooldown
        if should_alert:
            last_alert = self.alerts_sent[alert_type]
            if current_time - last_alert > self.alert_cooldown:
                self._send_alert(alert_type, health_report)
                self.alerts_sent[alert_type] = current_time

    def _send_alert(self, alert_type: str, health_report: Dict):
        """Send health alert (placeholder for actual alerting system)."""
        logger.warning(
            f"HEALTH ALERT [{alert_type.upper()}]: {health_report['summary']}"
        )

        # In a real implementation, this would integrate with:
        # - Email notifications
        # - Slack/Teams webhooks
        # - PagerDuty/OpsGenie
        # - SMS alerts
        # - Monitoring systems (Prometheus AlertManager, etc.)
