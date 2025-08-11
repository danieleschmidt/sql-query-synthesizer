"""Auto-Scaling Engine for SQL Query Synthesizer.

This module implements intelligent auto-scaling capabilities that monitor
system load and automatically adjust resources to handle varying workloads.
"""

import asyncio
import logging
import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import subprocess
import psutil

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    DATABASE_CONNECTIONS = "database_connections"
    WORKER_PROCESSES = "worker_processes"
    CACHE_SIZE = "cache_size"
    MEMORY_LIMIT = "memory_limit"
    CONCURRENT_QUERIES = "concurrent_queries"


class ScalingTrigger(Enum):
    """Triggers that initiate scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUERY_QUEUE_LENGTH = "query_queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CONNECTION_POOL_USAGE = "connection_pool_usage"


@dataclass
class ScalingMetric:
    """Metric used for scaling decisions."""
    name: str
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    trend: List[float] = field(default_factory=list)
    last_updated: float = 0.0


@dataclass
class ScalingRule:
    """Rule that defines when and how to scale."""
    rule_id: str
    resource_type: ResourceType
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period: int  # seconds
    min_instances: int
    max_instances: int
    scale_factor: float = 1.5
    enabled: bool = True


@dataclass
class ScalingAction:
    """Represents a scaling action to be executed."""
    action_id: str
    resource_type: ResourceType
    direction: ScalingDirection
    target_value: int
    current_value: int
    reason: str
    timestamp: float
    executed: bool = False
    success: bool = False
    execution_time: float = 0.0


class MetricsCollector:
    """Collects system and application metrics for scaling decisions."""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics: Dict[str, ScalingMetric] = {}
        self.running = False
        self._collector_task = None
        self._lock = threading.Lock()
        
        # Initialize default metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize default system metrics."""
        self.metrics = {
            "cpu_utilization": ScalingMetric(
                name="CPU Utilization",
                current_value=0.0,
                threshold_up=80.0,
                threshold_down=20.0,
                weight=1.0
            ),
            "memory_utilization": ScalingMetric(
                name="Memory Utilization", 
                current_value=0.0,
                threshold_up=85.0,
                threshold_down=30.0,
                weight=1.2
            ),
            "query_queue_length": ScalingMetric(
                name="Query Queue Length",
                current_value=0.0,
                threshold_up=50.0,
                threshold_down=5.0,
                weight=1.5
            ),
            "response_time_ms": ScalingMetric(
                name="Average Response Time",
                current_value=0.0,
                threshold_up=2000.0,
                threshold_down=100.0,
                weight=1.3
            ),
            "connection_pool_usage": ScalingMetric(
                name="Connection Pool Usage",
                current_value=0.0,
                threshold_up=90.0,
                threshold_down=20.0,
                weight=1.1
            ),
            "error_rate": ScalingMetric(
                name="Error Rate",
                current_value=0.0,
                threshold_up=5.0,
                threshold_down=1.0,
                weight=2.0
            )
        }
    
    def start_collection(self):
        """Start metrics collection in background."""
        if self.running:
            return
        
        self.running = True
        self._collector_task = threading.Thread(target=self._collection_loop, daemon=True)
        self._collector_task.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self._collector_task and self._collector_task.is_alive():
            self._collector_task.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(5)  # Brief pause on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            self._update_metric("cpu_utilization", cpu_percent)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._update_metric("memory_utilization", memory_percent)
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # These would be integrated with your actual application metrics
            # For now, using placeholder values
            
            # Query queue length (would come from actual queue)
            queue_length = 0  # Placeholder
            self._update_metric("query_queue_length", queue_length)
            
            # Response time (would come from actual measurements)
            response_time = 100.0  # Placeholder
            self._update_metric("response_time_ms", response_time)
            
            # Connection pool usage (would come from actual pool)
            pool_usage = 0.0  # Placeholder  
            self._update_metric("connection_pool_usage", pool_usage)
            
            # Error rate (would come from actual error tracking)
            error_rate = 0.0  # Placeholder
            self._update_metric("error_rate", error_rate)
            
        except Exception as e:
            logger.warning(f"Failed to collect application metrics: {e}")
    
    def _update_metric(self, metric_name: str, value: float):
        """Update a metric with new value and maintain trend."""
        with self._lock:
            if metric_name not in self.metrics:
                return
            
            metric = self.metrics[metric_name]
            metric.current_value = value
            metric.last_updated = time.time()
            metric.trend.append(value)
            
            # Keep only recent trend data
            if len(metric.trend) > 100:
                metric.trend = metric.trend[-50:]
    
    def get_metric(self, metric_name: str) -> Optional[ScalingMetric]:
        """Get current metric value."""
        with self._lock:
            return self.metrics.get(metric_name)
    
    def get_all_metrics(self) -> Dict[str, ScalingMetric]:
        """Get all current metrics."""
        with self._lock:
            return self.metrics.copy()
    
    def update_external_metric(self, metric_name: str, value: float):
        """Update metric from external source."""
        self._update_metric(metric_name, value)


class ScalingDecisionEngine:
    """Makes intelligent scaling decisions based on collected metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: List[ScalingAction] = []
        self.current_resources: Dict[ResourceType, int] = {}
        self._last_scaling_time: Dict[str, float] = {}
        
        # Initialize default scaling rules
        self._initialize_scaling_rules()
        
        # Initialize current resource levels
        self.current_resources = {
            ResourceType.DATABASE_CONNECTIONS: 10,
            ResourceType.WORKER_PROCESSES: 2,
            ResourceType.CACHE_SIZE: 1000,
            ResourceType.MEMORY_LIMIT: 1024,
            ResourceType.CONCURRENT_QUERIES: 50
        }
    
    def _initialize_scaling_rules(self):
        """Initialize default scaling rules."""
        
        # Database connections scaling
        self.scaling_rules["db_connections_cpu"] = ScalingRule(
            rule_id="db_connections_cpu",
            resource_type=ResourceType.DATABASE_CONNECTIONS,
            trigger=ScalingTrigger.CPU_UTILIZATION,
            scale_up_threshold=70.0,
            scale_down_threshold=20.0,
            cooldown_period=300,  # 5 minutes
            min_instances=5,
            max_instances=100,
            scale_factor=1.5
        )
        
        # Worker processes scaling
        self.scaling_rules["workers_queue"] = ScalingRule(
            rule_id="workers_queue",
            resource_type=ResourceType.WORKER_PROCESSES,
            trigger=ScalingTrigger.QUERY_QUEUE_LENGTH,
            scale_up_threshold=20.0,
            scale_down_threshold=2.0,
            cooldown_period=180,  # 3 minutes
            min_instances=1,
            max_instances=20,
            scale_factor=2.0
        )
        
        # Cache size scaling
        self.scaling_rules["cache_memory"] = ScalingRule(
            rule_id="cache_memory",
            resource_type=ResourceType.CACHE_SIZE,
            trigger=ScalingTrigger.MEMORY_UTILIZATION,
            scale_up_threshold=60.0,
            scale_down_threshold=30.0,
            cooldown_period=600,  # 10 minutes
            min_instances=500,
            max_instances=10000,
            scale_factor=1.3
        )
        
        # Concurrent queries scaling
        self.scaling_rules["concurrent_response"] = ScalingRule(
            rule_id="concurrent_response",
            resource_type=ResourceType.CONCURRENT_QUERIES,
            trigger=ScalingTrigger.RESPONSE_TIME,
            scale_up_threshold=1500.0,
            scale_down_threshold=200.0,
            cooldown_period=120,  # 2 minutes
            min_instances=10,
            max_instances=500,
            scale_factor=1.4
        )
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules[rule.rule_id] = rule
        logger.info(f"Added scaling rule: {rule.rule_id}")
    
    def evaluate_scaling_decisions(self) -> List[ScalingAction]:
        """Evaluate all scaling rules and return recommended actions."""
        actions = []
        current_time = time.time()
        
        for rule_id, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown period
            last_scaling = self._last_scaling_time.get(rule_id, 0)
            if current_time - last_scaling < rule.cooldown_period:
                continue
            
            # Get relevant metric
            metric = self._get_metric_for_trigger(rule.trigger)
            if not metric:
                continue
            
            # Evaluate scaling decision
            action = self._evaluate_rule(rule, metric, current_time)
            if action:
                actions.append(action)
        
        return actions
    
    def _get_metric_for_trigger(self, trigger: ScalingTrigger) -> Optional[ScalingMetric]:
        """Get metric associated with a scaling trigger."""
        trigger_metric_map = {
            ScalingTrigger.CPU_UTILIZATION: "cpu_utilization",
            ScalingTrigger.MEMORY_UTILIZATION: "memory_utilization",
            ScalingTrigger.QUERY_QUEUE_LENGTH: "query_queue_length",
            ScalingTrigger.RESPONSE_TIME: "response_time_ms",
            ScalingTrigger.ERROR_RATE: "error_rate",
            ScalingTrigger.CONNECTION_POOL_USAGE: "connection_pool_usage"
        }
        
        metric_name = trigger_metric_map.get(trigger)
        if metric_name:
            return self.metrics_collector.get_metric(metric_name)
        return None
    
    def _evaluate_rule(self, rule: ScalingRule, metric: ScalingMetric, current_time: float) -> Optional[ScalingAction]:
        """Evaluate a single scaling rule."""
        current_value = metric.current_value
        current_instances = self.current_resources.get(rule.resource_type, 1)
        
        # Determine scaling direction
        direction = ScalingDirection.STABLE
        target_instances = current_instances
        reason = ""
        
        if current_value >= rule.scale_up_threshold and current_instances < rule.max_instances:
            direction = ScalingDirection.UP
            target_instances = min(
                int(current_instances * rule.scale_factor),
                rule.max_instances
            )
            reason = f"{metric.name} ({current_value:.1f}) above threshold ({rule.scale_up_threshold})"
            
        elif current_value <= rule.scale_down_threshold and current_instances > rule.min_instances:
            direction = ScalingDirection.DOWN
            target_instances = max(
                int(current_instances / rule.scale_factor),
                rule.min_instances
            )
            reason = f"{metric.name} ({current_value:.1f}) below threshold ({rule.scale_down_threshold})"
        
        # Check if trend supports the scaling decision
        if direction != ScalingDirection.STABLE:
            if not self._trend_supports_scaling(metric, direction):
                logger.debug(f"Trend doesn't support scaling for rule {rule.rule_id}")
                return None
        
        if direction != ScalingDirection.STABLE and target_instances != current_instances:
            import uuid
            return ScalingAction(
                action_id=str(uuid.uuid4())[:8],
                resource_type=rule.resource_type,
                direction=direction,
                target_value=target_instances,
                current_value=current_instances,
                reason=reason,
                timestamp=current_time
            )
        
        return None
    
    def _trend_supports_scaling(self, metric: ScalingMetric, direction: ScalingDirection) -> bool:
        """Check if metric trend supports the scaling decision."""
        if len(metric.trend) < 3:
            return True  # Not enough data, allow scaling
        
        # Get recent trend
        recent_values = metric.trend[-5:]
        
        if direction == ScalingDirection.UP:
            # Check if trend is increasing
            increasing = sum(1 for i in range(1, len(recent_values)) 
                           if recent_values[i] > recent_values[i-1])
            return increasing >= len(recent_values) - 2
            
        elif direction == ScalingDirection.DOWN:
            # Check if trend is decreasing
            decreasing = sum(1 for i in range(1, len(recent_values)) 
                           if recent_values[i] < recent_values[i-1])
            return decreasing >= len(recent_values) - 2
        
        return True
    
    def execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action."""
        start_time = time.time()
        
        try:
            success = self._perform_scaling(action)
            
            if success:
                # Update current resource levels
                self.current_resources[action.resource_type] = action.target_value
                self._last_scaling_time[action.resource_type.value] = action.timestamp
                
                logger.info(
                    f"Scaling {action.direction.value}: {action.resource_type.value} "
                    f"from {action.current_value} to {action.target_value} - {action.reason}"
                )
            
            action.executed = True
            action.success = success
            action.execution_time = time.time() - start_time
            
            # Record in history
            self.scaling_history.append(action)
            
            # Keep history limited
            if len(self.scaling_history) > 1000:
                self.scaling_history = self.scaling_history[-500:]
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute scaling action {action.action_id}: {e}")
            action.executed = True
            action.success = False
            action.execution_time = time.time() - start_time
            self.scaling_history.append(action)
            return False
    
    def _perform_scaling(self, action: ScalingAction) -> bool:
        """Perform the actual scaling operation."""
        
        # This would integrate with your actual infrastructure
        # For now, we'll simulate the scaling operations
        
        resource_type = action.resource_type
        target_value = action.target_value
        
        if resource_type == ResourceType.DATABASE_CONNECTIONS:
            # Scale database connection pool
            logger.info(f"Scaling database connections to {target_value}")
            # Here you would adjust the connection pool size
            return True
            
        elif resource_type == ResourceType.WORKER_PROCESSES:
            # Scale worker processes
            logger.info(f"Scaling worker processes to {target_value}")
            # Here you would start/stop worker processes
            return True
            
        elif resource_type == ResourceType.CACHE_SIZE:
            # Scale cache size
            logger.info(f"Scaling cache size to {target_value}")
            # Here you would adjust cache memory limits
            return True
            
        elif resource_type == ResourceType.MEMORY_LIMIT:
            # Scale memory limits
            logger.info(f"Scaling memory limit to {target_value}MB")
            # Here you would adjust memory limits
            return True
            
        elif resource_type == ResourceType.CONCURRENT_QUERIES:
            # Scale concurrent query limit
            logger.info(f"Scaling concurrent queries to {target_value}")
            # Here you would adjust query concurrency limits
            return True
        
        return False
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling performance statistics."""
        if not self.scaling_history:
            return {"total_actions": 0}
        
        total_actions = len(self.scaling_history)
        successful_actions = sum(1 for a in self.scaling_history if a.success)
        
        # Actions by resource type
        by_resource = {}
        for action in self.scaling_history:
            resource = action.resource_type.value
            if resource not in by_resource:
                by_resource[resource] = {"up": 0, "down": 0}
            by_resource[resource][action.direction.value] += 1
        
        # Recent scaling activity
        recent_time = time.time() - 3600  # Last hour
        recent_actions = [a for a in self.scaling_history if a.timestamp > recent_time]
        
        # Average execution time
        execution_times = [a.execution_time for a in self.scaling_history if a.execution_time > 0]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": (successful_actions / total_actions * 100) if total_actions > 0 else 0,
            "actions_by_resource": by_resource,
            "recent_actions_count": len(recent_actions),
            "average_execution_time_ms": avg_execution_time * 1000,
            "current_resource_levels": {k.value: v for k, v in self.current_resources.items()}
        }


class AutoScalingEngine:
    """Main auto-scaling engine that coordinates metrics collection and scaling decisions."""
    
    def __init__(
        self, 
        metrics_interval: int = 30,
        evaluation_interval: int = 60,
        enabled: bool = True
    ):
        self.metrics_collector = MetricsCollector(metrics_interval)
        self.decision_engine = ScalingDecisionEngine(self.metrics_collector)
        self.evaluation_interval = evaluation_interval
        self.enabled = enabled
        self.running = False
        
        self._engine_task = None
        self._shutdown_event = threading.Event()
    
    def start(self):
        """Start the auto-scaling engine."""
        if self.running or not self.enabled:
            return
        
        self.running = True
        self.metrics_collector.start_collection()
        
        # Start scaling evaluation loop
        self._engine_task = threading.Thread(target=self._scaling_loop, daemon=True)
        self._engine_task.start()
        
        logger.info("Auto-scaling engine started")
    
    def stop(self):
        """Stop the auto-scaling engine."""
        if not self.running:
            return
        
        self.running = False
        self._shutdown_event.set()
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Wait for engine task to finish
        if self._engine_task and self._engine_task.is_alive():
            self._engine_task.join(timeout=10)
        
        logger.info("Auto-scaling engine stopped")
    
    def _scaling_loop(self):
        """Main scaling evaluation loop."""
        while self.running and not self._shutdown_event.is_set():
            try:
                # Evaluate scaling decisions
                actions = self.decision_engine.evaluate_scaling_decisions()
                
                # Execute recommended actions
                for action in actions:
                    if not self.running:
                        break
                    
                    logger.info(f"Executing scaling action: {action.action_id}")
                    success = self.decision_engine.execute_scaling_action(action)
                    
                    if success:
                        logger.info(f"Scaling action {action.action_id} completed successfully")
                    else:
                        logger.error(f"Scaling action {action.action_id} failed")
                
                # Wait for next evaluation
                self._shutdown_event.wait(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                self._shutdown_event.wait(30)  # Brief pause on error
    
    def update_metric(self, metric_name: str, value: float):
        """Update a metric from external source."""
        self.metrics_collector.update_external_metric(metric_name, value)
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.decision_engine.add_scaling_rule(rule)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        metrics = self.metrics_collector.get_all_metrics()
        scaling_stats = self.decision_engine.get_scaling_statistics()
        
        return {
            "enabled": self.enabled,
            "running": self.running,
            "metrics": {
                name: {
                    "current_value": metric.current_value,
                    "threshold_up": metric.threshold_up,
                    "threshold_down": metric.threshold_down,
                    "last_updated": metric.last_updated
                }
                for name, metric in metrics.items()
            },
            "scaling_statistics": scaling_stats,
            "active_rules": len(self.decision_engine.scaling_rules),
            "evaluation_interval": self.evaluation_interval
        }
    
    def force_scaling_evaluation(self) -> List[ScalingAction]:
        """Force immediate scaling evaluation."""
        return self.decision_engine.evaluate_scaling_decisions()


# Global auto-scaling engine instance
auto_scaling_engine = AutoScalingEngine()