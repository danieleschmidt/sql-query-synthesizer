"""
Auto-Scaling System
Provides intelligent auto-scaling based on load patterns and resource utilization.
"""

import time
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
import weakref

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Different auto-scaling strategies."""
    REACTIVE = "reactive"  # Scale based on current metrics
    PREDICTIVE = "predictive"  # Scale based on predicted load
    SCHEDULED = "scheduled"  # Scale based on time schedules
    HYBRID = "hybrid"  # Combination of strategies


class ScalingMetric(Enum):
    """Metrics used for scaling decisions."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_DEPTH = "queue_depth"
    ERROR_RATE = "error_rate"
    CONNECTION_POOL_USAGE = "connection_pool_usage"


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    name: str
    metric: ScalingMetric
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_adjustment: int
    scale_down_adjustment: int
    cooldown_period_seconds: int = 300
    evaluation_periods: int = 2
    enabled: bool = True


class LoadBalancer:
    """Simple load balancer for distributing requests across instances."""
    
    def __init__(self):
        self.instances = []
        self.current_index = 0
        self.instance_stats = defaultdict(lambda: {
            'requests': 0,
            'failures': 0,
            'avg_response_time': 0,
            'last_health_check': datetime.min
        })
        self.lock = threading.RLock()
    
    def add_instance(self, instance_id: str, instance_data: Dict[str, Any]):
        """Add an instance to the load balancer."""
        with self.lock:
            if instance_id not in [inst['id'] for inst in self.instances]:
                self.instances.append({
                    'id': instance_id,
                    'data': instance_data,
                    'healthy': True,
                    'weight': 1.0
                })
                logger.info(f"Added instance {instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str):
        """Remove an instance from the load balancer."""
        with self.lock:
            self.instances = [inst for inst in self.instances if inst['id'] != instance_id]
            if instance_id in self.instance_stats:
                del self.instance_stats[instance_id]
            logger.info(f"Removed instance {instance_id} from load balancer")
    
    def get_next_instance(self) -> Optional[Dict[str, Any]]:
        """Get next instance using round-robin with health checking."""
        with self.lock:
            if not self.instances:
                return None
            
            healthy_instances = [inst for inst in self.instances if inst['healthy']]
            
            if not healthy_instances:
                logger.warning("No healthy instances available")
                return None
            
            # Simple round-robin
            selected = healthy_instances[self.current_index % len(healthy_instances)]
            self.current_index = (self.current_index + 1) % len(healthy_instances)
            
            return selected
    
    def record_request(self, instance_id: str, response_time_ms: float, success: bool):
        """Record request statistics for an instance."""
        with self.lock:
            stats = self.instance_stats[instance_id]
            stats['requests'] += 1
            
            if not success:
                stats['failures'] += 1
            
            # Update average response time
            current_avg = stats['avg_response_time']
            request_count = stats['requests']
            stats['avg_response_time'] = (current_avg * (request_count - 1) + response_time_ms) / request_count
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across instances."""
        with self.lock:
            total_requests = sum(stats['requests'] for stats in self.instance_stats.values())
            
            distribution = {}
            for instance in self.instances:
                inst_id = instance['id']
                stats = self.instance_stats[inst_id]
                
                distribution[inst_id] = {
                    'healthy': instance['healthy'],
                    'weight': instance['weight'],
                    'requests': stats['requests'],
                    'request_percentage': (stats['requests'] / max(total_requests, 1)) * 100,
                    'failure_rate': (stats['failures'] / max(stats['requests'], 1)) * 100,
                    'avg_response_time_ms': stats['avg_response_time']
                }
            
            return {
                'total_instances': len(self.instances),
                'healthy_instances': len([inst for inst in self.instances if inst['healthy']]),
                'total_requests': total_requests,
                'distribution': distribution
            }


class ResourcePool:
    """Pool of resources that can be scaled up or down."""
    
    def __init__(self, pool_name: str, min_size: int = 1, max_size: int = 100):
        self.pool_name = pool_name
        self.min_size = min_size
        self.max_size = max_size
        
        self.resources = {}  # resource_id -> resource_data
        self.available_resources = deque()
        self.busy_resources = set()
        
        self.creation_function: Optional[Callable] = None
        self.destruction_function: Optional[Callable] = None
        self.health_check_function: Optional[Callable] = None
        
        self.statistics = {
            'created': 0,
            'destroyed': 0,
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.lock = threading.RLock()
    
    def set_resource_management_functions(self, 
                                        creation_func: Callable,
                                        destruction_func: Callable,
                                        health_check_func: Optional[Callable] = None):
        """Set functions for resource lifecycle management."""
        self.creation_function = creation_func
        self.destruction_function = destruction_func
        self.health_check_function = health_check_func
    
    def acquire_resource(self, timeout: float = 30) -> Optional[Any]:
        """Acquire a resource from the pool."""
        with self.lock:
            self.statistics['requests'] += 1
            
            # Try to get from available resources
            if self.available_resources:
                resource_id = self.available_resources.popleft()
                self.busy_resources.add(resource_id)
                self.statistics['cache_hits'] += 1
                return self.resources[resource_id]
            
            # Create new resource if under max size
            if len(self.resources) < self.max_size and self.creation_function:
                resource_id = f"{self.pool_name}_{len(self.resources)}_{int(time.time())}"
                
                try:
                    resource = self.creation_function()
                    self.resources[resource_id] = resource
                    self.busy_resources.add(resource_id)
                    self.statistics['created'] += 1
                    self.statistics['cache_misses'] += 1
                    
                    logger.debug(f"Created new resource {resource_id} in pool {self.pool_name}")
                    return resource
                    
                except Exception as e:
                    logger.error(f"Failed to create resource in pool {self.pool_name}: {e}")
                    return None
            
            # Pool is at max capacity
            logger.warning(f"Resource pool {self.pool_name} is at maximum capacity ({self.max_size})")
            return None
    
    def release_resource(self, resource: Any):
        """Release a resource back to the pool."""
        with self.lock:
            # Find resource by object reference
            resource_id = None
            for rid, res in self.resources.items():
                if res is resource:
                    resource_id = rid
                    break
            
            if resource_id and resource_id in self.busy_resources:
                self.busy_resources.remove(resource_id)
                
                # Health check if function is provided
                if self.health_check_function:
                    try:
                        if self.health_check_function(resource):
                            self.available_resources.append(resource_id)
                        else:
                            # Resource is unhealthy, destroy it
                            self._destroy_resource(resource_id)
                    except Exception as e:
                        logger.error(f"Health check failed for resource {resource_id}: {e}")
                        self._destroy_resource(resource_id)
                else:
                    self.available_resources.append(resource_id)
    
    def scale_pool(self, target_size: int) -> int:
        """Scale the pool to target size."""
        with self.lock:
            current_size = len(self.resources)
            target_size = max(self.min_size, min(target_size, self.max_size))
            
            if target_size > current_size:
                # Scale up
                return self._scale_up(target_size - current_size)
            elif target_size < current_size:
                # Scale down
                return self._scale_down(current_size - target_size)
            else:
                return 0
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get pool statistics and status."""
        with self.lock:
            total_requests = self.statistics['requests']
            hit_rate = (self.statistics['cache_hits'] / max(total_requests, 1)) * 100
            
            return {
                'pool_name': self.pool_name,
                'size_limits': {'min': self.min_size, 'max': self.max_size},
                'current_size': len(self.resources),
                'available_resources': len(self.available_resources),
                'busy_resources': len(self.busy_resources),
                'utilization_percent': (len(self.busy_resources) / max(len(self.resources), 1)) * 100,
                'statistics': self.statistics.copy(),
                'hit_rate_percent': hit_rate
            }
    
    def _scale_up(self, count: int) -> int:
        """Scale up the pool by creating new resources."""
        if not self.creation_function:
            logger.warning(f"Cannot scale up pool {self.pool_name}: no creation function")
            return 0
        
        created = 0
        for _ in range(count):
            if len(self.resources) >= self.max_size:
                break
            
            resource_id = f"{self.pool_name}_{len(self.resources)}_{int(time.time())}"
            
            try:
                resource = self.creation_function()
                self.resources[resource_id] = resource
                self.available_resources.append(resource_id)
                self.statistics['created'] += 1
                created += 1
                
            except Exception as e:
                logger.error(f"Failed to create resource during scale-up: {e}")
                break
        
        if created > 0:
            logger.info(f"Scaled up pool {self.pool_name} by {created} resources")
        
        return created
    
    def _scale_down(self, count: int) -> int:
        """Scale down the pool by destroying resources."""
        destroyed = 0
        
        # Remove from available resources first
        while destroyed < count and self.available_resources:
            resource_id = self.available_resources.popleft()
            self._destroy_resource(resource_id)
            destroyed += 1
        
        # If we need to remove more and have idle resources, remove them
        # (This is simplified - in production you'd wait for resources to become available)
        
        if destroyed > 0:
            logger.info(f"Scaled down pool {self.pool_name} by {destroyed} resources")
        
        return destroyed
    
    def _destroy_resource(self, resource_id: str):
        """Destroy a specific resource."""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            
            if self.destruction_function:
                try:
                    self.destruction_function(resource)
                except Exception as e:
                    logger.error(f"Error destroying resource {resource_id}: {e}")
            
            del self.resources[resource_id]
            self.statistics['destroyed'] += 1


class AutoScaler:
    """Main auto-scaling system."""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.REACTIVE):
        self.strategy = strategy
        self.scaling_rules: List[ScalingRule] = []
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.load_balancer = LoadBalancer()
        
        # Scaling state
        self.scaling_active = False
        self.scaling_thread = None
        self.scaling_interval = 30  # seconds
        
        # Metrics history for decision making
        self.metrics_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        
        # Cooldown tracking
        self.last_scale_action = defaultdict(lambda: datetime.min)
        
        self._setup_default_scaling_rules()
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
    
    def register_resource_pool(self, pool: ResourcePool):
        """Register a resource pool for auto-scaling."""
        self.resource_pools[pool.pool_name] = pool
        logger.info(f"Registered resource pool: {pool.pool_name}")
    
    def start_auto_scaling(self):
        """Start the auto-scaling system."""
        if not self.scaling_active:
            self.scaling_active = True
            self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
            self.scaling_thread.start()
            logger.info("Auto-scaling system started")
    
    def stop_auto_scaling(self):
        """Stop the auto-scaling system."""
        self.scaling_active = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        logger.info("Auto-scaling system stopped")
    
    def record_metrics(self, metrics: Dict[ScalingMetric, float]):
        """Record metrics for scaling decisions."""
        metric_record = {
            'timestamp': datetime.utcnow(),
            'metrics': metrics.copy()
        }
        
        self.metrics_history.append(metric_record)
    
    def evaluate_scaling_decision(self) -> List[Dict[str, Any]]:
        """Evaluate whether scaling is needed based on rules and metrics."""
        if len(self.metrics_history) < 2:
            return []
        
        scaling_decisions = []
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if (datetime.utcnow() - self.last_scale_action[rule.name] < 
                timedelta(seconds=rule.cooldown_period_seconds)):
                continue
            
            # Get recent metrics for this rule's metric
            recent_metrics = list(self.metrics_history)[-rule.evaluation_periods:]
            metric_values = [
                record['metrics'].get(rule.metric, 0) 
                for record in recent_metrics
                if rule.metric in record['metrics']
            ]
            
            if len(metric_values) < rule.evaluation_periods:
                continue
            
            # Evaluate scaling decision
            avg_value = statistics.mean(metric_values)
            decision = self._evaluate_rule_decision(rule, avg_value, metric_values)
            
            if decision:
                scaling_decisions.append(decision)
        
        return scaling_decisions
    
    def apply_scaling_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply scaling decisions to resource pools."""
        results = []
        
        for decision in decisions:
            try:
                result = self._apply_single_scaling_decision(decision)
                results.append(result)
                
                # Record scaling action
                scaling_record = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'decision': decision,
                    'result': result
                }
                self.scaling_history.append(scaling_record)
                
            except Exception as e:
                logger.error(f"Error applying scaling decision: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'decision': decision
                })
        
        return results
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        return {
            'scaling_active': self.scaling_active,
            'strategy': self.strategy.value,
            'scaling_rules': [asdict(rule) for rule in self.scaling_rules],
            'resource_pools': {
                name: pool.get_pool_statistics() 
                for name, pool in self.resource_pools.items()
            },
            'load_balancer': self.load_balancer.get_load_distribution(),
            'recent_scaling_actions': list(self.scaling_history)[-10:],
            'metrics_collected': len(self.metrics_history)
        }
    
    def predict_scaling_needs(self, time_horizon_minutes: int = 30) -> Dict[str, Any]:
        """Predict future scaling needs (simplified predictive model)."""
        if len(self.metrics_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Simple trend analysis
        recent_metrics = list(self.metrics_history)[-20:]
        
        predictions = {}
        for metric_type in ScalingMetric:
            values = [
                record['metrics'].get(metric_type, 0)
                for record in recent_metrics
                if metric_type in record['metrics']
            ]
            
            if len(values) >= 5:
                # Simple linear trend
                trend = (values[-1] - values[0]) / len(values)
                predicted_value = values[-1] + (trend * time_horizon_minutes / 5)  # Assuming 5-min intervals
                
                predictions[metric_type.value] = {
                    'current': values[-1],
                    'predicted': predicted_value,
                    'trend': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
                    'confidence': min(len(values) / 20, 1.0)
                }
        
        return {
            'time_horizon_minutes': time_horizon_minutes,
            'predictions': predictions,
            'prediction_timestamp': datetime.utcnow().isoformat()
        }
    
    def _setup_default_scaling_rules(self):
        """Setup default scaling rules."""
        # CPU-based scaling
        self.add_scaling_rule(ScalingRule(
            name="cpu_scaling",
            metric=ScalingMetric.CPU_USAGE,
            scale_up_threshold=75.0,
            scale_down_threshold=30.0,
            scale_up_adjustment=2,
            scale_down_adjustment=1,
            cooldown_period_seconds=300,
            evaluation_periods=2
        ))
        
        # Memory-based scaling
        self.add_scaling_rule(ScalingRule(
            name="memory_scaling",
            metric=ScalingMetric.MEMORY_USAGE,
            scale_up_threshold=80.0,
            scale_down_threshold=40.0,
            scale_up_adjustment=2,
            scale_down_adjustment=1,
            cooldown_period_seconds=300,
            evaluation_periods=2
        ))
        
        # Response time-based scaling
        self.add_scaling_rule(ScalingRule(
            name="response_time_scaling",
            metric=ScalingMetric.RESPONSE_TIME,
            scale_up_threshold=2000.0,  # 2 seconds
            scale_down_threshold=500.0,   # 0.5 seconds
            scale_up_adjustment=3,
            scale_down_adjustment=1,
            cooldown_period_seconds=180,
            evaluation_periods=3
        ))
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self.scaling_active:
            try:
                # Evaluate scaling decisions
                decisions = self.evaluate_scaling_decision()
                
                if decisions:
                    logger.info(f"Auto-scaler found {len(decisions)} scaling decisions")
                    results = self.apply_scaling_decisions(decisions)
                    
                    successful = len([r for r in results if r.get('success', False)])
                    logger.info(f"Applied {successful}/{len(results)} scaling decisions successfully")
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
            
            time.sleep(self.scaling_interval)
    
    def _evaluate_rule_decision(self, rule: ScalingRule, avg_value: float, 
                              metric_values: List[float]) -> Optional[Dict[str, Any]]:
        """Evaluate a single scaling rule."""
        scale_up = avg_value > rule.scale_up_threshold
        scale_down = avg_value < rule.scale_down_threshold
        
        if not (scale_up or scale_down):
            return None
        
        # Additional checks
        if scale_up:
            # Ensure it's not a spike - check that most recent values are high
            recent_high = sum(1 for v in metric_values[-2:] if v > rule.scale_up_threshold)
            if recent_high < len(metric_values[-2:]) * 0.5:
                return None
        
        if scale_down:
            # Ensure stable low values
            recent_low = sum(1 for v in metric_values[-2:] if v < rule.scale_down_threshold)
            if recent_low < len(metric_values[-2:]) * 0.8:
                return None
        
        return {
            'rule_name': rule.name,
            'metric': rule.metric.value,
            'action': 'scale_up' if scale_up else 'scale_down',
            'adjustment': rule.scale_up_adjustment if scale_up else rule.scale_down_adjustment,
            'current_value': avg_value,
            'threshold': rule.scale_up_threshold if scale_up else rule.scale_down_threshold,
            'confidence': min(len(metric_values) / rule.evaluation_periods, 1.0)
        }
    
    def _apply_single_scaling_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single scaling decision."""
        action = decision['action']
        adjustment = decision['adjustment']
        rule_name = decision['rule_name']
        
        # Update cooldown
        self.last_scale_action[rule_name] = datetime.utcnow()
        
        # Apply to all relevant resource pools
        results = []
        for pool_name, pool in self.resource_pools.items():
            current_size = len(pool.resources)
            
            if action == 'scale_up':
                target_size = current_size + adjustment
                scaled = pool.scale_pool(target_size)
                results.append({
                    'pool': pool_name,
                    'action': 'scale_up',
                    'previous_size': current_size,
                    'target_size': target_size,
                    'actual_change': scaled
                })
            
            elif action == 'scale_down':
                target_size = max(pool.min_size, current_size - adjustment)
                scaled = pool.scale_pool(target_size)
                results.append({
                    'pool': pool_name,
                    'action': 'scale_down',
                    'previous_size': current_size,
                    'target_size': target_size,
                    'actual_change': -scaled  # Negative for scale down
                })
        
        return {
            'success': True,
            'rule_name': rule_name,
            'pool_results': results,
            'timestamp': datetime.utcnow().isoformat()
        }


# Global auto-scaler instance
auto_scaler = AutoScaler()