"""
Auto-scaling and load balancing for quantum components
"""

import time
import threading
import asyncio
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import collections

from .exceptions import QuantumResourceError, QuantumSchedulerError
from .performance import QuantumPerformanceMonitor, PerformanceMetrics


class ScalingMode(Enum):
    """Auto-scaling modes"""
    MANUAL = "manual"           # No auto-scaling
    REACTIVE = "reactive"       # Scale based on current load
    PREDICTIVE = "predictive"   # Scale based on predicted load
    ADAPTIVE = "adaptive"       # Learn and adapt scaling patterns


@dataclass
class WorkerNode:
    """Represents a quantum worker node"""
    node_id: str
    capacity: float
    current_load: float = 0.0
    is_healthy: bool = True
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    total_tasks_completed: int = 0
    total_processing_time: float = 0.0
    
    @property
    def utilization(self) -> float:
        """Current utilization as percentage"""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 0
    
    @property
    def average_task_time(self) -> float:
        """Average task processing time"""
        return (self.total_processing_time / self.total_tasks_completed 
                if self.total_tasks_completed > 0 else 0.0)
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp"""
        self.last_heartbeat = time.time()
    
    def assign_load(self, load: float) -> bool:
        """Assign load to worker if capacity allows"""
        if self.current_load + load <= self.capacity and self.is_healthy:
            self.current_load += load
            return True
        return False
    
    def release_load(self, load: float):
        """Release load from worker"""
        self.current_load = max(0, self.current_load - load)
    
    def complete_task(self, processing_time: float):
        """Record task completion"""
        self.total_tasks_completed += 1
        self.total_processing_time += processing_time


class QuantumLoadBalancer:
    """
    Intelligent load balancer for quantum operations
    """
    
    def __init__(self, rebalance_interval: float = 30.0):
        self.rebalance_interval = rebalance_interval
        
        # Worker management
        self._workers: Dict[str, WorkerNode] = {}
        self._worker_lock = threading.RLock()
        
        # Load balancing algorithms
        self._algorithms = {
            "round_robin": self._round_robin,
            "least_connections": self._least_connections,
            "weighted_round_robin": self._weighted_round_robin,
            "least_response_time": self._least_response_time,
            "adaptive": self._adaptive_selection
        }
        
        self._current_algorithm = "adaptive"
        self._round_robin_index = 0
        
        # Performance tracking
        self._request_history: collections.deque = collections.deque(maxlen=1000)
        self._last_rebalance = time.time()
        
        # Health monitoring
        self._health_check_interval = 10.0
        self._unhealthy_threshold = 60.0  # Seconds without heartbeat
        
        # Start background tasks
        self._running = True
        self._background_thread = threading.Thread(target=self._background_tasks, daemon=True)
        self._background_thread.start()
    
    def register_worker(self, node_id: str, capacity: float = 1.0) -> bool:
        """
        Register a new worker node
        
        Args:
            node_id: Unique identifier for the worker
            capacity: Worker capacity (default 1.0)
            
        Returns:
            True if registered successfully
        """
        with self._worker_lock:
            if node_id in self._workers:
                return False  # Worker already exists
            
            self._workers[node_id] = WorkerNode(node_id=node_id, capacity=capacity)
            return True
    
    def unregister_worker(self, node_id: str) -> bool:
        """
        Unregister a worker node
        
        Args:
            node_id: Worker identifier
            
        Returns:
            True if unregistered successfully
        """
        with self._worker_lock:
            if node_id in self._workers:
                del self._workers[node_id]
                return True
            return False
    
    def select_worker(self, task_requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Select optimal worker for a task
        
        Args:
            task_requirements: Task requirements (load, capabilities, etc.)
            
        Returns:
            Selected worker node ID or None if no workers available
        """
        with self._worker_lock:
            healthy_workers = [w for w in self._workers.values() if w.is_healthy]
            
            if not healthy_workers:
                return None
            
            # Use configured algorithm
            algorithm = self._algorithms.get(self._current_algorithm, self._adaptive_selection)
            selected_worker = algorithm(healthy_workers, task_requirements)
            
            return selected_worker.node_id if selected_worker else None
    
    def assign_task(self, node_id: str, load: float = 1.0) -> bool:
        """
        Assign task to specific worker
        
        Args:
            node_id: Worker identifier
            load: Task load weight
            
        Returns:
            True if assigned successfully
        """
        with self._worker_lock:
            worker = self._workers.get(node_id)
            if worker and worker.assign_load(load):
                # Record request
                self._request_history.append({
                    "timestamp": time.time(),
                    "worker_id": node_id,
                    "load": load
                })
                return True
            return False
    
    def complete_task(self, node_id: str, load: float = 1.0, processing_time: float = 0.0):
        """
        Mark task as completed on worker
        
        Args:
            node_id: Worker identifier
            load: Task load weight
            processing_time: Time taken to process task
        """
        with self._worker_lock:
            worker = self._workers.get(node_id)
            if worker:
                worker.release_load(load)
                worker.complete_task(processing_time)
    
    def update_worker_heartbeat(self, node_id: str, health_data: Optional[Dict[str, Any]] = None):
        """
        Update worker heartbeat and health data
        
        Args:
            node_id: Worker identifier
            health_data: Optional health metrics
        """
        with self._worker_lock:
            worker = self._workers.get(node_id)
            if worker:
                worker.update_heartbeat()
                
                # Update health based on health_data if provided
                if health_data:
                    worker.is_healthy = health_data.get("healthy", True)
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across workers"""
        with self._worker_lock:
            distribution = {}
            
            for node_id, worker in self._workers.items():
                distribution[node_id] = {
                    "current_load": worker.current_load,
                    "capacity": worker.capacity,
                    "utilization": worker.utilization,
                    "is_healthy": worker.is_healthy,
                    "tasks_completed": worker.total_tasks_completed,
                    "average_task_time": worker.average_task_time
                }
            
            # Calculate overall statistics
            total_load = sum(w.current_load for w in self._workers.values())
            total_capacity = sum(w.capacity for w in self._workers.values() if w.is_healthy)
            
            distribution["overall"] = {
                "total_workers": len(self._workers),
                "healthy_workers": sum(1 for w in self._workers.values() if w.is_healthy),
                "total_load": total_load,
                "total_capacity": total_capacity,
                "overall_utilization": (total_load / total_capacity * 100) if total_capacity > 0 else 0,
                "algorithm": self._current_algorithm
            }
            
            return distribution
    
    def _round_robin(self, workers: List[WorkerNode], 
                    task_requirements: Optional[Dict[str, Any]]) -> Optional[WorkerNode]:
        """Round-robin worker selection"""
        if not workers:
            return None
        
        self._round_robin_index = (self._round_robin_index + 1) % len(workers)
        return workers[self._round_robin_index]
    
    def _least_connections(self, workers: List[WorkerNode], 
                          task_requirements: Optional[Dict[str, Any]]) -> Optional[WorkerNode]:
        """Select worker with least current load"""
        return min(workers, key=lambda w: w.current_load)
    
    def _weighted_round_robin(self, workers: List[WorkerNode], 
                             task_requirements: Optional[Dict[str, Any]]) -> Optional[WorkerNode]:
        """Weighted round-robin based on capacity"""
        if not workers:
            return None
        
        # Calculate weights based on available capacity
        weights = []
        for worker in workers:
            available_capacity = worker.capacity - worker.current_load
            weights.append(max(0.1, available_capacity))  # Minimum weight 0.1
        
        # Weighted selection
        total_weight = sum(weights)
        if total_weight == 0:
            return workers[0]  # Fallback
        
        import random
        random_value = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return workers[i]
        
        return workers[-1]  # Fallback
    
    def _least_response_time(self, workers: List[WorkerNode], 
                            task_requirements: Optional[Dict[str, Any]]) -> Optional[WorkerNode]:
        """Select worker with least average response time"""
        # Filter workers with task history
        workers_with_history = [w for w in workers if w.total_tasks_completed > 0]
        
        if not workers_with_history:
            # Fallback to least connections
            return self._least_connections(workers, task_requirements)
        
        return min(workers_with_history, key=lambda w: w.average_task_time)
    
    def _adaptive_selection(self, workers: List[WorkerNode], 
                           task_requirements: Optional[Dict[str, Any]]) -> Optional[WorkerNode]:
        """Adaptive selection combining multiple factors"""
        if not workers:
            return None
        
        # Calculate composite score for each worker
        best_worker = None
        best_score = float('inf')
        
        for worker in workers:
            # Load factor (lower is better)
            load_factor = worker.utilization / 100.0
            
            # Response time factor (lower is better)
            response_time_factor = (
                worker.average_task_time / 10.0 if worker.total_tasks_completed > 0 else 0.5
            )
            
            # Availability factor (available capacity)
            availability_factor = max(0, worker.capacity - worker.current_load) / worker.capacity
            
            # Combine factors (lower total score is better)
            score = (load_factor * 0.4 + 
                    response_time_factor * 0.3 + 
                    (1.0 - availability_factor) * 0.3)
            
            if score < best_score:
                best_score = score
                best_worker = worker
        
        return best_worker
    
    def _background_tasks(self):
        """Background thread for health checks and rebalancing"""
        while self._running:
            try:
                current_time = time.time()
                
                # Health check
                if current_time - self._last_rebalance > self._health_check_interval:
                    self._check_worker_health()
                
                # Rebalancing
                if current_time - self._last_rebalance > self.rebalance_interval:
                    self._rebalance_if_needed()
                    self._last_rebalance = current_time
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception:
                pass  # Don't let background thread crash
    
    def _check_worker_health(self):
        """Check health of all workers"""
        current_time = time.time()
        
        with self._worker_lock:
            for worker in self._workers.values():
                # Mark as unhealthy if no heartbeat for too long
                if (current_time - worker.last_heartbeat) > self._unhealthy_threshold:
                    worker.is_healthy = False
    
    def _rebalance_if_needed(self):
        """Rebalance load if imbalance detected"""
        with self._worker_lock:
            healthy_workers = [w for w in self._workers.values() if w.is_healthy]
            
            if len(healthy_workers) < 2:
                return  # Can't rebalance with fewer than 2 workers
            
            # Calculate load imbalance
            utilizations = [w.utilization for w in healthy_workers]
            avg_utilization = statistics.mean(utilizations)
            max_utilization = max(utilizations)
            min_utilization = min(utilizations)
            
            # Rebalance if imbalance is significant
            imbalance_threshold = 30.0  # 30% difference
            if (max_utilization - min_utilization) > imbalance_threshold:
                # Switch to more balanced algorithm temporarily
                if self._current_algorithm == "round_robin":
                    self._current_algorithm = "adaptive"
    
    def set_algorithm(self, algorithm: str):
        """Set load balancing algorithm"""
        if algorithm in self._algorithms:
            self._current_algorithm = algorithm
    
    def shutdown(self):
        """Shutdown load balancer"""
        self._running = False
        if self._background_thread.is_alive():
            self._background_thread.join(timeout=5.0)


class QuantumAutoScaler:
    """
    Auto-scaling system for quantum workers
    """
    
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = 20,
                 target_utilization: float = 70.0,
                 scale_cooldown: float = 300.0):  # 5 minutes
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_cooldown = scale_cooldown
        
        # Load balancer integration
        self.load_balancer = QuantumLoadBalancer()
        
        # Scaling history
        self._scaling_history: collections.deque = collections.deque(maxlen=100)
        self._last_scale_action = 0.0
        
        # Performance monitoring
        self.performance_monitor = QuantumPerformanceMonitor()
        
        # Worker factory
        self._worker_factory: Optional[Callable] = None
        self._worker_destroyer: Optional[Callable] = None
        
        # Scaling lock
        self._scaling_lock = threading.RLock()
        
        # Auto-scaling thread
        self._auto_scale_enabled = True
        self._auto_scale_thread = threading.Thread(target=self._auto_scale_loop, daemon=True)
        self._auto_scale_thread.start()
    
    def set_worker_factory(self, factory: Callable, destroyer: Callable):
        """
        Set worker factory and destroyer functions
        
        Args:
            factory: Function to create new workers
            destroyer: Function to destroy workers
        """
        self._worker_factory = factory
        self._worker_destroyer = destroyer
    
    def evaluate_scaling_need(self) -> Tuple[str, int]:
        """
        Evaluate if scaling is needed
        
        Returns:
            Tuple of (action, count) where action is 'scale_up', 'scale_down', or 'maintain'
        """
        with self._scaling_lock:
            load_dist = self.load_balancer.get_load_distribution()
            overall = load_dist.get("overall", {})
            
            current_workers = overall.get("healthy_workers", 0)
            current_utilization = overall.get("overall_utilization", 0)
            
            # Don't scale if in cooldown period
            if time.time() - self._last_scale_action < self.scale_cooldown:
                return "maintain", 0
            
            # Scale up conditions
            if (current_utilization > self.target_utilization + 20 and 
                current_workers < self.max_workers):
                # Calculate how many workers to add (aggressive scaling for high load)
                if current_utilization > 90:
                    scale_count = min(3, self.max_workers - current_workers)
                else:
                    scale_count = min(2, self.max_workers - current_workers)
                return "scale_up", scale_count
            
            # Scale down conditions
            elif (current_utilization < self.target_utilization - 20 and 
                  current_workers > self.min_workers):
                # Conservative scale down (one at a time)
                return "scale_down", 1
            
            return "maintain", 0
    
    def scale_up(self, count: int) -> int:
        """
        Scale up by adding workers
        
        Args:
            count: Number of workers to add
            
        Returns:
            Number of workers actually added
        """
        if not self._worker_factory:
            return 0
        
        with self._scaling_lock:
            current_count = len(self.load_balancer._workers)
            max_add = min(count, self.max_workers - current_count)
            
            added = 0
            for i in range(max_add):
                try:
                    # Create new worker
                    worker_id = f"quantum_worker_{current_count + i + 1}_{int(time.time())}"
                    
                    # Use factory to create worker
                    worker_instance = self._worker_factory()
                    
                    # Register with load balancer
                    if self.load_balancer.register_worker(worker_id, capacity=1.0):
                        added += 1
                        
                        # Record scaling action
                        self._scaling_history.append({
                            "action": "scale_up",
                            "timestamp": time.time(),
                            "worker_id": worker_id,
                            "reason": "high_utilization"
                        })
                        
                except Exception as e:
                    # Log error but continue with other workers
                    pass
            
            if added > 0:
                self._last_scale_action = time.time()
            
            return added
    
    def scale_down(self, count: int) -> int:
        """
        Scale down by removing workers
        
        Args:
            count: Number of workers to remove
            
        Returns:
            Number of workers actually removed
        """
        if not self._worker_destroyer:
            return 0
        
        with self._scaling_lock:
            current_workers = len(self.load_balancer._workers)
            max_remove = min(count, current_workers - self.min_workers)
            
            if max_remove <= 0:
                return 0
            
            # Select workers to remove (prefer least loaded)
            workers_by_load = sorted(
                self.load_balancer._workers.items(),
                key=lambda x: x[1].current_load
            )
            
            removed = 0
            for worker_id, worker in workers_by_load[:max_remove]:
                try:
                    # Only remove if worker has low load
                    if worker.current_load < 0.1:
                        # Unregister from load balancer
                        if self.load_balancer.unregister_worker(worker_id):
                            # Use destroyer to clean up worker
                            self._worker_destroyer(worker_id)
                            removed += 1
                            
                            # Record scaling action
                            self._scaling_history.append({
                                "action": "scale_down",
                                "timestamp": time.time(),
                                "worker_id": worker_id,
                                "reason": "low_utilization"
                            })
                            
                except Exception:
                    pass
            
            if removed > 0:
                self._last_scale_action = time.time()
            
            return removed
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        with self._scaling_lock:
            scale_ups = sum(1 for h in self._scaling_history if h["action"] == "scale_up")
            scale_downs = sum(1 for h in self._scaling_history if h["action"] == "scale_down")
            
            recent_actions = [
                h for h in self._scaling_history 
                if time.time() - h["timestamp"] < 3600  # Last hour
            ]
            
            return {
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "target_utilization": self.target_utilization,
                "current_workers": len(self.load_balancer._workers),
                "total_scale_ups": scale_ups,
                "total_scale_downs": scale_downs,
                "recent_actions": len(recent_actions),
                "last_scale_action": self._last_scale_action,
                "cooldown_remaining": max(0, self.scale_cooldown - (time.time() - self._last_scale_action)),
                "auto_scale_enabled": self._auto_scale_enabled
            }
    
    def _auto_scale_loop(self):
        """Background auto-scaling loop"""
        while self._auto_scale_enabled:
            try:
                action, count = self.evaluate_scaling_need()
                
                if action == "scale_up" and count > 0:
                    self.scale_up(count)
                elif action == "scale_down" and count > 0:
                    self.scale_down(count)
                
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception:
                pass  # Don't let auto-scaling crash
    
    def enable_auto_scaling(self):
        """Enable automatic scaling"""
        self._auto_scale_enabled = True
    
    def disable_auto_scaling(self):
        """Disable automatic scaling"""
        self._auto_scale_enabled = False
    
    def shutdown(self):
        """Shutdown auto-scaler"""
        self._auto_scale_enabled = False
        self.load_balancer.shutdown()


# Global scaling system
quantum_auto_scaler = QuantumAutoScaler()