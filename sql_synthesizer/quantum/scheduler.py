"""
Quantum-Inspired Task Scheduler

Implements quantum computing principles for distributed task scheduling
and resource optimization in the SQL synthesizer system.
"""

import asyncio
import heapq
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .core import QuantumState


class TaskPriority(Enum):
    """Task priority levels using quantum energy states"""

    GROUND_STATE = 0  # Highest priority (lowest energy)
    EXCITED_1 = 1  # High priority
    EXCITED_2 = 2  # Medium priority
    EXCITED_3 = 3  # Low priority
    IONIZED = 4  # Lowest priority (highest energy)


@dataclass
class QuantumTask:
    """Represents a task in quantum superposition"""

    id: str
    priority: TaskPriority
    execution_time: float
    dependencies: List[str] = field(default_factory=list)
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    probability: float = 1.0
    entangled_with: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __lt__(self, other):
        """For priority queue ordering"""
        return (self.priority.value, self.created_at) < (
            other.priority.value,
            other.created_at,
        )


class QuantumResource:
    """Represents a computational resource in quantum superposition"""

    def __init__(self, resource_id: str, capacity: float = 1.0):
        self.resource_id = resource_id
        self.capacity = capacity
        self.utilization = 0.0
        self.quantum_state = QuantumState.SUPERPOSITION
        self.entangled_tasks: List[str] = []
        self.coherence_time = 100.0  # Time before decoherence
        self.last_measurement = time.time()

    def is_available(self, required_capacity: float = 1.0) -> bool:
        """Check if resource can handle the required capacity"""
        return (self.capacity - self.utilization) >= required_capacity

    def allocate(self, capacity: float, task_id: str):
        """Allocate resource capacity to a task"""
        if self.is_available(capacity):
            self.utilization += capacity
            self.entangled_tasks.append(task_id)
            return True
        return False

    def release(self, capacity: float, task_id: str):
        """Release resource capacity from a task"""
        self.utilization = max(0, self.utilization - capacity)
        if task_id in self.entangled_tasks:
            self.entangled_tasks.remove(task_id)

    def measure_state(self) -> float:
        """Collapse quantum state to get actual utilization"""
        current_time = time.time()
        decoherence = (current_time - self.last_measurement) / self.coherence_time

        if decoherence > 1.0:
            # Complete decoherence - classical behavior
            self.quantum_state = QuantumState.COLLAPSED
            return self.utilization

        # Quantum fluctuation based on coherence
        fluctuation = 0.1 * (1 - decoherence) * random.uniform(-1, 1)
        measured_utilization = max(0, min(1, self.utilization + fluctuation))

        self.last_measurement = current_time
        return measured_utilization


class QuantumTaskScheduler:
    """
    Quantum-inspired task scheduler using superposition and entanglement
    """

    def __init__(self, num_resources: int = 4, quantum_coherence_time: float = 100.0):
        self.resources = [
            QuantumResource(f"quantum_worker_{i}", capacity=1.0)
            for i in range(num_resources)
        ]
        self.task_queue = []
        self.running_tasks: Dict[str, QuantumTask] = {}
        self.completed_tasks: Dict[str, QuantumTask] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.quantum_coherence_time = quantum_coherence_time
        self.executor = ThreadPoolExecutor(max_workers=num_resources)
        self.scheduler_running = False

    async def submit_task(self, task: QuantumTask) -> str:
        """Submit a task to the quantum scheduler"""
        # Create superposition of possible execution paths
        task.quantum_state = QuantumState.SUPERPOSITION
        task.probability = self._calculate_initial_probability(task)

        # Add to dependency graph
        self.dependency_graph[task.id] = task.dependencies.copy()

        # Apply quantum interference for priority adjustment
        self._apply_quantum_interference(task)

        heapq.heappush(self.task_queue, task)

        if not self.scheduler_running:
            asyncio.create_task(self._quantum_scheduler_loop())

        return task.id

    def _calculate_initial_probability(self, task: QuantumTask) -> float:
        """Calculate initial quantum probability based on task characteristics"""
        # Higher probability for higher priority tasks
        priority_factor = (5 - task.priority.value) / 5.0

        # Consider execution time (shorter tasks have slightly higher probability)
        time_factor = 1.0 / (1.0 + task.execution_time / 10.0)

        # Dependency factor (fewer deps = higher probability)
        dep_factor = 1.0 / (1.0 + len(task.dependencies))

        return priority_factor * 0.6 + time_factor * 0.2 + dep_factor * 0.2

    def _apply_quantum_interference(self, new_task: QuantumTask):
        """Apply quantum interference between tasks"""
        for existing_task in self.task_queue:
            if existing_task.priority == new_task.priority:
                # Constructive interference for same priority
                phase_diff = (
                    abs(hash(existing_task.id) - hash(new_task.id)) % 100 / 100.0
                )
                interference = math.cos(2 * math.pi * phase_diff) ** 2

                # Amplify probabilities
                existing_task.probability *= 1 + 0.1 * interference
                new_task.probability *= 1 + 0.1 * interference

    async def _quantum_scheduler_loop(self):
        """Main quantum scheduling loop"""
        self.scheduler_running = True

        try:
            while self.task_queue or self.running_tasks:
                await self._quantum_schedule_step()
                await asyncio.sleep(0.1)  # Allow other coroutines to run
        finally:
            self.scheduler_running = False

    async def _quantum_schedule_step(self):
        """Single step of quantum scheduling"""
        # Measure quantum states and collapse superpositions
        available_resources = self._measure_resource_states()

        # Find ready tasks (dependencies satisfied)
        ready_tasks = self._find_ready_tasks()

        if not ready_tasks or not available_resources:
            return

        # Apply quantum tunneling for deadlock prevention
        tunneled_tasks = self._apply_quantum_tunneling(ready_tasks)

        # Schedule tasks using quantum annealing
        scheduled_pairs = self._quantum_annealing_schedule(
            tunneled_tasks, available_resources
        )

        # Execute scheduled tasks
        for task, resource in scheduled_pairs:
            await self._execute_task(task, resource)

    def _measure_resource_states(self) -> List[QuantumResource]:
        """Measure quantum resource states"""
        available = []
        for resource in self.resources:
            measured_util = resource.measure_state()
            if resource.is_available():
                available.append(resource)
        return available

    def _find_ready_tasks(self) -> List[QuantumTask]:
        """Find tasks with satisfied dependencies"""
        ready = []
        temp_queue = []

        while self.task_queue:
            task = heapq.heappop(self.task_queue)

            # Check if dependencies are satisfied
            deps_satisfied = all(
                dep_id in self.completed_tasks for dep_id in task.dependencies
            )

            if deps_satisfied:
                ready.append(task)
            else:
                temp_queue.append(task)

        # Restore non-ready tasks to queue
        for task in temp_queue:
            heapq.heappush(self.task_queue, task)

        return ready

    def _apply_quantum_tunneling(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Apply quantum tunneling to escape scheduling constraints"""
        tunneled = []

        for task in tasks:
            # Quantum tunneling probability
            tunnel_prob = math.exp(-task.priority.value / 2.0)

            if random.random() < tunnel_prob:
                # Tunneling: temporarily boost priority
                original_priority = task.priority
                if task.priority.value > 0:
                    task.priority = TaskPriority(task.priority.value - 1)
                tunneled.append(task)
                # Restore original priority after scheduling
                task.priority = original_priority
            else:
                tunneled.append(task)

        return tunneled

    def _quantum_annealing_schedule(
        self, tasks: List[QuantumTask], resources: List[QuantumResource]
    ) -> List[Tuple[QuantumTask, QuantumResource]]:
        """Use quantum annealing to find optimal task-resource assignments"""
        if not tasks or not resources:
            return []

        assignments = []
        temperature = 100.0
        cooling_rate = 0.9

        # Initial random assignment
        available_resources = resources.copy()
        remaining_tasks = tasks.copy()

        for _ in range(min(100, len(tasks) * len(resources))):
            if not remaining_tasks or not available_resources:
                break

            # Energy-based selection (lower energy = better fit)
            best_energy = float("inf")
            best_assignment = None

            for task in remaining_tasks[:3]:  # Limit search space
                for resource in available_resources[:3]:
                    energy = self._calculate_assignment_energy(task, resource)

                    # Quantum acceptance with temperature
                    if energy < best_energy or random.random() < math.exp(
                        -energy / temperature
                    ):
                        best_energy = energy
                        best_assignment = (task, resource)

            if best_assignment:
                task, resource = best_assignment
                assignments.append((task, resource))
                remaining_tasks.remove(task)
                if not resource.is_available():
                    available_resources.remove(resource)

            temperature *= cooling_rate

        return assignments

    def _calculate_assignment_energy(
        self, task: QuantumTask, resource: QuantumResource
    ) -> float:
        """Calculate energy of task-resource assignment"""
        # Priority energy (higher priority = lower energy)
        priority_energy = task.priority.value * 10

        # Resource utilization energy
        util_energy = resource.utilization * 20

        # Entanglement energy (prefer entangled resources for related tasks)
        entanglement_bonus = 0
        if any(
            entangled_task in resource.entangled_tasks
            for entangled_task in task.entangled_with
        ):
            entanglement_bonus = -15

        # Quantum fluctuation
        quantum_noise = random.uniform(-2, 2)

        return priority_energy + util_energy + entanglement_bonus + quantum_noise

    async def _execute_task(self, task: QuantumTask, resource: QuantumResource):
        """Execute a task on a quantum resource"""
        # Collapse quantum state
        task.quantum_state = QuantumState.COLLAPSED
        task.scheduled_at = time.time()

        # Allocate resource
        resource.allocate(1.0, task.id)
        self.running_tasks[task.id] = task

        # Simulate async execution
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                self.executor, self._simulate_task_execution, task
            )

            # Task completed successfully
            task.completed_at = time.time()
            self.completed_tasks[task.id] = task

        except Exception:
            # Task failed - return to queue with lower priority
            task.priority = TaskPriority(min(4, task.priority.value + 1))
            task.quantum_state = QuantumState.SUPERPOSITION
            heapq.heappush(self.task_queue, task)

        finally:
            # Release resource
            resource.release(1.0, task.id)
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]

    def _simulate_task_execution(self, task: QuantumTask):
        """Simulate task execution with quantum effects"""
        # Base execution time with quantum uncertainty
        uncertainty = random.uniform(0.8, 1.2)
        actual_time = task.execution_time * uncertainty

        time.sleep(actual_time / 1000.0)  # Convert to seconds for simulation

    def entangle_tasks(self, task_id1: str, task_id2: str):
        """Create quantum entanglement between tasks"""
        # Find tasks in queue or running
        task1 = None
        task2 = None

        for task in self.task_queue:
            if task.id == task_id1:
                task1 = task
            elif task.id == task_id2:
                task2 = task

        if task_id1 in self.running_tasks:
            task1 = self.running_tasks[task_id1]
        if task_id2 in self.running_tasks:
            task2 = self.running_tasks[task_id2]

        if task1 and task2:
            task1.entangled_with.append(task_id2)
            task2.entangled_with.append(task_id1)
            task1.quantum_state = QuantumState.ENTANGLED
            task2.quantum_state = QuantumState.ENTANGLED

    def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum scheduling metrics"""
        total_tasks = (
            len(self.task_queue) + len(self.running_tasks) + len(self.completed_tasks)
        )

        quantum_states = {"superposition": 0, "entangled": 0, "collapsed": 0}

        all_tasks = (
            list(self.task_queue)
            + list(self.running_tasks.values())
            + list(self.completed_tasks.values())
        )
        for task in all_tasks:
            if task.quantum_state == QuantumState.SUPERPOSITION:
                quantum_states["superposition"] += 1
            elif task.quantum_state == QuantumState.ENTANGLED:
                quantum_states["entangled"] += 1
            else:
                quantum_states["collapsed"] += 1

        resource_utilization = sum(r.utilization for r in self.resources) / len(
            self.resources
        )

        return {
            "total_tasks": total_tasks,
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "quantum_states": quantum_states,
            "average_resource_utilization": resource_utilization,
            "quantum_coherence": sum(
                1
                for r in self.resources
                if r.quantum_state == QuantumState.SUPERPOSITION
            )
            / len(self.resources),
        }

    async def shutdown(self):
        """Gracefully shutdown the quantum scheduler"""
        self.scheduler_running = False
        self.executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
