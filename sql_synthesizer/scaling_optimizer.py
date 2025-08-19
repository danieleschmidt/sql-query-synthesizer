"""Quantum-Inspired Scaling Optimizer for SQL Query Synthesizer.

This module implements advanced scaling optimization with quantum-inspired algorithms
for predictive resource allocation, intelligent load balancing, and autonomous
performance optimization.
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for different scenarios."""

    REACTIVE = "reactive"  # React to current load
    PREDICTIVE = "predictive"  # Predict future load
    QUANTUM_INSPIRED = "quantum_inspired"  # Quantum superposition optimization
    HYBRID = "hybrid"  # Combination of strategies


class ResourcePriority(Enum):
    """Priority levels for resource allocation."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class ScalingConfiguration:
    """Configuration for scaling optimization."""

    # Scaling thresholds
    cpu_scale_up_threshold: float = 70.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 40.0
    queue_scale_up_threshold: int = 50
    queue_scale_down_threshold: int = 10

    # Response time thresholds
    response_time_p95_threshold: float = 1000.0  # ms
    response_time_p99_threshold: float = 5000.0  # ms

    # Scaling factors
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.8
    max_scale_factor: float = 10.0
    min_scale_factor: float = 0.1

    # Cooldown periods
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes

    # Prediction settings
    prediction_window: int = 300  # 5 minutes
    historical_data_points: int = 1000
    confidence_threshold: float = 0.75

    # Quantum-inspired settings
    quantum_coherence_time: float = 30.0
    entanglement_strength: float = 0.8
    superposition_states: int = 8


@dataclass
class ResourceState:
    """Current state of a resource."""

    resource_id: str
    current_capacity: float
    target_capacity: float
    utilization: float
    last_scaled: float
    scaling_trend: str = "stable"
    prediction_confidence: float = 0.0
    quantum_state: Optional[Dict[str, float]] = None


@dataclass
class ScalingDecision:
    """A scaling decision with reasoning."""

    resource_id: str
    action: str  # "scale_up", "scale_down", "maintain"
    current_capacity: float
    target_capacity: float
    reasoning: str
    confidence: float
    priority: ResourcePriority
    estimated_impact: Dict[str, float]
    quantum_probability: float = 0.0


class QuantumScalingOptimizer:
    """Quantum-inspired scaling optimizer using superposition and entanglement."""

    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self._resource_states: Dict[str, ResourceState] = {}
        self._scaling_history: deque = deque(maxlen=config.historical_data_points)
        self._last_optimization = 0.0
        self._quantum_coherence_start = time.time()
        self._entangled_resources: Dict[str, List[str]] = defaultdict(list)
        self._performance_metrics: deque = deque(maxlen=1000)

    def register_resource(self, resource_id: str, initial_capacity: float):
        """Register a resource for optimization."""
        self._resource_states[resource_id] = ResourceState(
            resource_id=resource_id,
            current_capacity=initial_capacity,
            target_capacity=initial_capacity,
            utilization=0.0,
            last_scaled=time.time(),
            quantum_state=self._initialize_quantum_state(),
        )

    def _initialize_quantum_state(self) -> Dict[str, float]:
        """Initialize quantum state for a resource."""
        # Create superposition of possible scaling states
        states = {}
        for i in range(self.config.superposition_states):
            # Each state represents a different scaling level
            scale_factor = 0.5 + (i / (self.config.superposition_states - 1)) * 1.5
            states[f"state_{i}"] = 1.0 / self.config.superposition_states
        return states

    def update_metrics(
        self,
        resource_id: str,
        utilization: float,
        response_time_p95: float,
        queue_length: int,
        error_rate: float,
    ):
        """Update performance metrics for a resource."""
        if resource_id not in self._resource_states:
            self.register_resource(resource_id, 1.0)

        state = self._resource_states[resource_id]
        state.utilization = utilization

        # Store performance metrics
        metrics = {
            "timestamp": time.time(),
            "resource_id": resource_id,
            "utilization": utilization,
            "response_time_p95": response_time_p95,
            "queue_length": queue_length,
            "error_rate": error_rate,
        }
        self._performance_metrics.append(metrics)

        # Update quantum state based on performance
        self._update_quantum_state(resource_id, metrics)

    def _update_quantum_state(self, resource_id: str, metrics: Dict[str, Any]):
        """Update quantum state based on performance metrics."""
        state = self._resource_states[resource_id]
        quantum_state = state.quantum_state or {}

        # Calculate quantum coherence (how well-defined the state is)
        coherence_time_elapsed = time.time() - self._quantum_coherence_start
        coherence = math.exp(
            -coherence_time_elapsed / self.config.quantum_coherence_time
        )

        # Update state probabilities based on performance
        utilization = metrics["utilization"]
        response_time = metrics["response_time_p95"]

        for state_key, probability in quantum_state.items():
            state_index = int(state_key.split("_")[1])
            scale_factor = (
                0.5 + (state_index / (self.config.superposition_states - 1)) * 1.5
            )

            # Calculate fitness for this scaling state
            fitness = self._calculate_state_fitness(
                scale_factor, utilization, response_time
            )

            # Update probability using quantum evolution
            quantum_state[state_key] = probability * fitness * coherence

        # Normalize probabilities
        total_probability = sum(quantum_state.values())
        if total_probability > 0:
            for state_key in quantum_state:
                quantum_state[state_key] /= total_probability

        state.quantum_state = quantum_state

    def _calculate_state_fitness(
        self, scale_factor: float, utilization: float, response_time: float
    ) -> float:
        """Calculate fitness of a quantum state based on performance."""
        # Ideal utilization is around 70%
        utilization_fitness = 1.0 - abs(utilization - 70.0) / 100.0

        # Lower response time is better
        response_time_fitness = max(
            0.0, 1.0 - response_time / self.config.response_time_p95_threshold
        )

        # Consider resource efficiency (avoid over-provisioning)
        efficiency_fitness = 1.0 / (1.0 + abs(scale_factor - 1.0))

        # Combine fitness components
        total_fitness = (
            utilization_fitness * 0.4
            + response_time_fitness * 0.4
            + efficiency_fitness * 0.2
        )

        return max(0.1, total_fitness)  # Ensure non-zero fitness

    def optimize_scaling(self) -> List[ScalingDecision]:
        """Perform quantum-inspired scaling optimization."""
        current_time = time.time()

        # Check if optimization is needed
        if current_time - self._last_optimization < 30.0:  # 30-second cooldown
            return []

        decisions = []

        for resource_id, state in self._resource_states.items():
            decision = self._make_scaling_decision(resource_id, state)
            if decision:
                decisions.append(decision)

        self._last_optimization = current_time
        return decisions

    def _make_scaling_decision(
        self, resource_id: str, state: ResourceState
    ) -> Optional[ScalingDecision]:
        """Make a scaling decision for a specific resource."""
        current_time = time.time()

        # Check cooldown periods
        time_since_scale = current_time - state.last_scaled
        if time_since_scale < self.config.scale_up_cooldown:
            return None

        # Get quantum state probabilities
        quantum_state = state.quantum_state or {}
        if not quantum_state:
            return None

        # Find the most probable scaling state
        best_state_key = max(quantum_state.keys(), key=lambda k: quantum_state[k])
        best_probability = quantum_state[best_state_key]
        state_index = int(best_state_key.split("_")[1])
        target_scale_factor = (
            0.5 + (state_index / (self.config.superposition_states - 1)) * 1.5
        )

        target_capacity = state.current_capacity * target_scale_factor

        # Apply bounds
        target_capacity = max(
            state.current_capacity * self.config.min_scale_factor,
            min(target_capacity, state.current_capacity * self.config.max_scale_factor),
        )

        # Determine action
        capacity_change = (
            target_capacity - state.current_capacity
        ) / state.current_capacity

        if abs(capacity_change) < 0.1:  # Less than 10% change
            action = "maintain"
        elif capacity_change > 0:
            action = "scale_up"
        else:
            action = "scale_down"

        # Calculate confidence and priority
        confidence = best_probability
        priority = self._determine_priority(state, capacity_change)

        # Estimate impact
        estimated_impact = self._estimate_scaling_impact(
            resource_id, state.current_capacity, target_capacity
        )

        # Generate reasoning
        reasoning = self._generate_scaling_reasoning(
            resource_id, action, state, confidence
        )

        return ScalingDecision(
            resource_id=resource_id,
            action=action,
            current_capacity=state.current_capacity,
            target_capacity=target_capacity,
            reasoning=reasoning,
            confidence=confidence,
            priority=priority,
            estimated_impact=estimated_impact,
            quantum_probability=best_probability,
        )

    def _determine_priority(
        self, state: ResourceState, capacity_change: float
    ) -> ResourcePriority:
        """Determine priority of scaling decision."""
        utilization = state.utilization

        if utilization > 90.0 or capacity_change > 0.5:
            return ResourcePriority.CRITICAL
        elif utilization > 80.0 or capacity_change > 0.3:
            return ResourcePriority.HIGH
        elif utilization > 60.0 or abs(capacity_change) > 0.2:
            return ResourcePriority.NORMAL
        elif abs(capacity_change) > 0.1:
            return ResourcePriority.LOW
        else:
            return ResourcePriority.BACKGROUND

    def _estimate_scaling_impact(
        self, resource_id: str, current_capacity: float, target_capacity: float
    ) -> Dict[str, float]:
        """Estimate the impact of scaling on system performance."""
        capacity_ratio = target_capacity / current_capacity

        # Estimate performance improvements
        estimated_response_time_improvement = (
            (capacity_ratio - 1.0) * 0.3 if capacity_ratio > 1.0 else 0.0
        )
        estimated_throughput_improvement = (
            (capacity_ratio - 1.0) * 0.5 if capacity_ratio > 1.0 else 0.0
        )
        estimated_resource_cost = (
            capacity_ratio - 1.0
        ) * 1.2  # Cost scales slightly higher

        return {
            "response_time_improvement": estimated_response_time_improvement,
            "throughput_improvement": estimated_throughput_improvement,
            "resource_cost_change": estimated_resource_cost,
            "capacity_change": capacity_ratio - 1.0,
        }

    def _generate_scaling_reasoning(
        self, resource_id: str, action: str, state: ResourceState, confidence: float
    ) -> str:
        """Generate human-readable reasoning for scaling decision."""
        utilization = state.utilization

        if action == "scale_up":
            if utilization > 80.0:
                return (
                    f"High utilization ({utilization:.1f}%) requires immediate scaling"
                )
            else:
                return f"Quantum analysis predicts performance degradation (confidence: {confidence:.2f})"
        elif action == "scale_down":
            if utilization < 30.0:
                return (
                    f"Low utilization ({utilization:.1f}%) allows resource optimization"
                )
            else:
                return f"Quantum optimization suggests capacity reduction (confidence: {confidence:.2f})"
        else:
            return f"Current capacity optimal for workload (utilization: {utilization:.1f}%)"

    def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Apply a scaling decision to the system."""
        if decision.resource_id not in self._resource_states:
            logger.error(f"Resource {decision.resource_id} not found")
            return False

        state = self._resource_states[decision.resource_id]

        # Update resource state
        state.current_capacity = decision.target_capacity
        state.target_capacity = decision.target_capacity
        state.last_scaled = time.time()
        state.scaling_trend = decision.action
        state.prediction_confidence = decision.confidence

        # Record scaling history
        self._scaling_history.append(
            {
                "timestamp": time.time(),
                "resource_id": decision.resource_id,
                "action": decision.action,
                "previous_capacity": decision.current_capacity,
                "new_capacity": decision.target_capacity,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
                "quantum_probability": decision.quantum_probability,
            }
        )

        logger.info(
            f"Scaling {decision.action} applied to {decision.resource_id}: "
            f"{decision.current_capacity:.2f} -> {decision.target_capacity:.2f} "
            f"(confidence: {decision.confidence:.2f})"
        )

        return True

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        current_time = time.time()

        # Calculate scaling statistics
        total_scalings = len(self._scaling_history)
        scale_up_count = sum(
            1 for h in self._scaling_history if h["action"] == "scale_up"
        )
        scale_down_count = sum(
            1 for h in self._scaling_history if h["action"] == "scale_down"
        )

        # Calculate average confidence
        avg_confidence = (
            sum(h["confidence"] for h in self._scaling_history) / total_scalings
            if total_scalings > 0
            else 0.0
        )

        # Calculate quantum coherence
        coherence_time_elapsed = current_time - self._quantum_coherence_start
        current_coherence = math.exp(
            -coherence_time_elapsed / self.config.quantum_coherence_time
        )

        # Resource utilization summary
        resource_summary = {}
        for resource_id, state in self._resource_states.items():
            resource_summary[resource_id] = {
                "current_capacity": state.current_capacity,
                "utilization": state.utilization,
                "scaling_trend": state.scaling_trend,
                "prediction_confidence": state.prediction_confidence,
                "quantum_entropy": self._calculate_quantum_entropy(state.quantum_state),
            }

        return {
            "total_scaling_actions": total_scalings,
            "scale_up_actions": scale_up_count,
            "scale_down_actions": scale_down_count,
            "average_confidence": avg_confidence,
            "quantum_coherence": current_coherence,
            "active_resources": len(self._resource_states),
            "resource_summary": resource_summary,
            "last_optimization": self._last_optimization,
            "optimization_interval": current_time - self._last_optimization,
        }

    def _calculate_quantum_entropy(
        self, quantum_state: Optional[Dict[str, float]]
    ) -> float:
        """Calculate quantum entropy of a state."""
        if not quantum_state:
            return 0.0

        entropy = 0.0
        for probability in quantum_state.values():
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def reset_quantum_coherence(self):
        """Reset quantum coherence timer."""
        self._quantum_coherence_start = time.time()
        logger.info("Quantum coherence reset - starting fresh optimization cycle")

    def entangle_resources(
        self, resource1: str, resource2: str, strength: float = None
    ):
        """Create quantum entanglement between resources for correlated scaling."""
        if strength is None:
            strength = self.config.entanglement_strength

        self._entangled_resources[resource1].append(resource2)
        self._entangled_resources[resource2].append(resource1)

        logger.info(
            f"Quantum entanglement created between {resource1} and {resource2} "
            f"(strength: {strength:.2f})"
        )


# Global optimizer instance
quantum_scaling_optimizer = QuantumScalingOptimizer(ScalingConfiguration())


def get_quantum_scaling_optimizer() -> QuantumScalingOptimizer:
    """Get the global quantum scaling optimizer instance."""
    return quantum_scaling_optimizer


def optimize_system_scaling(
    cpu_utilization: float,
    memory_utilization: float,
    queue_length: int,
    response_time_p95: float,
    error_rate: float = 0.0,
) -> List[ScalingDecision]:
    """Convenience function for system-wide scaling optimization."""
    optimizer = get_quantum_scaling_optimizer()

    # Register default resources if not already registered
    default_resources = [
        "database_connections",
        "worker_processes",
        "cache_size",
        "concurrent_queries",
    ]

    for resource_id in default_resources:
        if resource_id not in optimizer._resource_states:
            optimizer.register_resource(resource_id, 1.0)

        # Update metrics for each resource
        if resource_id == "database_connections":
            utilization = cpu_utilization  # DB connections often correlate with CPU
        elif resource_id == "worker_processes":
            utilization = max(cpu_utilization, memory_utilization)
        elif resource_id == "cache_size":
            utilization = memory_utilization
        else:  # concurrent_queries
            utilization = (queue_length / 100.0) * 100.0  # Convert to percentage

        optimizer.update_metrics(
            resource_id, utilization, response_time_p95, queue_length, error_rate
        )

    return optimizer.optimize_scaling()
