"""
Comprehensive tests for quantum-inspired optimization components
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from sql_synthesizer.quantum.core import (
    QuantumQueryOptimizer,
    QuantumQueryPlanGenerator,
    QuantumState,
    Qubit,
    QueryPlan,
)
from sql_synthesizer.quantum.integration import QuantumSQLSynthesizer
from sql_synthesizer.quantum.scheduler import (
    QuantumResource,
    QuantumTask,
    QuantumTaskScheduler,
    TaskPriority,
)


class TestQubit:
    """Test quantum bit functionality"""

    def test_qubit_initialization(self):
        """Test qubit starts in superposition"""
        qubit = Qubit()
        assert not qubit.measured
        assert qubit.value is None
        assert abs(abs(qubit.amplitude_0) ** 2 - 0.5) < 0.01
        assert abs(abs(qubit.amplitude_1) ** 2 - 0.5) < 0.01

    def test_qubit_measurement(self):
        """Test qubit measurement collapses state"""
        qubit = Qubit()
        value = qubit.measure()

        assert qubit.measured
        assert value in [0, 1]
        assert qubit.value == value

        # Second measurement should return same value
        assert qubit.measure() == value

    def test_qubit_reset(self):
        """Test qubit reset to superposition"""
        qubit = Qubit()
        qubit.measure()  # Collapse to classical state

        qubit.reset()

        assert not qubit.measured
        assert qubit.value is None


class TestQueryPlan:
    """Test quantum query plan functionality"""

    def test_query_plan_creation(self):
        """Test query plan initialization"""
        plan = QueryPlan(
            joins=[("users", "orders")],
            filters=[{"column": "id", "value": 1}],
            aggregations=["count"],
            cost=10.5,
            probability=0.8,
        )

        assert len(plan.joins) == 1
        assert plan.joins[0] == ("users", "orders")
        assert plan.cost == 10.5
        assert plan.probability == 0.8
        assert plan.quantum_state == QuantumState.SUPERPOSITION


class TestQuantumQueryOptimizer:
    """Test quantum query optimizer"""

    def test_optimizer_initialization(self):
        """Test optimizer setup"""
        optimizer = QuantumQueryOptimizer(num_qubits=8, temperature=500.0)

        assert len(optimizer.qubits) == 8
        assert optimizer.temperature == 500.0
        assert optimizer.cooling_rate == 0.95

    def test_create_superposition(self):
        """Test creating quantum superposition of plans"""
        optimizer = QuantumQueryOptimizer()
        plans = [
            QueryPlan([], [], [], 10.0, 0.0),
            QueryPlan([], [], [], 20.0, 0.0),
            QueryPlan([], [], [], 30.0, 0.0),
        ]

        superposition_plans = optimizer.create_superposition(plans)

        assert len(superposition_plans) == 3
        for plan in superposition_plans:
            assert abs(plan.probability - 1 / 3) < 0.01
            assert plan.quantum_state == QuantumState.SUPERPOSITION

    def test_quantum_interference(self):
        """Test quantum interference effects"""
        optimizer = QuantumQueryOptimizer()
        plans = [
            QueryPlan(
                [], [], [], 10.0, 0.5
            ),  # Low cost - should get higher probability
            QueryPlan(
                [], [], [], 50.0, 0.5
            ),  # High cost - should get lower probability
        ]

        interfered_plans = optimizer.quantum_interference(plans)

        # Lower cost plan should have higher probability after interference
        low_cost_prob = next(p.probability for p in interfered_plans if p.cost == 10.0)
        high_cost_prob = next(p.probability for p in interfered_plans if p.cost == 50.0)

        assert low_cost_prob > high_cost_prob

    def test_quantum_annealing(self):
        """Test quantum annealing optimization"""
        optimizer = QuantumQueryOptimizer()
        plans = [
            QueryPlan([], [], [], 10.0, 0.3),
            QueryPlan([], [], [], 5.0, 0.7),  # Best plan
            QueryPlan([], [], [], 20.0, 0.2),
        ]

        optimal_plan = optimizer.quantum_annealing(plans, iterations=100)

        assert optimal_plan.quantum_state == QuantumState.COLLAPSED
        # Should tend to select the lowest cost plan
        assert optimal_plan.cost <= min(p.cost for p in plans)

    @pytest.mark.asyncio
    async def test_optimize_query_async(self):
        """Test async quantum optimization"""
        optimizer = QuantumQueryOptimizer()
        plans = [
            QueryPlan([], [], [], 15.0, 0.0),
            QueryPlan([], [], [], 8.0, 0.0),
            QueryPlan([], [], [], 25.0, 0.0),
        ]

        optimal_plan = await optimizer.optimize_query_async(plans)

        assert optimal_plan.quantum_state == QuantumState.COLLAPSED
        assert optimal_plan in plans

    def test_entangle_queries(self):
        """Test quantum entanglement between plans"""
        optimizer = QuantumQueryOptimizer()
        plan1 = QueryPlan([], [], [], 10.0, 0.5)
        plan2 = QueryPlan([], [], [], 20.0, 0.5)

        entangled_1, entangled_2 = optimizer.entangle_queries(plan1, plan2)

        assert entangled_1.quantum_state == QuantumState.ENTANGLED
        assert entangled_2.quantum_state == QuantumState.ENTANGLED
        assert entangled_1.probability + entangled_2.probability == 1.0

    def test_measure_optimization(self):
        """Test measurement collapse"""
        optimizer = QuantumQueryOptimizer()
        plans = [QueryPlan([], [], [], 10.0, 0.7), QueryPlan([], [], [], 20.0, 0.3)]

        measured_plan = optimizer.measure_optimization(plans)

        assert measured_plan.quantum_state == QuantumState.COLLAPSED
        assert measured_plan in plans

    def test_get_quantum_metrics(self):
        """Test quantum metrics collection"""
        optimizer = QuantumQueryOptimizer(num_qubits=10)

        # Measure some qubits
        optimizer.qubits[0].measure()
        optimizer.qubits[1].measure()

        metrics = optimizer.get_quantum_metrics()

        assert metrics["total_qubits"] == 10
        assert metrics["measured_qubits"] == 2
        assert metrics["superposition_qubits"] == 8
        assert metrics["quantum_coherence"] == 0.8

    def test_reset_quantum_state(self):
        """Test quantum state reset"""
        optimizer = QuantumQueryOptimizer()

        # Measure some qubits
        optimizer.qubits[0].measure()
        optimizer.temperature = 100.0

        optimizer.reset_quantum_state()

        assert not optimizer.qubits[0].measured
        assert optimizer.temperature == 1000.0


class TestQuantumQueryPlanGenerator:
    """Test quantum query plan generation"""

    def test_generator_initialization(self):
        """Test generator setup"""
        generator = QuantumQueryPlanGenerator()

        assert "table_scan" in generator.base_cost_factors
        assert "hash_join" in generator.base_cost_factors

    def test_generate_plans(self):
        """Test plan generation"""
        generator = QuantumQueryPlanGenerator()

        tables = ["users", "orders"]
        joins = [("users", "orders")]
        filters = [{"column": "status", "selectivity": 0.1}]
        aggregations = ["count"]

        plans = generator.generate_plans(tables, joins, filters, aggregations)

        assert len(plans) > 0
        for plan in plans:
            assert isinstance(plan, QueryPlan)
            assert plan.cost > 0
            assert plan.joins == joins
            assert len(plan.filters) > 0
            assert plan.aggregations == aggregations


class TestQuantumResource:
    """Test quantum resource management"""

    def test_resource_initialization(self):
        """Test resource setup"""
        resource = QuantumResource("worker_1", capacity=2.0)

        assert resource.resource_id == "worker_1"
        assert resource.capacity == 2.0
        assert resource.utilization == 0.0
        assert resource.quantum_state == QuantumState.SUPERPOSITION

    def test_resource_allocation(self):
        """Test resource allocation"""
        resource = QuantumResource("worker_1", capacity=2.0)

        success = resource.allocate(1.0, "task_1")
        assert success
        assert resource.utilization == 1.0
        assert "task_1" in resource.entangled_tasks

        # Should fail if over capacity
        success = resource.allocate(2.0, "task_2")
        assert not success

    def test_resource_release(self):
        """Test resource release"""
        resource = QuantumResource("worker_1", capacity=2.0)
        resource.allocate(1.5, "task_1")

        resource.release(1.0, "task_1")

        assert resource.utilization == 0.5
        assert "task_1" not in resource.entangled_tasks

    def test_measure_state(self):
        """Test quantum state measurement"""
        resource = QuantumResource("worker_1")
        resource.utilization = 0.5

        measured_util = resource.measure_state()

        # Should be close to actual utilization with some quantum fluctuation
        assert 0.0 <= measured_util <= 1.0


class TestQuantumTask:
    """Test quantum task functionality"""

    def test_task_creation(self):
        """Test task initialization"""
        task = QuantumTask(
            id="test_task",
            priority=TaskPriority.EXCITED_1,
            execution_time=100.0,
            dependencies=["dep1", "dep2"],
        )

        assert task.id == "test_task"
        assert task.priority == TaskPriority.EXCITED_1
        assert task.execution_time == 100.0
        assert task.dependencies == ["dep1", "dep2"]
        assert task.quantum_state == QuantumState.SUPERPOSITION

    def test_task_comparison(self):
        """Test task priority comparison"""
        task1 = QuantumTask("t1", TaskPriority.GROUND_STATE, 10.0, created_at=1.0)
        task2 = QuantumTask("t2", TaskPriority.EXCITED_1, 10.0, created_at=2.0)

        assert task1 < task2  # Higher priority (lower value) comes first


class TestQuantumTaskScheduler:
    """Test quantum task scheduler"""

    def test_scheduler_initialization(self):
        """Test scheduler setup"""
        scheduler = QuantumTaskScheduler(num_resources=3)

        assert len(scheduler.resources) == 3
        assert len(scheduler.task_queue) == 0
        assert len(scheduler.running_tasks) == 0

    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test task submission"""
        scheduler = QuantumTaskScheduler()
        task = QuantumTask("test_task", TaskPriority.EXCITED_1, 50.0)

        task_id = await scheduler.submit_task(task)

        assert task_id == "test_task"
        assert len(scheduler.task_queue) == 1
        assert task.quantum_state == QuantumState.SUPERPOSITION
        assert task.probability > 0

    def test_calculate_initial_probability(self):
        """Test initial probability calculation"""
        scheduler = QuantumTaskScheduler()

        high_priority_task = QuantumTask("hp", TaskPriority.GROUND_STATE, 10.0)
        low_priority_task = QuantumTask("lp", TaskPriority.IONIZED, 10.0)

        hp_prob = scheduler._calculate_initial_probability(high_priority_task)
        lp_prob = scheduler._calculate_initial_probability(low_priority_task)

        assert hp_prob > lp_prob

    def test_find_ready_tasks(self):
        """Test dependency resolution"""
        scheduler = QuantumTaskScheduler()

        # Add completed dependency
        completed_task = QuantumTask("completed", TaskPriority.GROUND_STATE, 10.0)
        scheduler.completed_tasks["completed"] = completed_task

        # Add ready task (no dependencies)
        ready_task = QuantumTask("ready", TaskPriority.EXCITED_1, 20.0)
        scheduler.task_queue.append(ready_task)

        # Add blocked task (has dependency)
        blocked_task = QuantumTask(
            "blocked", TaskPriority.EXCITED_1, 30.0, ["missing_dep"]
        )
        scheduler.task_queue.append(blocked_task)

        # Add unblocked task (dependency completed)
        unblocked_task = QuantumTask(
            "unblocked", TaskPriority.EXCITED_1, 40.0, ["completed"]
        )
        scheduler.task_queue.append(unblocked_task)

        ready_tasks = scheduler._find_ready_tasks()

        ready_ids = [task.id for task in ready_tasks]
        assert "ready" in ready_ids
        assert "unblocked" in ready_ids
        assert "blocked" not in ready_ids

    def test_calculate_assignment_energy(self):
        """Test energy calculation for task-resource assignments"""
        scheduler = QuantumTaskScheduler()

        high_priority_task = QuantumTask("hp", TaskPriority.GROUND_STATE, 10.0)
        low_priority_task = QuantumTask("lp", TaskPriority.IONIZED, 10.0)

        resource = QuantumResource("worker", capacity=1.0)

        hp_energy = scheduler._calculate_assignment_energy(high_priority_task, resource)
        lp_energy = scheduler._calculate_assignment_energy(low_priority_task, resource)

        assert hp_energy < lp_energy  # Higher priority = lower energy

    def test_entangle_tasks(self):
        """Test task entanglement"""
        scheduler = QuantumTaskScheduler()

        task1 = QuantumTask("t1", TaskPriority.EXCITED_1, 10.0)
        task2 = QuantumTask("t2", TaskPriority.EXCITED_1, 20.0)

        scheduler.task_queue.extend([task1, task2])

        scheduler.entangle_tasks("t1", "t2")

        assert "t2" in task1.entangled_with
        assert "t1" in task2.entangled_with
        assert task1.quantum_state == QuantumState.ENTANGLED
        assert task2.quantum_state == QuantumState.ENTANGLED

    def test_get_quantum_metrics(self):
        """Test quantum metrics collection"""
        scheduler = QuantumTaskScheduler()

        # Add tasks in different states
        scheduler.task_queue.append(QuantumTask("queued", TaskPriority.EXCITED_1, 10.0))

        running_task = QuantumTask("running", TaskPriority.EXCITED_1, 20.0)
        running_task.quantum_state = QuantumState.COLLAPSED
        scheduler.running_tasks["running"] = running_task

        completed_task = QuantumTask("completed", TaskPriority.EXCITED_1, 30.0)
        completed_task.quantum_state = QuantumState.COLLAPSED
        scheduler.completed_tasks["completed"] = completed_task

        metrics = scheduler.get_quantum_metrics()

        assert metrics["total_tasks"] == 3
        assert metrics["queued_tasks"] == 1
        assert metrics["running_tasks"] == 1
        assert metrics["completed_tasks"] == 1
        assert "quantum_states" in metrics
        assert "average_resource_utilization" in metrics

    @pytest.mark.asyncio
    async def test_scheduler_shutdown(self):
        """Test graceful shutdown"""
        scheduler = QuantumTaskScheduler()

        await scheduler.shutdown()

        assert not scheduler.scheduler_running


class TestQuantumSQLSynthesizer:
    """Test quantum SQL synthesizer integration"""

    def test_synthesizer_initialization(self):
        """Test synthesizer setup"""
        mock_base = Mock()
        synthesizer = QuantumSQLSynthesizer(mock_base, enable_quantum=True)

        assert synthesizer.base_synthesizer == mock_base
        assert synthesizer.enable_quantum
        assert synthesizer.quantum_optimizer is not None
        assert synthesizer.quantum_scheduler is not None

    def test_synthesizer_disabled_quantum(self):
        """Test synthesizer with quantum disabled"""
        mock_base = Mock()
        synthesizer = QuantumSQLSynthesizer(mock_base, enable_quantum=False)

        assert not synthesizer.enable_quantum
        assert synthesizer.quantum_optimizer is None
        assert synthesizer.quantum_scheduler is None

    @pytest.mark.asyncio
    async def test_query_with_quantum_disabled(self):
        """Test query execution with quantum disabled"""
        mock_base = AsyncMock()
        mock_result = Mock()
        mock_result.sql = "SELECT 1"
        mock_result.data = [{"result": 1}]
        mock_result.explanation = "Test query"
        mock_result.error = None
        mock_result.execution_time = 0.1
        mock_base.query.return_value = mock_result

        synthesizer = QuantumSQLSynthesizer(mock_base, enable_quantum=False)

        result = await synthesizer.query("test question")

        assert result.sql == "SELECT 1"
        assert result.quantum_metrics == {}
        assert result.quantum_cost_reduction == 0.0

    def test_get_cache_key(self):
        """Test cache key generation"""
        mock_base = Mock()
        synthesizer = QuantumSQLSynthesizer(mock_base)

        key1 = synthesizer._get_cache_key("test query", {"param": "value"})
        key2 = synthesizer._get_cache_key("test query", {"param": "value"})
        key3 = synthesizer._get_cache_key("different query", {"param": "value"})

        assert key1 == key2
        assert key1 != key3

    def test_get_quantum_statistics(self):
        """Test quantum statistics collection"""
        mock_base = Mock()
        synthesizer = QuantumSQLSynthesizer(mock_base, enable_quantum=True)

        stats = synthesizer.get_quantum_statistics()

        assert stats["quantum_enabled"]
        assert "optimizer_metrics" in stats
        assert "scheduler_metrics" in stats

    def test_clear_optimization_cache(self):
        """Test cache clearing"""
        mock_base = Mock()
        synthesizer = QuantumSQLSynthesizer(mock_base, enable_quantum=True)

        # Add something to cache
        synthesizer.optimization_cache["test"] = {"data": "value"}

        synthesizer.clear_optimization_cache()

        assert len(synthesizer.optimization_cache) == 0

    @pytest.mark.asyncio
    async def test_synthesizer_shutdown(self):
        """Test graceful shutdown"""
        mock_base = Mock()
        synthesizer = QuantumSQLSynthesizer(mock_base, enable_quantum=True)

        await synthesizer.shutdown()

        # Should complete without error


# Integration tests
class TestQuantumIntegration:
    """Test quantum system integration"""

    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self):
        """Test complete quantum optimization flow"""
        # Create mock base synthesizer
        mock_base = AsyncMock()
        mock_result = Mock()
        mock_result.sql = "SELECT * FROM users ORDER BY id"
        mock_result.data = [{"id": 1, "name": "test"}]
        mock_result.explanation = "Test query explanation"
        mock_result.error = None
        mock_result.execution_time = 0.05
        mock_base.query.return_value = mock_result

        # Create quantum synthesizer
        synthesizer = QuantumSQLSynthesizer(mock_base, enable_quantum=True)

        # Execute query (should use fallback since no real schema analysis)
        result = await synthesizer.query("Show me all users")

        assert result.sql is not None
        assert hasattr(result, "quantum_metrics")
        assert hasattr(result, "optimization_time")
        assert hasattr(result, "quantum_cost_reduction")

    def test_quantum_plan_cost_estimation(self):
        """Test quantum cost estimation accuracy"""
        generator = QuantumQueryPlanGenerator()

        # Simple query should have lower cost
        simple_plans = generator.generate_plans(
            tables=["users"], joins=[], filters=[], aggregations=[]
        )

        # Complex query should have higher cost
        complex_plans = generator.generate_plans(
            tables=["users", "orders", "products"],
            joins=[("users", "orders"), ("orders", "products")],
            filters=[{"selectivity": 0.1}, {"selectivity": 0.05}],
            aggregations=["count", "sum"],
        )

        avg_simple_cost = sum(p.cost for p in simple_plans) / len(simple_plans)
        avg_complex_cost = sum(p.cost for p in complex_plans) / len(complex_plans)

        assert avg_complex_cost > avg_simple_cost

    @pytest.mark.asyncio
    async def test_concurrent_quantum_optimization(self):
        """Test concurrent quantum optimizations"""
        optimizer = QuantumQueryOptimizer()

        # Create multiple sets of plans
        plans_sets = [
            [QueryPlan([], [], [], 10.0 + i, 0.0) for i in range(3)] for _ in range(5)
        ]

        # Run concurrent optimizations
        tasks = [optimizer.optimize_query_async(plans) for plans in plans_sets]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert result.quantum_state == QuantumState.COLLAPSED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
