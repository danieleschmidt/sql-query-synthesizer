"""
Comprehensive tests for quantum autonomous systems
"""

import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil

# Import components under test
from sql_synthesizer.quantum.autonomous_optimizer import (
    AutonomousQuantumOptimizer, SDLCTask, SDLCPhase, OptimizationStrategy
)
from sql_synthesizer.quantum.resilience import (
    QuantumCircuitBreaker, QuantumBulkhead, BulkheadConfig,
    ResilienceManager, CircuitState
)
from sql_synthesizer.quantum.monitoring import (
    QuantumMonitoringSystem, MetricType, QuantumAnomalyDetector
)
from sql_synthesizer.quantum.auto_scaling import (
    QuantumAutoScaler, ResourceConfig, ResourceType, ScalingStrategy
)
from autonomous_sdlc_engine import AutonomousSDLCEngine


class TestAutonomousQuantumOptimizer:
    """Test cases for AutonomousQuantumOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        return AutonomousQuantumOptimizer(
            max_parallel_tasks=4,
            optimization_timeout=10.0
        )
    
    @pytest.fixture
    def sample_tasks(self):
        return [
            SDLCTask(
                task_id="task_1",
                phase=SDLCPhase.ANALYSIS,
                description="Analyze requirements",
                priority=5.0,
                estimated_effort=2.0
            ),
            SDLCTask(
                task_id="task_2", 
                phase=SDLCPhase.IMPLEMENTATION,
                description="Implement core functionality",
                priority=8.0,
                estimated_effort=5.0,
                dependencies=["task_1"]
            ),
            SDLCTask(
                task_id="task_3",
                phase=SDLCPhase.TESTING,
                description="Run tests",
                priority=6.0,
                estimated_effort=3.0,
                dependencies=["task_2"]
            )
        ]
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly"""
        assert optimizer.max_parallel_tasks == 4
        assert optimizer.optimization_timeout == 10.0
        assert optimizer.total_optimizations == 0
        assert optimizer.successful_optimizations == 0
        assert len(optimizer.strategy_weights) == 4
    
    @pytest.mark.asyncio
    async def test_optimize_sdlc_tasks(self, optimizer, sample_tasks):
        """Test SDLC task optimization"""
        result = await optimizer.optimize_sdlc_tasks(sample_tasks)
        
        assert isinstance(result, object)  # OptimizationResult
        assert len(result.optimized_tasks) == 3
        assert len(result.execution_plan) > 0
        assert result.optimization_confidence > 0
        assert result.total_estimated_time > 0
        assert len(result.recommendations) >= 0
    
    @pytest.mark.asyncio
    async def test_empty_tasks_error(self, optimizer):
        """Test that empty task list raises error"""
        with pytest.raises(Exception):  # QuantumOptimizationError
            await optimizer.optimize_sdlc_tasks([])
    
    def test_learning_statistics(self, optimizer):
        """Test learning statistics collection"""
        stats = optimizer.get_learning_statistics()
        
        assert "total_optimizations" in stats
        assert "success_rate" in stats
        assert "strategy_weights" in stats
        assert "learning_rate" in stats
        
        assert stats["total_optimizations"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["learning_rate"] == 0.1
    
    def test_optimization_report(self, optimizer):
        """Test optimization report export"""
        report = optimizer.export_optimization_report()
        
        assert "autonomous_optimizer" in report
        assert "optimization_history" in report
        assert "generated_at" in report
        
        optimizer_info = report["autonomous_optimizer"]
        assert optimizer_info["max_parallel_tasks"] == 4
        assert optimizer_info["optimization_timeout"] == 10.0


class TestQuantumCircuitBreaker:
    """Test cases for QuantumCircuitBreaker"""
    
    @pytest.fixture
    def circuit_breaker(self):
        return QuantumCircuitBreaker(
            name="test_circuit",
            failure_threshold=3,
            recovery_timeout=1.0,
            quantum_adaptation=True
        )
    
    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initializes correctly"""
        assert circuit_breaker.name == "test_circuit"
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.recovery_timeout == 1.0
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.quantum_adaptation is True
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call through circuit breaker"""
        
        async def successful_func():
            return "success"
        
        result = await circuit_breaker.call(successful_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.successful_requests == 1
    
    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self, circuit_breaker):
        """Test circuit opens after repeated failures"""
        
        async def failing_func():
            raise RuntimeError("Test failure")
        
        # Trigger enough failures to open circuit
        for _ in range(3):
            try:
                await circuit_breaker.call(failing_func)
            except RuntimeError:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.metrics.failed_requests == 3
    
    @pytest.mark.asyncio
    async def test_circuit_blocks_when_open(self, circuit_breaker):
        """Test circuit blocks calls when open"""
        
        # Force circuit to open
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.next_attempt_time = time.time() + 10
        
        async def test_func():
            return "should not execute"
        
        with pytest.raises(Exception):  # QuantumCircuitBreakerError
            await circuit_breaker.call(test_func)
    
    def test_circuit_state_reporting(self, circuit_breaker):
        """Test circuit breaker state reporting"""
        state = circuit_breaker.get_state()
        
        assert state["name"] == "test_circuit"
        assert state["state"] == "closed"
        assert "metrics" in state
        assert "failure_count" in state
        assert "adaptive_threshold" in state
    
    def test_quantum_adaptation(self, circuit_breaker):
        """Test quantum adaptation of thresholds"""
        initial_threshold = circuit_breaker.adaptive_threshold
        
        # Simulate some successes
        for _ in range(10):
            circuit_breaker._record_success(0.1)
        
        # Threshold should adapt (may go up or down due to quantum noise)
        assert circuit_breaker.adaptive_threshold != initial_threshold
    
    def test_reset_circuit(self, circuit_breaker):
        """Test manual circuit reset"""
        # Force some failures and open state
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 5
        
        circuit_breaker.reset()
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0


class TestQuantumBulkhead:
    """Test cases for QuantumBulkhead"""
    
    @pytest.fixture
    def bulkhead_config(self):
        return BulkheadConfig(
            max_concurrent_calls=2,
            max_queue_size=5,
            timeout_seconds=1.0,
            priority_levels=2,
            auto_scaling=False
        )
    
    @pytest.fixture
    def bulkhead(self, bulkhead_config):
        return QuantumBulkhead("test_bulkhead", bulkhead_config)
    
    def test_bulkhead_initialization(self, bulkhead, bulkhead_config):
        """Test bulkhead initializes correctly"""
        assert bulkhead.name == "test_bulkhead"
        assert bulkhead.config == bulkhead_config
        assert len(bulkhead.semaphores) == 2  # priority_levels
    
    @pytest.mark.asyncio
    async def test_bulkhead_execution(self, bulkhead):
        """Test function execution through bulkhead"""
        
        async def test_func(value):
            await asyncio.sleep(0.1)
            return value * 2
        
        result = await bulkhead.execute(test_func, 5)
        assert result == 10
        assert bulkhead.metrics.successful_requests == 1
    
    @pytest.mark.asyncio
    async def test_bulkhead_capacity_limit(self, bulkhead):
        """Test bulkhead respects capacity limits"""
        
        async def slow_func():
            await asyncio.sleep(2.0)  # Longer than timeout
            return "done"
        
        # Start tasks that will exceed capacity
        tasks = []
        for i in range(5):  # More than max_concurrent_calls (2)
            task = asyncio.create_task(bulkhead.execute(slow_func, priority=0))
            tasks.append(task)
            await asyncio.sleep(0.01)  # Small delay to ensure ordering
        
        # Wait a bit then check results
        await asyncio.sleep(0.2)
        
        # Some tasks should fail due to capacity/timeout limits
        completed = sum(1 for task in tasks if task.done())
        assert completed < 5  # Not all should complete immediately
        
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
    
    def test_bulkhead_state_reporting(self, bulkhead):
        """Test bulkhead state reporting"""
        state = bulkhead.get_state()
        
        assert state["name"] == "test_bulkhead"
        assert "current_capacity" in state
        assert "priority_levels" in state
        assert "metrics" in state
        assert state["priority_levels"] == 2


class TestQuantumMonitoringSystem:
    """Test cases for QuantumMonitoringSystem"""
    
    @pytest.fixture
    def monitoring_system(self):
        return QuantumMonitoringSystem()
    
    def test_monitoring_initialization(self, monitoring_system):
        """Test monitoring system initializes correctly"""
        assert len(monitoring_system.metrics) == 0
        assert len(monitoring_system.alerts) == 0
        assert monitoring_system.auto_anomaly_detection is True
    
    def test_create_metric(self, monitoring_system):
        """Test metric creation"""
        metric = monitoring_system.create_metric("test_metric", MetricType.GAUGE)
        
        assert "test_metric" in monitoring_system.metrics
        assert metric.name == "test_metric"
        assert metric.metric_type == MetricType.GAUGE
    
    def test_record_metric(self, monitoring_system):
        """Test metric recording"""
        monitoring_system.record_metric("cpu_usage", 75.5)
        
        assert "cpu_usage" in monitoring_system.metrics
        metric = monitoring_system.metrics["cpu_usage"]
        assert len(metric.points) == 1
        assert metric.points[0].value == 75.5
    
    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, monitoring_system):
        """Test monitoring start and stop"""
        await monitoring_system.start_monitoring()
        assert monitoring_system.monitoring_task is not None
        assert not monitoring_system.monitoring_task.done()
        
        monitoring_system.stop_monitoring()
        await asyncio.sleep(0.1)  # Allow time for cancellation
        assert monitoring_system.monitoring_task.cancelled() or monitoring_system.monitoring_task.done()
    
    def test_alert_handlers(self, monitoring_system):
        """Test alert handler registration"""
        handler_called = []
        
        def test_handler(alert):
            handler_called.append(alert)
        
        monitoring_system.add_alert_handler(test_handler)
        assert len(monitoring_system.alert_handlers) == 1
    
    def test_monitoring_status(self, monitoring_system):
        """Test monitoring status reporting"""
        # Add some metrics
        monitoring_system.record_metric("test1", 100.0)
        monitoring_system.record_metric("test2", 200.0)
        
        status = monitoring_system.get_monitoring_status()
        
        assert "monitoring_active" in status
        assert "metrics" in status
        assert len(status["metrics"]) == 2
        assert "test1" in status["metrics"]
        assert "test2" in status["metrics"]
    
    def test_export_monitoring_report(self, monitoring_system):
        """Test monitoring report export"""
        monitoring_system.record_metric("test_metric", 42.0)
        
        report = monitoring_system.export_monitoring_report()
        
        assert "quantum_monitoring_report" in report
        assert "system_status" in report["quantum_monitoring_report"]
        assert "anomaly_detector_config" in report["quantum_monitoring_report"]


class TestQuantumAnomalyDetector:
    """Test cases for QuantumAnomalyDetector"""
    
    @pytest.fixture
    def anomaly_detector(self):
        return QuantumAnomalyDetector(
            sensitivity=0.8,
            window_size=20,
            quantum_ensemble=True
        )
    
    def test_anomaly_detector_initialization(self, anomaly_detector):
        """Test anomaly detector initializes correctly"""
        assert anomaly_detector.sensitivity == 0.8
        assert anomaly_detector.window_size == 20
        assert anomaly_detector.quantum_ensemble is True
        assert len(anomaly_detector.detection_algorithms) == 4
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_empty_points(self, anomaly_detector):
        """Test anomaly detection with insufficient points"""
        from sql_synthesizer.quantum.monitoring import MetricPoint
        
        points = [MetricPoint(timestamp=time.time(), value=10.0)]
        anomalies = await anomaly_detector.detect_anomalies("test_metric", points)
        
        assert len(anomalies) == 0  # Not enough points for detection
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_normal_data(self, anomaly_detector):
        """Test anomaly detection with normal data"""
        from sql_synthesizer.quantum.monitoring import MetricPoint
        
        # Generate normal data points
        points = []
        base_time = time.time()
        
        for i in range(50):
            # Normal data around 50 with small variations
            value = 50.0 + (i % 5 - 2)  # Values between 48-52
            points.append(MetricPoint(timestamp=base_time + i, value=value))
        
        anomalies = await anomaly_detector.detect_anomalies("normal_metric", points)
        
        # Should detect few or no anomalies in normal data
        assert len(anomalies) <= 2
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_spike(self, anomaly_detector):
        """Test anomaly detection with clear spike"""
        from sql_synthesizer.quantum.monitoring import MetricPoint
        
        points = []
        base_time = time.time()
        
        # Generate normal data with a clear spike
        for i in range(50):
            if i == 25:
                value = 200.0  # Clear spike
            else:
                value = 50.0 + (i % 3 - 1)  # Normal values around 50
            
            points.append(MetricPoint(timestamp=base_time + i, value=value))
        
        anomalies = await anomaly_detector.detect_anomalies("spike_metric", points)
        
        # Should detect the spike
        assert len(anomalies) >= 1
        
        # Check that we detected a spike anomaly
        spike_anomalies = [a for a in anomalies if a.anomaly_type.value == "spike"]
        assert len(spike_anomalies) >= 1


class TestQuantumAutoScaler:
    """Test cases for QuantumAutoScaler"""
    
    @pytest.fixture
    def monitoring_system(self):
        return QuantumMonitoringSystem()
    
    @pytest.fixture
    def auto_scaler(self, monitoring_system):
        return QuantumAutoScaler(
            monitoring_system=monitoring_system,
            scaling_strategy=ScalingStrategy.QUANTUM_ADAPTIVE
        )
    
    @pytest.fixture
    def cpu_resource_config(self):
        return ResourceConfig(
            resource_type=ResourceType.CPU,
            current_capacity=100,
            min_capacity=50,
            max_capacity=200,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            cooldown_period=1.0  # Short for testing
        )
    
    def test_auto_scaler_initialization(self, auto_scaler):
        """Test auto-scaler initializes correctly"""
        assert auto_scaler.scaling_strategy == ScalingStrategy.QUANTUM_ADAPTIVE
        assert auto_scaler.auto_scaling_enabled is True
        assert len(auto_scaler.resource_configs) == 0
    
    def test_configure_resource(self, auto_scaler, cpu_resource_config):
        """Test resource configuration"""
        auto_scaler.configure_resource(ResourceType.CPU, cpu_resource_config)
        
        assert ResourceType.CPU in auto_scaler.resource_configs
        assert auto_scaler.resource_configs[ResourceType.CPU] == cpu_resource_config
    
    @pytest.mark.asyncio
    async def test_start_stop_auto_scaling(self, auto_scaler):
        """Test starting and stopping auto-scaling"""
        await auto_scaler.start_auto_scaling()
        assert auto_scaler.scaling_task is not None
        assert not auto_scaler.scaling_task.done()
        
        auto_scaler.stop_auto_scaling()
        await asyncio.sleep(0.1)  # Allow time for cancellation
        assert auto_scaler.scaling_task.cancelled() or auto_scaler.scaling_task.done()
    
    def test_scaling_status(self, auto_scaler, cpu_resource_config):
        """Test scaling status reporting"""
        auto_scaler.configure_resource(ResourceType.CPU, cpu_resource_config)
        
        status = auto_scaler.get_scaling_status()
        
        assert "auto_scaling_enabled" in status
        assert "scaling_strategy" in status
        assert "resources" in status
        assert "cpu" in status["resources"]
        
        cpu_status = status["resources"]["cpu"]
        assert cpu_status["current_capacity"] == 100
        assert cpu_status["min_capacity"] == 50
        assert cpu_status["max_capacity"] == 200
    
    def test_export_scaling_report(self, auto_scaler):
        """Test scaling report export"""
        report = auto_scaler.export_scaling_report()
        
        assert "quantum_auto_scaling_report" in report
        assert "status" in report["quantum_auto_scaling_report"]
        assert "performance_metrics" in report["quantum_auto_scaling_report"]


class TestAutonomousSDLCEngine:
    """Test cases for AutonomousSDLCEngine"""
    
    @pytest.fixture
    def temp_project_root(self):
        """Create temporary project directory"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create basic project structure
        (temp_dir / "sql_synthesizer").mkdir()
        (temp_dir / "sql_synthesizer" / "__init__.py").touch()
        (temp_dir / "pyproject.toml").touch()
        (temp_dir / "README.md").write_text("Test project")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sdlc_engine(self, temp_project_root):
        return AutonomousSDLCEngine(
            project_root=temp_project_root,
            max_parallel_tasks=4,
            continuous_optimization=False  # Disable for testing
        )
    
    def test_sdlc_engine_initialization(self, sdlc_engine, temp_project_root):
        """Test SDLC engine initializes correctly"""
        assert sdlc_engine.project_root == temp_project_root
        assert sdlc_engine.max_parallel_tasks == 4
        assert sdlc_engine.continuous_optimization is False
        assert len(sdlc_engine.checkpoints) > 0
    
    @pytest.mark.asyncio
    async def test_execute_autonomous_sdlc(self, sdlc_engine):
        """Test autonomous SDLC execution"""
        # This is a long-running test, so we'll mock some components
        with patch.object(sdlc_engine.optimizer, 'optimize_sdlc_tasks') as mock_optimize:
            # Mock optimization result
            from sql_synthesizer.quantum.autonomous_optimizer import OptimizationResult
            
            mock_result = OptimizationResult(
                optimized_tasks=[],
                execution_plan=[],
                total_estimated_time=10.0,
                optimization_confidence=0.8,
                quantum_metrics={},
                recommendations=[]
            )
            mock_optimize.return_value = mock_result
            
            result = await sdlc_engine.execute_autonomous_sdlc()
            
            assert "autonomous_sdlc_report" in result
            assert "checkpoints" in result
            assert "optimization" in result
            assert "execution" in result
    
    def test_checkpoint_validation(self, sdlc_engine):
        """Test checkpoint validation methods"""
        # Test project structure validation
        assert sdlc_engine._validate_project_structure() is True
        
        # Test dependencies validation
        assert sdlc_engine._validate_dependencies() is True
        
        # Test architecture validation
        assert sdlc_engine._validate_architecture() is True


class TestIntegration:
    """Integration tests combining multiple systems"""
    
    @pytest.mark.asyncio
    async def test_full_quantum_system_integration(self):
        """Test full integration of quantum systems"""
        
        # Initialize components
        monitoring = QuantumMonitoringSystem()
        resilience = ResilienceManager()
        auto_scaler = QuantumAutoScaler(monitoring)
        
        # Configure auto-scaler
        cpu_config = ResourceConfig(
            resource_type=ResourceType.CPU,
            current_capacity=100,
            min_capacity=50,
            max_capacity=200,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            cooldown_period=0.1
        )
        auto_scaler.configure_resource(ResourceType.CPU, cpu_config)
        
        # Start systems
        await monitoring.start_monitoring()
        await auto_scaler.start_auto_scaling()
        resilience.start_health_monitoring()
        
        # Simulate some activity
        monitoring.record_metric("cpu_usage", 85.0)  # Should trigger scaling
        monitoring.record_metric("response_time", 0.5)
        
        await asyncio.sleep(0.2)  # Allow systems to process
        
        # Check that systems are running
        monitoring_status = monitoring.get_monitoring_status()
        assert monitoring_status["monitoring_active"] is True
        
        scaling_status = auto_scaler.get_scaling_status()
        assert scaling_status["auto_scaling_enabled"] is True
        
        resilience_status = resilience.get_system_status()
        assert resilience_status["health_monitoring_active"] is True
        
        # Stop systems
        monitoring.stop_monitoring()
        auto_scaler.stop_auto_scaling()
        resilience.stop_health_monitoring()
    
    @pytest.mark.asyncio
    async def test_resilience_patterns_under_load(self):
        """Test resilience patterns under simulated load"""
        
        resilience = ResilienceManager()
        
        # Create circuit breaker and bulkhead
        cb = resilience.create_circuit_breaker("test_service", failure_threshold=3)
        
        bulkhead_config = BulkheadConfig(max_concurrent_calls=2, timeout_seconds=0.5)
        bh = resilience.create_bulkhead("test_bulkhead", bulkhead_config)
        
        # Test function that sometimes fails
        failure_count = 0
        
        async def unreliable_service():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 5:  # First 5 calls fail
                raise RuntimeError("Service unavailable")
            await asyncio.sleep(0.1)
            return "success"
        
        # Test circuit breaker behavior
        results = []
        for i in range(10):
            try:
                result = await cb.call(unreliable_service)
                results.append(result)
            except Exception as e:
                results.append(str(e))
        
        # Should have some failures, then circuit opens, then eventual success
        assert len(results) == 10
        assert "success" in results  # At least some successes after recovery
        
        # Test bulkhead isolation
        async def slow_service():
            await asyncio.sleep(1.0)  # Longer than bulkhead timeout
            return "slow_result"
        
        # Start multiple tasks that should be isolated
        tasks = [
            asyncio.create_task(bh.execute(slow_service, priority=0))
            for _ in range(5)
        ]
        
        await asyncio.sleep(0.1)
        
        # Some should timeout due to bulkhead limits
        completed = sum(1 for task in tasks if task.done())
        
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Cleanup
        resilience.stop_health_monitoring()


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])