"""
Tests for Autonomous SDLC Enhancement Module
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from sql_synthesizer.autonomous_sdlc import (
    AutonomousQualityGateEngine,
    EnhancedErrorHandler,
    AutonomousPerformanceOptimizer,
    AutonomousDeploymentEngine
)


@pytest.mark.asyncio
async def test_quality_gate_engine():
    """Test autonomous quality gate engine"""
    engine = AutonomousQualityGateEngine()
    
    # Execute quality gates
    results = await engine.execute_all_gates()
    
    # Validate results structure
    assert "overall_passed" in results
    assert "overall_score" in results
    assert "gates" in results
    assert isinstance(results["overall_score"], float)
    assert 0.0 <= results["overall_score"] <= 1.0


@pytest.mark.asyncio 
async def test_quality_gate_auto_healing():
    """Test quality gate auto-healing capabilities"""
    engine = AutonomousQualityGateEngine()
    
    # Test auto-healing
    healing_results = await engine.auto_heal()
    
    # Validate healing results
    assert "ruff_autofix" in healing_results
    assert "black_format" in healing_results
    assert isinstance(healing_results["ruff_autofix"], bool)
    assert isinstance(healing_results["black_format"], bool)


def test_enhanced_error_handler():
    """Test enhanced error handling"""
    handler = EnhancedErrorHandler()
    
    # Test error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_context = handler.handle_error(e)
        
        # Validate error context
        assert error_context.error_type == "ValueError"
        assert error_context.error_message == "Test error"
        assert error_context.severity in ["low", "medium", "high", "critical"]
        assert isinstance(error_context.recovery_suggestions, list)


@pytest.mark.asyncio
async def test_enhanced_error_handler_recovery():
    """Test auto-recovery functionality"""
    handler = EnhancedErrorHandler()
    
    # Create a mock error context
    try:
        raise ConnectionError("Database connection failed")
    except Exception as e:
        error_context = handler.handle_error(e)
        
        # Test auto-recovery
        recovery_success = await handler.attempt_auto_recovery(error_context)
        assert isinstance(recovery_success, bool)


def test_error_analytics():
    """Test error analytics and insights"""
    handler = EnhancedErrorHandler()
    
    # Generate some errors
    for i in range(5):
        try:
            if i % 2:
                raise ValueError(f"Test error {i}")
            else:
                raise ConnectionError(f"Connection error {i}")
        except Exception as e:
            handler.handle_error(e)
    
    # Get analytics
    analytics = handler.get_error_analytics()
    
    # Validate analytics
    assert "total_errors" in analytics
    assert analytics["total_errors"] == 5
    assert "most_common_errors" in analytics
    assert "severity_distribution" in analytics


@pytest.mark.asyncio
async def test_performance_optimizer():
    """Test autonomous performance optimizer"""
    optimizer = AutonomousPerformanceOptimizer(monitoring_interval=1)
    
    # Collect some metrics
    for i in range(3):
        await optimizer.profiler.collect_metrics()
        await asyncio.sleep(0.1)
    
    # Test scaling recommendation
    recommendation = await optimizer.scaler.analyze_scaling_needs()
    
    # Validate recommendation
    assert recommendation.action in ["scale_up", "scale_down", "optimize", "maintain"]
    assert 0.0 <= recommendation.confidence <= 1.0
    assert recommendation.urgency in ["low", "medium", "high", "critical"]
    assert isinstance(recommendation.implementation_steps, list)


@pytest.mark.asyncio
async def test_performance_report():
    """Test performance report generation"""
    optimizer = AutonomousPerformanceOptimizer(monitoring_interval=1)
    
    # Collect metrics and generate recommendation
    for i in range(5):
        await optimizer.profiler.collect_metrics()
        recommendation = await optimizer.scaler.analyze_scaling_needs()
        
        # Simulate optimization history
        optimizer.optimization_history.append({
            "timestamp": asyncio.get_event_loop().time(),
            "metrics": {
                "cpu_percent": 50 + i * 5,
                "response_time_ms": 100 + i * 10,
                "memory_percent": 60 + i * 2,
                "throughput_qps": 90 - i * 5
            },
            "recommendation": {"action": "maintain"},
            "auto_executed": False
        })
    
    # Generate report
    report = optimizer.get_performance_report()
    
    # Validate report
    assert report["status"] == "success"
    assert "performance_summary" in report
    assert "optimization_opportunities" in report


@pytest.mark.asyncio
async def test_deployment_engine():
    """Test autonomous deployment engine"""
    engine = AutonomousDeploymentEngine()
    
    # Test deployment readiness analysis
    readiness = await engine.analyze_deployment_readiness("development")
    
    # Validate readiness analysis
    assert "ready" in readiness
    assert "quality_score" in readiness
    assert "deployment_success_rate" in readiness
    assert "recommendation" in readiness
    assert readiness["recommendation"] in ["proceed", "wait"]


@pytest.mark.asyncio 
async def test_deployment_analytics():
    """Test deployment analytics"""
    engine = AutonomousDeploymentEngine()
    
    # Get analytics (should handle empty history)
    analytics = engine.get_deployment_analytics()
    
    # Validate analytics structure
    assert "message" in analytics or "total_deployments" in analytics


def test_deployment_config_loading():
    """Test deployment configuration loading"""
    engine = AutonomousDeploymentEngine()
    
    # Validate default configs
    assert "development" in engine.configs
    assert "staging" in engine.configs
    assert "production" in engine.configs
    
    # Validate config structure
    dev_config = engine.configs["development"]
    assert dev_config.environment == "development"
    assert dev_config.replicas >= 1
    assert isinstance(dev_config.resources, dict)


@pytest.mark.asyncio
async def test_blue_green_deployment():
    """Test blue-green deployment mechanism"""
    from sql_synthesizer.autonomous_sdlc.autonomous_deployment import BlueGreenDeployer, DeploymentConfig
    
    deployer = BlueGreenDeployer()
    
    # Create test config
    config = DeploymentConfig(
        environment="test",
        image_tag="test-v1.0.0",
        replicas=1,
        resources={"cpu": "100m", "memory": "256Mi"},
        health_checks={"interval": 10, "timeout": 5}
    )
    
    # Test deployment
    result = await deployer.deploy(config)
    
    # Validate deployment result
    assert result.deployment_id.startswith("deploy-")
    assert isinstance(result.success, bool)
    assert isinstance(result.duration_seconds, float)
    assert result.health_status in ["healthy", "unhealthy", "failed", "rollback_completed"]


@pytest.mark.asyncio
async def test_health_checker():
    """Test deployment health checking"""
    from sql_synthesizer.autonomous_sdlc.autonomous_deployment import HealthChecker, DeploymentConfig
    
    health_checker = HealthChecker()
    
    # Create test config
    config = DeploymentConfig(
        environment="test",
        image_tag="test-v1.0.0",
        replicas=1,
        resources={"cpu": "100m", "memory": "256Mi"},
        health_checks={"interval": 10, "timeout": 5}
    )
    
    # Test health check
    is_healthy, metrics = await health_checker.check_deployment_health(config)
    
    # Validate health check results
    assert isinstance(is_healthy, bool)
    assert "checks" in metrics
    assert "overall_healthy" in metrics
    assert "success_rate" in metrics


def test_circuit_breaker():
    """Test resilient circuit breaker"""
    from sql_synthesizer.autonomous_sdlc.enhanced_error_handling import ResilientCircuitBreaker
    
    breaker = ResilientCircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
    
    # Initial state should be closed
    assert breaker.state == "closed"
    assert breaker.failure_count == 0


@pytest.mark.asyncio
async def test_adaptive_retry():
    """Test adaptive retry mechanism"""
    from sql_synthesizer.autonomous_sdlc.enhanced_error_handling import AdaptiveRetry
    
    retry_handler = AdaptiveRetry(max_retries=2, base_delay=0.1)
    
    # Test successful function
    async def success_func():
        return "success"
    
    result = await retry_handler.execute(success_func)
    assert result == "success"
    
    # Test function that fails initially
    call_count = 0
    
    async def failing_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Temporary failure")
        return "eventual_success"
    
    result = await retry_handler.execute(failing_func)
    assert result == "eventual_success"
    assert call_count == 2


def test_scaling_recommendation():
    """Test scaling recommendation generation"""
    from sql_synthesizer.autonomous_sdlc.scaling_optimizer import (
        PerformanceProfiler, 
        IntelligentScaler,
        PerformanceMetrics
    )
    
    profiler = PerformanceProfiler()
    scaler = IntelligentScaler(profiler)
    
    # Add some test metrics
    for i in range(5):
        metrics = PerformanceMetrics(
            timestamp=1000 + i,
            cpu_percent=50 + i * 10,
            memory_percent=60 + i * 5,
            memory_available_gb=4.0,
            disk_io_read_mb=100,
            disk_io_write_mb=50,
            network_sent_mb=200,
            network_recv_mb=150,
            active_connections=20,
            response_time_ms=200 + i * 50,
            throughput_qps=100 - i * 10,
            error_rate=0.01
        )
        profiler.metrics_history.append(metrics)
    
    # Test scaling decision logic (synchronous parts)
    current_metrics = profiler.metrics_history[-1]
    trends = profiler.get_performance_trends()
    
    # Test scale up score calculation
    scale_up_score = scaler._calculate_scale_up_score(current_metrics, trends)
    assert 0.0 <= scale_up_score <= 1.0
    
    # Test scale down score calculation  
    scale_down_score = scaler._calculate_scale_down_score(current_metrics, trends)
    assert 0.0 <= scale_down_score <= 1.0
    
    # Test optimize score calculation
    optimize_score = scaler._calculate_optimize_score(current_metrics, trends)
    assert 0.0 <= optimize_score <= 1.0