"""
Autonomous SDLC Enhancement Module
Progressive quality gates, self-healing, and autonomous development capabilities
"""

from .autonomous_deployment import (
    AutonomousDeploymentEngine,
    BlueGreenDeployer,
    DeploymentConfig,
    DeploymentResult,
    HealthChecker,
)
from .enhanced_error_handling import (
    AdaptiveRetry,
    EnhancedErrorHandler,
    ErrorContext,
    RecoveryAction,
    ResilientCircuitBreaker,
    adaptive_retry,
    enhanced_error_handler,
    global_error_handler,
    resilient_circuit_breaker,
)
from .quality_gates import (
    AutonomousQualityGateEngine,
    CodeQualityGate,
    PerformanceGate,
    QualityGateResult,
    SecurityGate,
    TestCoverageGate,
)
from .scaling_optimizer import (
    AutonomousPerformanceOptimizer,
    IntelligentScaler,
    PerformanceMetrics,
    PerformanceProfiler,
    ScalingRecommendation,
)

__all__ = [
    # Quality Gates
    "AutonomousQualityGateEngine",
    "QualityGateResult",
    "CodeQualityGate",
    "SecurityGate",
    "TestCoverageGate",
    "PerformanceGate",
    # Error Handling
    "EnhancedErrorHandler",
    "ErrorContext",
    "RecoveryAction",
    "ResilientCircuitBreaker",
    "AdaptiveRetry",
    "enhanced_error_handler",
    "resilient_circuit_breaker",
    "adaptive_retry",
    "global_error_handler",
    # Performance Optimization
    "AutonomousPerformanceOptimizer",
    "PerformanceProfiler",
    "IntelligentScaler",
    "PerformanceMetrics",
    "ScalingRecommendation",
    # Deployment
    "AutonomousDeploymentEngine",
    "BlueGreenDeployer",
    "HealthChecker",
    "DeploymentConfig",
    "DeploymentResult",
]
