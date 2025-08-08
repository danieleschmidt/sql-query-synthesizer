"""
Reliability Module - Error Recovery and Fault Tolerance

Provides comprehensive error handling, automatic recovery mechanisms,
and system resilience capabilities for production environments.
"""

from .error_recovery import (
    ErrorSeverity,
    RecoveryStrategy,
    ErrorContext,
    RecoveryAction,
    ErrorClassifier,
    CircuitBreaker,
    CircuitBreakerOpenError,
    RetryManager,
    ErrorRecoveryManager,
    error_recovery_manager,
    resilient_operation
)

from .health_monitoring import (
    HealthStatus,
    HealthCheck,
    ComponentHealth,
    HealthMonitor,
    DependencyChecker,
    PerformanceMonitor,
    SystemHealthManager
)

from .graceful_degradation import (
    DegradationLevel,
    DegradationStrategy,
    GracefulDegradationManager,
    ServiceCapability,
    degradation_manager
)

__all__ = [
    "ErrorSeverity",
    "RecoveryStrategy", 
    "ErrorContext",
    "RecoveryAction",
    "ErrorClassifier",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "RetryManager",
    "ErrorRecoveryManager",
    "error_recovery_manager",
    "resilient_operation",
    "HealthStatus",
    "HealthCheck",
    "ComponentHealth",
    "HealthMonitor",
    "DependencyChecker", 
    "PerformanceMonitor",
    "SystemHealthManager",
    "DegradationLevel",
    "DegradationStrategy",
    "GracefulDegradationManager",
    "ServiceCapability",
    "degradation_manager"
]