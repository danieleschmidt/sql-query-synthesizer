"""
Reliability Module - Error Recovery and Fault Tolerance

Provides comprehensive error handling, automatic recovery mechanisms,
and system resilience capabilities for production environments.
"""

from .error_recovery import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    ErrorClassifier,
    ErrorContext,
    ErrorRecoveryManager,
    ErrorSeverity,
    RecoveryAction,
    RecoveryStrategy,
    RetryManager,
    error_recovery_manager,
    resilient_operation,
)
from .graceful_degradation import (
    DegradationLevel,
    DegradationStrategy,
    GracefulDegradationManager,
    ServiceCapability,
    degradation_manager,
)
from .health_monitoring import (
    ComponentHealth,
    DependencyChecker,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    PerformanceMonitor,
    SystemHealthManager,
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
    "degradation_manager",
]
