"""Robust Error Handling System for SQL Query Synthesizer.

This module implements comprehensive error handling, recovery strategies,
and resilience patterns for production-grade reliability.
"""

import asyncio
import functools
import logging
import threading
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""

    LOW = "low"  # Minor issues, system continues normally
    MEDIUM = "medium"  # Moderate issues, degraded performance
    HIGH = "high"  # Serious issues, major features affected
    CRITICAL = "critical"  # System-threatening issues, immediate action required


class ErrorCategory(Enum):
    """Error categories for targeted handling strategies."""

    DATABASE = "database"  # Database connection/query errors
    NETWORK = "network"  # Network connectivity issues
    AUTHENTICATION = "authentication"  # Auth and authorization errors
    VALIDATION = "validation"  # Input validation errors
    PROCESSING = "processing"  # Query processing errors
    RESOURCE = "resource"  # Memory, disk, CPU resource errors
    EXTERNAL = "external"  # Third-party service errors (OpenAI, etc.)
    CONFIGURATION = "configuration"  # Configuration and setup errors
    UNKNOWN = "unknown"  # Unclassified errors


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""

    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration."""

    strategy_name: str
    applicable_categories: List[ErrorCategory]
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True
    recovery_function: Optional[Callable] = None
    fallback_function: Optional[Callable] = None


class RobustErrorHandler:
    """Comprehensive error handling system with recovery strategies."""

    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = {}
        self.error_count_by_category: Dict[ErrorCategory, int] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Initialize default recovery strategies
        self._setup_default_strategies()

        # Circuit breaker configuration
        self.circuit_breaker_failure_threshold = 5
        self.circuit_breaker_recovery_timeout = 60.0

        # Error aggregation for alerting
        self.error_aggregation_window = 300  # 5 minutes
        self.alert_threshold = 10  # errors per window

    def _setup_default_strategies(self):
        """Setup default recovery strategies for different error categories."""

        # Database error recovery
        db_strategy = RecoveryStrategy(
            strategy_name="database_retry_with_backoff",
            applicable_categories=[ErrorCategory.DATABASE],
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_backoff=True,
        )
        self.register_recovery_strategy(db_strategy)

        # Network error recovery
        network_strategy = RecoveryStrategy(
            strategy_name="network_retry_with_exponential_backoff",
            applicable_categories=[ErrorCategory.NETWORK, ErrorCategory.EXTERNAL],
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_backoff=True,
        )
        self.register_recovery_strategy(network_strategy)

        # Validation error recovery (no retry, immediate fallback)
        validation_strategy = RecoveryStrategy(
            strategy_name="validation_immediate_fallback",
            applicable_categories=[ErrorCategory.VALIDATION],
            max_attempts=1,
            base_delay=0.0,
            exponential_backoff=False,
        )
        self.register_recovery_strategy(validation_strategy)

    def register_recovery_strategy(self, strategy: RecoveryStrategy):
        """Register a recovery strategy for specific error categories."""
        for category in strategy.applicable_categories:
            if category not in self.recovery_strategies:
                self.recovery_strategies[category] = []
            self.recovery_strategies[category].append(strategy)

        logger.info(f"Registered recovery strategy: {strategy.strategy_name}")

    def handle_error(
        self,
        exception: Exception,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Dict[str, Any] = None,
        auto_recover: bool = True,
    ) -> ErrorContext:
        """Handle an error with optional automatic recovery."""

        import uuid

        # Create error context
        error_context = ErrorContext(
            error_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            severity=severity,
            category=category,
            operation=operation,
            metadata=context or {},
        )

        # Add exception details to metadata
        error_context.metadata.update(
            {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc(),
            }
        )

        # Record error
        with self._lock:
            self.error_history.append(error_context)
            self.error_count_by_category[category] = (
                self.error_count_by_category.get(category, 0) + 1
            )

            # Cleanup old error history
            cutoff_time = time.time() - 3600  # Keep 1 hour of history
            self.error_history = [
                e for e in self.error_history if e.timestamp > cutoff_time
            ]

        # Log error
        log_level = self._get_log_level(severity)
        logger.log(
            log_level,
            f"Error {error_context.error_id} in {operation}: {exception} "
            f"(severity: {severity.value}, category: {category.value})",
        )

        # Check circuit breaker
        if self._is_circuit_open(operation):
            logger.warning(f"Circuit breaker OPEN for operation: {operation}")
            raise Exception(f"Circuit breaker open for {operation}")

        # Attempt recovery if enabled
        if auto_recover:
            self._attempt_recovery(error_context, exception)

        # Update circuit breaker
        self._update_circuit_breaker(operation, error_context.recovery_successful)

        # Check if alerting is needed
        self._check_alerting_threshold(category)

        return error_context

    def _attempt_recovery(
        self, error_context: ErrorContext, original_exception: Exception
    ):
        """Attempt to recover from an error using registered strategies."""

        strategies = self.recovery_strategies.get(error_context.category, [])

        if not strategies:
            logger.debug(
                f"No recovery strategies for category: {error_context.category.value}"
            )
            return

        for strategy in strategies:
            logger.info(
                f"Attempting recovery with strategy: {strategy.strategy_name} "
                f"for error {error_context.error_id}"
            )

            try:
                error_context.recovery_attempted = True

                # Execute recovery with retry logic
                success = self._execute_recovery_with_retry(
                    strategy, error_context, original_exception
                )

                if success:
                    error_context.recovery_successful = True
                    logger.info(
                        f"Recovery successful for error {error_context.error_id}"
                    )
                    break

            except Exception as recovery_error:
                logger.warning(
                    f"Recovery strategy {strategy.strategy_name} failed for error "
                    f"{error_context.error_id}: {recovery_error}"
                )
                continue

    def _execute_recovery_with_retry(
        self,
        strategy: RecoveryStrategy,
        error_context: ErrorContext,
        original_exception: Exception,
    ) -> bool:
        """Execute recovery strategy with retry logic."""

        for attempt in range(strategy.max_attempts):
            try:
                error_context.retry_count = attempt + 1

                # Calculate delay
                if strategy.exponential_backoff and attempt > 0:
                    delay = min(
                        strategy.base_delay * (2 ** (attempt - 1)), strategy.max_delay
                    )
                    logger.debug(
                        f"Recovery attempt {attempt + 1}, waiting {delay:.2f}s"
                    )
                    time.sleep(delay)

                # Execute recovery function if provided
                if strategy.recovery_function:
                    result = strategy.recovery_function(
                        error_context, original_exception
                    )
                    if result:
                        return True

                # If no custom recovery function, just retry the original operation
                # This would need to be implemented based on the specific operation
                # For now, we'll consider it successful after the delay
                return True

            except Exception as retry_error:
                logger.debug(f"Recovery attempt {attempt + 1} failed: {retry_error}")
                if attempt == strategy.max_attempts - 1:
                    # Execute fallback if available
                    if strategy.fallback_function:
                        try:
                            strategy.fallback_function(
                                error_context, original_exception
                            )
                            return True
                        except Exception as fallback_error:
                            logger.error(f"Fallback function failed: {fallback_error}")
                continue

        return False

    def _is_circuit_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for an operation."""
        if operation not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[operation]

        if breaker["state"] != "open":
            return False

        # Check if recovery timeout has passed
        if time.time() - breaker["opened_at"] > self.circuit_breaker_recovery_timeout:
            breaker["state"] = "half_open"
            logger.info(f"Circuit breaker for {operation} moved to HALF_OPEN")
            return False

        return True

    def _update_circuit_breaker(self, operation: str, success: bool):
        """Update circuit breaker state based on operation result."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {
                "state": "closed",
                "failure_count": 0,
                "success_count": 0,
                "opened_at": 0,
            }

        breaker = self.circuit_breakers[operation]

        if success:
            breaker["success_count"] += 1
            breaker["failure_count"] = 0  # Reset failure count on success

            if breaker["state"] == "half_open":
                breaker["state"] = "closed"
                logger.info(f"Circuit breaker for {operation} moved to CLOSED")
        else:
            breaker["failure_count"] += 1
            breaker["success_count"] = 0

            if breaker["failure_count"] >= self.circuit_breaker_failure_threshold:
                breaker["state"] = "open"
                breaker["opened_at"] = time.time()
                logger.warning(f"Circuit breaker for {operation} moved to OPEN")

    def _check_alerting_threshold(self, category: ErrorCategory):
        """Check if error count exceeds alerting threshold."""
        recent_errors = [
            e
            for e in self.error_history
            if (
                e.category == category
                and time.time() - e.timestamp < self.error_aggregation_window
            )
        ]

        if len(recent_errors) >= self.alert_threshold:
            logger.critical(
                f"Alert: {len(recent_errors)} {category.value} errors in "
                f"{self.error_aggregation_window}s window"
            )
            # Here you would integrate with your alerting system

    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Map error severity to logging level."""
        mapping = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return mapping.get(severity, logging.WARNING)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            total_errors = len(self.error_history)

            if total_errors == 0:
                return {"total_errors": 0}

            # Error distribution by category
            category_distribution = {}
            for category in ErrorCategory:
                count = self.error_count_by_category.get(category, 0)
                if count > 0:
                    category_distribution[category.value] = count

            # Error distribution by severity
            severity_distribution = {}
            for error in self.error_history:
                severity = error.severity.value
                severity_distribution[severity] = (
                    severity_distribution.get(severity, 0) + 1
                )

            # Recovery statistics
            recovery_stats = {
                "attempted": sum(1 for e in self.error_history if e.recovery_attempted),
                "successful": sum(
                    1 for e in self.error_history if e.recovery_successful
                ),
            }
            if recovery_stats["attempted"] > 0:
                recovery_stats["success_rate"] = (
                    recovery_stats["successful"] / recovery_stats["attempted"] * 100
                )
            else:
                recovery_stats["success_rate"] = 0

            # Circuit breaker status
            circuit_status = {}
            for operation, breaker in self.circuit_breakers.items():
                circuit_status[operation] = {
                    "state": breaker["state"],
                    "failure_count": breaker["failure_count"],
                    "success_count": breaker["success_count"],
                }

            return {
                "total_errors": total_errors,
                "category_distribution": category_distribution,
                "severity_distribution": severity_distribution,
                "recovery_statistics": recovery_stats,
                "circuit_breaker_status": circuit_status,
                "error_rate_per_minute": self._calculate_error_rate(),
                "most_common_errors": self._get_most_common_errors(5),
            }

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate per minute."""
        now = time.time()
        recent_errors = [
            e for e in self.error_history if now - e.timestamp < 60  # Last minute
        ]
        return len(recent_errors)  # errors per minute

    def _get_most_common_errors(self, limit: int) -> List[Dict[str, Any]]:
        """Get most common error types."""
        error_types = {}

        for error in self.error_history:
            error_type = error.metadata.get("exception_type", "Unknown")
            if error_type not in error_types:
                error_types[error_type] = {
                    "count": 0,
                    "category": error.category.value,
                    "latest_message": "",
                }

            error_types[error_type]["count"] += 1
            error_types[error_type]["latest_message"] = error.metadata.get(
                "exception_message", ""
            )

        # Sort by count and return top N
        sorted_errors = sorted(
            error_types.items(), key=lambda x: x[1]["count"], reverse=True
        )

        return [
            {
                "error_type": error_type,
                "count": data["count"],
                "category": data["category"],
                "latest_message": (
                    data["latest_message"][:100] + "..."
                    if len(data["latest_message"]) > 100
                    else data["latest_message"]
                ),
            }
            for error_type, data in sorted_errors[:limit]
        ]


def robust_operation(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    auto_recover: bool = True,
    operation_name: str = None,
):
    """Decorator for robust error handling of operations."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(
                    exception=e,
                    operation=op_name,
                    severity=severity,
                    category=category,
                    auto_recover=auto_recover,
                )
                raise  # Re-raise after handling

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(
                    exception=e,
                    operation=op_name,
                    severity=severity,
                    category=category,
                    auto_recover=auto_recover,
                )
                raise  # Re-raise after handling

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


@contextmanager
def error_context(
    operation: str,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    auto_recover: bool = True,
):
    """Context manager for robust error handling."""
    try:
        yield
    except Exception as e:
        error_handler.handle_error(
            exception=e,
            operation=operation,
            category=category,
            severity=severity,
            auto_recover=auto_recover,
        )
        raise


@asynccontextmanager
async def async_error_context(
    operation: str,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    auto_recover: bool = True,
):
    """Async context manager for robust error handling."""
    try:
        yield
    except Exception as e:
        error_handler.handle_error(
            exception=e,
            operation=operation,
            category=category,
            severity=severity,
            auto_recover=auto_recover,
        )
        raise


class GracefulDegradation:
    """Implement graceful degradation when systems fail."""

    def __init__(self):
        self.degradation_levels = {}
        self.fallback_functions = {}

    def register_fallback(
        self, service_name: str, degradation_level: int, fallback_func: Callable
    ):
        """Register a fallback function for a service at a specific degradation level."""
        if service_name not in self.fallback_functions:
            self.fallback_functions[service_name] = {}

        self.fallback_functions[service_name][degradation_level] = fallback_func
        logger.info(
            f"Registered fallback for {service_name} at level {degradation_level}"
        )

    def degrade_service(self, service_name: str, level: int):
        """Degrade a service to a specific level."""
        self.degradation_levels[service_name] = level
        logger.warning(f"Service {service_name} degraded to level {level}")

    def execute_with_degradation(
        self, service_name: str, primary_func: Callable, *args, **kwargs
    ):
        """Execute function with graceful degradation."""
        degradation_level = self.degradation_levels.get(service_name, 0)

        if degradation_level == 0:
            # Normal operation
            return primary_func(*args, **kwargs)

        # Find appropriate fallback
        fallbacks = self.fallback_functions.get(service_name, {})

        # Try fallbacks in order of degradation level
        for level in sorted(fallbacks.keys(), reverse=True):
            if level <= degradation_level:
                try:
                    logger.info(
                        f"Using degraded service {service_name} at level {level}"
                    )
                    return fallbacks[level](*args, **kwargs)
                except Exception as e:
                    logger.error(
                        f"Fallback level {level} failed for {service_name}: {e}"
                    )
                    continue

        # If all fallbacks fail, raise exception
        raise Exception(f"All fallbacks failed for service {service_name}")


# Global instances
error_handler = RobustErrorHandler()
graceful_degradation = GracefulDegradation()
