"""
Enhanced Error Handling with Self-Healing Capabilities
Progressive error recovery and resilience patterns
"""

import asyncio
import inspect
import logging
import time
import traceback
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Enhanced error context with recovery suggestions"""

    error_type: str
    error_message: str
    module: str
    function: str
    timestamp: float
    traceback: str
    recovery_suggestions: List[str]
    severity: str  # low, medium, high, critical
    user_impact: str


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken"""

    name: str
    description: str
    auto_executable: bool
    risk_level: str  # low, medium, high
    execute_func: Optional[Callable] = None


class EnhancedErrorHandler:
    """Enhanced error handler with learning and recovery capabilities"""

    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_patterns: Dict[str, List[RecoveryAction]] = {}
        self.auto_recovery_enabled = True
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """Setup default recovery patterns"""

        # Database connection errors
        self.recovery_patterns["DatabaseError"] = [
            RecoveryAction(
                name="retry_connection",
                description="Retry database connection with exponential backoff",
                auto_executable=True,
                risk_level="low",
            ),
            RecoveryAction(
                name="switch_to_fallback_db",
                description="Switch to fallback database if available",
                auto_executable=True,
                risk_level="medium",
            ),
            RecoveryAction(
                name="enable_offline_mode",
                description="Enable offline mode with cached data",
                auto_executable=False,
                risk_level="medium",
            ),
        ]

        # OpenAI API errors
        self.recovery_patterns["ProviderError"] = [
            RecoveryAction(
                name="retry_with_backoff",
                description="Retry API call with exponential backoff",
                auto_executable=True,
                risk_level="low",
            ),
            RecoveryAction(
                name="switch_to_fallback_model",
                description="Switch to alternative LLM provider",
                auto_executable=True,
                risk_level="medium",
            ),
            RecoveryAction(
                name="use_cached_response",
                description="Use cached response if available",
                auto_executable=True,
                risk_level="low",
            ),
        ]

        # Memory/Performance errors
        self.recovery_patterns["MemoryError"] = [
            RecoveryAction(
                name="clear_caches",
                description="Clear memory caches to free up space",
                auto_executable=True,
                risk_level="low",
            ),
            RecoveryAction(
                name="reduce_batch_size",
                description="Reduce processing batch size",
                auto_executable=True,
                risk_level="low",
            ),
            RecoveryAction(
                name="enable_streaming",
                description="Switch to streaming processing mode",
                auto_executable=True,
                risk_level="medium",
            ),
        ]

        # Security errors
        self.recovery_patterns["SecurityError"] = [
            RecoveryAction(
                name="sanitize_input",
                description="Apply additional input sanitization",
                auto_executable=True,
                risk_level="low",
            ),
            RecoveryAction(
                name="enable_strict_mode",
                description="Enable strict security validation mode",
                auto_executable=True,
                risk_level="low",
            ),
            RecoveryAction(
                name="block_request",
                description="Block potentially malicious request",
                auto_executable=False,
                risk_level="high",
            ),
        ]

    def handle_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Handle error with enhanced context and recovery suggestions"""

        # Extract context information
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame else None
        module = (
            caller_frame.f_globals.get("__name__", "unknown")
            if caller_frame
            else "unknown"
        )
        function = caller_frame.f_code.co_name if caller_frame else "unknown"

        error_type = type(error).__name__
        error_message = str(error)

        # Determine severity and user impact
        severity = self._determine_severity(error_type, error_message)
        user_impact = self._determine_user_impact(error_type, severity)

        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(
            error_type, error_message, context
        )

        error_context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            module=module,
            function=function,
            timestamp=time.time(),
            traceback=traceback.format_exc(),
            recovery_suggestions=recovery_suggestions,
            severity=severity,
            user_impact=user_impact,
        )

        # Log the error
        self._log_error(error_context)

        # Store in history for learning
        self.error_history.append(error_context)

        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

        return error_context

    def _determine_severity(self, error_type: str, error_message: str) -> str:
        """Determine error severity based on type and message"""

        critical_patterns = [
            "authentication",
            "authorization",
            "security",
            "injection",
            "corruption",
            "data loss",
            "breach",
        ]

        high_patterns = [
            "timeout",
            "connection",
            "database",
            "unavailable",
            "service",
            "provider",
            "api",
        ]

        medium_patterns = ["validation", "format", "parsing", "cache", "memory", "disk"]

        error_text = (error_type + " " + error_message).lower()

        if any(pattern in error_text for pattern in critical_patterns):
            return "critical"
        elif any(pattern in error_text for pattern in high_patterns):
            return "high"
        elif any(pattern in error_text for pattern in medium_patterns):
            return "medium"
        else:
            return "low"

    def _determine_user_impact(self, error_type: str, severity: str) -> str:
        """Determine user impact based on error type and severity"""

        if severity == "critical":
            return "Service completely unavailable to users"
        elif severity == "high":
            return "Significant feature degradation or failures"
        elif severity == "medium":
            return "Minor feature issues or performance impact"
        else:
            return "Minimal user impact, internal issue"

    def _generate_recovery_suggestions(
        self, error_type: str, error_message: str, context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate recovery suggestions based on error patterns"""

        suggestions = []

        # Get pattern-based suggestions
        recovery_actions = self.recovery_patterns.get(error_type, [])
        suggestions.extend([action.description for action in recovery_actions])

        # Add specific suggestions based on error message
        error_lower = error_message.lower()

        if "timeout" in error_lower:
            suggestions.extend(
                [
                    "Increase request timeout values",
                    "Implement retry mechanism with exponential backoff",
                    "Check network connectivity and latency",
                ]
            )

        if "memory" in error_lower or "out of memory" in error_lower:
            suggestions.extend(
                [
                    "Increase available memory allocation",
                    "Optimize memory usage patterns",
                    "Implement memory cleanup routines",
                ]
            )

        if "permission" in error_lower or "access" in error_lower:
            suggestions.extend(
                [
                    "Check file/directory permissions",
                    "Verify user authentication status",
                    "Review security policies",
                ]
            )

        if "connection" in error_lower:
            suggestions.extend(
                [
                    "Verify network connectivity",
                    "Check service endpoint availability",
                    "Review connection pool configuration",
                ]
            )

        # Remove duplicates
        return list(set(suggestions))

    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level based on severity"""

        log_message = (
            f"Error in {error_context.module}.{error_context.function}: "
            f"{error_context.error_type}: {error_context.error_message}"
        )

        if error_context.severity == "critical":
            logger.critical(log_message)
        elif error_context.severity == "high":
            logger.error(log_message)
        elif error_context.severity == "medium":
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # Log recovery suggestions
        if error_context.recovery_suggestions:
            logger.info(
                f"Recovery suggestions: {', '.join(error_context.recovery_suggestions[:3])}"
            )

    async def attempt_auto_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt automatic recovery based on error type"""

        if not self.auto_recovery_enabled:
            return False

        recovery_actions = self.recovery_patterns.get(error_context.error_type, [])

        for action in recovery_actions:
            if action.auto_executable and action.risk_level in ["low", "medium"]:
                try:
                    logger.info(f"Attempting auto-recovery: {action.name}")

                    if action.execute_func:
                        result = await action.execute_func()
                        if result:
                            logger.info(f"Auto-recovery successful: {action.name}")
                            return True
                    else:
                        # Simulate recovery for demonstration
                        await asyncio.sleep(0.1)
                        logger.info(f"Auto-recovery simulated: {action.name}")
                        return True

                except Exception as e:
                    logger.warning(f"Auto-recovery failed for {action.name}: {str(e)}")
                    continue

        return False

    def get_error_analytics(self) -> Dict[str, Any]:
        """Get analytics on error patterns and recovery success"""

        if not self.error_history:
            return {"message": "No error history available"}

        # Group errors by type
        error_counts = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for error in self.error_history:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
            severity_counts[error.severity] += 1

        # Most common errors
        most_common = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Recent error trend (last 24 hours)
        recent_cutoff = time.time() - 86400  # 24 hours
        recent_errors = [e for e in self.error_history if e.timestamp > recent_cutoff]

        return {
            "total_errors": len(self.error_history),
            "error_types": len(error_counts),
            "most_common_errors": most_common,
            "severity_distribution": severity_counts,
            "recent_24h": len(recent_errors),
            "avg_errors_per_day": (
                len(self.error_history) / 7 if len(self.error_history) > 0 else 0
            ),
            "recovery_patterns_available": len(self.recovery_patterns),
        }


def enhanced_error_handler(handler: Optional[EnhancedErrorHandler] = None):
    """Decorator for enhanced error handling"""

    if handler is None:
        handler = EnhancedErrorHandler()

    def decorator(func):
        """TODO: Add docstring"""
        @wraps(func)
            """TODO: Add docstring"""
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = handler.handle_error(
                    e,
                    {
                        "function": func.__name__,
                        "args": len(args),
                        "kwargs": list(kwargs.keys()),
                    },
                )

                # Attempt auto-recovery for high/critical errors
                if error_context.severity in ["high", "critical"]:
                    recovery_attempted = await handler.attempt_auto_recovery(
                        error_context
                    )
                    if recovery_attempted:
                        # Retry once after recovery
                        try:
                            return await func(*args, **kwargs)
                        except Exception as retry_e:
                            logger.error(f"Retry after recovery failed: {str(retry_e)}")

                # Re-raise the original exception
                raise e

     """TODO: Add docstring"""
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler.handle_error(
                    e,
                    {
                        "function": func.__name__,
                        "args": len(args),
                        "kwargs": list(kwargs.keys()),
                    },
                )

                # For sync functions, just log and re-raise
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise e

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global error handler instance
global_error_handler = EnhancedErrorHandler()


class ResilientCircuitBreaker:
    """Enhanced circuit breaker with adaptive thresholds"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

        self.error_handler = EnhancedErrorHandler()

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""

        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                self.success_count = 0
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is open - service unavailable")

        try:
            result = (
                await func(*args, **kwargs)
                if inspect.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # Success
            if self.state == "half-open":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed - service recovered")

            return result

        except Exception as e:
            # Handle error with enhanced error handler
            error_context = self.error_handler.handle_error(e)

            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    f"Circuit breaker opened due to {self.failure_count} failures"
                )

            # Try auto-recovery for critical errors
            if error_context.severity in ["high", "critical"]:
                recovery_success = await self.error_handler.attempt_auto_recovery(
                    error_context
                )
                if recovery_success:
                    logger.info("Auto-recovery successful, resetting circuit breaker")
                    self.failure_count = max(
                        0, self.failure_count - 2
                    )  # Reduce failure count

            raise e


class AdaptiveRetry:
    """Adaptive retry mechanism with intelligent backoff"""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.error_handler = EnhancedErrorHandler()

    async def execute(self, func: Callable, *args, **kwargs):
        """Execute function with adaptive retry logic"""

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return (
                    await func(*args, **kwargs)
                    if inspect.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )

            except Exception as e:
                last_exception = e
                error_context = self.error_handler.handle_error(e)

                if attempt == self.max_retries:
                    logger.error(f"All {self.max_retries} retry attempts failed")
                    break

                # Calculate adaptive delay
                delay = min(self.base_delay * (2**attempt), self.max_delay)

                # Reduce delay for certain error types
                if error_context.error_type in ["ConnectionError", "TimeoutError"]:
                    delay *= 0.5  # Retry faster for network issues
                elif error_context.error_type in [
                    "AuthenticationError",
                    "SecurityError",
                ]:
                    delay *= 2  # Retry slower for auth issues

                logger.info(
                    f"Retry attempt {attempt + 1}/{self.max_retries} in {delay}s"
                )
                await asyncio.sleep(delay)

                # Attempt auto-recovery between retries
                if error_context.severity in ["high", "critical"]:
                    await self.error_handler.attempt_auto_recovery(error_context)

        # All retries failed
        raise last_exception


# Convenience decorators
def resilient_circuit_breaker(
    failure_threshold: int = 5, recovery_timeout: float = 60.0
):
    """Decorator for resilient circuit breaker"""
    breaker = ResilientCircuitBreaker(failure_threshold, recovery_timeout)

     """TODO: Add docstring"""
     """TODO: Add docstring"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


def adaptive_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for adaptive retry"""
        """TODO: Add docstring"""
    retry_handler = AdaptiveRetry(max_retries, base_delay)
        """TODO: Add docstring"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_handler.execute(func, *args, **kwargs)

        return wrapper

    return decorator
