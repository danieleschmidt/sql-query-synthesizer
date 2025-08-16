"""Circuit breaker pattern for LLM provider resilience."""

import logging
import time
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failures detected, requests blocked
    HALF_OPEN = "half_open"  # Recovery testing, limited requests allowed


class CircuitBreaker:
    """Circuit breaker for LLM provider reliability.

    Implements the circuit breaker pattern to prevent cascading failures
    when LLM providers are unavailable or experiencing issues.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Provider is failing, requests are blocked to allow recovery
    - HALF_OPEN: Testing if provider has recovered, limited requests allowed
    """

    # Class-level instances dictionary for singleton pattern per provider
    _instances: Dict[str, "CircuitBreaker"] = {}

    def __new__(cls, provider_name: str, *args, **kwargs) -> "CircuitBreaker":
        """Ensure singleton per provider name."""
        if provider_name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[provider_name] = instance
        return cls._instances[provider_name]

    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            provider_name: Name of the LLM provider (e.g., 'openai')
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        # Only initialize once per provider (singleton pattern)
        if hasattr(self, "_initialized"):
            return

        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        # Circuit state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0

        self._initialized = True

        logger.info(
            f"Circuit breaker initialized for {provider_name}: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s"
        )

    def is_request_allowed(self) -> bool:
        """Check if requests are allowed through the circuit breaker.

        Returns:
            bool: True if request should be allowed, False if blocked
        """
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.last_failure_time >= self.recovery_timeout:
                logger.info(
                    f"Moving {self.provider_name} circuit to HALF_OPEN for recovery test"
                )
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Allow one request to test recovery
            return True

        return False

    def record_success(self) -> None:
        """Record a successful operation."""
        self.success_count += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Success in half-open state means provider has recovered
            logger.info(f"Recovery confirmed for {self.provider_name}, closing circuit")
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on any success in closed state
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failure in half-open state means provider still failing
            logger.warning(
                f"Recovery failed for {self.provider_name}, reopening circuit"
            )
            self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.CLOSED:
            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"Opening circuit for {self.provider_name}: "
                    f"{self.failure_count} failures reached threshold {self.failure_threshold}"
                )
                self.state = CircuitBreakerState.OPEN

    @property
    def total_requests(self) -> int:
        """Get total number of requests processed."""
        return self.success_count + self.failure_count

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status.

        Returns:
            Dict containing circuit breaker metrics and state
        """
        return {
            "provider_name": self.provider_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self.last_failure_time,
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state (for testing/admin)."""
        logger.info(f"Manually resetting circuit breaker for {self.provider_name}")
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
