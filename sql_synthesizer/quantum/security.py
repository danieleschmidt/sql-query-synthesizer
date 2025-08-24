"""
Security features and circuit breaker patterns for quantum components
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .exceptions import QuantumCircuitBreakerError, QuantumSecurityError


class ThreatLevel(Enum):
    """Security threat levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class SecurityEvent:
    """Represents a security event"""

    event_type: str
    timestamp: float
    threat_level: ThreatLevel
    client_id: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """TODO: Add docstring"""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "threat_level": self.threat_level.value,
            "client_id": self.client_id,
            "details": self.details,
        }


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state_change_time: float = 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        total = self.failure_count + self.success_count
        return self.failure_count / total if total > 0 else 0.0


class QuantumCircuitBreaker:
    """
    Circuit breaker pattern for quantum operations
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitBreakerState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()

        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_failure: Optional[Callable] = None
        self._fallback_handler: Optional[Callable] = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker

        Args:
            func: Function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            QuantumCircuitBreakerError: If circuit is open
        """
        with self._lock:
            current_state = self._get_current_state()

            if current_state == CircuitBreakerState.OPEN:
                # Circuit is open - fail fast
                if self._fallback_handler:
                    return self._fallback_handler(*args, **kwargs)

                raise QuantumCircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open",
                    circuit_name=self.name,
                    failure_count=self._stats.failure_count,
                    failure_threshold=self.failure_threshold,
                    details={
                        "last_failure_time": self._stats.last_failure_time,
                        "recovery_timeout": self.recovery_timeout,
                    },
                )

            # Try to execute the function
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result

            except Exception as e:
                self._on_failure_occurred(e)
                raise

    def _get_current_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        if self._state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if time.time() - self._stats.last_failure_time >= self.recovery_timeout:
                self._change_state(CircuitBreakerState.HALF_OPEN)

        return self._state

    def _on_success(self):
        """Handle successful operation"""
        with self._lock:
            self._stats.success_count += 1
            self._stats.last_success_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Check if we've had enough successes to close the circuit
                if self._stats.success_count >= self.success_threshold:
                    self._stats.failure_count = 0  # Reset failure count
                    self._change_state(CircuitBreakerState.CLOSED)

    def _on_failure_occurred(self, exception: Exception):
        """Handle failed operation"""
        with self._lock:
            self._stats.failure_count += 1
            self._stats.last_failure_time = time.time()

            # Call failure callback if set
            if self._on_failure:
                self._on_failure(exception, self._stats)

            # Check if we should open the circuit
            if (
                self._state == CircuitBreakerState.CLOSED
                and self._stats.failure_count >= self.failure_threshold
            ):
                self._change_state(CircuitBreakerState.OPEN)
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Failed during recovery - go back to open
                self._change_state(CircuitBreakerState.OPEN)

    def _change_state(self, new_state: CircuitBreakerState):
        """Change circuit breaker state"""
        old_state = self._state
        self._state = new_state
        self._stats.state_change_time = time.time()

        # Reset success count when changing states
        if new_state in (CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN):
            self._stats.success_count = 0

        # Call state change callback if set
        if self._on_state_change:
            self._on_state_change(old_state, new_state, self._stats)

    def set_fallback_handler(self, handler: Callable):
        """Set fallback handler for when circuit is open"""
        self._fallback_handler = handler

    def set_on_state_change(self, callback: Callable):
        """Set callback for state changes"""
        self._on_state_change = callback

    def set_on_failure(self, callback: Callable):
        """Set callback for failures"""
        self._on_failure = callback

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics"""
        with self._lock:
            return CircuitBreakerStats(
                failure_count=self._stats.failure_count,
                success_count=self._stats.success_count,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                state_change_time=self._stats.state_change_time,
            )

    def get_state(self) -> CircuitBreakerState:
        """Get current state"""
        with self._lock:
            return self._get_current_state()

    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self._stats.failure_count = 0
            self._stats.success_count = 0
            self._change_state(CircuitBreakerState.CLOSED)


class QuantumSecurityManager:
    """
    Comprehensive security manager for quantum operations
    """

    def __init__(
        self,
        rate_limit_per_minute: int = 60,
        max_request_size: int = 1024 * 1024,  # 1MB
        enable_audit_logging: bool = True,
    ):
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_request_size = max_request_size
        self.enable_audit_logging = enable_audit_logging

        # Security tracking
        self._client_requests: Dict[str, List[float]] = {}
        self._security_events: List[SecurityEvent] = []
        self._blocked_clients: Dict[str, float] = {}  # client_id -> block_until_time
        self._lock = threading.RLock()

        # Security callbacks
        self._security_event_handler: Optional[Callable] = None

        # Circuit breakers for different operations
        self.circuit_breakers = {
            "quantum_optimization": QuantumCircuitBreaker(
                "quantum_optimization", failure_threshold=3, recovery_timeout=30.0
            ),
            "quantum_scheduling": QuantumCircuitBreaker(
                "quantum_scheduling", failure_threshold=5, recovery_timeout=60.0
            ),
            "plan_generation": QuantumCircuitBreaker(
                "plan_generation", failure_threshold=10, recovery_timeout=15.0
            ),
        }

    def authenticate_request(
        self, client_id: str, api_key: Optional[str] = None
    ) -> bool:
        """
        Authenticate quantum operation request

        Args:
            client_id: Unique client identifier
            api_key: Optional API key for authentication

        Returns:
            True if authenticated, False otherwise
        """
        with self._lock:
            # Check if client is blocked
            if client_id in self._blocked_clients:
                if time.time() < self._blocked_clients[client_id]:
                    self._log_security_event(
                        "blocked_client_attempt",
                        client_id,
                        ThreatLevel.MEDIUM,
                        {"reason": "client_still_blocked"},
                    )
                    return False
                else:
                    # Unblock client
                    del self._blocked_clients[client_id]

            # Simple authentication - in production, this would integrate with proper auth
            if api_key and len(api_key) < 10:
                self._log_security_event(
                    "weak_api_key",
                    client_id,
                    ThreatLevel.HIGH,
                    {"api_key_length": len(api_key)},
                )
                return False

            return True

    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limit

        Args:
            client_id: Unique client identifier

        Returns:
            True if within rate limit, False otherwise
        """
        with self._lock:
            current_time = time.time()
            minute_ago = current_time - 60.0

            # Initialize client request history
            if client_id not in self._client_requests:
                self._client_requests[client_id] = []

            # Clean old requests
            self._client_requests[client_id] = [
                req_time
                for req_time in self._client_requests[client_id]
                if req_time > minute_ago
            ]

            # Check rate limit
            request_count = len(self._client_requests[client_id])

            if request_count >= self.rate_limit_per_minute:
                # Rate limit exceeded
                self._log_security_event(
                    "rate_limit_exceeded",
                    client_id,
                    ThreatLevel.HIGH,
                    {
                        "request_count": request_count,
                        "rate_limit": self.rate_limit_per_minute,
                        "time_window": "1_minute",
                    },
                )

                # Block client for 5 minutes
                self._blocked_clients[client_id] = current_time + 300.0
                return False

            # Add current request
            self._client_requests[client_id].append(current_time)
            return True

    def validate_request_size(self, data: Any, client_id: str) -> bool:
        """
        Validate request size to prevent DoS attacks

        Args:
            data: Request data to validate
            client_id: Client identifier

        Returns:
            True if size is acceptable, False otherwise
        """
        # Estimate data size (simplified)
        data_size = len(str(data))

        if data_size > self.max_request_size:
            self._log_security_event(
                "oversized_request",
                client_id,
                ThreatLevel.MEDIUM,
                {"data_size": data_size, "max_size": self.max_request_size},
            )
            return False

        return True

    def detect_anomalies(
        self, operation: str, client_id: str, metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Detect anomalous behavior in quantum operations

        Args:
            operation: Type of operation
            client_id: Client identifier
            metrics: Operation metrics

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Check for unusual execution times
        if "execution_time" in metrics:
            exec_time = metrics["execution_time"]
            if exec_time > 30.0:  # More than 30 seconds
                anomalies.append("excessive_execution_time")

        # Check for unusual resource usage
        if "resource_usage" in metrics:
            usage = metrics["resource_usage"]
            if usage > 0.9:  # More than 90% resource usage
                anomalies.append("high_resource_usage")

        # Check for unusual quantum parameters
        if "qubit_count" in metrics and metrics["qubit_count"] > 100:
            anomalies.append("excessive_qubit_count")

        if "temperature" in metrics and metrics["temperature"] > 5000:
            anomalies.append("excessive_temperature")

        # Log anomalies
        if anomalies:
            self._log_security_event(
                "anomaly_detected",
                client_id,
                ThreatLevel.MEDIUM,
                {"operation": operation, "anomalies": anomalies, "metrics": metrics},
            )

        return anomalies

    def secure_execute(
        self, operation: str, client_id: str, func: Callable, *args, **kwargs
    ) -> Any:
        """
        Securely execute a quantum operation

        Args:
            operation: Operation type
            client_id: Client identifier
            func: Function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            QuantumSecurityError: If security checks fail
            QuantumCircuitBreakerError: If circuit breaker is open
        """
        # Authentication check
        if not self.authenticate_request(client_id):
            raise QuantumSecurityError(
                "Authentication failed",
                security_check="authentication",
                threat_level="high",
                client_info={"client_id": client_id},
            )

        # Rate limiting check
        if not self.check_rate_limit(client_id):
            raise QuantumSecurityError(
                "Rate limit exceeded",
                security_check="rate_limit",
                threat_level="high",
                client_info={"client_id": client_id},
            )

        # Request size validation
        request_data = {"args": args, "kwargs": kwargs}
        if not self.validate_request_size(request_data, client_id):
            raise QuantumSecurityError(
                "Request size too large",
                security_check="request_size",
                threat_level="medium",
                client_info={"client_id": client_id},
            )

        # Execute through circuit breaker if available
        circuit_breaker = self.circuit_breakers.get(operation)

        start_time = time.time()
        try:
            if circuit_breaker:
                result = circuit_breaker.call(func, *args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Check for anomalies
            execution_time = time.time() - start_time
            metrics = {"execution_time": execution_time, "operation": operation}

            anomalies = self.detect_anomalies(operation, client_id, metrics)

            # Log successful operation
            self._log_security_event(
                "operation_completed",
                client_id,
                ThreatLevel.LOW,
                {
                    "operation": operation,
                    "execution_time": execution_time,
                    "anomalies": anomalies,
                },
            )

            return result

        except Exception as e:
            # Log failed operation
            execution_time = time.time() - start_time
            self._log_security_event(
                "operation_failed",
                client_id,
                ThreatLevel.MEDIUM,
                {
                    "operation": operation,
                    "execution_time": execution_time,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    def _log_security_event(
        self,
        event_type: str,
        client_id: str,
        threat_level: ThreatLevel,
        details: Dict[str, Any],
    ):
        """Log a security event"""
        if not self.enable_audit_logging:
            return

        event = SecurityEvent(
            event_type=event_type,
            timestamp=time.time(),
            threat_level=threat_level,
            client_id=client_id,
            details=details,
        )

        with self._lock:
            self._security_events.append(event)

            # Keep only last 1000 events to prevent memory issues
            if len(self._security_events) > 1000:
                self._security_events = self._security_events[-1000:]

        # Call event handler if set
        if self._security_event_handler:
            self._security_event_handler(event)

    def get_security_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events"""
        with self._lock:
            return self._security_events[-limit:] if self._security_events else []

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        with self._lock:
            total_events = len(self._security_events)

            # Count by threat level
            threat_counts = {level.value: 0 for level in ThreatLevel}
            for event in self._security_events:
                threat_counts[event.threat_level.value] += 1

            # Count by event type
            event_type_counts = {}
            for event in self._security_events:
                event_type_counts[event.event_type] = (
                    event_type_counts.get(event.event_type, 0) + 1
                )

            # Circuit breaker stats
            circuit_stats = {}
            for name, cb in self.circuit_breakers.items():
                stats = cb.get_stats()
                circuit_stats[name] = {
                    "state": cb.get_state().value,
                    "failure_count": stats.failure_count,
                    "success_count": stats.success_count,
                    "failure_rate": stats.failure_rate,
                }

            return {
                "total_events": total_events,
                "threat_level_counts": threat_counts,
                "event_type_counts": event_type_counts,
                "blocked_clients": len(self._blocked_clients),
                "circuit_breaker_stats": circuit_stats,
                "rate_limit_per_minute": self.rate_limit_per_minute,
                "max_request_size": self.max_request_size,
            }

    def set_security_event_handler(self, handler: Callable[[SecurityEvent], None]):
        """Set handler for security events"""
        self._security_event_handler = handler

    def block_client(self, client_id: str, duration_seconds: float):
        """Manually block a client"""
        with self._lock:
            self._blocked_clients[client_id] = time.time() + duration_seconds

        self._log_security_event(
            "client_blocked",
            client_id,
            ThreatLevel.HIGH,
            {"duration_seconds": duration_seconds, "reason": "manual_block"},
        )

    def unblock_client(self, client_id: str):
        """Manually unblock a client"""
        with self._lock:
            if client_id in self._blocked_clients:
                del self._blocked_clients[client_id]

        self._log_security_event(
            "client_unblocked", client_id, ThreatLevel.LOW, {"reason": "manual_unblock"}
        )


# Global security manager instance
quantum_security = QuantumSecurityManager()


# Security decorators
def secure_quantum_operation(operation_name: str):
    """
    Decorator to secure quantum operations

    Usage:
        @secure_quantum_operation("optimization")
        def optimize_query(client_id, ...):
            ...
    """

     """TODO: Add docstring"""
         """TODO: Add docstring"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract client_id from arguments
            client_id = kwargs.get("client_id", "unknown")
            if not client_id and args:
                # Assume first argument is client_id if not in kwargs
                client_id = str(args[0]) if args else "unknown"

            return quantum_security.secure_execute(
                operation_name, client_id, func, *args, **kwargs
            )

        return wrapper

    return decorator


def with_circuit_breaker(breaker_name: str):
    """
    Decorator to wrap function with circuit breaker

    Usage:
        @with_circuit_breaker("optimization")
        def optimize_query(...):
            ...
    """
        """TODO: Add docstring"""
        """TODO: Add docstring"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            circuit_breaker = quantum_security.circuit_breakers.get(breaker_name)
            if circuit_breaker:
                return circuit_breaker.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
