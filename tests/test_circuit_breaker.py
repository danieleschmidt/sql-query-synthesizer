"""Tests for circuit breaker functionality in LLM providers."""

import time
from unittest.mock import Mock, patch

import pytest

from sql_synthesizer.circuit_breaker import CircuitBreaker, CircuitBreakerState


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset circuit breaker state between tests
        CircuitBreaker._instances = {}

    def test_circuit_breaker_closed_state_initial(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker("test-provider", failure_threshold=3, recovery_timeout=60)

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.is_request_allowed()

    def test_circuit_breaker_record_success(self):
        """Test recording successful operations."""
        cb = CircuitBreaker("test-provider", failure_threshold=3, recovery_timeout=60)

        cb.record_success()

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_record_failure_under_threshold(self):
        """Test recording failures under threshold keeps circuit closed."""
        cb = CircuitBreaker("test-provider", failure_threshold=3, recovery_timeout=60)

        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 2
        assert cb.is_request_allowed()

    def test_circuit_breaker_opens_on_threshold(self):
        """Test circuit breaker opens when failure threshold is reached."""
        cb = CircuitBreaker("test-provider", failure_threshold=3, recovery_timeout=60)

        # Record failures up to threshold
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 3
        assert not cb.is_request_allowed()

    def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker moves to half-open after recovery timeout."""
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0

            cb = CircuitBreaker(
                "test-provider", failure_threshold=3, recovery_timeout=60
            )

            # Open the circuit
            cb.record_failure()
            cb.record_failure()
            cb.record_failure()

            assert cb.state == CircuitBreakerState.OPEN

            # Advance time beyond recovery timeout
            mock_time.return_value = 1070.0  # 70 seconds later

            assert cb.is_request_allowed()
            assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_half_open_success_closes_circuit(self):
        """Test successful request in half-open state closes circuit."""
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0

            cb = CircuitBreaker(
                "test-provider", failure_threshold=3, recovery_timeout=60
            )

            # Open the circuit
            cb.record_failure()
            cb.record_failure()
            cb.record_failure()

            # Move to half-open
            mock_time.return_value = 1070.0
            cb.is_request_allowed()  # Triggers half-open

            # Record success in half-open state
            cb.record_success()

            assert cb.state == CircuitBreakerState.CLOSED
            assert cb.failure_count == 0

    def test_circuit_breaker_half_open_failure_reopens_circuit(self):
        """Test failure in half-open state reopens circuit."""
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0

            cb = CircuitBreaker(
                "test-provider", failure_threshold=3, recovery_timeout=60
            )

            # Open the circuit
            cb.record_failure()
            cb.record_failure()
            cb.record_failure()

            # Move to half-open
            mock_time.return_value = 1070.0
            cb.is_request_allowed()  # Triggers half-open

            # Record failure in half-open state
            cb.record_failure()

            assert cb.state == CircuitBreakerState.OPEN
            assert not cb.is_request_allowed()

    def test_circuit_breaker_singleton_per_provider(self):
        """Test circuit breaker is singleton per provider name."""
        cb1 = CircuitBreaker("provider-a", failure_threshold=3, recovery_timeout=60)
        cb2 = CircuitBreaker("provider-a", failure_threshold=5, recovery_timeout=120)
        cb3 = CircuitBreaker("provider-b", failure_threshold=3, recovery_timeout=60)

        # Same provider should return same instance
        assert cb1 is cb2
        # Different provider should return different instance
        assert cb1 is not cb3

    def test_circuit_breaker_success_resets_failure_count(self):
        """Test success resets failure count in closed state."""
        cb = CircuitBreaker("test-provider", failure_threshold=3, recovery_timeout=60)

        # Record some failures
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        # Record success
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_metrics_tracking(self):
        """Test circuit breaker tracks metrics correctly."""
        cb = CircuitBreaker("test-provider", failure_threshold=3, recovery_timeout=60)

        # Test success metrics
        cb.record_success()
        assert cb.success_count == 1

        # Test failure metrics
        cb.record_failure()
        assert cb.failure_count == 1

        # Test total requests
        assert cb.total_requests == 2

    def test_circuit_breaker_get_status(self):
        """Test circuit breaker status reporting."""
        cb = CircuitBreaker("test-provider", failure_threshold=3, recovery_timeout=60)

        status = cb.get_status()

        expected_keys = {
            "provider_name",
            "state",
            "failure_count",
            "success_count",
            "total_requests",
            "failure_threshold",
            "recovery_timeout",
        }
        assert set(status.keys()) == expected_keys
        assert status["provider_name"] == "test-provider"
        assert status["state"] == CircuitBreakerState.CLOSED.value
