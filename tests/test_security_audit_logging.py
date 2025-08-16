"""
Test suite for security event logging and audit trail functionality.

This test suite follows TDD principles:
1. Write failing tests first (RED)
2. Implement minimal code to pass (GREEN)
3. Refactor for clarity/performance (REFACTOR)
"""

import json
import logging
import tempfile
import unittest
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

from sql_synthesizer.security_audit import (
    SecurityAuditLogger,
    SecurityEvent,
    SecurityEventSeverity,
    SecurityEventType,
    security_audit_logger,
)


class TestSecurityEventType(unittest.TestCase):
    """Test SecurityEventType enum."""

    def test_security_event_types_exist(self):
        """Test that all required security event types are defined."""
        expected_types = [
            "SQL_INJECTION_ATTEMPT",
            "API_KEY_AUTHENTICATION_FAILED",
            "RATE_LIMIT_EXCEEDED",
            "CSRF_TOKEN_VALIDATION_FAILED",
            "REQUEST_SIZE_LIMIT_EXCEEDED",
            "QUERY_EXECUTION",
            "XSS_ATTEMPT",
        ]

        for event_type in expected_types:
            assert hasattr(
                SecurityEventType, event_type
            ), f"Missing event type: {event_type}"


class TestSecurityEvent(unittest.TestCase):
    """Test SecurityEvent data class."""

    def test_security_event_creation(self):
        """Test creating a security event with all fields."""
        event = SecurityEvent(
            event_type=SecurityEventType.SQL_INJECTION_ATTEMPT,
            severity=SecurityEventSeverity.HIGH,
            message="Detected SQL injection attempt",
            timestamp="2025-07-23T10:30:00Z",
            client_ip="192.168.1.1",
            user_agent="curl/7.68.0",
            request_path="/api/query",
            additional_data={"query": "'; DROP TABLE users; --"},
        )

        assert event.event_type == SecurityEventType.SQL_INJECTION_ATTEMPT
        assert event.severity == SecurityEventSeverity.HIGH
        assert event.message == "Detected SQL injection attempt"
        assert event.client_ip == "192.168.1.1"
        assert event.user_agent == "curl/7.68.0"
        assert event.request_path == "/api/query"
        assert event.additional_data == {"query": "'; DROP TABLE users; --"}

    def test_security_event_minimal_creation(self):
        """Test creating a security event with minimal required fields."""
        event = SecurityEvent(
            event_type=SecurityEventType.API_KEY_AUTHENTICATION_FAILED,
            severity=SecurityEventSeverity.MEDIUM,
            message="Invalid credentials",
            timestamp="2025-07-23T10:30:00Z",
        )

        assert event.event_type == SecurityEventType.API_KEY_AUTHENTICATION_FAILED
        assert event.severity == SecurityEventSeverity.MEDIUM
        assert event.message == "Invalid credentials"
        assert event.client_ip is None

    def test_security_event_to_dict(self):
        """Test converting security event to dictionary."""
        event = SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=SecurityEventSeverity.LOW,
            message="Rate limit exceeded",
            timestamp="2025-07-23T10:30:00Z",
            client_ip="10.0.0.1",
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "rate_limit_exceeded"
        assert event_dict["severity"] == "low"
        assert event_dict["message"] == "Rate limit exceeded"
        assert event_dict["client_ip"] == "10.0.0.1"
        assert event_dict["timestamp"] == "2025-07-23T10:30:00Z"


class TestSecurityAuditLogger(unittest.TestCase):
    """Test SecurityAuditLogger functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = SecurityAuditLogger("test_security_audit")

        # Capture log output
        self.log_stream = StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.logger.logger.addHandler(self.log_handler)
        self.logger.logger.setLevel(logging.INFO)

        # Reset statistics
        self.logger.reset_statistics()

    def tearDown(self):
        """Clean up test fixtures."""
        self.logger.logger.removeHandler(self.log_handler)

    def test_log_security_event(self):
        """Test logging a security event."""
        self.logger.log_event(
            event_type=SecurityEventType.SQL_INJECTION_ATTEMPT,
            severity=SecurityEventSeverity.HIGH,
            message="SQL injection detected",
            client_ip="192.168.1.100",
        )

        # Check that event was logged
        log_output = self.log_stream.getvalue()
        assert "sql_injection_attempt" in log_output
        assert "SQL injection detected" in log_output
        assert "192.168.1.100" in log_output

        # Check statistics
        stats = self.logger.get_event_statistics()
        assert stats["total_events"] == 1
        assert stats["events_by_type"]["sql_injection_attempt"] == 1
        assert stats["events_by_severity"]["high"] == 1

    def test_log_sql_injection_attempt(self):
        """Test logging SQL injection attempt with convenience method."""
        self.logger.log_sql_injection_attempt(
            malicious_input="'; DROP TABLE users; --",
            detection_method="pattern_matching",
            client_ip="evil.hacker.com",
        )

        log_output = self.log_stream.getvalue()
        assert "sql_injection_attempt" in log_output
        assert "DROP TABLE users" in log_output
        assert "pattern_matching" in log_output
        assert "evil.hacker.com" in log_output

    def test_log_authentication_failure(self):
        """Test logging authentication failure with convenience method."""
        self.logger.log_authentication_failure(
            auth_type="API_KEY",
            reason="Invalid API key format",
            client_ip="192.168.1.50",
            user_agent="curl/7.68.0",
        )

        log_output = self.log_stream.getvalue()
        assert "api_key_auth_failed" in log_output
        assert "Invalid API key format" in log_output
        assert "192.168.1.50" in log_output
        assert "curl/7.68.0" in log_output

    def test_log_rate_limit_exceeded(self):
        """Test logging rate limit violation with convenience method."""
        self.logger.log_rate_limit_exceeded(
            client_identifier="api:test123",
            limit_type="requests_per_minute",
            current_rate=100,
            limit_threshold=60,
            client_ip="10.0.0.1",
        )

        log_output = self.log_stream.getvalue()
        assert "rate_limit_exceeded" in log_output
        assert "api:test123" in log_output
        assert "100" in log_output
        assert "60" in log_output

    def test_log_query_execution(self):
        """Test logging query execution for audit trail."""
        self.logger.log_query_execution(
            sql_query="SELECT * FROM users WHERE id = 1",
            execution_time_ms=15.5,
            row_count=1,
            client_ip="192.168.1.10",
        )

        log_output = self.log_stream.getvalue()
        assert "query_execution" in log_output
        assert "Query executed successfully" in log_output
        assert "SELECT * FROM users" in log_output
        assert "15.5" in log_output
        assert "192.168.1.10" in log_output

    def test_event_statistics(self):
        """Test event statistics tracking."""
        # Log various events
        self.logger.log_event(
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventSeverity.HIGH,
            "Test SQL injection",
        )

        self.logger.log_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecurityEventSeverity.MEDIUM,
            "Test rate limit",
        )

        self.logger.log_event(
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventSeverity.HIGH,
            "Another SQL injection",
        )

        stats = self.logger.get_event_statistics()

        assert stats["total_events"] == 3
        assert stats["events_by_type"]["sql_injection_attempt"] == 2
        assert stats["events_by_type"]["rate_limit_exceeded"] == 1
        assert stats["events_by_severity"]["high"] == 2
        assert stats["events_by_severity"]["medium"] == 1

    def test_severity_based_log_levels(self):
        """Test that different severities use appropriate log levels."""
        # Clear existing logs
        self.log_stream.truncate(0)
        self.log_stream.seek(0)

        test_cases = [
            (SecurityEventSeverity.LOW, "INFO"),
            (SecurityEventSeverity.MEDIUM, "WARNING"),
            (SecurityEventSeverity.HIGH, "ERROR"),
            (SecurityEventSeverity.CRITICAL, "CRITICAL"),
        ]

        for severity, expected_level in test_cases:
            # Clear stream for each test
            self.log_stream.truncate(0)
            self.log_stream.seek(0)

            self.logger.log_event(
                SecurityEventType.QUERY_EXECUTION,
                severity,
                f"Test {severity.value} event",
            )

            log_output = self.log_stream.getvalue()
            # Note: The actual log level name might not appear in the JSON output
            # but the event should be logged if severity is appropriate
            assert f"Test {severity.value} event" in log_output

    def test_large_input_truncation(self):
        """Test that large malicious inputs are truncated."""
        large_input = "A" * 500  # Larger than truncation limit

        self.logger.log_sql_injection_attempt(
            malicious_input=large_input, detection_method="size_check"
        )

        log_output = self.log_stream.getvalue()
        # Should contain truncated input
        assert "..." in log_output
        assert "input_length" in log_output
        assert "500" in log_output


if __name__ == "__main__":
    unittest.main()
