"""Tests for security audit logging functionality."""

import json
import logging
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from sql_synthesizer.security_audit import (
    SecurityAuditLogger,
    SecurityEvent,
    SecurityEventSeverity,
    SecurityEventType,
    get_security_audit_logger,
)


class TestSecurityEvent:
    """Test SecurityEvent data class."""

    def test_security_event_creation(self):
        """Test creating a security event."""
        event = SecurityEvent(
            event_type=SecurityEventType.SQL_INJECTION_ATTEMPT,
            severity=SecurityEventSeverity.HIGH,
            message="Test SQL injection detected",
            timestamp="2025-07-21T10:00:00Z",
            client_ip="192.168.1.100",
            trace_id="trace-123",
        )

        assert event.event_type == SecurityEventType.SQL_INJECTION_ATTEMPT
        assert event.severity == SecurityEventSeverity.HIGH
        assert event.message == "Test SQL injection detected"
        assert event.client_ip == "192.168.1.100"
        assert event.trace_id == "trace-123"

    def test_security_event_to_dict(self):
        """Test converting security event to dictionary."""
        event = SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=SecurityEventSeverity.MEDIUM,
            message="Rate limit exceeded",
            timestamp="2025-07-21T10:00:00Z",
            client_ip="10.0.0.1",
            user_agent="TestAgent/1.0",
            additional_data={"limit": 100, "current": 150},
        )

        result = event.to_dict()

        assert result["event_type"] == "rate_limit_exceeded"
        assert result["severity"] == "medium"
        assert result["message"] == "Rate limit exceeded"
        assert result["client_ip"] == "10.0.0.1"
        assert result["user_agent"] == "TestAgent/1.0"
        assert result["additional_data"]["limit"] == 100
        assert result["additional_data"]["current"] == 150

    def test_security_event_with_minimal_data(self):
        """Test creating security event with minimal required data."""
        event = SecurityEvent(
            event_type=SecurityEventType.QUERY_EXECUTION,
            severity=SecurityEventSeverity.LOW,
            message="Query executed",
            timestamp="2025-07-21T10:00:00Z",
        )

        result = event.to_dict()

        assert result["event_type"] == "query_execution"
        assert result["severity"] == "low"
        assert result["message"] == "Query executed"
        assert result["client_ip"] is None
        assert result["additional_data"] == {}


class TestSecurityAuditLogger:
    """Test SecurityAuditLogger functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a fresh logger for each test
        self.audit_logger = SecurityAuditLogger("test_security_audit")

        # Mock the logger to capture log calls
        self.mock_logger = Mock()
        self.audit_logger.logger = self.mock_logger

    def test_logger_initialization(self):
        """Test security audit logger initialization."""
        logger = SecurityAuditLogger("custom_logger")
        assert logger.logger.name == "custom_logger"
        assert logger._event_counts == {}
        assert logger._severity_counts == {}

    def test_log_basic_event(self):
        """Test logging a basic security event."""
        self.audit_logger.log_event(
            event_type=SecurityEventType.UNSAFE_INPUT_DETECTED,
            severity=SecurityEventSeverity.MEDIUM,
            message="Unsafe input detected in query",
            client_ip="192.168.1.50",
            trace_id="trace-456",
        )

        # Verify logger was called
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args[0][0]

        # Parse the JSON log message
        log_data = json.loads(call_args)

        assert log_data["event_type"] == "unsafe_input_detected"
        assert log_data["severity"] == "medium"
        assert log_data["message"] == "Unsafe input detected in query"
        assert log_data["client_ip"] == "192.168.1.50"
        assert log_data["trace_id"] == "trace-456"

        # Verify statistics updated
        assert (
            self.audit_logger._event_counts[SecurityEventType.UNSAFE_INPUT_DETECTED]
            == 1
        )
        assert self.audit_logger._severity_counts[SecurityEventSeverity.MEDIUM] == 1

    def test_log_event_with_additional_data(self):
        """Test logging event with additional context data."""
        self.audit_logger.log_event(
            event_type=SecurityEventType.API_KEY_AUTHENTICATION_FAILED,
            severity=SecurityEventSeverity.HIGH,
            message="Invalid API key",
            client_ip="10.0.0.5",
            request_path="/api/query",
            request_method="POST",
            auth_type="bearer",
            provided_key="invalid-key-123",
        )

        self.mock_logger.error.assert_called_once()
        call_args = self.mock_logger.error.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["request_path"] == "/api/query"
        assert log_data["request_method"] == "POST"
        assert log_data["additional_data"]["auth_type"] == "bearer"
        assert log_data["additional_data"]["provided_key"] == "invalid-key-123"

    def test_log_levels_based_on_severity(self):
        """Test that appropriate log levels are used based on severity."""
        # Test LOW severity -> info
        self.audit_logger.log_event(
            SecurityEventType.QUERY_EXECUTION,
            SecurityEventSeverity.LOW,
            "Query executed",
        )
        self.mock_logger.info.assert_called_once()

        # Test MEDIUM severity -> warning
        self.audit_logger.log_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecurityEventSeverity.MEDIUM,
            "Rate limit exceeded",
        )
        self.mock_logger.warning.assert_called_once()

        # Test HIGH severity -> error
        self.audit_logger.log_event(
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventSeverity.HIGH,
            "SQL injection detected",
        )
        self.mock_logger.error.assert_called_once()

        # Test CRITICAL severity -> critical
        self.audit_logger.log_event(
            SecurityEventType.UNEXPECTED_ERROR,
            SecurityEventSeverity.CRITICAL,
            "System compromise detected",
        )
        self.mock_logger.critical.assert_called_once()

    def test_log_sql_injection_attempt(self):
        """Test specialized SQL injection logging."""
        malicious_input = "'; DROP TABLE users; --"

        self.audit_logger.log_sql_injection_attempt(
            malicious_input=malicious_input,
            detection_method="pattern_matching",
            client_ip="192.168.1.100",
            trace_id="trace-sql-123",
            user_question="Show all users",
        )

        self.mock_logger.error.assert_called_once()
        call_args = self.mock_logger.error.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event_type"] == "sql_injection_attempt"
        assert log_data["severity"] == "high"
        assert "SQL injection attempt detected" in log_data["message"]
        assert log_data["additional_data"]["malicious_input"] == malicious_input
        assert log_data["additional_data"]["detection_method"] == "pattern_matching"
        assert log_data["additional_data"]["input_length"] == len(malicious_input)
        assert log_data["additional_data"]["user_question"] == "Show all users"

    def test_log_sql_injection_with_long_input(self):
        """Test SQL injection logging with long input truncation."""
        # Create input longer than 200 characters
        long_input = "SELECT * FROM users WHERE id = 1 " + "OR 1=1 " * 50

        self.audit_logger.log_sql_injection_attempt(
            malicious_input=long_input,
            detection_method="ast_analysis",
            client_ip="10.0.0.1",
        )

        call_args = self.mock_logger.error.call_args[0][0]
        log_data = json.loads(call_args)

        # Verify truncation
        logged_input = log_data["additional_data"]["malicious_input"]
        assert len(logged_input) <= 203  # 200 + "..."
        assert logged_input.endswith("...")
        assert log_data["additional_data"]["input_length"] == len(long_input)

    def test_log_authentication_failure(self):
        """Test authentication failure logging."""
        self.audit_logger.log_authentication_failure(
            auth_type="api_key",
            reason="key_not_found",
            client_ip="172.16.0.10",
            user_agent="curl/7.68.0",
            trace_id="auth-trace-456",
            provided_key_prefix="sk-test",
        )

        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event_type"] == "api_key_auth_failed"
        assert log_data["severity"] == "medium"
        assert "Authentication failed: api_key - key_not_found" in log_data["message"]
        assert log_data["user_agent"] == "curl/7.68.0"
        assert log_data["additional_data"]["auth_type"] == "api_key"
        assert log_data["additional_data"]["failure_reason"] == "key_not_found"
        assert log_data["additional_data"]["provided_key_prefix"] == "sk-test"

    def test_log_rate_limit_exceeded(self):
        """Test rate limit violation logging."""
        self.audit_logger.log_rate_limit_exceeded(
            client_identifier="192.168.1.200",
            limit_type="requests_per_minute",
            current_rate=150,
            limit_threshold=100,
            client_ip="192.168.1.200",
            trace_id="rate-trace-789",
            window_size=60,
        )

        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event_type"] == "rate_limit_exceeded"
        assert (
            "Rate limit exceeded: requests_per_minute - 150/100" in log_data["message"]
        )
        assert log_data["additional_data"]["client_identifier"] == "192.168.1.200"
        assert log_data["additional_data"]["current_rate"] == 150
        assert log_data["additional_data"]["limit_threshold"] == 100
        assert log_data["additional_data"]["window_size"] == 60

    def test_log_query_execution(self):
        """Test query execution audit logging."""
        sql_query = (
            "SELECT id, name FROM users WHERE status = 'active' ORDER BY name LIMIT 10"
        )

        self.audit_logger.log_query_execution(
            sql_query=sql_query,
            execution_time_ms=45.2,
            row_count=8,
            client_ip="10.0.0.50",
            trace_id="query-trace-123",
            user_question="Show active users",
        )

        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event_type"] == "query_execution"
        assert log_data["severity"] == "low"
        assert log_data["message"] == "Query executed successfully"
        assert log_data["additional_data"]["sql_query"] == sql_query
        assert log_data["additional_data"]["execution_time_ms"] == 45.2
        assert log_data["additional_data"]["row_count"] == 8
        assert log_data["additional_data"]["query_length"] == len(sql_query)

    def test_log_query_execution_with_long_query(self):
        """Test query execution logging with long query truncation."""
        # Create a long SQL query
        long_query = (
            "SELECT " + ", ".join([f"column_{i}" for i in range(100)]) + " FROM users"
        )

        self.audit_logger.log_query_execution(
            sql_query=long_query, execution_time_ms=100.5, row_count=50
        )

        call_args = self.mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)

        # Verify truncation
        logged_query = log_data["additional_data"]["sql_query"]
        assert len(logged_query) <= 503  # 500 + "..."
        if len(long_query) > 500:
            assert logged_query.endswith("...")
        assert log_data["additional_data"]["query_length"] == len(long_query)

    def test_get_event_statistics(self):
        """Test retrieving event statistics."""
        # Log several events
        self.audit_logger.log_event(
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventSeverity.HIGH,
            "Test 1",
        )
        self.audit_logger.log_event(
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventSeverity.HIGH,
            "Test 2",
        )
        self.audit_logger.log_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecurityEventSeverity.MEDIUM,
            "Test 3",
        )

        stats = self.audit_logger.get_event_statistics()

        assert stats["total_events"] == 3
        assert stats["events_by_type"]["sql_injection_attempt"] == 2
        assert stats["events_by_type"]["rate_limit_exceeded"] == 1
        assert stats["events_by_severity"]["high"] == 2
        assert stats["events_by_severity"]["medium"] == 1
        assert "statistics_since" in stats

    def test_reset_statistics(self):
        """Test resetting event statistics."""
        # Log some events
        self.audit_logger.log_event(
            SecurityEventType.QUERY_EXECUTION, SecurityEventSeverity.LOW, "Test event"
        )

        # Verify statistics exist
        stats = self.audit_logger.get_event_statistics()
        assert stats["total_events"] == 1

        # Reset statistics
        self.audit_logger.reset_statistics()

        # Verify statistics are cleared
        stats = self.audit_logger.get_event_statistics()
        assert stats["total_events"] == 0
        assert stats["events_by_type"] == {}
        assert stats["events_by_severity"] == {}

    @patch("sql_synthesizer.security_audit.datetime")
    def test_timestamp_format(self, mock_datetime):
        """Test that timestamps are properly formatted."""
        mock_now = Mock()
        mock_now.isoformat.return_value = "2025-07-21T15:30:45.123456+00:00"
        mock_datetime.now.return_value = mock_now

        self.audit_logger.log_event(
            SecurityEventType.QUERY_EXECUTION,
            SecurityEventSeverity.LOW,
            "Test timestamp",
        )

        call_args = self.mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["timestamp"] == "2025-07-21T15:30:45.123456Z"


class TestGlobalSecurityAuditLogger:
    """Test the global security audit logger instance."""

    def test_global_logger_exists(self):
        """Test that global security audit logger is available."""
        logger = get_security_audit_logger()
        assert logger is not None
        assert isinstance(logger, SecurityAuditLogger)
        assert logger.logger.name == "security_audit"

    def test_global_logger_functionality(self):
        """Test that global logger functions correctly."""
        # This is a basic smoke test since we're using the global instance
        logger = get_security_audit_logger()
        initial_stats = logger.get_event_statistics()

        # The global logger might have been used by other tests
        # so we just check that it's functional
        assert "total_events" in initial_stats
        assert "events_by_type" in initial_stats
        assert "events_by_severity" in initial_stats


class TestSecurityEventTypes:
    """Test security event type enumerations."""

    def test_all_event_types_have_values(self):
        """Test that all security event types have string values."""
        for event_type in SecurityEventType:
            assert isinstance(event_type.value, str)
            assert len(event_type.value) > 0

    def test_all_severity_levels_have_values(self):
        """Test that all severity levels have string values."""
        for severity in SecurityEventSeverity:
            assert isinstance(severity.value, str)
            assert len(severity.value) > 0

    def test_event_type_values_are_unique(self):
        """Test that all event type values are unique."""
        values = [event_type.value for event_type in SecurityEventType]
        assert len(values) == len(set(values))

    def test_severity_values_are_unique(self):
        """Test that all severity values are unique."""
        values = [severity.value for severity in SecurityEventSeverity]
        assert len(values) == len(set(values))
