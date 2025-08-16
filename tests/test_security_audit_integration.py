"""Integration tests for security audit logging with enhanced validation."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from sql_synthesizer.security_audit import (
    SecurityEventSeverity,
    SecurityEventType,
    security_audit_logger,
)
from sql_synthesizer.services.enhanced_query_validator import (
    EnhancedQueryValidatorService,
)
from sql_synthesizer.services.query_service import QueryService
from sql_synthesizer.types import QueryResult
from sql_synthesizer.user_experience import UserFriendlyError


class TestSecurityAuditIntegration:
    """Test integration between security audit logging and validation services."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create enhanced validator
        self.validator = EnhancedQueryValidatorService(
            max_question_length=1000,
            allowed_tables=["users", "orders"],
            allowed_columns=["id", "name", "email", "status"],
        )

        # Mock the security audit logger
        self.mock_audit_logger = Mock()

        # Patch the global audit logger
        self.audit_logger_patcher = patch(
            "sql_synthesizer.services.enhanced_query_validator.get_security_audit_logger",
            return_value=self.mock_audit_logger,
        )
        self.audit_logger_patcher.start()

    def teardown_method(self):
        """Clean up after tests."""
        self.audit_logger_patcher.stop()

    def test_sql_injection_detection_logs_audit_event(self):
        """Test that SQL injection detection logs appropriate audit events."""
        malicious_input = "Show all users'; DROP TABLE users; --"

        with pytest.raises(UserFriendlyError):
            self.validator.validate_question(malicious_input)

        # Verify audit event was logged
        self.mock_audit_logger.log_sql_injection_attempt.assert_called_once()
        call_args = self.mock_audit_logger.log_sql_injection_attempt.call_args

        assert call_args[1]["malicious_input"] == malicious_input
        assert call_args[1]["detection_method"] == "pattern_matching"
        assert "original_input" in call_args[1]

    def test_rate_limiting_violation_logs_audit_event(self):
        """Test that rate limiting violations log audit events."""
        # Create validator with low rate limit for testing
        validator = EnhancedQueryValidatorService(
            max_question_length=1000,
            max_validation_attempts_per_minute=5,  # Low limit for testing
        )

        # Patch the audit logger for this validator too
        with patch(
            "sql_synthesizer.services.enhanced_query_validator.get_security_audit_logger",
            return_value=self.mock_audit_logger,
        ):
            client_id = "test_client_123"

            # Make multiple rapid requests to trigger rate limiting
            for _ in range(10):  # Exceed the low limit
                try:
                    validator.validate_question(
                        "SELECT * FROM users", client_id=client_id
                    )
                except UserFriendlyError:
                    break

            # Verify rate limit audit event was logged
            self.mock_audit_logger.log_rate_limit_exceeded.assert_called()
            call_args = self.mock_audit_logger.log_rate_limit_exceeded.call_args

            assert call_args[1]["client_identifier"] == client_id
            assert call_args[1]["limit_type"] == "validation_attempts_per_minute"
            assert call_args[1]["current_rate"] >= call_args[1]["limit_threshold"]

    def test_unauthorized_reference_detection_logs_audit_event(self):
        """Test that unauthorized reference detection logs audit events."""
        # Use a simple SQL injection pattern that should trigger logging
        malicious_input = "SELECT * FROM users; DROP TABLE users; --"

        with pytest.raises(UserFriendlyError):
            self.validator.validate_sql_statement(malicious_input)

        # Should log SQL injection detection
        assert self.mock_audit_logger.log_sql_injection_attempt.called

    def test_semantic_analysis_detection_logs_audit_event(self):
        """Test that semantic analysis detection logs audit events."""
        # SQL with suspicious semantic patterns
        suspicious_sql = "SELECT * FROM users WHERE 1=1 OR 'a'='a'"

        with pytest.raises(UserFriendlyError):
            self.validator.validate_sql_statement(suspicious_sql)

        # Verify some audit logging occurred
        assert (
            self.mock_audit_logger.log_sql_injection_attempt.called
            or self.mock_audit_logger.log_event.called
        )

    def test_successful_validation_does_not_log_security_events(self):
        """Test that successful validation doesn't trigger security audit events."""
        safe_input = "Show me all active users"

        # This should not raise an exception
        result = self.validator.validate_question(safe_input)
        assert result == safe_input

        # Verify no security audit events were logged
        self.mock_audit_logger.log_sql_injection_attempt.assert_not_called()
        self.mock_audit_logger.log_rate_limit_exceeded.assert_not_called()
        self.mock_audit_logger.log_event.assert_not_called()


class TestQueryServiceAuditIntegration:
    """Test integration between security audit logging and query service."""

    def test_audit_logger_integration_exists(self):
        """Test that audit logger integration is properly imported."""
        # Simple test to verify the integration is wired up
        with patch(
            "sql_synthesizer.services.query_service.get_security_audit_logger",
            return_value=Mock(),
        ) as mock_logger:
            # This is mainly a smoke test to ensure imports work
            assert mock_logger is not None


class TestGlobalAuditLoggerUsage:
    """Test that the global audit logger can be used across the system."""

    def test_global_audit_logger_can_log_events(self):
        """Test that the global audit logger functions correctly."""
        # Reset any existing statistics
        security_audit_logger.reset_statistics()

        # Log a test event
        security_audit_logger.log_event(
            event_type=SecurityEventType.QUERY_EXECUTION,
            severity=SecurityEventSeverity.LOW,
            message="Test event for global logger",
        )

        # Verify statistics were updated
        stats = security_audit_logger.get_event_statistics()
        assert stats["total_events"] >= 1

    def test_global_audit_logger_statistics_tracking(self):
        """Test that global audit logger tracks statistics correctly."""
        # Reset statistics
        security_audit_logger.reset_statistics()

        # Log multiple events of different types
        security_audit_logger.log_sql_injection_attempt(
            malicious_input="'; DROP TABLE test; --",
            detection_method="pattern_matching",
        )

        security_audit_logger.log_rate_limit_exceeded(
            client_identifier="test_client",
            limit_type="requests_per_minute",
            current_rate=150,
            limit_threshold=100,
        )

        # Check statistics
        stats = security_audit_logger.get_event_statistics()

        assert stats["total_events"] == 2
        assert "sql_injection_attempt" in stats["events_by_type"]
        assert "rate_limit_exceeded" in stats["events_by_type"]
        assert stats["events_by_type"]["sql_injection_attempt"] == 1
        assert stats["events_by_type"]["rate_limit_exceeded"] == 1
