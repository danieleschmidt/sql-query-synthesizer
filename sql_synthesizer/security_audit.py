"""
Security audit logging for SQL Query Synthesizer.

This module provides centralized security event logging and audit trail
functionality for monitoring and compliance purposes.
"""

import json
import logging
import logging.handlers
import time
import os
import glob
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


class SecurityEventType(Enum):
    """Types of security events that can be logged."""
    
    # Authentication and authorization
    API_KEY_AUTHENTICATION_FAILED = "api_key_auth_failed"
    API_KEY_AUTHENTICATION_SUCCESS = "api_key_auth_success"
    CSRF_TOKEN_VALIDATION_FAILED = "csrf_validation_failed"
    
    # Input validation and injection attempts
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    UNSAFE_INPUT_DETECTED = "unsafe_input_detected"
    INVALID_REQUEST_STRUCTURE = "invalid_request_structure"
    
    # Rate limiting and DoS protection
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    REQUEST_SIZE_LIMIT_EXCEEDED = "request_size_exceeded"
    
    # Data access and query events
    QUERY_EXECUTION = "query_execution"
    SCHEMA_ACCESS = "schema_access"
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    
    # System security events
    SECURITY_HEADER_VIOLATION = "security_header_violation"
    SUSPICIOUS_USER_AGENT = "suspicious_user_agent"
    UNEXPECTED_ERROR = "unexpected_error"


class SecurityEventSeverity(Enum):
    """Severity levels for security events."""
    
    LOW = "low"           # Informational, normal operations
    MEDIUM = "medium"     # Suspicious activity, potential threat
    HIGH = "high"         # Confirmed security violation
    CRITICAL = "critical" # Immediate security threat requiring response


@dataclass
class SecurityEvent:
    """Represents a security event to be logged."""
    
    event_type: SecurityEventType
    severity: SecurityEventSeverity
    message: str
    timestamp: str
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    trace_id: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging."""
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "trace_id": self.trace_id,
            "request_path": self.request_path,
            "request_method": self.request_method,
            "additional_data": self.additional_data or {},
        }


class SecurityAuditLogger:
    """Centralized security audit logging system with rotation and retention."""
    
    def __init__(self, logger_name: str = "security_audit", config=None):
        """Initialize the security audit logger.
        
        Args:
            logger_name: Name for the security audit logger
            config: Configuration object with audit logging settings
        """
        self.logger = logging.getLogger(logger_name)
        self.config = config
        
        # Configure logger level
        if config and hasattr(config, 'security_log_level'):
            level = getattr(logging, config.security_log_level.upper(), logging.INFO)
            self.logger.setLevel(level)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Create dedicated handlers for security events if they don't exist
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Track event statistics
        self._event_counts: Dict[SecurityEventType, int] = {}
        self._severity_counts: Dict[SecurityEventSeverity, int] = {}
        
        # Perform initial log cleanup if configured
        self._cleanup_old_logs()
    
    def _setup_handlers(self) -> None:
        """Set up logging handlers with rotation and formatting."""
        # Always add console handler for immediate visibility
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler with rotation if configured
        if self.config and hasattr(self.config, 'audit_log_file'):
            try:
                # Ensure log directory exists
                log_path = Path(self.config.audit_log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Configure rotating file handler
                max_bytes = getattr(self.config, 'audit_log_max_bytes', 50 * 1024 * 1024)
                backup_count = getattr(self.config, 'audit_log_backup_count', 10)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=self.config.audit_log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                
                # Configure formatter based on config
                if hasattr(self.config, 'security_log_format') and self.config.security_log_format.lower() == 'json':
                    # JSON formatter for structured logging
                    file_formatter = logging.Formatter('%(message)s')
                else:
                    # Standard formatter
                    file_formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
                
            except (OSError, PermissionError) as e:
                # If file logging fails, log to console only
                self.logger.warning(f"Failed to setup file logging: {e}. Using console only.")
    
    def _cleanup_old_logs(self) -> None:
        """Clean up old audit log files based on retention policy."""
        if not (self.config and hasattr(self.config, 'audit_log_file') and 
                hasattr(self.config, 'audit_log_retention_days')):
            return
        
        try:
            log_path = Path(self.config.audit_log_file)
            log_dir = log_path.parent
            log_pattern = f"{log_path.stem}.*"
            
            retention_days = self.config.audit_log_retention_days
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Find and remove old log files
            removed_count = 0
            for log_file in log_dir.glob(log_pattern):
                if log_file.is_file():
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        try:
                            log_file.unlink()
                            removed_count += 1
                        except OSError as e:
                            self.logger.warning(f"Failed to remove old log file {log_file}: {e}")
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old audit log files")
                
        except Exception as e:
            self.logger.warning(f"Log cleanup failed: {e}")
    
    def log_event(
        self,
        event_type: SecurityEventType,
        severity: SecurityEventSeverity,
        message: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        trace_id: Optional[str] = None,
        request_path: Optional[str] = None,
        request_method: Optional[str] = None,
        **additional_data
    ) -> None:
        """Log a security event.
        
        Args:
            event_type: Type of security event
            severity: Severity level of the event
            message: Human-readable description of the event
            client_ip: Client IP address if available
            user_agent: User agent string if available
            trace_id: Request trace ID for correlation
            request_path: HTTP request path if available
            request_method: HTTP request method if available
            **additional_data: Additional data to include in the log
        """
        # Create security event
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            client_ip=client_ip,
            user_agent=user_agent,
            trace_id=trace_id,
            request_path=request_path,
            request_method=request_method,
            additional_data=additional_data if additional_data else None
        )
        
        # Update statistics
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
        self._severity_counts[severity] = self._severity_counts.get(severity, 0) + 1
        
        # Log the event
        log_data = event.to_dict()
        log_message = json.dumps(log_data, separators=(',', ':'))
        
        # Use appropriate log level based on severity
        if severity == SecurityEventSeverity.LOW:
            self.logger.info(log_message)
        elif severity == SecurityEventSeverity.MEDIUM:
            self.logger.warning(log_message)
        elif severity == SecurityEventSeverity.HIGH:
            self.logger.error(log_message)
        elif severity == SecurityEventSeverity.CRITICAL:
            self.logger.critical(log_message)
    
    def log_sql_injection_attempt(
        self,
        malicious_input: str,
        detection_method: str,
        client_ip: Optional[str] = None,
        trace_id: Optional[str] = None,
        **additional_data
    ) -> None:
        """Log SQL injection attempt with specific context.
        
        Args:
            malicious_input: The malicious input that was detected
            detection_method: How the injection was detected
            client_ip: Client IP address
            trace_id: Request trace ID
            **additional_data: Additional context data
        """
        # Truncate malicious input for logging
        truncated_input = malicious_input[:200] + "..." if len(malicious_input) > 200 else malicious_input
        
        self.log_event(
            event_type=SecurityEventType.SQL_INJECTION_ATTEMPT,
            severity=SecurityEventSeverity.HIGH,
            message=f"SQL injection attempt detected: {detection_method}",
            client_ip=client_ip,
            trace_id=trace_id,
            malicious_input=truncated_input,
            detection_method=detection_method,
            input_length=len(malicious_input),
            **additional_data
        )
    
    def log_authentication_failure(
        self,
        auth_type: str,
        reason: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        trace_id: Optional[str] = None,
        **additional_data
    ) -> None:
        """Log authentication failure.
        
        Args:
            auth_type: Type of authentication that failed
            reason: Reason for authentication failure
            client_ip: Client IP address
            user_agent: User agent string
            trace_id: Request trace ID
            **additional_data: Additional context data
        """
        self.log_event(
            event_type=SecurityEventType.API_KEY_AUTHENTICATION_FAILED,
            severity=SecurityEventSeverity.MEDIUM,
            message=f"Authentication failed: {auth_type} - {reason}",
            client_ip=client_ip,
            user_agent=user_agent,
            trace_id=trace_id,
            auth_type=auth_type,
            failure_reason=reason,
            **additional_data
        )
    
    def log_rate_limit_exceeded(
        self,
        client_identifier: str,
        limit_type: str,
        current_rate: int,
        limit_threshold: int,
        client_ip: Optional[str] = None,
        trace_id: Optional[str] = None,
        **additional_data
    ) -> None:
        """Log rate limit violation.
        
        Args:
            client_identifier: Identifier for the client (IP, API key, etc.)
            limit_type: Type of rate limit that was exceeded
            current_rate: Current request rate
            limit_threshold: The rate limit threshold
            client_ip: Client IP address
            trace_id: Request trace ID
            **additional_data: Additional context data
        """
        self.log_event(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=SecurityEventSeverity.MEDIUM,
            message=f"Rate limit exceeded: {limit_type} - {current_rate}/{limit_threshold}",
            client_ip=client_ip,
            trace_id=trace_id,
            client_identifier=client_identifier,
            limit_type=limit_type,
            current_rate=current_rate,
            limit_threshold=limit_threshold,
            **additional_data
        )
    
    def log_query_execution(
        self,
        sql_query: str,
        execution_time_ms: float,
        row_count: int,
        client_ip: Optional[str] = None,
        trace_id: Optional[str] = None,
        **additional_data
    ) -> None:
        """Log query execution for audit trail.
        
        Args:
            sql_query: SQL query that was executed
            execution_time_ms: Query execution time in milliseconds
            row_count: Number of rows returned
            client_ip: Client IP address
            trace_id: Request trace ID
            **additional_data: Additional context data
        """
        # Truncate query for logging
        truncated_query = sql_query[:500] + "..." if len(sql_query) > 500 else sql_query
        
        self.log_event(
            event_type=SecurityEventType.QUERY_EXECUTION,
            severity=SecurityEventSeverity.LOW,
            message=f"Query executed successfully",
            client_ip=client_ip,
            trace_id=trace_id,
            sql_query=truncated_query,
            execution_time_ms=execution_time_ms,
            row_count=row_count,
            query_length=len(sql_query),
            **additional_data
        )
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged security events.
        
        Returns:
            Dictionary containing event counts by type and severity
        """
        return {
            "total_events": sum(self._event_counts.values()),
            "events_by_type": {event_type.value: count for event_type, count in self._event_counts.items()},
            "events_by_severity": {severity.value: count for severity, count in self._severity_counts.items()},
            "statistics_since": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }
    
    def reset_statistics(self) -> None:
        """Reset event statistics counters."""
        self._event_counts.clear()
        self._severity_counts.clear()


# Global security audit logger instance (will be initialized with config)
security_audit_logger = None

def get_security_audit_logger(config=None):
    """Get or create the global security audit logger instance."""
    global security_audit_logger
    if security_audit_logger is None:
        security_audit_logger = SecurityAuditLogger(config=config)
    return security_audit_logger