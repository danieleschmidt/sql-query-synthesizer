"""
Security middleware and utilities for web application.

This module provides comprehensive security features including:
- CSRF protection
- Rate limiting
- Security headers
- Input validation and sanitization
- API key authentication
"""

import logging
import secrets
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

from flask import g, jsonify, request
from markupsafe import escape

from .config import config
from .security_audit import (
    SecurityEventSeverity,
    SecurityEventType,
    get_security_audit_logger,
)

logger = logging.getLogger(__name__)


class CSRFProtection:
    """CSRF protection implementation using synchronizer tokens."""

    def __init__(self, secret_key: str = None):
        """Initialize CSRF protection with secret key."""
        self.secret_key = secret_key or self._generate_secret_key()

    def _generate_secret_key(self) -> str:
        """Generate a secure random secret key."""
        return secrets.token_urlsafe(32)

    def generate_token(self) -> str:
        """Generate a CSRF token for the current session."""
        return secrets.token_urlsafe(32)

    def validate_token(self, token: str, session_token: str) -> bool:
        """Validate a CSRF token against the session token."""
        if not token or not session_token:
            return False
        return secrets.compare_digest(token, session_token)


class RateLimiter:
    """Rate limiting implementation using sliding window algorithm."""

    def __init__(self, requests_per_minute: int = 60, window_size: int = 60):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per client
            window_size: Window size in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.window_size = window_size
        self.clients: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if client is allowed to make a request.

        Args:
            client_id: Unique identifier for the client (IP, API key, etc.)

        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()
        client_requests = self.clients[client_id]

        # Remove requests outside the current window
        while client_requests and client_requests[0] <= now - self.window_size:
            client_requests.popleft()

        # Check if client has exceeded rate limit
        if len(client_requests) >= self.requests_per_minute:
            return False

        # Add current request
        client_requests.append(now)
        return True

    def get_rate_limit_headers(self, client_id: str) -> Dict[str, str]:
        """Get rate limit headers for response."""
        client_requests = self.clients[client_id]
        remaining = max(0, self.requests_per_minute - len(client_requests))

        # Calculate reset time (next window)
        if client_requests:
            reset_time = int(client_requests[0] + self.window_size)
        else:
            reset_time = int(time.time() + self.window_size)

        return {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
        }


class SecurityHeaders:
    """Security headers implementation."""

    @staticmethod
    def get_security_headers(enable_hsts: bool = False) -> Dict[str, str]:
        """
        Get comprehensive security headers.

        Args:
            enable_hsts: Whether to include HSTS header

        Returns:
            Dictionary of security headers
        """
        headers = {
            # Content Security Policy - restrictive policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "script-src 'self'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # XSS Protection
            "X-XSS-Protection": "1; mode=block",
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Remove server information
            "Server": "SQL-Synthesizer",
            # Cache control for sensitive pages
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        # Add HSTS header if HTTPS is enabled
        if enable_hsts:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return headers


class InputValidator:
    """Input validation and sanitization utilities."""

    @staticmethod
    def validate_question_length(question: str, max_length: int = None) -> bool:
        """
        Validate question length.

        Args:
            question: User question to validate
            max_length: Maximum allowed length

        Returns:
            True if valid, False if too long
        """
        max_length = max_length or config.max_question_length
        return len(question) <= max_length

    @staticmethod
    def sanitize_question(question: str) -> str:
        """
        Sanitize user question for safe processing.

        Args:
            question: User question to sanitize

        Returns:
            Sanitized question
        """
        if not question:
            return ""

        # HTML escape to prevent XSS
        sanitized = escape(question)

        # Remove potential template injection patterns
        dangerous_patterns = [
            "{{",
            "}}",
            "{%",
            "%}",
            "${",
            "}",
            "<%",
            "%>",
            "<script",
            "</script",
            "javascript:",
            "data:",
            "vbscript:",
        ]

        sanitized_str = str(sanitized)
        for pattern in dangerous_patterns:
            sanitized_str = sanitized_str.replace(pattern, "")

        return sanitized_str.strip()

    @staticmethod
    def validate_json_structure(
        data: Dict[str, Any], required_fields: list
    ) -> tuple[bool, str]:
        """
        Validate JSON request structure.

        Args:
            data: JSON data to validate
            required_fields: List of required field names

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Request must be a JSON object"

        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        return True, ""


class APIKeyAuth:
    """API key authentication implementation."""

    def __init__(self, api_key: str = None):
        """Initialize API key authentication."""
        self.api_key = api_key or config.webapp_api_key
        self.enabled = config.webapp_api_key_required and self.api_key is not None

    def validate_api_key(self, provided_key: str) -> bool:
        """
        Validate provided API key.

        Args:
            provided_key: API key provided in request

        Returns:
            True if valid, False otherwise
        """
        if not self.enabled:
            return True  # API key not required

        if not provided_key or not self.api_key:
            return False

        return secrets.compare_digest(provided_key, self.api_key)

    def extract_api_key(self, request) -> Optional[str]:
        """
        Extract API key from request headers.

        Args:
            request: Flask request object

        Returns:
            API key if found, None otherwise
        """
        # Try different header formats
        headers_to_check = [
            "X-API-Key",
            "Authorization",  # Bearer token format
            "X-API-TOKEN",
        ]

        for header in headers_to_check:
            value = request.headers.get(header)
            if value:
                # Handle Bearer token format
                if header == "Authorization" and value.startswith("Bearer "):
                    return value[7:]  # Remove 'Bearer ' prefix
                return value

        return None


class SecurityMiddleware:
    """Comprehensive security middleware for Flask applications."""

    def __init__(self, app=None):
        """Initialize security middleware."""
        self.csrf = CSRFProtection()
        self.rate_limiter = RateLimiter(config.webapp_rate_limit)
        self.api_auth = APIKeyAuth()
        self.validator = InputValidator()

        if app:
            self.init_app(app)

    def init_app(self, app):
        """Initialize middleware with Flask app."""
        # Set secret key for session management
        if not app.secret_key:
            if config.webapp_secret_key:
                app.secret_key = config.webapp_secret_key
            else:
                # Generate a warning and use a temporary key
                logger.warning(
                    "No SECRET_KEY configured. Using temporary key. Set QUERY_AGENT_SECRET_KEY environment variable."
                )
                app.secret_key = secrets.token_urlsafe(32)

        # Configure request size limit
        app.config["MAX_CONTENT_LENGTH"] = config.webapp_max_request_size * 1024 * 1024

        # Register security handlers
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.errorhandler(413)(self._handle_request_too_large)
        app.errorhandler(429)(self._handle_rate_limit)

    def _get_client_id(self, request) -> str:
        """Get unique client identifier for rate limiting."""
        # Use API key if available, otherwise use IP address
        api_key = self.api_auth.extract_api_key(request)
        if api_key:
            return f"api:{api_key[:8]}"  # Use first 8 chars for privacy

        # Use forwarded IP if behind proxy
        forwarded_ip = request.headers.get("X-Forwarded-For")
        if forwarded_ip:
            return f"ip:{forwarded_ip.split(',')[0].strip()}"

        return f"ip:{request.remote_addr}"

    def _before_request(self):
        """Handle security checks before request processing."""
        # Skip security for internal routes
        if request.endpoint in ["metrics"]:
            return

        # Rate limiting
        client_id = self._get_client_id(request)
        if not self.rate_limiter.is_allowed(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")

            # Log security event for rate limit violation
            get_security_audit_logger(config).log_rate_limit_exceeded(
                client_identifier=client_id,
                limit_type="requests_per_minute",
                current_rate=len(self.rate_limiter.clients.get(client_id, [])),
                limit_threshold=self.rate_limiter.requests_per_minute,
                client_ip=request.remote_addr,
                trace_id=getattr(g, "trace_id", None),
            )

            return jsonify({"error": "Rate limit exceeded"}), 429

        # Store rate limit info for response headers
        g.rate_limit_headers = self.rate_limiter.get_rate_limit_headers(client_id)

        # API key validation for API endpoints
        if request.endpoint and request.endpoint.startswith("api_"):
            api_key = self.api_auth.extract_api_key(request)
            if not self.api_auth.validate_api_key(api_key):
                logger.warning(f"Invalid API key attempt from client: {client_id}")

                # Log security event for authentication failure
                get_security_audit_logger(config).log_authentication_failure(
                    auth_type="API_KEY",
                    reason="Invalid or missing API key",
                    client_ip=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                    trace_id=getattr(g, "trace_id", None),
                )

                return jsonify({"error": "Invalid or missing API key"}), 401

        # CSRF protection for forms (not API endpoints)
        if (
            request.method == "POST"
            and request.endpoint
            and not request.endpoint.startswith("api_")
            and config.webapp_csrf_enabled
        ):

            # Check CSRF token
            csrf_token = request.form.get("csrf_token") or request.headers.get(
                "X-CSRF-Token"
            )
            session_token = request.cookies.get("csrf_token")

            if not self.csrf.validate_token(csrf_token, session_token):
                logger.warning(f"CSRF token validation failed for client: {client_id}")

                # Log security event for CSRF token validation failure
                get_security_audit_logger(config).log_event(
                    event_type=SecurityEventType.CSRF_TOKEN_VALIDATION_FAILED,
                    severity=SecurityEventSeverity.MEDIUM,
                    message="CSRF token validation failed",
                    client_ip=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                    trace_id=getattr(g, "trace_id", None),
                    request_path=request.path,
                    request_method=request.method,
                )

                return jsonify({"error": "CSRF token missing or invalid"}), 400

    def _after_request(self, response):
        """Add security headers to response."""
        # Add security headers
        security_headers = SecurityHeaders.get_security_headers(
            config.webapp_enable_hsts
        )
        for header, value in security_headers.items():
            response.headers[header] = value

        # Add rate limit headers
        if hasattr(g, "rate_limit_headers"):
            for header, value in g.rate_limit_headers.items():
                response.headers[header] = value

        # Add CSRF token to cookies for form-based requests
        if (
            request.endpoint == "index"
            and request.method == "GET"
            and config.webapp_csrf_enabled
        ):
            csrf_token = self.csrf.generate_token()
            response.set_cookie(
                "csrf_token",
                csrf_token,
                httponly=True,
                secure=request.is_secure,
                samesite="Strict",
            )

        return response

    def _handle_request_too_large(self, error):
        """Handle request too large errors."""
        client_id = self._get_client_id(request)
        logger.warning(f"Request too large from client: {client_id}")

        # Log security event for request size limit
        get_security_audit_logger(config).log_event(
            event_type=SecurityEventType.REQUEST_SIZE_LIMIT_EXCEEDED,
            severity=SecurityEventSeverity.MEDIUM,
            message="Request size limit exceeded",
            client_ip=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
            trace_id=getattr(g, "trace_id", None),
            request_path=request.path,
            request_method=request.method,
            content_length=request.headers.get("Content-Length"),
        )

        return jsonify({"error": "Request too large"}), 413

    def _handle_rate_limit(self, error):
        """Handle rate limit errors."""
        return jsonify({"error": "Rate limit exceeded"}), 429


# Global security middleware instance
security_middleware = SecurityMiddleware()
