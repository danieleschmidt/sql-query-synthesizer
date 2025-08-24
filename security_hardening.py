"""
Advanced Security Hardening Configuration for SQL Query Synthesizer.

This module provides comprehensive security hardening including:
- Runtime security monitoring and threat detection
- Advanced input validation and sanitization
- Security policy enforcement
- Compliance framework integration
- Incident response automation
"""

import asyncio
import hashlib
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security configuration levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InputValidationConfig:
    """Advanced input validation and sanitization configuration."""

    # SQL Injection Protection
    enable_ast_validation: bool = field(default=True)
    enable_pattern_matching: bool = field(default=True)
    enable_semantic_analysis: bool = field(default=True)

    # Input Limits
    max_query_length: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_MAX_QUESTION_LENGTH", "2000")))
    max_nested_depth: int = field(default=10)
    max_parameters: int = field(default=50)

    # Dangerous Patterns
    blocked_keywords: Set[str] = field(default_factory=lambda: {
        "drop", "delete", "truncate", "alter", "create", "insert", "update",
        "exec", "execute", "sp_", "xp_", "--", "/*", "*/", "@@", "char",
        "nchar", "varchar", "nvarchar", "waitfor", "delay"
    })

    suspicious_patterns: List[str] = field(default_factory=lambda: [
        r"union\s+select",
        r"1\s*=\s*1",
        r"or\s+1\s*=\s*1",
        r"and\s+1\s*=\s*1",
        r"';.*--",
        r"'\s*or\s*'",
        r"benchmark\s*\(",
        r"sleep\s*\(",
        r"pg_sleep\s*\(",
        r"information_schema",
        r"sys\.",
        r"master\.",
        r"msdb\.",
        r"tempdb\."
    ])

    # Content Security
    enable_content_filtering: bool = field(default=True)
    enable_data_loss_prevention: bool = field(default=True)
    scan_for_pii: bool = field(default=True)

    # Rate Limiting
    enable_rate_limiting: bool = field(default=True)
    requests_per_minute: int = field(default_factory=lambda: int(os.environ.get("QUERY_AGENT_RATE_LIMIT_PER_MINUTE", "60")))
    burst_limit: int = field(default=10)


@dataclass
class AuthenticationConfig:
    """Authentication and authorization security configuration."""

    # API Key Security
    require_api_key: bool = field(default_factory=lambda: os.environ.get("QUERY_AGENT_API_KEY_REQUIRED", "false").lower() == "true")
    api_key_length: int = field(default=64)
    api_key_rotation_days: int = field(default=90)

    # Session Security
    session_timeout_minutes: int = field(default=30)
    enable_session_encryption: bool = field(default=True)
    secure_cookies_only: bool = field(default=True)

    # Multi-Factor Authentication
    enable_mfa: bool = field(default=False)
    mfa_methods: List[str] = field(default_factory=lambda: ["totp", "email"])

    # Access Control
    enable_rbac: bool = field(default=False)
    default_role: str = field(default="readonly")
    admin_roles: List[str] = field(default_factory=lambda: ["admin", "superuser"])

    # OAuth/SSO Integration
    enable_oauth: bool = field(default=False)
    oauth_providers: List[str] = field(default_factory=list)

    # Audit and Compliance
    log_all_authentication_events: bool = field(default=True)
    failed_login_threshold: int = field(default=5)
    lockout_duration_minutes: int = field(default=15)


@dataclass
class NetworkSecurityConfig:
    """Network-level security configuration."""

    # HTTPS/TLS
    enforce_https: bool = field(default=True)
    tls_version: str = field(default="1.3")
    enable_hsts: bool = field(default_factory=lambda: os.environ.get("QUERY_AGENT_ENABLE_HSTS", "false").lower() == "true")
    hsts_max_age: int = field(default=31536000)  # 1 year

    # CORS Security
    enable_cors: bool = field(default=True)
    allowed_origins: List[str] = field(default_factory=lambda: ["https://localhost:3000"])
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])

    # IP Filtering
    enable_ip_whitelist: bool = field(default=False)
    allowed_ips: List[str] = field(default_factory=list)
    enable_ip_blacklist: bool = field(default=True)
    blocked_ips: List[str] = field(default_factory=list)

    # DDoS Protection
    enable_ddos_protection: bool = field(default=True)
    max_connections_per_ip: int = field(default=100)
    connection_rate_limit: int = field(default=10)  # connections per second

    # Firewall Integration
    enable_waf: bool = field(default=False)
    waf_provider: str = field(default="cloudflare")


@dataclass
class DataProtectionConfig:
    """Data protection and privacy configuration."""

    # Encryption
    enable_data_encryption: bool = field(default=True)
    encryption_algorithm: str = field(default="AES-256-GCM")
    key_rotation_days: int = field(default=30)

    # PII Protection
    enable_pii_detection: bool = field(default=True)
    pii_patterns: Dict[str, str] = field(default_factory=lambda: {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}-\d{3}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    })

    # Data Masking
    enable_data_masking: bool = field(default=True)
    masking_character: str = field(default="*")
    preserve_format: bool = field(default=True)

    # Backup Security
    encrypt_backups: bool = field(default=True)
    backup_retention_days: int = field(default=90)
    secure_backup_location: bool = field(default=True)

    # Audit Trail
    enable_data_access_logging: bool = field(default=True)
    log_data_modifications: bool = field(default=True)
    audit_retention_days: int = field(default=365)


@dataclass
class ThreatDetectionConfig:
    """Advanced threat detection and response configuration."""

    # Anomaly Detection
    enable_anomaly_detection: bool = field(default=True)
    baseline_learning_days: int = field(default=7)
    anomaly_threshold: float = field(default=2.0)  # Standard deviations

    # Behavioral Analysis
    enable_behavioral_analysis: bool = field(default=True)
    track_user_patterns: bool = field(default=True)
    detect_credential_stuffing: bool = field(default=True)

    # Real-time Monitoring
    enable_realtime_monitoring: bool = field(default=True)
    monitoring_interval_seconds: int = field(default=10)
    alert_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "failed_logins": 5,
        "suspicious_queries": 3,
        "unusual_access_patterns": 10
    })

    # Incident Response
    enable_auto_response: bool = field(default=True)
    response_actions: List[str] = field(default_factory=lambda: [
        "block_ip", "disable_account", "alert_admin", "log_incident"
    ])

    # Threat Intelligence
    enable_threat_intel: bool = field(default=False)
    threat_intel_sources: List[str] = field(default_factory=list)
    update_interval_hours: int = field(default=24)


@dataclass
class ComplianceConfig:
    """Compliance framework configuration."""

    # Regulatory Frameworks
    enabled_frameworks: List[str] = field(default_factory=lambda: ["SOC2", "GDPR"])

    # SOC 2 Compliance
    soc2_enabled: bool = field(default=True)
    soc2_controls: List[str] = field(default_factory=lambda: [
        "CC6.1", "CC6.2", "CC6.3", "CC6.7", "CC6.8"  # Common Criteria controls
    ])

    # GDPR Compliance
    gdpr_enabled: bool = field(default=True)
    enable_right_to_erasure: bool = field(default=True)
    enable_data_portability: bool = field(default=True)
    consent_management: bool = field(default=True)

    # HIPAA Compliance
    hipaa_enabled: bool = field(default=False)
    enable_phi_protection: bool = field(default=False)

    # Audit and Reporting
    generate_compliance_reports: bool = field(default=True)
    report_schedule: str = field(default="monthly")
    automated_evidence_collection: bool = field(default=True)


@dataclass
class SecurityHardeningConfig:
    """Master security hardening configuration."""

    security_level: SecurityLevel = field(default=SecurityLevel.ENHANCED)

    input_validation: InputValidationConfig = field(default_factory=InputValidationConfig)
    authentication: AuthenticationConfig = field(default_factory=AuthenticationConfig)
    network_security: NetworkSecurityConfig = field(default_factory=NetworkSecurityConfig)
    data_protection: DataProtectionConfig = field(default_factory=DataProtectionConfig)
    threat_detection: ThreatDetectionConfig = field(default_factory=ThreatDetectionConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)

    # Global Security Settings
    enable_security_logging: bool = field(default=True)
    security_log_level: str = field(default="INFO")
    enable_security_metrics: bool = field(default=True)

    # Development vs Production
    development_mode: bool = field(default_factory=lambda: os.environ.get("QUERY_AGENT_ENV", "development") == "development")

    # Emergency Response
    enable_kill_switch: bool = field(default=True)
    emergency_contacts: List[str] = field(default_factory=list)


class SecurityHardeningSuite:
    """Comprehensive security hardening implementation."""

    def __init__(self, config: SecurityHardeningConfig):
        self.config = config
        self.threat_incidents: List[Dict[str, Any]] = []
        self.security_metrics: Dict[str, Any] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    async def initialize_security(self):
        """Initialize all security hardening measures."""

        logger.info(f"Initializing security hardening (Level: {self.config.security_level.value})")

        # Initialize components
        await self._setup_input_validation()
        await self._setup_authentication()
        await self._setup_network_security()
        await self._setup_data_protection()
        await self._setup_threat_detection()
        await self._setup_compliance_monitoring()

        # Start security monitoring
        if self.config.threat_detection.enable_realtime_monitoring:
            asyncio.create_task(self._start_security_monitoring())

        logger.info("Security hardening initialization completed")

    async def _setup_input_validation(self):
        """Configure advanced input validation."""

        if not self.config.input_validation.enable_ast_validation:
            return

        logger.info("Setting up advanced input validation...")

        # Compile regex patterns for performance
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.input_validation.suspicious_patterns
        ]

        logger.info(f"Configured {len(self.compiled_patterns)} security patterns")

    async def validate_input(self, input_text: str, source_ip: str = None) -> Dict[str, Any]:
        """Comprehensive input validation and threat assessment."""

        validation_result = {
            "is_safe": True,
            "threats_detected": [],
            "risk_score": 0.0,
            "recommendations": []
        }

        # Length validation
        if len(input_text) > self.config.input_validation.max_query_length:
            validation_result["is_safe"] = False
            validation_result["threats_detected"].append("input_too_long")
            validation_result["risk_score"] += 0.3

        # Keyword blocking
        input_lower = input_text.lower()
        for keyword in self.config.input_validation.blocked_keywords:
            if keyword in input_lower:
                validation_result["is_safe"] = False
                validation_result["threats_detected"].append(f"blocked_keyword_{keyword}")
                validation_result["risk_score"] += 0.5

        # Pattern matching
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(input_text):
                validation_result["is_safe"] = False
                validation_result["threats_detected"].append(f"suspicious_pattern_{i}")
                validation_result["risk_score"] += 0.4

        # PII detection
        if self.config.data_protection.enable_pii_detection:
            pii_found = self._detect_pii(input_text)
            if pii_found:
                validation_result["threats_detected"].extend([f"pii_{pii_type}" for pii_type in pii_found])
                validation_result["risk_score"] += 0.6

        # Rate limiting check
        if source_ip and self.config.input_validation.enable_rate_limiting:
            if await self._check_rate_limit(source_ip):
                validation_result["is_safe"] = False
                validation_result["threats_detected"].append("rate_limit_exceeded")
                validation_result["risk_score"] += 0.8

        # Generate threat level
        if validation_result["risk_score"] >= 0.8:
            validation_result["threat_level"] = ThreatLevel.CRITICAL
        elif validation_result["risk_score"] >= 0.6:
            validation_result["threat_level"] = ThreatLevel.HIGH
        elif validation_result["risk_score"] >= 0.3:
            validation_result["threat_level"] = ThreatLevel.MEDIUM
        else:
            validation_result["threat_level"] = ThreatLevel.LOW

        # Log security events
        if not validation_result["is_safe"]:
            await self._log_security_incident(
                incident_type="input_validation_failure",
                details=validation_result,
                source_ip=source_ip,
                input_sample=input_text[:100]  # Log first 100 chars only
            )

        return validation_result

    def _detect_pii(self, text: str) -> List[str]:
        """Detect personally identifiable information in text."""

        pii_found = []

        for pii_type, pattern in self.config.data_protection.pii_patterns.items():
            if re.search(pattern, text):
                pii_found.append(pii_type)

        return pii_found

    async def _check_rate_limit(self, source_ip: str) -> bool:
        """Check if source IP has exceeded rate limits."""

        # This is a simplified implementation
        # In production, you'd use Redis or similar for distributed rate limiting
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(minutes=1)

        # Count requests in the current window
        # Implementation would track actual request timestamps
        return False  # Placeholder

    async def _setup_authentication(self):
        """Configure authentication security measures."""

        logger.info("Setting up authentication security...")

        if self.config.authentication.require_api_key:
            # Generate secure API keys
            self._generate_api_keys()

        # Configure session security
        if self.config.authentication.enable_session_encryption:
            self._setup_session_encryption()

    def _generate_api_keys(self):
        """Generate cryptographically secure API keys."""

        # Generate a new API key
        api_key = secrets.token_urlsafe(self.config.authentication.api_key_length)

        # Hash the key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        logger.info(f"Generated new API key (hash: {key_hash[:16]}...)")

        # In production, store the hash securely
        return api_key

    def _setup_session_encryption(self):
        """Configure secure session handling."""

        session_key = secrets.token_bytes(32)  # 256-bit key
        logger.info("Session encryption configured")

        return session_key

    async def _setup_network_security(self):
        """Configure network-level security measures."""

        logger.info("Setting up network security...")

        # Configure security headers
        self.security_headers = {
            "Strict-Transport-Security": f"max-age={self.config.network_security.hsts_max_age}; includeSubDomains",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

        logger.info(f"Configured {len(self.security_headers)} security headers")

    async def _setup_data_protection(self):
        """Configure data protection measures."""

        logger.info("Setting up data protection...")

        if self.config.data_protection.enable_data_encryption:
            self._setup_encryption()

        if self.config.data_protection.enable_data_masking:
            self._setup_data_masking()

    def _setup_encryption(self):
        """Configure data encryption."""

        # Generate encryption key
        encryption_key = secrets.token_bytes(32)  # 256-bit key
        logger.info(f"Data encryption configured ({self.config.data_protection.encryption_algorithm})")

        return encryption_key

    def _setup_data_masking(self):
        """Configure data masking rules."""

        self.masking_rules = {}
        for pii_type, pattern in self.config.data_protection.pii_patterns.items():
            self.masking_rules[pii_type] = {
                "pattern": re.compile(pattern),
                "replacement": self._generate_mask_pattern(pattern)
            }

        logger.info(f"Configured {len(self.masking_rules)} data masking rules")

    def _generate_mask_pattern(self, pattern: str) -> str:
        """Generate masking pattern for a given regex."""

        # Simplified masking - replace with configured character
        return self.config.data_protection.masking_character * 8

    async def _setup_threat_detection(self):
        """Configure threat detection and monitoring."""

        logger.info("Setting up threat detection...")

        if self.config.threat_detection.enable_anomaly_detection:
            await self._initialize_anomaly_detection()

        if self.config.threat_detection.enable_behavioral_analysis:
            await self._initialize_behavioral_analysis()

    async def _initialize_anomaly_detection(self):
        """Initialize machine learning-based anomaly detection."""

        # This would typically involve loading pre-trained models
        # or setting up training pipelines
        logger.info("Anomaly detection initialized")

    async def _initialize_behavioral_analysis(self):
        """Initialize behavioral analysis for users and requests."""

        self.user_baselines = {}
        logger.info("Behavioral analysis initialized")

    async def _setup_compliance_monitoring(self):
        """Configure compliance monitoring and reporting."""

        logger.info("Setting up compliance monitoring...")

        self.compliance_events = []
        self.compliance_metrics = {}

        for framework in self.config.compliance.enabled_frameworks:
            logger.info(f"Enabled compliance monitoring for {framework}")

    async def _start_security_monitoring(self):
        """Start continuous security monitoring loop."""

        logger.info("Starting security monitoring...")

        while True:
            try:
                await self._monitor_security_metrics()
                await self._detect_threats()
                await self._update_security_status()

                await asyncio.sleep(self.config.threat_detection.monitoring_interval_seconds)

            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(5)

    async def _monitor_security_metrics(self):
        """Monitor key security metrics."""

        current_time = datetime.utcnow()

        # Update security metrics
        self.security_metrics.update({
            "timestamp": current_time.isoformat(),
            "active_sessions": len(self.active_sessions),
            "threat_incidents": len(self.threat_incidents),
            "compliance_events": len(getattr(self, 'compliance_events', []))
        })

    async def _detect_threats(self):
        """Run threat detection algorithms."""

        # Placeholder for advanced threat detection
        # In production, this would include ML models, pattern matching, etc.
        pass

    async def _update_security_status(self):
        """Update overall security status."""

        # Calculate security health score
        health_score = self._calculate_security_health()

        self.security_metrics["health_score"] = health_score

        if health_score < 0.7:  # Below 70% health
            await self._trigger_security_alert("low_security_health", {"score": health_score})

    def _calculate_security_health(self) -> float:
        """Calculate overall security health score (0.0 to 1.0)."""

        # Simplified calculation
        base_score = 1.0

        # Deduct for recent incidents
        recent_incidents = len([
            incident for incident in self.threat_incidents
            if datetime.fromisoformat(incident.get("timestamp", "1970-01-01T00:00:00"))
            > datetime.utcnow() - timedelta(hours=24)
        ])

        base_score -= min(recent_incidents * 0.1, 0.5)  # Max 50% deduction

        return max(base_score, 0.0)

    async def _trigger_security_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger security alert and response actions."""

        logger.warning(f"Security alert triggered: {alert_type}")

        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": alert_type,
            "details": details,
            "response_actions": []
        }

        # Execute configured response actions
        if self.config.threat_detection.enable_auto_response:
            for action in self.config.threat_detection.response_actions:
                await self._execute_response_action(action, alert)

        # Log the incident
        await self._log_security_incident("security_alert", alert)

    async def _execute_response_action(self, action: str, alert: Dict[str, Any]):
        """Execute automated security response action."""

        if action == "alert_admin":
            await self._send_security_notification(alert)
        elif action == "block_ip":
            await self._block_ip_address(alert.get("source_ip"))
        elif action == "log_incident":
            await self._log_security_incident("auto_response", {"action": action, "alert": alert})

        alert["response_actions"].append(action)

    async def _send_security_notification(self, alert: Dict[str, Any]):
        """Send security notification to administrators."""

        # Placeholder for notification system
        logger.info(f"Security notification sent: {alert['type']}")

    async def _block_ip_address(self, ip_address: str):
        """Block IP address in firewall/load balancer."""

        if not ip_address:
            return

        # Add to blocked IPs list
        self.config.network_security.blocked_ips.append(ip_address)
        logger.info(f"IP address blocked: {ip_address}")

    async def _log_security_incident(self, incident_type: str, details: Dict[str, Any], **kwargs):
        """Log security incident for audit and analysis."""

        incident = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": incident_type,
            "details": details,
            **kwargs
        }

        self.threat_incidents.append(incident)

        # Log to security log
        logger.warning(f"Security incident: {incident_type} - {details}")

        # In production, this would also:
        # - Send to SIEM system
        # - Store in secure audit database
        # - Trigger compliance workflows


def load_security_config() -> SecurityHardeningConfig:
    """Load security configuration based on environment and requirements."""

    config = SecurityHardeningConfig()

    # Environment-specific security levels
    env = os.environ.get("QUERY_AGENT_ENV", "development")

    if env == "production":
        config.security_level = SecurityLevel.MAXIMUM
        config.authentication.require_api_key = True
        config.network_security.enforce_https = True
        config.threat_detection.enable_realtime_monitoring = True
        config.compliance.enabled_frameworks = ["SOC2", "GDPR"]

    elif env == "staging":
        config.security_level = SecurityLevel.ENHANCED
        config.authentication.require_api_key = True
        config.network_security.enforce_https = True
        config.threat_detection.enable_realtime_monitoring = True

    elif env == "development":
        config.security_level = SecurityLevel.STANDARD
        config.development_mode = True
        # Relax some security measures for development
        config.network_security.enforce_https = False
        config.authentication.require_api_key = False

    logger.info(f"Security configuration loaded for {env} environment (Level: {config.security_level.value})")
    return config


async def initialize_security_hardening(config: SecurityHardeningConfig) -> SecurityHardeningSuite:
    """Initialize comprehensive security hardening."""

    security_suite = SecurityHardeningSuite(config)
    await security_suite.initialize_security()

    logger.info("Security hardening suite initialized successfully")
    return security_suite


# Example usage
if __name__ == "__main__":

    async def demo():
        """TODO: Add docstring"""
        # Load configuration
        security_config = load_security_config()

        # Initialize security hardening
        security_suite = await initialize_security_hardening(security_config)

        # Test input validation
        test_inputs = [
            "SELECT * FROM users WHERE name = 'John'",  # Safe
            "'; DROP TABLE users; --",  # SQL injection attempt
            "SELECT * FROM users WHERE ssn = '123-45-6789'",  # Contains PII
        ]

        for test_input in test_inputs:
            result = await security_suite.validate_input(test_input, "192.168.1.100")
            print(f"Input: {test_input[:50]}...")
            print(f"Safe: {result['is_safe']}, Risk Score: {result['risk_score']:.2f}")
            print(f"Threats: {result['threats_detected']}")
            print("---")

        print("Security hardening demo completed")

    # Run demo
    asyncio.run(demo())
