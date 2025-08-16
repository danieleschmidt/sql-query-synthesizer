"""
Graceful Degradation System
Manages service degradation strategies to maintain core functionality during failures.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation."""

    NORMAL = "normal"
    MINOR_DEGRADATION = "minor_degradation"
    MAJOR_DEGRADATION = "major_degradation"
    CRITICAL_DEGRADATION = "critical_degradation"
    EMERGENCY_MODE = "emergency_mode"


class DegradationStrategy(Enum):
    """Different degradation strategies."""

    FEATURE_TOGGLE = "feature_toggle"
    FALLBACK_DATA = "fallback_data"
    REDUCED_FUNCTIONALITY = "reduced_functionality"
    CACHED_RESPONSES = "cached_responses"
    SIMPLIFIED_PROCESSING = "simplified_processing"
    RATE_LIMITING = "rate_limiting"
    READ_ONLY_MODE = "read_only_mode"


@dataclass
class ServiceCapability:
    """Represents a service capability that can be degraded."""

    name: str
    priority: int  # 1=highest priority, 10=lowest
    degradation_strategy: DegradationStrategy
    enabled: bool = True
    fallback_function: Optional[Callable] = None
    degraded_since: Optional[datetime] = None
    degradation_reason: Optional[str] = None
    impact_description: str = ""
    recovery_check: Optional[Callable] = None


class GracefulDegradationManager:
    """Manages graceful degradation of services during system stress or failures."""

    def __init__(self):
        self.capabilities: Dict[str, ServiceCapability] = {}
        self.current_level = DegradationLevel.NORMAL
        self.degradation_history: List[Dict] = []

        # Degradation triggers
        self.triggers = {
            "cpu_threshold": 85,
            "memory_threshold": 90,
            "error_rate_threshold": 10,  # errors per minute
            "response_time_threshold": 5000,  # ms
            "dependency_failure_threshold": 3,
        }

        # Monitoring state
        self.error_counts = defaultdict(int)
        self.response_times = []
        self.dependency_failures = defaultdict(int)

        self.lock = threading.RLock()

        # Register default capabilities
        self._register_default_capabilities()

    def register_capability(self, capability: ServiceCapability):
        """Register a service capability for degradation management."""
        with self.lock:
            self.capabilities[capability.name] = capability
            logger.info(
                f"Registered service capability: {capability.name} (priority: {capability.priority})"
            )

    def trigger_degradation(
        self,
        level: DegradationLevel,
        reason: str,
        affected_capabilities: Optional[List[str]] = None,
    ):
        """Manually trigger service degradation."""
        with self.lock:
            if (
                level.value == self.current_level.value
                and affected_capabilities is None
            ):
                logger.info(f"Already at degradation level: {level.value}")
                return

            previous_level = self.current_level
            self.current_level = level

            # Apply degradation strategies
            degraded_capabilities = self._apply_degradation_level(
                level, affected_capabilities
            )

            # Record degradation event
            degradation_event = {
                "timestamp": datetime.utcnow().isoformat(),
                "previous_level": previous_level.value,
                "new_level": level.value,
                "reason": reason,
                "affected_capabilities": degraded_capabilities,
                "trigger_type": "manual",
            }

            self.degradation_history.append(degradation_event)

            logger.warning(f"Service degradation triggered: {level.value} - {reason}")
            logger.info(f"Affected capabilities: {degraded_capabilities}")

    def auto_assess_degradation_need(self, system_metrics: Dict[str, Any]) -> bool:
        """Automatically assess if degradation is needed based on system metrics."""
        with self.lock:
            degradation_needed = False
            reasons = []
            suggested_level = self.current_level

            # Check CPU usage
            cpu_percent = system_metrics.get("cpu_percent", 0)
            if cpu_percent > self.triggers["cpu_threshold"]:
                degradation_needed = True
                reasons.append(f"High CPU usage: {cpu_percent}%")
                if cpu_percent > 95:
                    suggested_level = DegradationLevel.CRITICAL_DEGRADATION
                elif cpu_percent > 90:
                    suggested_level = DegradationLevel.MAJOR_DEGRADATION
                else:
                    suggested_level = DegradationLevel.MINOR_DEGRADATION

            # Check memory usage
            memory_percent = system_metrics.get("memory_percent", 0)
            if memory_percent > self.triggers["memory_threshold"]:
                degradation_needed = True
                reasons.append(f"High memory usage: {memory_percent}%")
                if memory_percent > 95:
                    suggested_level = max(
                        suggested_level,
                        DegradationLevel.CRITICAL_DEGRADATION,
                        key=lambda x: list(DegradationLevel).index(x),
                    )
                elif memory_percent > 92:
                    suggested_level = max(
                        suggested_level,
                        DegradationLevel.MAJOR_DEGRADATION,
                        key=lambda x: list(DegradationLevel).index(x),
                    )

            # Check error rates
            recent_errors = self._get_recent_error_count()
            if recent_errors > self.triggers["error_rate_threshold"]:
                degradation_needed = True
                reasons.append(f"High error rate: {recent_errors}/min")
                if recent_errors > 50:
                    suggested_level = max(
                        suggested_level,
                        DegradationLevel.MAJOR_DEGRADATION,
                        key=lambda x: list(DegradationLevel).index(x),
                    )
                else:
                    suggested_level = max(
                        suggested_level,
                        DegradationLevel.MINOR_DEGRADATION,
                        key=lambda x: list(DegradationLevel).index(x),
                    )

            # Auto-trigger if needed
            if degradation_needed and suggested_level != self.current_level:
                self.trigger_degradation(
                    suggested_level, f"Auto-triggered: {'; '.join(reasons)}", None
                )
                return True

            return False

    def attempt_recovery(self) -> bool:
        """Attempt to recover from degraded state."""
        with self.lock:
            if self.current_level == DegradationLevel.NORMAL:
                return True

            # Check if recovery is possible
            recovery_possible = self._check_recovery_conditions()

            if recovery_possible:
                # Gradually restore capabilities
                restored_capabilities = self._attempt_capability_recovery()

                if restored_capabilities:
                    # Update degradation level
                    new_level = self._calculate_degradation_level_from_capabilities()

                    recovery_event = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "previous_level": self.current_level.value,
                        "new_level": new_level.value,
                        "restored_capabilities": restored_capabilities,
                        "trigger_type": "recovery",
                    }

                    self.current_level = new_level
                    self.degradation_history.append(recovery_event)

                    logger.info(f"Service recovery: {new_level.value}")
                    logger.info(f"Restored capabilities: {restored_capabilities}")

                    return new_level == DegradationLevel.NORMAL

            return False

    def get_capability_status(self, capability_name: str) -> Optional[Dict[str, Any]]:
        """Get current status of a specific capability."""
        capability = self.capabilities.get(capability_name)
        if not capability:
            return None

        return {
            "name": capability.name,
            "enabled": capability.enabled,
            "priority": capability.priority,
            "strategy": capability.degradation_strategy.value,
            "degraded_since": (
                capability.degraded_since.isoformat()
                if capability.degraded_since
                else None
            ),
            "degradation_reason": capability.degradation_reason,
            "impact_description": capability.impact_description,
        }

    def get_degradation_status(self) -> Dict[str, Any]:
        """Get comprehensive degradation status."""
        with self.lock:
            enabled_capabilities = [
                name for name, cap in self.capabilities.items() if cap.enabled
            ]
            degraded_capabilities = [
                name for name, cap in self.capabilities.items() if not cap.enabled
            ]

            return {
                "current_level": self.current_level.value,
                "capabilities": {
                    "total": len(self.capabilities),
                    "enabled": len(enabled_capabilities),
                    "degraded": len(degraded_capabilities),
                    "enabled_list": enabled_capabilities,
                    "degraded_list": degraded_capabilities,
                },
                "degradation_since": self._get_degradation_start_time(),
                "recent_events": self.degradation_history[-5:],
                "recovery_possible": self._check_recovery_conditions(),
            }

    def record_error(self, component: str, error_type: str = "general"):
        """Record an error for degradation assessment."""
        with self.lock:
            error_key = f"{component}_{error_type}"
            self.error_counts[error_key] += 1

            # Clean old error counts (keep only last 5 minutes)
            # In a production system, you'd use a time-based sliding window

    def record_response_time(self, response_time_ms: float):
        """Record response time for performance assessment."""
        with self.lock:
            self.response_times.append(
                {"timestamp": time.time(), "response_time_ms": response_time_ms}
            )

            # Keep only recent response times
            cutoff = time.time() - 300  # 5 minutes
            self.response_times = [
                rt for rt in self.response_times if rt["timestamp"] > cutoff
            ]

    def execute_with_degradation(
        self, capability_name: str, primary_function: Callable, *args, **kwargs
    ) -> Any:
        """Execute a function with degradation support."""
        capability = self.capabilities.get(capability_name)

        if not capability:
            logger.warning(f"Unknown capability: {capability_name}")
            return primary_function(*args, **kwargs)

        if capability.enabled:
            # Capability is enabled, execute normally
            try:
                return primary_function(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in capability {capability_name}: {e}")
                # On error, try fallback if available
                if capability.fallback_function:
                    logger.info(f"Attempting fallback for {capability_name}")
                    return capability.fallback_function(*args, **kwargs)
                else:
                    raise
        else:
            # Capability is degraded, use fallback
            if capability.fallback_function:
                logger.debug(f"Using degraded mode for {capability_name}")
                return capability.fallback_function(*args, **kwargs)
            else:
                # No fallback available
                logger.warning(
                    f"Capability {capability_name} unavailable and no fallback defined"
                )
                return self._get_default_degraded_response(capability_name)

    def _register_default_capabilities(self):
        """Register default service capabilities."""
        # LLM-based SQL generation
        self.register_capability(
            ServiceCapability(
                name="llm_sql_generation",
                priority=2,  # High priority but not critical
                degradation_strategy=DegradationStrategy.FALLBACK_DATA,
                impact_description="Falls back to template-based SQL generation",
                fallback_function=self._fallback_sql_generation,
            )
        )

        # Advanced query insights
        self.register_capability(
            ServiceCapability(
                name="query_insights",
                priority=5,  # Medium priority
                degradation_strategy=DegradationStrategy.FEATURE_TOGGLE,
                impact_description="Query optimization insights temporarily disabled",
                fallback_function=self._fallback_basic_insights,
            )
        )

        # Semantic caching
        self.register_capability(
            ServiceCapability(
                name="semantic_caching",
                priority=7,  # Lower priority
                degradation_strategy=DegradationStrategy.SIMPLIFIED_PROCESSING,
                impact_description="Uses simple key-based caching instead of semantic similarity",
                fallback_function=self._fallback_simple_caching,
            )
        )

        # Adaptive learning
        self.register_capability(
            ServiceCapability(
                name="adaptive_learning",
                priority=8,  # Lower priority
                degradation_strategy=DegradationStrategy.FEATURE_TOGGLE,
                impact_description="Machine learning features temporarily disabled",
                fallback_function=self._fallback_no_learning,
            )
        )

        # Query validation
        self.register_capability(
            ServiceCapability(
                name="advanced_validation",
                priority=3,  # Important for security
                degradation_strategy=DegradationStrategy.SIMPLIFIED_PROCESSING,
                impact_description="Uses basic validation instead of advanced AST analysis",
                fallback_function=self._fallback_basic_validation,
            )
        )

    def _apply_degradation_level(
        self, level: DegradationLevel, specific_capabilities: Optional[List[str]] = None
    ) -> List[str]:
        """Apply degradation strategies based on level."""
        degraded_capabilities = []

        if specific_capabilities:
            # Degrade specific capabilities
            for cap_name in specific_capabilities:
                if cap_name in self.capabilities:
                    self.capabilities[cap_name].enabled = False
                    self.capabilities[cap_name].degraded_since = datetime.utcnow()
                    self.capabilities[cap_name].degradation_reason = (
                        f"Specific degradation: {level.value}"
                    )
                    degraded_capabilities.append(cap_name)
        else:
            # Degrade based on priority and level
            priority_threshold = self._get_priority_threshold_for_level(level)

            for cap_name, capability in self.capabilities.items():
                if capability.priority >= priority_threshold and capability.enabled:
                    capability.enabled = False
                    capability.degraded_since = datetime.utcnow()
                    capability.degradation_reason = f"Auto degradation: {level.value}"
                    degraded_capabilities.append(cap_name)

        return degraded_capabilities

    def _get_priority_threshold_for_level(self, level: DegradationLevel) -> int:
        """Get priority threshold for degradation level."""
        thresholds = {
            DegradationLevel.MINOR_DEGRADATION: 8,  # Disable priority 8-10
            DegradationLevel.MAJOR_DEGRADATION: 5,  # Disable priority 5-10
            DegradationLevel.CRITICAL_DEGRADATION: 3,  # Disable priority 3-10
            DegradationLevel.EMERGENCY_MODE: 1,  # Disable all except priority 1
        }
        return thresholds.get(level, 10)

    def _check_recovery_conditions(self) -> bool:
        """Check if conditions allow for recovery."""
        # Simple recovery condition check
        # In production, this would check system metrics, dependency health, etc.

        recent_errors = self._get_recent_error_count()
        if recent_errors > self.triggers["error_rate_threshold"] * 0.8:
            return False

        # Check if enough time has passed since degradation
        for capability in self.capabilities.values():
            if (
                not capability.enabled
                and capability.degraded_since
                and datetime.utcnow() - capability.degraded_since < timedelta(minutes=2)
            ):
                return False  # Too soon to recover

        return True

    def _attempt_capability_recovery(self) -> List[str]:
        """Attempt to recover degraded capabilities."""
        recovered = []

        # Sort capabilities by priority (recover high priority first)
        sorted_capabilities = sorted(
            [(name, cap) for name, cap in self.capabilities.items() if not cap.enabled],
            key=lambda x: x[1].priority,
        )

        for cap_name, capability in sorted_capabilities:
            # Check if this specific capability can be recovered
            if capability.recovery_check:
                try:
                    if capability.recovery_check():
                        capability.enabled = True
                        capability.degraded_since = None
                        capability.degradation_reason = None
                        recovered.append(cap_name)
                except Exception as e:
                    logger.error(f"Recovery check failed for {cap_name}: {e}")
            else:
                # Default recovery - just re-enable
                capability.enabled = True
                capability.degraded_since = None
                capability.degradation_reason = None
                recovered.append(cap_name)

        return recovered

    def _calculate_degradation_level_from_capabilities(self) -> DegradationLevel:
        """Calculate current degradation level based on enabled capabilities."""
        disabled_capabilities = [
            cap for cap in self.capabilities.values() if not cap.enabled
        ]

        if not disabled_capabilities:
            return DegradationLevel.NORMAL

        # Check the highest priority of disabled capabilities
        highest_disabled_priority = min(cap.priority for cap in disabled_capabilities)

        if highest_disabled_priority == 1:
            return DegradationLevel.EMERGENCY_MODE
        elif highest_disabled_priority <= 3:
            return DegradationLevel.CRITICAL_DEGRADATION
        elif highest_disabled_priority <= 5:
            return DegradationLevel.MAJOR_DEGRADATION
        else:
            return DegradationLevel.MINOR_DEGRADATION

    def _get_recent_error_count(self) -> int:
        """Get recent error count for degradation assessment."""
        # Simple implementation - in production you'd use proper time windows
        return sum(self.error_counts.values())

    def _get_degradation_start_time(self) -> Optional[str]:
        """Get when degradation started."""
        earliest_degradation = None
        for capability in self.capabilities.values():
            if not capability.enabled and capability.degraded_since:
                if (
                    earliest_degradation is None
                    or capability.degraded_since < earliest_degradation
                ):
                    earliest_degradation = capability.degraded_since

        return earliest_degradation.isoformat() if earliest_degradation else None

    # Fallback functions
    def _fallback_sql_generation(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for LLM-based SQL generation."""
        user_question = args[0] if args else kwargs.get("user_question", "")

        # Very basic SQL generation based on common patterns
        if "count" in user_question.lower():
            sql = "SELECT COUNT(*) FROM table_name"
        elif "select" in user_question.lower() or "show" in user_question.lower():
            sql = "SELECT * FROM table_name LIMIT 10"
        else:
            sql = "SELECT 1 as status"

        return {
            "sql": sql,
            "explanation": "Basic SQL generated - advanced AI features temporarily unavailable",
            "confidence": 0.3,
            "degraded_mode": True,
        }

    def _fallback_basic_insights(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for query insights."""
        return {
            "insights": [
                {
                    "type": "info",
                    "message": "Advanced query insights temporarily unavailable",
                    "recommendation": "Query analysis running in basic mode",
                }
            ],
            "degraded_mode": True,
        }

    def _fallback_simple_caching(self, *args, **kwargs) -> Any:
        """Fallback for semantic caching."""
        # Use simple string-based caching instead of semantic similarity
        return None  # Let it fall through to normal caching

    def _fallback_no_learning(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for adaptive learning."""
        return {
            "patterns": [],
            "insights": [],
            "message": "Learning features temporarily disabled",
            "degraded_mode": True,
        }

    def _fallback_basic_validation(self, sql: str, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for advanced validation."""
        # Basic validation - just check for obvious SQL injection patterns
        dangerous_patterns = ["drop", "delete", "truncate", "exec", ";"]

        sql_lower = sql.lower()
        for pattern in dangerous_patterns:
            if pattern in sql_lower:
                return {
                    "valid": False,
                    "message": f"Potentially dangerous pattern detected: {pattern}",
                    "degraded_validation": True,
                }

        return {
            "valid": True,
            "message": "Basic validation passed",
            "degraded_validation": True,
        }

    def _get_default_degraded_response(self, capability_name: str) -> Dict[str, Any]:
        """Get default response when capability is unavailable."""
        return {
            "status": "unavailable",
            "message": f'Service capability "{capability_name}" is temporarily unavailable',
            "degraded_mode": True,
            "degradation_level": self.current_level.value,
        }


# Global degradation manager instance
degradation_manager = GracefulDegradationManager()
