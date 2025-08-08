"""
Advanced Error Recovery and Fault Tolerance System
Provides intelligent error handling, automatic recovery, and system resilience.
"""

import logging
import time
import json
import traceback
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import threading
from functools import wraps
import sys

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Different recovery strategies for various error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    BACKOFF = "exponential_backoff"


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""
    error_id: str
    error_type: str
    severity: ErrorSeverity
    message: str
    stack_trace: str
    component: str
    operation: str
    timestamp: datetime
    user_context: Optional[Dict] = None
    system_state: Optional[Dict] = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken for an error."""
    strategy: RecoveryStrategy
    action_function: Optional[Callable] = None
    parameters: Optional[Dict] = None
    timeout_seconds: int = 30
    retry_delay_seconds: float = 1.0
    max_retries: int = 3


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery strategies."""
    
    def __init__(self):
        self.error_patterns = {
            # Database-related errors
            'database_connection': {
                'patterns': ['connection', 'timeout', 'unreachable', 'refused'],
                'severity': ErrorSeverity.HIGH,
                'recovery': RecoveryStrategy.RETRY
            },
            'database_query': {
                'patterns': ['syntax error', 'invalid query', 'table does not exist'],
                'severity': ErrorSeverity.MEDIUM,
                'recovery': RecoveryStrategy.FALLBACK
            },
            'database_permission': {
                'patterns': ['permission denied', 'access denied', 'insufficient privileges'],
                'severity': ErrorSeverity.HIGH,
                'recovery': RecoveryStrategy.FAIL_FAST
            },
            
            # LLM/API errors
            'llm_rate_limit': {
                'patterns': ['rate limit', 'quota exceeded', 'too many requests'],
                'severity': ErrorSeverity.MEDIUM,
                'recovery': RecoveryStrategy.BACKOFF
            },
            'llm_authentication': {
                'patterns': ['unauthorized', 'invalid api key', 'authentication failed'],
                'severity': ErrorSeverity.HIGH,
                'recovery': RecoveryStrategy.FAIL_FAST
            },
            'llm_timeout': {
                'patterns': ['timeout', 'request timeout', 'read timeout'],
                'severity': ErrorSeverity.MEDIUM,
                'recovery': RecoveryStrategy.RETRY
            },
            
            # System errors
            'memory_error': {
                'patterns': ['out of memory', 'memory error', 'cannot allocate'],
                'severity': ErrorSeverity.CRITICAL,
                'recovery': RecoveryStrategy.GRACEFUL_DEGRADATION
            },
            'disk_space': {
                'patterns': ['no space left', 'disk full', 'storage full'],
                'severity': ErrorSeverity.CRITICAL,
                'recovery': RecoveryStrategy.GRACEFUL_DEGRADATION
            },
            
            # Network errors
            'network_error': {
                'patterns': ['network error', 'connection reset', 'host unreachable'],
                'severity': ErrorSeverity.MEDIUM,
                'recovery': RecoveryStrategy.RETRY
            }
        }
    
    def classify_error(self, error: Exception, context: Optional[Dict] = None) -> Tuple[str, ErrorSeverity, RecoveryStrategy]:
        """Classify an error and determine recovery strategy."""
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Check against known patterns
        for category, config in self.error_patterns.items():
            if any(pattern in error_message for pattern in config['patterns']):
                return category, config['severity'], config['recovery']
        
        # Default classification based on exception type
        if 'Connection' in error_type or 'Timeout' in error_type:
            return 'network_error', ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY
        elif 'Permission' in error_type or 'Auth' in error_type:
            return 'permission_error', ErrorSeverity.HIGH, RecoveryStrategy.FAIL_FAST
        elif 'Memory' in error_type:
            return 'memory_error', ErrorSeverity.CRITICAL, RecoveryStrategy.GRACEFUL_DEGRADATION
        else:
            return 'unknown_error', ErrorSeverity.MEDIUM, RecoveryStrategy.FALLBACK


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._call_with_circuit_breaker_async(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self._should_reject_call():
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    async def _call_with_circuit_breaker_async(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self._should_reject_call():
            raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_reject_call(self) -> bool:
        """Determine if call should be rejected based on circuit breaker state."""
        if self.state == 'CLOSED':
            return False
        elif self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                return False
            return True
        elif self.state == 'HALF_OPEN':
            return False
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0
            self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry and exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Retry attempt {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}")
                time.sleep(delay)
        
        raise last_exception
    
    async def retry_with_backoff_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry and exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Async retry attempt {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff and jitter."""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add ±50% jitter
        
        return delay


class ErrorRecoveryManager:
    """Main manager for error recovery and fault tolerance."""
    
    def __init__(self):
        self.error_classifier = ErrorClassifier()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.error_counts: Counter = Counter()
        self.recovery_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'recovery_strategies': Counter()
        }
        
        # Recovery actions registry
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        
        self._setup_default_recovery_actions()
    
    def handle_error(self, error: Exception, context: Optional[Dict] = None,
                    component: str = "unknown", operation: str = "unknown") -> Any:
        """Main error handling entry point."""
        # Classify error
        error_category, severity, recovery_strategy = self.error_classifier.classify_error(error, context)
        
        # Create error context
        error_context = ErrorContext(
            error_id=self._generate_error_id(),
            error_type=type(error).__name__,
            severity=severity,
            message=str(error),
            stack_trace=traceback.format_exc(),
            component=component,
            operation=operation,
            timestamp=datetime.utcnow(),
            user_context=context,
            system_state=self._collect_system_state()
        )
        
        # Log error
        self._log_error(error_context)
        
        # Track error
        self._track_error(error_context, error_category)
        
        # Attempt recovery
        try:
            result = self._attempt_recovery(error_context, recovery_strategy, error)
            self.recovery_stats['recovered_errors'] += 1
            return result
        except Exception as recovery_error:
            self.recovery_stats['failed_recoveries'] += 1
            logger.error(f"Recovery failed for error {error_context.error_id}: {recovery_error}")
            
            # If recovery fails, decide whether to re-raise or gracefully degrade
            if severity == ErrorSeverity.CRITICAL:
                return self._graceful_degradation(error_context)
            else:
                raise error  # Re-raise original error
    
    def register_recovery_action(self, error_category: str, action: RecoveryAction):
        """Register a custom recovery action for an error category."""
        self.recovery_actions[error_category] = action
        logger.info(f"Registered recovery action for {error_category}: {action.strategy}")
    
    def get_circuit_breaker(self, component: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[component]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error and recovery statistics."""
        total_errors = self.recovery_stats['total_errors']
        
        return {
            'error_counts': dict(self.error_counts.most_common(10)),
            'recovery_stats': {
                'total_errors': total_errors,
                'recovered_errors': self.recovery_stats['recovered_errors'],
                'failed_recoveries': self.recovery_stats['failed_recoveries'],
                'recovery_rate': (self.recovery_stats['recovered_errors'] / max(total_errors, 1)) * 100
            },
            'recovery_strategies': dict(self.recovery_stats['recovery_strategies']),
            'circuit_breaker_states': {
                component: breaker.state 
                for component, breaker in self.circuit_breakers.items()
            },
            'recent_errors': [
                {
                    'error_id': ctx.error_id,
                    'error_type': ctx.error_type,
                    'severity': ctx.severity.value,
                    'component': ctx.component,
                    'timestamp': ctx.timestamp.isoformat()
                }
                for ctx in self.error_history[-10:]
            ]
        }
    
    def _attempt_recovery(self, error_context: ErrorContext, 
                         recovery_strategy: RecoveryStrategy, original_error: Exception) -> Any:
        """Attempt to recover from error using appropriate strategy."""
        self.recovery_stats['recovery_strategies'][recovery_strategy.value] += 1
        
        if recovery_strategy == RecoveryStrategy.RETRY:
            return self._retry_recovery(error_context, original_error)
        elif recovery_strategy == RecoveryStrategy.FALLBACK:
            return self._fallback_recovery(error_context)
        elif recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._circuit_breaker_recovery(error_context)
        elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(error_context)
        elif recovery_strategy == RecoveryStrategy.BACKOFF:
            return self._backoff_recovery(error_context, original_error)
        elif recovery_strategy == RecoveryStrategy.FAIL_FAST:
            raise original_error
        else:
            logger.warning(f"Unknown recovery strategy: {recovery_strategy}")
            raise original_error
    
    def _retry_recovery(self, error_context: ErrorContext, original_error: Exception) -> Any:
        """Retry-based recovery with exponential backoff."""
        # Get custom recovery action if registered
        recovery_action = self.recovery_actions.get(error_context.component)
        
        if recovery_action and recovery_action.action_function:
            return self.retry_manager.retry_with_backoff(
                recovery_action.action_function,
                error_context=error_context
            )
        else:
            # Default retry behavior - just re-raise for now
            # In a real implementation, you'd retry the original operation
            logger.info(f"Retry recovery attempted for {error_context.error_id}")
            raise original_error
    
    def _fallback_recovery(self, error_context: ErrorContext) -> Any:
        """Fallback-based recovery."""
        recovery_action = self.recovery_actions.get(error_context.component)
        
        if recovery_action and recovery_action.action_function:
            return recovery_action.action_function(error_context=error_context)
        else:
            # Default fallback - return safe default value
            logger.info(f"Fallback recovery applied for {error_context.error_id}")
            return self._get_safe_default(error_context)
    
    def _circuit_breaker_recovery(self, error_context: ErrorContext) -> Any:
        """Circuit breaker-based recovery."""
        component = error_context.component
        breaker = self.get_circuit_breaker(component)
        
        if breaker.state == 'OPEN':
            logger.warning(f"Circuit breaker OPEN for {component}, applying fallback")
            return self._fallback_recovery(error_context)
        else:
            raise CircuitBreakerOpenError(f"Circuit breaker prevents operation on {component}")
    
    def _graceful_degradation(self, error_context: ErrorContext) -> Any:
        """Graceful degradation recovery."""
        logger.warning(f"Graceful degradation applied for critical error {error_context.error_id}")
        
        # Return minimal functionality result
        return {
            'status': 'degraded',
            'message': 'Service operating in degraded mode due to system error',
            'error_id': error_context.error_id,
            'timestamp': error_context.timestamp.isoformat()
        }
    
    def _backoff_recovery(self, error_context: ErrorContext, original_error: Exception) -> Any:
        """Exponential backoff recovery for rate limiting scenarios."""
        logger.info(f"Backoff recovery applied for {error_context.error_id}")
        
        # Apply longer delay for rate limiting
        delay = min(60.0, 2.0 ** error_context.recovery_attempts)
        time.sleep(delay)
        
        # Try fallback after backoff
        return self._fallback_recovery(error_context)
    
    def _get_safe_default(self, error_context: ErrorContext) -> Any:
        """Get safe default value for fallback recovery."""
        if 'query' in error_context.operation.lower():
            return {
                'sql': 'SELECT 1 as status',
                'explanation': 'Safe fallback query due to error',
                'data': [{'status': 1}],
                'error_recovery_applied': True,
                'original_error_id': error_context.error_id
            }
        else:
            return {
                'status': 'error_recovery_applied',
                'message': 'Operation completed with fallback behavior',
                'error_id': error_context.error_id
            }
    
    def _setup_default_recovery_actions(self):
        """Setup default recovery actions for common scenarios."""
        # Database connection recovery
        self.recovery_actions['database'] = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            timeout_seconds=30,
            max_retries=3,
            retry_delay_seconds=2.0
        )
        
        # LLM service recovery
        self.recovery_actions['llm'] = RecoveryAction(
            strategy=RecoveryStrategy.BACKOFF,
            timeout_seconds=60,
            max_retries=5,
            retry_delay_seconds=1.0
        )
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        import uuid
        return f"err_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    def _collect_system_state(self) -> Dict[str, Any]:
        """Collect current system state for error context."""
        import psutil
        
        try:
            return {
                'memory_usage_percent': psutil.virtual_memory().percent,
                'cpu_usage_percent': psutil.cpu_percent(),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'active_threads': threading.active_count(),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception:
            return {'collection_error': 'Failed to collect system state'}
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level based on severity."""
        log_data = {
            'error_id': error_context.error_id,
            'component': error_context.component,
            'operation': error_context.operation,
            'error_type': error_context.error_type,
            'message': error_context.message[:500]  # Truncate long messages
        }
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {log_data}", extra=log_data)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {log_data}", extra=log_data)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {log_data}", extra=log_data)
        else:
            logger.info(f"LOW SEVERITY ERROR: {log_data}", extra=log_data)
    
    def _track_error(self, error_context: ErrorContext, error_category: str):
        """Track error for statistics and analysis."""
        self.recovery_stats['total_errors'] += 1
        self.error_counts[error_category] += 1
        self.error_history.append(error_context)
        
        # Keep only recent error history
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


def resilient_operation(component: str = "unknown", operation: str = "unknown"):
    """Decorator for making operations resilient with automatic error recovery."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = kwargs.get('context', {})
                return error_recovery_manager.handle_error(
                    e, context, component, operation
                )
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = kwargs.get('context', {})
                return error_recovery_manager.handle_error(
                    e, context, component, operation
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator