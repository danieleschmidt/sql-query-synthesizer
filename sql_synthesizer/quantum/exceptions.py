"""
Quantum-specific exceptions for robust error handling
"""

from typing import Any, Optional, Dict, List


class QuantumError(Exception):
    """Base class for all quantum-related errors"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "QUANTUM_ERROR"
        self.details = details or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/monitoring"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class QuantumOptimizationError(QuantumError):
    """Raised when quantum optimization fails"""
    
    def __init__(self, message: str, optimization_stage: str = None, 
                 plan_count: int = None, details: Dict[str, Any] = None):
        super().__init__(message, "QUANTUM_OPTIMIZATION_ERROR", details)
        self.optimization_stage = optimization_stage
        self.plan_count = plan_count
        
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["optimization_stage"] = self.optimization_stage
        data["plan_count"] = self.plan_count
        return data


class QuantumSchedulerError(QuantumError):
    """Raised when quantum task scheduler encounters an error"""
    
    def __init__(self, message: str, task_id: str = None, 
                 scheduler_state: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "QUANTUM_SCHEDULER_ERROR", details)
        self.task_id = task_id
        self.scheduler_state = scheduler_state
        
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["task_id"] = self.task_id
        data["scheduler_state"] = self.scheduler_state
        return data


class QuantumResourceError(QuantumError):
    """Raised when quantum resource allocation fails"""
    
    def __init__(self, message: str, resource_id: str = None, 
                 requested_capacity: float = None, available_capacity: float = None,
                 details: Dict[str, Any] = None):
        super().__init__(message, "QUANTUM_RESOURCE_ERROR", details)
        self.resource_id = resource_id
        self.requested_capacity = requested_capacity
        self.available_capacity = available_capacity
        
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["resource_id"] = self.resource_id
        data["requested_capacity"] = self.requested_capacity
        data["available_capacity"] = self.available_capacity
        return data


class QuantumDecoherenceError(QuantumError):
    """Raised when quantum coherence is lost unexpectedly"""
    
    def __init__(self, message: str, coherence_time: float = None, 
                 expected_coherence: float = None, details: Dict[str, Any] = None):
        super().__init__(message, "QUANTUM_DECOHERENCE_ERROR", details)
        self.coherence_time = coherence_time
        self.expected_coherence = expected_coherence
        
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["coherence_time"] = self.coherence_time
        data["expected_coherence"] = self.expected_coherence
        return data


class QuantumValidationError(QuantumError):
    """Raised when quantum input validation fails"""
    
    def __init__(self, message: str, field_name: str = None, 
                 field_value: Any = None, validation_rule: str = None,
                 details: Dict[str, Any] = None):
        super().__init__(message, "QUANTUM_VALIDATION_ERROR", details)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule
        
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["field_name"] = self.field_name
        data["field_value"] = str(self.field_value) if self.field_value is not None else None
        data["validation_rule"] = self.validation_rule
        return data


class QuantumTimeoutError(QuantumError):
    """Raised when quantum operations timeout"""
    
    def __init__(self, message: str, operation: str = None, 
                 timeout_seconds: float = None, elapsed_seconds: float = None,
                 details: Dict[str, Any] = None):
        super().__init__(message, "QUANTUM_TIMEOUT_ERROR", details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["operation"] = self.operation
        data["timeout_seconds"] = self.timeout_seconds
        data["elapsed_seconds"] = self.elapsed_seconds
        return data


class QuantumCircuitBreakerError(QuantumError):
    """Raised when quantum circuit breaker is open"""
    
    def __init__(self, message: str, circuit_name: str = None, 
                 failure_count: int = None, failure_threshold: int = None,
                 details: Dict[str, Any] = None):
        super().__init__(message, "QUANTUM_CIRCUIT_BREAKER_ERROR", details)
        self.circuit_name = circuit_name  
        self.failure_count = failure_count
        self.failure_threshold = failure_threshold
        
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["circuit_name"] = self.circuit_name
        data["failure_count"] = self.failure_count
        data["failure_threshold"] = self.failure_threshold
        return data


class QuantumSecurityError(QuantumError):
    """Raised when quantum security validation fails"""
    
    def __init__(self, message: str, security_check: str = None, 
                 threat_level: str = "medium", client_info: Dict[str, Any] = None,
                 details: Dict[str, Any] = None):
        super().__init__(message, "QUANTUM_SECURITY_ERROR", details)
        self.security_check = security_check
        self.threat_level = threat_level
        self.client_info = client_info or {}
        
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["security_check"] = self.security_check
        data["threat_level"] = self.threat_level
        data["client_info"] = self.client_info
        return data


class QuantumPlanGenerationError(QuantumError):
    """Raised when quantum plan generation fails"""
    
    def __init__(self, message: str, table_count: int = None, 
                 join_count: int = None, filter_count: int = None,
                 details: Dict[str, Any] = None):
        super().__init__(message, "QUANTUM_PLAN_GENERATION_ERROR", details)
        self.table_count = table_count
        self.join_count = join_count  
        self.filter_count = filter_count
        
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["table_count"] = self.table_count
        data["join_count"] = self.join_count
        data["filter_count"] = self.filter_count
        return data


# Error severity levels
class QuantumErrorSeverity:
    """Quantum error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    @classmethod
    def get_severity(cls, error: QuantumError) -> str:
        """Determine error severity based on error type"""
        severity_mapping = {
            "QuantumValidationError": cls.LOW,
            "QuantumPlanGenerationError": cls.MEDIUM,
            "QuantumOptimizationError": cls.MEDIUM,
            "QuantumResourceError": cls.MEDIUM,
            "QuantumTimeoutError": cls.HIGH,
            "QuantumSchedulerError": cls.HIGH,
            "QuantumDecoherenceError": cls.HIGH,
            "QuantumCircuitBreakerError": cls.HIGH,
            "QuantumSecurityError": cls.CRITICAL
        }
        
        return severity_mapping.get(error.__class__.__name__, cls.MEDIUM)


# Error recovery strategies
class QuantumErrorRecovery:
    """Quantum error recovery strategies"""
    
    @staticmethod
    def should_retry(error: QuantumError, attempt_count: int = 0, max_attempts: int = 3) -> bool:
        """Determine if operation should be retried based on error type"""
        if attempt_count >= max_attempts:
            return False
            
        # Don't retry validation or security errors
        no_retry_types = {
            "QuantumValidationError",
            "QuantumSecurityError"
        }
        
        if error.__class__.__name__ in no_retry_types:
            return False
            
        # Retry timeout and resource errors with exponential backoff
        retry_types = {
            "QuantumTimeoutError",
            "QuantumResourceError", 
            "QuantumSchedulerError",
            "QuantumOptimizationError"
        }
        
        return error.__class__.__name__ in retry_types
    
    @staticmethod
    def get_retry_delay(attempt_count: int, base_delay: float = 1.0) -> float:
        """Calculate retry delay with exponential backoff"""
        return base_delay * (2 ** attempt_count)
    
    @staticmethod
    def get_fallback_strategy(error: QuantumError) -> Optional[str]:
        """Get fallback strategy for different error types"""
        fallback_strategies = {
            "QuantumOptimizationError": "use_classical_optimization",
            "QuantumSchedulerError": "use_simple_scheduler",
            "QuantumResourceError": "reduce_resource_requirements",
            "QuantumDecoherenceError": "restart_quantum_system",
            "QuantumCircuitBreakerError": "bypass_quantum_optimization",
            "QuantumTimeoutError": "reduce_complexity"
        }
        
        return fallback_strategies.get(error.__class__.__name__)


# Exception context manager for quantum operations
class QuantumExceptionContext:
    """Context manager for quantum exception handling"""
    
    def __init__(self, operation_name: str, logger=None, 
                 enable_recovery: bool = True, max_retries: int = 3):
        self.operation_name = operation_name
        self.logger = logger
        self.enable_recovery = enable_recovery
        self.max_retries = max_retries
        self.attempt_count = 0
        
    def __enter__(self):
        self.attempt_count += 1
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, QuantumError):
            # Log the quantum error
            if self.logger:
                severity = QuantumErrorSeverity.get_severity(exc_val)
                self.logger.error(f"Quantum operation '{self.operation_name}' failed",
                                extra={
                                    "quantum_error": exc_val.to_dict(),
                                    "severity": severity,
                                    "attempt": self.attempt_count
                                })
            
            # Handle recovery if enabled
            if self.enable_recovery:
                should_retry = QuantumErrorRecovery.should_retry(
                    exc_val, self.attempt_count, self.max_retries
                )
                
                if should_retry:
                    if self.logger:
                        retry_delay = QuantumErrorRecovery.get_retry_delay(self.attempt_count)
                        self.logger.info(f"Retrying quantum operation '{self.operation_name}' "
                                       f"in {retry_delay}s (attempt {self.attempt_count + 1})")
                    
                    # For this context manager, we don't actually retry here
                    # The caller should handle the retry logic
                    return False  # Re-raise the exception
                
                # Get fallback strategy
                fallback = QuantumErrorRecovery.get_fallback_strategy(exc_val)
                if fallback and self.logger:
                    self.logger.info(f"Quantum operation '{self.operation_name}' "
                                   f"will use fallback strategy: {fallback}")
            
            return False  # Re-raise the exception
        
        return False  # Don't suppress other exceptions