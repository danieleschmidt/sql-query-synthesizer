"""
Input validation and sanitization for quantum components
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .exceptions import QuantumValidationError


@dataclass
class ValidationRule:
    """Represents a validation rule"""

    name: str
    message: str
    validator: callable


class QuantumValidator:
    """Comprehensive input validation for quantum operations"""

    # Security patterns to detect malicious input
    MALICIOUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # XSS attempts
        r"javascript:",  # JavaScript protocol
        r"on\w+\s*=",  # Event handlers
        r"eval\s*\(",  # Code evaluation
        r"exec\s*\(",  # Code execution
        r"__import__",  # Python imports
        r"subprocess",  # System calls
        r"os\.",  # OS operations
        r"file://",  # File protocol
        r"ftp://",  # FTP protocol
    ]

    def __init__(self):
        self.validation_rules = self._initialize_rules()

    def _initialize_rules(self) -> Dict[str, List[ValidationRule]]:
        """Initialize validation rules for different quantum components"""
        return {
            "qubit_count": [
                ValidationRule(
                    "positive_integer",
                    "Qubit count must be a positive integer",
                    lambda x: isinstance(x, int) and x > 0,
                ),
                ValidationRule(
                    "reasonable_limit",
                    "Qubit count must be between 1 and 1000 for practical reasons",
                    lambda x: 1 <= x <= 1000,
                ),
            ],
            "temperature": [
                ValidationRule(
                    "positive_number",
                    "Temperature must be a positive number",
                    lambda x: isinstance(x, (int, float)) and x > 0,
                ),
                ValidationRule(
                    "reasonable_range",
                    "Temperature must be between 0.1 and 10000",
                    lambda x: 0.1 <= x <= 10000,
                ),
            ],
            "probability": [
                ValidationRule(
                    "valid_probability",
                    "Probability must be between 0 and 1",
                    lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
                )
            ],
            "cost": [
                ValidationRule(
                    "non_negative_number",
                    "Cost must be a non-negative number",
                    lambda x: isinstance(x, (int, float)) and x >= 0,
                ),
                ValidationRule(
                    "finite_value",
                    "Cost must be a finite value",
                    lambda x: math.isfinite(x),
                ),
            ],
            "execution_time": [
                ValidationRule(
                    "positive_number",
                    "Execution time must be positive",
                    lambda x: isinstance(x, (int, float)) and x > 0,
                ),
                ValidationRule(
                    "reasonable_limit",
                    "Execution time must be less than 24 hours (86400 seconds)",
                    lambda x: x <= 86400,
                ),
            ],
            "task_id": [
                ValidationRule(
                    "non_empty_string",
                    "Task ID must be a non-empty string",
                    lambda x: isinstance(x, str) and len(x.strip()) > 0,
                ),
                ValidationRule(
                    "max_length",
                    "Task ID must be 256 characters or less",
                    lambda x: len(x) <= 256,
                ),
                ValidationRule(
                    "safe_characters",
                    "Task ID contains unsafe characters",
                    lambda x: self._is_safe_string(x),
                ),
            ],
            "table_name": [
                ValidationRule(
                    "valid_identifier",
                    "Table name must be a valid SQL identifier",
                    lambda x: self._is_valid_sql_identifier(x),
                ),
                ValidationRule(
                    "max_length",
                    "Table name must be 128 characters or less",
                    lambda x: len(x) <= 128,
                ),
            ],
            "column_name": [
                ValidationRule(
                    "valid_identifier",
                    "Column name must be a valid SQL identifier",
                    lambda x: self._is_valid_sql_identifier(x),
                ),
                ValidationRule(
                    "max_length",
                    "Column name must be 128 characters or less",
                    lambda x: len(x) <= 128,
                ),
            ],
            "query_question": [
                ValidationRule(
                    "non_empty_string",
                    "Query question must be a non-empty string",
                    lambda x: isinstance(x, str) and len(x.strip()) > 0,
                ),
                ValidationRule(
                    "max_length",
                    "Query question must be 10000 characters or less",
                    lambda x: len(x) <= 10000,
                ),
                ValidationRule(
                    "no_malicious_content",
                    "Query question contains potentially malicious content",
                    lambda x: not self._contains_malicious_patterns(x),
                ),
            ],
            "resource_capacity": [
                ValidationRule(
                    "positive_number",
                    "Resource capacity must be positive",
                    lambda x: isinstance(x, (int, float)) and x > 0,
                ),
                ValidationRule(
                    "reasonable_limit",
                    "Resource capacity must be between 0.1 and 1000",
                    lambda x: 0.1 <= x <= 1000,
                ),
            ],
        }

    def validate_field(self, field_name: str, value: Any) -> None:
        """
        Validate a single field value

        Args:
            field_name: Name of the field to validate
            value: Value to validate

        Raises:
            QuantumValidationError: If validation fails
        """
        if field_name not in self.validation_rules:
            return  # No validation rules defined for this field

        rules = self.validation_rules[field_name]

        for rule in rules:
            try:
                if not rule.validator(value):
                    raise QuantumValidationError(
                        rule.message,
                        field_name=field_name,
                        field_value=value,
                        validation_rule=rule.name,
                        details={"expected_format": self._get_field_format(field_name)},
                    )
            except Exception as e:
                if isinstance(e, QuantumValidationError):
                    raise
                # Convert other exceptions to validation errors
                raise QuantumValidationError(
                    f"Validation error for {field_name}: {str(e)}",
                    field_name=field_name,
                    field_value=value,
                    validation_rule=rule.name,
                )

    def validate_multiple(self, fields: Dict[str, Any]) -> None:
        """
        Validate multiple fields at once

        Args:
            fields: Dictionary of field_name -> value pairs

        Raises:
            QuantumValidationError: If any validation fails
        """
        errors = []

        for field_name, value in fields.items():
            try:
                self.validate_field(field_name, value)
            except QuantumValidationError as e:
                errors.append(e)

        if errors:
            # Combine multiple errors into one
            error_messages = [e.message for e in errors]
            combined_message = (
                f"Multiple validation errors: {'; '.join(error_messages)}"
            )

            raise QuantumValidationError(
                combined_message,
                details={
                    "field_errors": [e.to_dict() for e in errors],
                    "error_count": len(errors),
                },
            )

    def sanitize_string(self, value: str, max_length: int = None) -> str:
        """
        Sanitize string input by removing potentially dangerous content

        Args:
            value: String to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            raise QuantumValidationError(
                "Value must be a string",
                field_value=value,
                validation_rule="type_check",
            )

        # Remove malicious patterns
        sanitized = value
        for pattern in self.MALICIOUS_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Strip whitespace
        sanitized = sanitized.strip()

        # Limit length if specified
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    def validate_quantum_plan(self, plan: Dict[str, Any]) -> None:
        """
        Validate a quantum query plan structure

        Args:
            plan: Dictionary representing a query plan

        Raises:
            QuantumValidationError: If plan structure is invalid
        """
        required_fields = ["joins", "filters", "aggregations", "cost", "probability"]

        for field in required_fields:
            if field not in plan:
                raise QuantumValidationError(
                    f"Missing required field: {field}",
                    field_name=field,
                    validation_rule="required_field",
                    details={"required_fields": required_fields},
                )

        # Validate individual fields
        self.validate_field("cost", plan["cost"])
        self.validate_field("probability", plan["probability"])

        # Validate joins structure
        if not isinstance(plan["joins"], list):
            raise QuantumValidationError(
                "Joins must be a list",
                field_name="joins",
                field_value=type(plan["joins"]).__name__,
                validation_rule="type_check",
            )

        for i, join in enumerate(plan["joins"]):
            if not isinstance(join, (list, tuple)) or len(join) != 2:
                raise QuantumValidationError(
                    f"Join {i} must be a tuple/list of 2 table names",
                    field_name=f"joins[{i}]",
                    field_value=join,
                    validation_rule="structure_check",
                )

            # Validate table names in join
            for j, table in enumerate(join):
                try:
                    self.validate_field("table_name", table)
                except QuantumValidationError as e:
                    e.field_name = f"joins[{i}][{j}]"
                    raise

        # Validate filters structure
        if not isinstance(plan["filters"], list):
            raise QuantumValidationError(
                "Filters must be a list",
                field_name="filters",
                field_value=type(plan["filters"]).__name__,
                validation_rule="type_check",
            )

        for i, filter_def in enumerate(plan["filters"]):
            if not isinstance(filter_def, dict):
                raise QuantumValidationError(
                    f"Filter {i} must be a dictionary",
                    field_name=f"filters[{i}]",
                    field_value=type(filter_def).__name__,
                    validation_rule="type_check",
                )

    def validate_quantum_task(self, task: Dict[str, Any]) -> None:
        """
        Validate a quantum task structure

        Args:
            task: Dictionary representing a quantum task

        Raises:
            QuantumValidationError: If task structure is invalid
        """
        required_fields = ["id", "execution_time"]

        for field in required_fields:
            if field not in task:
                raise QuantumValidationError(
                    f"Missing required field: {field}",
                    field_name=field,
                    validation_rule="required_field",
                    details={"required_fields": required_fields},
                )

        # Validate individual fields
        self.validate_field("task_id", task["id"])
        self.validate_field("execution_time", task["execution_time"])

        # Validate dependencies if present
        if "dependencies" in task:
            if not isinstance(task["dependencies"], list):
                raise QuantumValidationError(
                    "Dependencies must be a list",
                    field_name="dependencies",
                    field_value=type(task["dependencies"]).__name__,
                    validation_rule="type_check",
                )

            for i, dep in enumerate(task["dependencies"]):
                try:
                    self.validate_field("task_id", dep)
                except QuantumValidationError as e:
                    e.field_name = f"dependencies[{i}]"
                    raise

    def _is_safe_string(self, value: str) -> bool:
        """Check if string contains only safe characters"""
        # Allow alphanumeric, underscore, hyphen, and dot
        safe_pattern = r"^[a-zA-Z0-9_\-\.]+$"
        return re.match(safe_pattern, value) is not None

    def _is_valid_sql_identifier(self, value: str) -> bool:
        """Check if string is a valid SQL identifier"""
        if not isinstance(value, str) or not value:
            return False

        # SQL identifier pattern: letter followed by letters, digits, or underscores
        identifier_pattern = r"^[a-zA-Z][a-zA-Z0-9_]*$"
        return re.match(identifier_pattern, value) is not None

    def _contains_malicious_patterns(self, value: str) -> bool:
        """Check if string contains malicious patterns"""
        for pattern in self.MALICIOUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False

    def _get_field_format(self, field_name: str) -> str:
        """Get expected format description for a field"""
        format_descriptions = {
            "qubit_count": "positive integer between 1 and 1000",
            "temperature": "positive number between 0.1 and 10000",
            "probability": "number between 0.0 and 1.0",
            "cost": "non-negative finite number",
            "execution_time": "positive number <= 86400",
            "task_id": "non-empty string with safe characters, max 256 chars",
            "table_name": "valid SQL identifier, max 128 chars",
            "column_name": "valid SQL identifier, max 128 chars",
            "query_question": "non-empty string without malicious content, max 10000 chars",
            "resource_capacity": "positive number between 0.1 and 1000",
        }

        return format_descriptions.get(field_name, "valid value")


# Pre-configured validator instance
quantum_validator = QuantumValidator()


# Validation decorators
def validate_quantum_input(**field_mappings):
    """
    Decorator to validate quantum function inputs

    Usage:
        @validate_quantum_input(num_qubits="qubit_count", temp="temperature")
        def create_optimizer(num_qubits, temp):
            ...
    """

    def decorator(func):
        """TODO: Add docstring"""
            """TODO: Add docstring"""
        def wrapper(*args, **kwargs):
            # Get function argument names
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate specified fields
            validation_data = {}
            for arg_name, field_type in field_mappings.items():
                if arg_name in bound_args.arguments:
                    validation_data[field_type] = bound_args.arguments[arg_name]

            if validation_data:
                quantum_validator.validate_multiple(validation_data)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def sanitize_quantum_input(*field_names):
    """
    Decorator to sanitize string inputs

    Usage:
        @sanitize_quantum_input("question", "task_id")
        def process_query(question, task_id):
            ...
    """

     """TODO: Add docstring"""
     """TODO: Add docstring"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function argument names
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Sanitize specified string fields
            for field_name in field_names:
                if field_name in bound_args.arguments:
                    value = bound_args.arguments[field_name]
                    if isinstance(value, str):
                        sanitized = quantum_validator.sanitize_string(value)
                        bound_args.arguments[field_name] = sanitized

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator
