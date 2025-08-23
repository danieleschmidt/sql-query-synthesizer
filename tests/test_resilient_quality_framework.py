"""
Comprehensive tests for Resilient Quality Framework
Testing error handling, validation, and recovery mechanisms
"""

import asyncio
import pytest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from sql_synthesizer.autonomous_sdlc.resilient_quality_framework import (
    ResilientErrorHandler,
    ComprehensiveValidator,
    ErrorSeverity,
    RecoveryStrategy,
    QualityError,
    ValidationRule,
    CircuitBreaker,
    resilient_quality_context,
    resilient_quality_operation
)


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        cb = CircuitBreaker(failure_threshold=3, timeout=60.0)
        
        assert cb.state == "closed"
        
        # Should allow function calls
        def test_func():
            return "success"
        
        result = cb.call(test_func)
        assert result == "success"
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_failure_counting(self):
        """Test circuit breaker failure counting"""
        cb = CircuitBreaker(failure_threshold=2, timeout=60.0)
        
        def failing_func():
            raise ValueError("Test error")
        
        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.failure_count == 1
        assert cb.state == "closed"
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.failure_count == 2
        assert cb.state == "open"
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state"""
        cb = CircuitBreaker(failure_threshold=1, timeout=1.0)
        
        def failing_func():
            raise ValueError("Test error")
        
        # Trigger circuit opening
        with pytest.raises(ValueError):
            cb.call(failing_func)
        
        assert cb.state == "open"
        
        # Should raise circuit breaker exception
        with pytest.raises(Exception, match="Circuit breaker open"):
            cb.call(lambda: "should not execute")
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open and recovery"""
        cb = CircuitBreaker(failure_threshold=1, timeout=0.1)
        
        def failing_func():
            raise ValueError("Test error")
        
        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.state == "open"
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should transition to half-open
        def success_func():
            return "recovered"
        
        result = cb.call(success_func)
        assert result == "recovered"
        assert cb.state == "closed"
        assert cb.failure_count == 0


class TestQualityError:
    """Test QualityError dataclass functionality"""
    
    def test_quality_error_creation(self):
        """Test QualityError creation and attributes"""
        error = QualityError(
            error_id="test_001",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category="test_category",
            operation="test_operation",
            message="Test error message",
            technical_details={"key": "value"}
        )
        
        assert error.error_id == "test_001"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == "test_category"
        assert error.operation == "test_operation"
        assert error.message == "Test error message"
        assert error.technical_details == {"key": "value"}
        assert error.recovery_attempted is False
        assert error.recovery_strategy is None
        assert error.recovery_success is False


class TestResilientErrorHandler:
    """Test resilient error handler functionality"""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing"""
        return ResilientErrorHandler()
    
    @pytest.mark.asyncio
    async def test_handle_error_basic(self, error_handler):
        """Test basic error handling"""
        test_error = ValueError("Test error")
        
        quality_error = await error_handler.handle_error(
            error=test_error,
            operation="test_operation",
            category="test_category",
            severity=ErrorSeverity.MEDIUM,
            context={"test_key": "test_value"}
        )
        
        assert quality_error.error_id.startswith("err_")
        assert quality_error.severity == ErrorSeverity.MEDIUM
        assert quality_error.category == "test_category"
        assert quality_error.operation == "test_operation"
        assert quality_error.message == "Test error"
        assert quality_error.context == {"test_key": "test_value"}
        
        # Should be added to error list
        assert len(error_handler.errors) == 1
        assert error_handler.errors[0].error_id == quality_error.error_id
    
    @pytest.mark.asyncio
    async def test_error_pattern_tracking(self, error_handler):
        """Test error pattern tracking"""
        # Create multiple similar errors
        for i in range(3):
            await error_handler.handle_error(
                error=ValueError(f"Error {i}"),
                operation="test_operation",
                category="validation",
                recovery_strategy=RecoveryStrategy.SKIP
            )
        
        # Check pattern tracking
        pattern = "validation::ValueError"
        assert pattern in error_handler.error_patterns
        assert error_handler.error_patterns[pattern] == 3
    
    @pytest.mark.asyncio
    async def test_retry_recovery_strategy(self, error_handler):
        """Test retry recovery strategy"""
        test_error = ConnectionError("Connection timeout")
        
        quality_error = await error_handler.handle_error(
            error=test_error,
            operation="database_connection",
            category="connection",
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        assert quality_error.recovery_attempted is True
        # Note: Actual recovery success depends on mock implementation
    
    @pytest.mark.asyncio
    async def test_fallback_recovery_strategy(self, error_handler):
        """Test fallback recovery strategy"""
        test_error = RuntimeError("Primary service unavailable")
        
        quality_error = await error_handler.handle_error(
            error=test_error,
            operation="code_quality",
            category="code_quality",
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.FALLBACK
        )
        
        assert quality_error.recovery_attempted is True
        assert "fallback_strategy" in quality_error.technical_details
    
    @pytest.mark.asyncio
    async def test_graceful_degrade_strategy(self, error_handler):
        """Test graceful degradation strategy"""
        test_error = TimeoutError("Analysis timeout")
        
        quality_error = await error_handler.handle_error(
            error=test_error,
            operation="advanced_analysis",
            category="analysis",
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADE
        )
        
        assert quality_error.recovery_attempted is True
        assert "degraded_to" in quality_error.technical_details
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_strategy(self, error_handler):
        """Test circuit breaker recovery strategy"""
        test_error = Exception("Service failure")
        
        quality_error = await error_handler.handle_error(
            error=test_error,
            operation="external_service",
            category="service",
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.CIRCUIT_BREAK
        )
        
        assert quality_error.recovery_attempted is True
        assert "external_service" in error_handler.circuit_breakers
    
    @pytest.mark.asyncio
    async def test_skip_recovery_strategy(self, error_handler):
        """Test skip recovery strategy"""
        test_error = ValueError("Non-critical validation error")
        
        quality_error = await error_handler.handle_error(
            error=test_error,
            operation="optional_validation",
            category="validation",
            severity=ErrorSeverity.LOW,
            recovery_strategy=RecoveryStrategy.SKIP
        )
        
        assert quality_error.recovery_attempted is True
        assert quality_error.recovery_success is True  # Skip is considered successful
    
    def test_get_error_statistics(self, error_handler):
        """Test error statistics generation"""
        # Add some mock errors
        error1 = QualityError(
            error_id="err_1", timestamp=time.time(), severity=ErrorSeverity.HIGH,
            category="security", operation="test", message="Error 1",
            technical_details={}, recovery_attempted=True, recovery_success=True
        )
        error2 = QualityError(
            error_id="err_2", timestamp=time.time(), severity=ErrorSeverity.MEDIUM,
            category="performance", operation="test", message="Error 2", 
            technical_details={}, recovery_attempted=True, recovery_success=False
        )
        error3 = QualityError(
            error_id="err_3", timestamp=time.time(), severity=ErrorSeverity.HIGH,
            category="security", operation="test", message="Error 3",
            technical_details={}, recovery_attempted=False, recovery_success=False
        )
        
        error_handler.errors = [error1, error2, error3]
        error_handler.error_patterns = {"security::TestError": 2, "performance::TestError": 1}
        
        stats = error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert stats["severity_distribution"]["high"] == 2
        assert stats["severity_distribution"]["medium"] == 1
        assert stats["category_distribution"]["security"] == 2
        assert stats["category_distribution"]["performance"] == 1
        assert stats["recovery_success_rate"] == 1/3  # Only 1 out of 3 recovered successfully
        assert stats["error_patterns"]["security::TestError"] == 2
    
    def test_empty_error_statistics(self, error_handler):
        """Test error statistics with no errors"""
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 0


class TestValidationRule:
    """Test ValidationRule functionality"""
    
    def test_validation_rule_creation(self):
        """Test ValidationRule creation"""
        def dummy_validator(context):
            return True, {}
        
        rule = ValidationRule(
            rule_id="test_rule",
            name="Test Rule",
            category="test",
            severity=ErrorSeverity.MEDIUM,
            validator_func=dummy_validator,
            error_message="Test error message",
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.category == "test"
        assert rule.severity == ErrorSeverity.MEDIUM
        assert rule.validator_func == dummy_validator
        assert rule.error_message == "Test error message"
        assert rule.recovery_strategy == RecoveryStrategy.RETRY
        assert rule.enabled is True


class TestComprehensiveValidator:
    """Test comprehensive validator functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create validator for testing"""
        return ComprehensiveValidator()
    
    @pytest.fixture
    def temp_project_root(self):
        """Create temporary project structure for validation testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create basic project structure
            (project_root / "sql_synthesizer").mkdir()
            (project_root / "tests").mkdir()
            (project_root / "pyproject.toml").write_text("""
[tool.pytest.ini_options]
minversion = "6.0"

[tool.bandit]
exclude_dirs = ["tests"]
""")
            (project_root / "README.md").write_text("# Test Project")
            (project_root / ".pre-commit-config.yaml").write_text("""
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
""")
            
            yield project_root
    
    def test_register_rule(self, validator):
        """Test validation rule registration"""
        def test_validator(context):
            return True, {"status": "passed"}
        
        rule = ValidationRule(
            rule_id="test_rule",
            name="Test Rule",
            category="test",
            severity=ErrorSeverity.LOW,
            validator_func=test_validator,
            error_message="Test error",
            recovery_strategy=RecoveryStrategy.SKIP
        )
        
        validator.register_rule(rule)
        
        assert "test_rule" in validator.validation_rules
        assert validator.validation_rules["test_rule"] == rule
    
    def test_register_default_rules(self, validator):
        """Test default validation rules registration"""
        validator.register_default_rules()
        
        expected_rules = [
            "project_structure",
            "code_quality_tools", 
            "security_config",
            "performance_baseline",
            "test_environment"
        ]
        
        for rule_id in expected_rules:
            assert rule_id in validator.validation_rules
            rule = validator.validation_rules[rule_id]
            assert isinstance(rule.severity, ErrorSeverity)
            assert isinstance(rule.recovery_strategy, RecoveryStrategy)
    
    @pytest.mark.asyncio
    async def test_validate_all_success(self, validator, temp_project_root):
        """Test successful validation of all rules"""
        validator.register_default_rules()
        
        context = {"project_root": str(temp_project_root)}
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock tool availability checks
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = (b"version info", b"")
            mock_subprocess.return_value = mock_proc
            
            results = await validator.validate_all(context)
        
        assert "validation_id" in results
        assert "overall_status" in results
        assert "rule_results" in results
        assert len(results["rule_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_project_structure(self, validator, temp_project_root):
        """Test project structure validation"""
        context = {"project_root": str(temp_project_root)}
        
        is_valid, details = await validator._validate_project_structure(context)
        
        assert is_valid is True
        assert "required_dirs" in details
        assert "required_files" in details
        assert "missing_dirs" in details
        assert "missing_files" in details
        assert len(details["missing_dirs"]) == 0
        assert len(details["missing_files"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_project_structure_missing_items(self, validator):
        """Test project structure validation with missing items"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            # Only create partial structure
            (project_root / "sql_synthesizer").mkdir()
            # Missing tests directory and files
            
            context = {"project_root": str(project_root)}
            is_valid, details = await validator._validate_project_structure(context)
            
            assert is_valid is False
            assert len(details["missing_dirs"]) > 0 or len(details["missing_files"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_quality_tools(self, validator):
        """Test quality tools availability validation"""
        context = {"project_root": "/tmp"}
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock some tools available, some not
            def mock_subprocess_side_effect(*args, **kwargs):
                tool_name = args[0]
                mock_proc = AsyncMock()
                
                if tool_name in ["ruff", "black"]:
                    mock_proc.returncode = 0  # Available
                else:
                    mock_proc.returncode = 1  # Not available
                    
                mock_proc.communicate.return_value = (b"", b"")
                return mock_proc
            
            mock_subprocess.side_effect = mock_subprocess_side_effect
            
            is_valid, details = await validator._validate_quality_tools(context)
            
            assert "available_tools" in details
            assert "missing_tools" in details
            assert "availability_rate" in details
    
    @pytest.mark.asyncio
    async def test_validate_security_config(self, validator, temp_project_root):
        """Test security configuration validation"""
        context = {"project_root": str(temp_project_root)}
        
        is_valid, details = await validator._validate_security_config(context)
        
        assert is_valid is True  # Should pass due to bandit in pre-commit config
        assert "configured_files" in details
        assert len(details["configured_files"]) > 0
    
    @pytest.mark.asyncio 
    async def test_validate_performance_baseline(self, validator, temp_project_root):
        """Test performance baseline validation"""
        # Create a mock baseline file
        (temp_project_root / "performance_benchmark_report.json").write_text('{"baseline": "data"}')
        
        context = {"project_root": str(temp_project_root)}
        
        is_valid, details = await validator._validate_performance_baseline(context)
        
        assert is_valid is True
        assert "available_baselines" in details
        assert len(details["available_baselines"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_test_environment(self, validator, temp_project_root):
        """Test test environment validation"""
        context = {"project_root": str(temp_project_root)}
        
        is_valid, details = await validator._validate_test_environment(context)
        
        assert "test_indicators" in details
        assert "valid_indicators" in details
        
        # Should have at least test directory and pyproject.toml with pytest config
        assert details["valid_indicators"] >= 2
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_execute_validation_rule_success(self, validator):
        """Test successful validation rule execution"""
        async def success_validator(context):
            return True, {"status": "passed", "details": "Everything OK"}
        
        rule = ValidationRule(
            rule_id="success_rule",
            name="Success Rule",
            category="test",
            severity=ErrorSeverity.LOW,
            validator_func=success_validator,
            error_message="Should not see this",
            recovery_strategy=RecoveryStrategy.SKIP
        )
        
        context = {"test": "context"}
        result = await validator._execute_validation_rule(rule, context)
        
        assert result["rule_id"] == "success_rule"
        assert result["passed"] is True
        assert result["error"] is None
        assert result["details"]["status"] == "passed"
    
    @pytest.mark.asyncio
    async def test_execute_validation_rule_failure(self, validator):
        """Test validation rule execution with failure"""
        async def failing_validator(context):
            return False, {"status": "failed", "reason": "Test failure"}
        
        rule = ValidationRule(
            rule_id="failing_rule", 
            name="Failing Rule",
            category="test",
            severity=ErrorSeverity.HIGH,
            validator_func=failing_validator,
            error_message="Expected failure",
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        context = {"test": "context"}
        result = await validator._execute_validation_rule(rule, context)
        
        assert result["rule_id"] == "failing_rule"
        assert result["passed"] is False
        assert result["details"]["status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_execute_validation_rule_exception(self, validator):
        """Test validation rule execution with exception"""
        async def exception_validator(context):
            raise RuntimeError("Validator crashed")
        
        rule = ValidationRule(
            rule_id="exception_rule",
            name="Exception Rule", 
            category="test",
            severity=ErrorSeverity.MEDIUM,
            validator_func=exception_validator,
            error_message="Rule crashed",
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADE
        )
        
        context = {"test": "context"}
        result = await validator._execute_validation_rule(rule, context)
        
        assert result["rule_id"] == "exception_rule"
        assert result["passed"] is False
        assert result["error"] is not None
        assert result["recovery_applied"] is True
    
    @pytest.mark.asyncio
    async def test_validation_overall_status_determination(self, validator):
        """Test overall validation status determination"""
        # Create rules with different severities
        async def critical_fail(context):
            return False, {"status": "critical_failure"}
        
        async def high_fail(context):
            return False, {"status": "high_failure"}
        
        async def medium_fail(context):
            return False, {"status": "medium_failure"}
        
        async def success(context):
            return True, {"status": "success"}
        
        rules = [
            ValidationRule("critical", "Critical", "test", ErrorSeverity.CRITICAL, 
                          critical_fail, "Critical error", RecoveryStrategy.FAIL_FAST),
            ValidationRule("high", "High", "test", ErrorSeverity.HIGH,
                          high_fail, "High error", RecoveryStrategy.RETRY),
            ValidationRule("medium", "Medium", "test", ErrorSeverity.MEDIUM,
                          medium_fail, "Medium error", RecoveryStrategy.FALLBACK),
            ValidationRule("success", "Success", "test", ErrorSeverity.LOW,
                          success, "Should pass", RecoveryStrategy.SKIP)
        ]
        
        for rule in rules:
            validator.register_rule(rule)
        
        context = {"test": "context"}
        results = await validator.validate_all(context)
        
        # With critical failure, overall status should be critical
        assert results["overall_status"] == "critical"
        assert results["critical_issues"] >= 1
        
        # Test without critical failure
        validator.validation_rules.pop("critical")
        results = await validator.validate_all(context)
        assert results["overall_status"] == "failed"  # Due to high severity
        
        # Test with only medium and low
        validator.validation_rules.pop("high")
        results = await validator.validate_all(context)
        assert results["overall_status"] == "warning"  # Due to medium severity


class TestResilientQualityContext:
    """Test resilient quality context manager"""
    
    @pytest.fixture
    def temp_project_root(self):
        """Create temporary project for context testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "sql_synthesizer").mkdir()
            (project_root / "tests").mkdir()
            (project_root / "pyproject.toml").touch()
            (project_root / "README.md").touch()
            yield project_root
    
    @pytest.mark.asyncio
    async def test_resilient_quality_context_success(self, temp_project_root):
        """Test successful resilient quality context usage"""
        async with resilient_quality_context(temp_project_root) as context:
            assert "project_root" in context
            assert "error_handler" in context
            assert "validator" in context
            assert "start_time" in context
            assert "initial_validation" in context
            
            # Context should contain working components
            error_handler = context["error_handler"]
            validator = context["validator"]
            
            assert isinstance(error_handler, ResilientErrorHandler)
            assert isinstance(validator, ComprehensiveValidator)
            
            # Should have validation rules registered
            assert len(validator.validation_rules) > 0
        
        # Context should have end time after completion
        assert "end_time" in context
        assert "total_execution_time" in context
    
    @pytest.mark.asyncio
    async def test_resilient_quality_context_with_error(self, temp_project_root):
        """Test resilient quality context with error handling"""
        try:
            async with resilient_quality_context(temp_project_root) as context:
                # Simulate an error within the context
                raise ValueError("Test error in context")
        except ValueError:
            pass  # Expected
        
        # Context should still have handled the error properly
        assert "end_time" in context
        assert context["end_time"] > context["start_time"]


class TestResilientQualityDecorator:
    """Test resilient quality operation decorator"""
    
    @pytest.mark.asyncio
    async def test_resilient_operation_success(self):
        """Test successful operation with resilient decorator"""
        @resilient_quality_operation(
            operation_name="test_operation",
            category="test",
            severity=ErrorSeverity.LOW,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        async def successful_operation():
            return {"status": "success", "data": "test_data"}
        
        result = await successful_operation()
        
        assert result["status"] == "success"
        assert result["data"] == "test_data"
    
    @pytest.mark.asyncio
    async def test_resilient_operation_with_recovery(self):
        """Test operation with error and recovery"""
        @resilient_quality_operation(
            operation_name="test_operation_recovery",
            category="test",
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.FALLBACK
        )
        async def operation_with_error():
            raise ConnectionError("Network unavailable")
        
        # Should handle error and return recovery result if successful
        with patch.object(ResilientErrorHandler, 'handle_error') as mock_handle:
            # Mock successful recovery
            mock_error = Mock()
            mock_error.recovery_success = True
            mock_error.error_id = "test_error_001"
            mock_handle.return_value = mock_error
            
            result = await operation_with_error()
            
            assert result["status"] == "recovered"
            assert result["error_id"] == "test_error_001"
    
    @pytest.mark.asyncio
    async def test_resilient_operation_recovery_failed(self):
        """Test operation where recovery fails"""
        @resilient_quality_operation(
            operation_name="test_operation_fail",
            category="test",
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        async def operation_with_unrecoverable_error():
            raise RuntimeError("Critical system error")
        
        with patch.object(ResilientErrorHandler, 'handle_error') as mock_handle:
            # Mock failed recovery
            mock_error = Mock()
            mock_error.recovery_success = False
            mock_error.error_id = "test_error_002"
            mock_handle.return_value = mock_error
            
            # Should re-raise the original exception
            with pytest.raises(RuntimeError, match="Critical system error"):
                await operation_with_unrecoverable_error()


class TestIntegration:
    """Integration tests for resilient quality framework components"""
    
    @pytest.fixture
    def temp_project_root(self):
        """Create comprehensive project structure for integration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create full project structure
            (project_root / "sql_synthesizer").mkdir()
            (project_root / "tests").mkdir()
            (project_root / "docs").mkdir()
            
            # Configuration files
            (project_root / "pyproject.toml").write_text("""
[build-system]
requires = ["setuptools", "wheel"]

[tool.pytest.ini_options]
minversion = "6.0"
testpaths = ["tests"]

[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101"]

[tool.coverage.run]
source = ["sql_synthesizer"]
""")
            
            (project_root / "README.md").write_text("# Integration Test Project")
            (project_root / "requirements.txt").write_text("pytest\nbandit\nsafety")
            (project_root / ".pre-commit-config.yaml").write_text("""
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
  - repo: https://github.com/pyupio/safety
    rev: 2.3.0
    hooks:
      - id: safety
""")
            
            # Performance baseline
            (project_root / "performance_benchmark_report.json").write_text("""
{
  "baseline_metrics": {
    "avg_response_time": 150,
    "throughput": 100,
    "memory_usage": 256
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
""")
            
            yield project_root
    
    @pytest.mark.asyncio
    async def test_complete_validation_workflow(self, temp_project_root):
        """Test complete validation workflow with error handling"""
        async with resilient_quality_context(temp_project_root) as context:
            validator = context["validator"]
            error_handler = context["error_handler"]
            
            # Mock tool availability for realistic testing
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                def subprocess_side_effect(*args, **kwargs):
                    tool = args[0]
                    mock_proc = AsyncMock()
                    
                    # Mock different tool availability
                    if tool in ["ruff", "black", "pytest"]:
                        mock_proc.returncode = 0
                    elif tool == "mypy":
                        mock_proc.returncode = 1  # Not available
                    else:
                        mock_proc.returncode = 0
                    
                    mock_proc.communicate.return_value = (b"tool output", b"")
                    return mock_proc
                
                mock_subprocess.side_effect = subprocess_side_effect
                
                # Run comprehensive validation
                validation_results = await validator.validate_all(context)
                
                # Should have completed successfully
                assert validation_results["overall_status"] in ["passed", "passed_with_warnings", "warning"]
                assert len(validation_results["rule_results"]) > 0
                
                # Check specific validations
                rule_results = validation_results["rule_results"]
                
                # Project structure should pass
                assert rule_results["project_structure"]["passed"] is True
                
                # Security config should pass (has bandit in pre-commit)
                assert rule_results["security_config"]["passed"] is True
                
                # Performance baseline should pass
                assert rule_results["performance_baseline"]["passed"] is True
                
                # Test environment should pass (has pytest config)
                assert rule_results["test_environment"]["passed"] is True
                
                # Quality tools might have mixed results
                tools_result = rule_results["code_quality_tools"]
                assert "availability_rate" in tools_result["details"]
    
    @pytest.mark.asyncio
    async def test_error_handling_during_validation(self, temp_project_root):
        """Test error handling during validation process"""
        validator = ComprehensiveValidator()
        error_handler = ResilientErrorHandler()
        
        # Add a rule that will always fail
        async def failing_validator(context):
            raise RuntimeError("Simulated validation failure")
        
        failing_rule = ValidationRule(
            rule_id="failing_test_rule",
            name="Failing Test Rule",
            category="test",
            severity=ErrorSeverity.MEDIUM,
            validator_func=failing_validator,
            error_message="This rule always fails",
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADE
        )
        
        validator.register_default_rules()
        validator.register_rule(failing_rule)
        
        context = {"project_root": str(temp_project_root)}
        
        # Should handle the failing rule gracefully
        results = await validator.validate_all(context)
        
        # Should still have results despite the failure
        assert "validation_id" in results
        assert "overall_status" in results
        assert "rule_results" in results
        
        # The failing rule should be marked as failed but handled
        failing_result = results["rule_results"]["failing_test_rule"]
        assert failing_result["passed"] is False
        assert failing_result["error"] is not None
        assert failing_result["recovery_applied"] is True
        
        # Overall validation should continue despite individual rule failure
        assert results["overall_status"] != "critical"  # Shouldn't be critical for medium severity
    
    @pytest.mark.asyncio
    async def test_concurrent_validation_execution(self, temp_project_root):
        """Test concurrent execution of multiple validations"""
        validator = ComprehensiveValidator()
        validator.register_default_rules()
        
        context = {"project_root": str(temp_project_root)}
        
        # Mock subprocess execution to be predictable
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_proc
            
            # Run multiple validations concurrently
            validation_tasks = []
            for i in range(3):
                task = validator.validate_all(context)
                validation_tasks.append(task)
            
            results_list = await asyncio.gather(*validation_tasks)
            
            # All validations should complete successfully
            assert len(results_list) == 3
            
            for results in results_list:
                assert "validation_id" in results
                assert "overall_status" in results
                assert len(results["rule_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with error handler"""
        error_handler = ResilientErrorHandler()
        
        # Simulate repeated failures to trigger circuit breaker
        for i in range(6):  # Exceed default threshold of 5
            await error_handler.handle_error(
                error=ConnectionError(f"Connection failed {i}"),
                operation="database_connection",
                category="connection",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.CIRCUIT_BREAK
            )
        
        # Circuit breaker should be open
        assert "database_connection" in error_handler.circuit_breakers
        cb = error_handler.circuit_breakers["database_connection"]
        assert cb.state == "open"
        
        # Statistics should reflect circuit breaker state
        stats = error_handler.get_error_statistics()
        assert stats["circuit_breaker_states"]["database_connection"] == "open"
    
    @pytest.mark.asyncio
    async def test_recovery_success_tracking(self):
        """Test tracking of recovery success rates"""
        error_handler = ResilientErrorHandler()
        
        # Simulate errors with different recovery outcomes
        test_cases = [
            (ConnectionError("Timeout 1"), RecoveryStrategy.RETRY, "connection"),
            (ValueError("Invalid input"), RecoveryStrategy.SKIP, "validation"),
            (RuntimeError("Service down"), RecoveryStrategy.FALLBACK, "service"),
            (TimeoutError("Too slow"), RecoveryStrategy.GRACEFUL_DEGRADE, "performance")
        ]
        
        for error, strategy, category in test_cases:
            await error_handler.handle_error(
                error=error,
                operation=f"test_operation_{category}",
                category=category,
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=strategy
            )
        
        # Get statistics
        stats = error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 4
        assert 0 <= stats["recovery_success_rate"] <= 1.0
        
        # Should have variety in categories
        assert len(stats["category_distribution"]) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])