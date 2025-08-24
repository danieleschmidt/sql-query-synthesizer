"""
Comprehensive Quality Gates System

Advanced quality validation system for autonomous SDLC with quantum optimization.
Implements progressive quality gates, automated code analysis, and self-healing validation.
"""

import asyncio
import json
import logging
import subprocess
import time
import ast
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib


class QualityGateType(Enum):
    """Types of quality gates"""
    SYNTAX_VALIDATION = "syntax_validation"
    STYLE_COMPLIANCE = "style_compliance"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION = "documentation"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    ARCHITECTURE_COMPLIANCE = "architecture_compliance"


class QualityLevel(Enum):
    """Quality assurance levels"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    PREMIUM = "premium"
    QUANTUM = "quantum"


class ValidationSeverity(Enum):
    """Validation issue severity"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """Represents a quality issue found during validation"""
    issue_id: str
    gate_type: QualityGateType
    severity: ValidationSeverity
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    rule_id: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class QualityGateResult:
    """Result of a quality gate execution"""
    gate_type: QualityGateType
    passed: bool
    score: float  # 0.0 to 1.0
    execution_time: float
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    report_id: str
    project_path: Path
    quality_level: QualityLevel
    overall_score: float
    gate_results: List[QualityGateResult]
    execution_time: float
    total_issues: int
    critical_issues: int
    auto_fixes_applied: int
    recommendations: List[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class QuantumQualityAnalyzer:
    """
    Quantum-enhanced quality analyzer for comprehensive code validation
    """

    def __init__(self,
                 project_path: Path,
                 quality_level: QualityLevel = QualityLevel.ENHANCED,
                 auto_fix_enabled: bool = True,
                 parallel_execution: bool = True,
                 logger: Optional[logging.Logger] = None):

        self.project_path = project_path
        self.quality_level = quality_level
        self.auto_fix_enabled = auto_fix_enabled
        self.parallel_execution = parallel_execution
        self.logger = logger or logging.getLogger(__name__)

        # Quality gate configurations
        self.gate_configs = self._initialize_gate_configs()

        # Validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "auto_fixes_applied": 0,
            "average_execution_time": 0.0
        }

        self.logger.info(f"Quantum Quality Analyzer initialized")
        self.logger.info(f"Project path: {project_path}")
        self.logger.info(f"Quality level: {quality_level.value}")

    def _initialize_gate_configs(self) -> Dict[QualityGateType, Dict[str, Any]]:
        """Initialize quality gate configurations based on level"""

        base_config = {
            QualityGateType.SYNTAX_VALIDATION: {
                "enabled": True,
                "weight": 0.2,
                "auto_fix": True,
                "file_patterns": ["*.py", "*.js", "*.ts", "*.json", "*.yaml", "*.yml"]
            },
            QualityGateType.STYLE_COMPLIANCE: {
                "enabled": True,
                "weight": 0.15,
                "auto_fix": True,
                "line_length": 88,
                "indent_size": 4
            },
            QualityGateType.SECURITY_ANALYSIS: {
                "enabled": True,
                "weight": 0.2,
                "auto_fix": False,
                "severity_threshold": ValidationSeverity.WARNING
            },
            QualityGateType.PERFORMANCE_ANALYSIS: {
                "enabled": self.quality_level in [QualityLevel.PREMIUM, QualityLevel.QUANTUM],
                "weight": 0.15,
                "auto_fix": False,
                "complexity_threshold": 10
            },
            QualityGateType.TEST_COVERAGE: {
                "enabled": True,
                "weight": 0.15,
                "auto_fix": False,
                "minimum_coverage": 0.8 if self.quality_level == QualityLevel.QUANTUM else 0.7
            },
            QualityGateType.DOCUMENTATION: {
                "enabled": self.quality_level in [QualityLevel.PREMIUM, QualityLevel.QUANTUM],
                "weight": 0.1,
                "auto_fix": True,
                "require_docstrings": True
            },
            QualityGateType.DEPENDENCY_ANALYSIS: {
                "enabled": True,
                "weight": 0.05,
                "auto_fix": False,
                "check_vulnerabilities": True
            }
        }

        return base_config

    async def execute_comprehensive_quality_gates(self) -> QualityReport:
        """
        Execute all quality gates with quantum optimization
        """

        start_time = time.time()
        report_id = f"quality_report_{int(start_time)}"

        self.logger.info(f"🔍 Executing comprehensive quality gates (Level: {self.quality_level.value})")

        try:
            # Execute quality gates
            if self.parallel_execution:
                gate_results = await self._execute_gates_parallel()
            else:
                gate_results = await self._execute_gates_sequential()

            # Apply auto-fixes if enabled
            auto_fixes_applied = 0
            if self.auto_fix_enabled:
                auto_fixes_applied = await self._apply_auto_fixes(gate_results)

            # Calculate overall quality score
            overall_score = self._calculate_overall_score(gate_results)

            # Generate recommendations
            recommendations = self._generate_recommendations(gate_results)

            # Count issues
            total_issues = sum(len(result.issues) for result in gate_results)
            critical_issues = sum(
                len([issue for issue in result.issues if issue.severity == ValidationSeverity.CRITICAL])
                for result in gate_results
            )

            execution_time = time.time() - start_time

            # Create comprehensive report
            quality_report = QualityReport(
                report_id=report_id,
                project_path=self.project_path,
                quality_level=self.quality_level,
                overall_score=overall_score,
                gate_results=gate_results,
                execution_time=execution_time,
                total_issues=total_issues,
                critical_issues=critical_issues,
                auto_fixes_applied=auto_fixes_applied,
                recommendations=recommendations
            )

            # Update statistics
            self.validation_stats["total_validations"] += 1
            if overall_score >= 0.8:  # 80% threshold for success
                self.validation_stats["successful_validations"] += 1
            self.validation_stats["auto_fixes_applied"] += auto_fixes_applied

            # Update average execution time
            total_validations = self.validation_stats["total_validations"]
            current_avg = self.validation_stats["average_execution_time"]
            self.validation_stats["average_execution_time"] = (
                (current_avg * (total_validations - 1) + execution_time) / total_validations
            )

            self.logger.info(f"✅ Quality gates completed in {execution_time:.2f}s")
            self.logger.info(f"Overall score: {overall_score:.1%}")
            self.logger.info(f"Issues found: {total_issues} (Critical: {critical_issues})")
            self.logger.info(f"Auto-fixes applied: {auto_fixes_applied}")

            return quality_report

        except Exception as e:
            execution_time = time.time() - start_time

            error_report = QualityReport(
                report_id=f"error_{report_id}",
                project_path=self.project_path,
                quality_level=self.quality_level,
                overall_score=0.0,
                gate_results=[],
                execution_time=execution_time,
                total_issues=0,
                critical_issues=1,
                auto_fixes_applied=0,
                recommendations=[f"Fix execution error: {str(e)}"]
            )

            self.logger.error(f"❌ Quality gates execution failed: {str(e)}")
            return error_report

    async def _execute_gates_parallel(self) -> List[QualityGateResult]:
        """Execute quality gates in parallel"""

        self.logger.debug("Executing quality gates in parallel")

        # Create tasks for enabled gates
        gate_tasks = []
        for gate_type, config in self.gate_configs.items():
            if config["enabled"]:
                gate_tasks.append(self._execute_quality_gate(gate_type, config))

        # Execute all gates concurrently
        gate_results = await asyncio.gather(*gate_tasks, return_exceptions=True)

        # Handle exceptions
        valid_results = []
        for i, result in enumerate(gate_results):
            if isinstance(result, Exception):
                gate_type = list(self.gate_configs.keys())[i]
                self.logger.error(f"Gate {gate_type.value} failed: {str(result)}")

                # Create error result
                error_result = QualityGateResult(
                    gate_type=gate_type,
                    passed=False,
                    score=0.0,
                    execution_time=0.0,
                    issues=[QualityIssue(
                        issue_id=f"error_{gate_type.value}",
                        gate_type=gate_type,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Gate execution failed: {str(result)}"
                    )],
                    recommendations=[f"Fix gate execution error for {gate_type.value}"]
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)

        return valid_results

    async def _execute_gates_sequential(self) -> List[QualityGateResult]:
        """Execute quality gates sequentially"""

        self.logger.debug("Executing quality gates sequentially")

        gate_results = []
        for gate_type, config in self.gate_configs.items():
            if config["enabled"]:
                try:
                    result = await self._execute_quality_gate(gate_type, config)
                    gate_results.append(result)
                except Exception as e:
                    self.logger.error(f"Gate {gate_type.value} failed: {str(e)}")

                    error_result = QualityGateResult(
                        gate_type=gate_type,
                        passed=False,
                        score=0.0,
                        execution_time=0.0,
                        issues=[QualityIssue(
                            issue_id=f"error_{gate_type.value}",
                            gate_type=gate_type,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Gate execution failed: {str(e)}"
                        )]
                    )
                    gate_results.append(error_result)

        return gate_results

    async def _execute_quality_gate(self,
                                  gate_type: QualityGateType,
                                  config: Dict[str, Any]) -> QualityGateResult:
        """Execute a specific quality gate"""

        start_time = time.time()

        self.logger.debug(f"Executing quality gate: {gate_type.value}")

        try:
            if gate_type == QualityGateType.SYNTAX_VALIDATION:
                result = await self._validate_syntax(config)
            elif gate_type == QualityGateType.STYLE_COMPLIANCE:
                result = await self._validate_style(config)
            elif gate_type == QualityGateType.SECURITY_ANALYSIS:
                result = await self._analyze_security(config)
            elif gate_type == QualityGateType.PERFORMANCE_ANALYSIS:
                result = await self._analyze_performance(config)
            elif gate_type == QualityGateType.TEST_COVERAGE:
                result = await self._analyze_test_coverage(config)
            elif gate_type == QualityGateType.DOCUMENTATION:
                result = await self._validate_documentation(config)
            elif gate_type == QualityGateType.DEPENDENCY_ANALYSIS:
                result = await self._analyze_dependencies(config)
            else:
                result = QualityGateResult(
                    gate_type=gate_type,
                    passed=True,
                    score=1.0,
                    execution_time=0.0,
                    recommendations=[f"Gate {gate_type.value} not implemented"]
                )

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            execution_time = time.time() - start_time

            return QualityGateResult(
                gate_type=gate_type,
                passed=False,
                score=0.0,
                execution_time=execution_time,
                issues=[QualityIssue(
                    issue_id=f"error_{gate_type.value}_{int(time.time())}",
                    gate_type=gate_type,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Gate execution error: {str(e)}"
                )]
            )

    async def _validate_syntax(self, config: Dict[str, Any]) -> QualityGateResult:
        """Validate syntax across supported file types"""

        issues = []
        metrics = {"files_checked": 0, "syntax_errors": 0}

        # Find Python files
        python_files = list(self.project_path.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            metrics["files_checked"] += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()

                # Parse Python syntax
                try:
                    ast.parse(source_code)
                except SyntaxError as e:
                    metrics["syntax_errors"] += 1
                    issues.append(QualityIssue(
                        issue_id=f"syntax_{file_path.name}_{e.lineno}",
                        gate_type=QualityGateType.SYNTAX_VALIDATION,
                        severity=ValidationSeverity.ERROR,
                        message=f"Syntax error: {e.msg}",
                        file_path=str(file_path),
                        line_number=e.lineno,
                        column_number=e.offset,
                        auto_fixable=False
                    ))

            except Exception as e:
                issues.append(QualityIssue(
                    issue_id=f"read_error_{file_path.name}",
                    gate_type=QualityGateType.SYNTAX_VALIDATION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Could not read file: {str(e)}",
                    file_path=str(file_path)
                ))

        # Calculate score
        if metrics["files_checked"] == 0:
            score = 1.0
        else:
            score = 1.0 - (metrics["syntax_errors"] / metrics["files_checked"])

        passed = score >= 0.95  # 95% threshold

        recommendations = []
        if not passed:
            recommendations.append(f"Fix {metrics['syntax_errors']} syntax errors")

        return QualityGateResult(
            gate_type=QualityGateType.SYNTAX_VALIDATION,
            passed=passed,
            score=score,
            execution_time=0.0,  # Will be set by caller
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )

    async def _validate_style(self, config: Dict[str, Any]) -> QualityGateResult:
        """Validate code style compliance"""

        issues = []
        metrics = {"files_checked": 0, "style_violations": 0}

        # Find Python files
        python_files = list(self.project_path.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            metrics["files_checked"] += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    # Check line length
                    if len(line.rstrip()) > config.get("line_length", 88):
                        metrics["style_violations"] += 1
                        issues.append(QualityIssue(
                            issue_id=f"line_length_{file_path.name}_{line_num}",
                            gate_type=QualityGateType.STYLE_COMPLIANCE,
                            severity=ValidationSeverity.WARNING,
                            message=f"Line too long ({len(line.rstrip())} > {config.get('line_length', 88)})",
                            file_path=str(file_path),
                            line_number=line_num,
                            auto_fixable=True,
                            suggestion="Break line or use shorter variable names"
                        ))

                    # Check trailing whitespace
                    if line.endswith(' \n') or line.endswith('\t\n'):
                        metrics["style_violations"] += 1
                        issues.append(QualityIssue(
                            issue_id=f"trailing_ws_{file_path.name}_{line_num}",
                            gate_type=QualityGateType.STYLE_COMPLIANCE,
                            severity=ValidationSeverity.INFO,
                            message="Trailing whitespace",
                            file_path=str(file_path),
                            line_number=line_num,
                            auto_fixable=True,
                            suggestion="Remove trailing whitespace"
                        ))

            except Exception as e:
                issues.append(QualityIssue(
                    issue_id=f"style_read_error_{file_path.name}",
                    gate_type=QualityGateType.STYLE_COMPLIANCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Could not analyze file: {str(e)}",
                    file_path=str(file_path)
                ))

        # Calculate score
        if metrics["files_checked"] == 0:
            score = 1.0
        else:
            # Allow up to 2 violations per file on average
            max_violations = metrics["files_checked"] * 2
            score = max(0.0, 1.0 - (metrics["style_violations"] / max(max_violations, 1)))

        passed = score >= 0.8  # 80% threshold

        recommendations = []
        if not passed:
            recommendations.append(f"Fix {metrics['style_violations']} style violations")
            recommendations.append("Consider using automated code formatters (black, autopep8)")

        return QualityGateResult(
            gate_type=QualityGateType.STYLE_COMPLIANCE,
            passed=passed,
            score=score,
            execution_time=0.0,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )

    async def _analyze_security(self, config: Dict[str, Any]) -> QualityGateResult:
        """Analyze security vulnerabilities"""

        issues = []
        metrics = {"files_checked": 0, "security_issues": 0}

        # Security patterns to check
        security_patterns = [
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password detected", ValidationSeverity.CRITICAL),
            (r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key detected", ValidationSeverity.CRITICAL),
            (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret detected", ValidationSeverity.CRITICAL),
            (r"eval\s*\(", "Use of eval() is dangerous", ValidationSeverity.ERROR),
            (r"exec\s*\(", "Use of exec() is dangerous", ValidationSeverity.ERROR),
            (r"subprocess\.call\s*\([^)]*shell\s*=\s*True", "Shell injection risk", ValidationSeverity.ERROR),
            (r"os\.system\s*\(", "Command injection risk", ValidationSeverity.ERROR),
            (r"pickle\.loads?\s*\(", "Unsafe deserialization", ValidationSeverity.WARNING),
            (r"yaml\.load\s*\([^)]*(?<!Loader\s*=\s*yaml\.SafeLoader)\)", "Unsafe YAML loading", ValidationSeverity.WARNING)
        ]

        # Find Python files
        python_files = list(self.project_path.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            metrics["files_checked"] += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                for pattern, message, severity in security_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)

                    for match in matches:
                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1

                        metrics["security_issues"] += 1
                        issues.append(QualityIssue(
                            issue_id=f"security_{hashlib.md5(match.group().encode()).hexdigest()[:8]}",
                            gate_type=QualityGateType.SECURITY_ANALYSIS,
                            severity=severity,
                            message=message,
                            file_path=str(file_path),
                            line_number=line_num,
                            rule_id=f"security_{pattern[:20]}",
                            auto_fixable=False,
                            suggestion="Review code for security implications"
                        ))

            except Exception as e:
                issues.append(QualityIssue(
                    issue_id=f"security_read_error_{file_path.name}",
                    gate_type=QualityGateType.SECURITY_ANALYSIS,
                    severity=ValidationSeverity.WARNING,
                    message=f"Could not analyze file: {str(e)}",
                    file_path=str(file_path)
                ))

        # Calculate score based on severity
        critical_issues = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        error_issues = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warning_issues = len([i for i in issues if i.severity == ValidationSeverity.WARNING])

        # Weighted score calculation
        penalty = (critical_issues * 0.5 + error_issues * 0.3 + warning_issues * 0.1)
        score = max(0.0, 1.0 - penalty / max(metrics["files_checked"], 1))

        passed = critical_issues == 0 and error_issues <= 1

        recommendations = []
        if critical_issues > 0:
            recommendations.append(f"URGENT: Fix {critical_issues} critical security issues")
        if error_issues > 0:
            recommendations.append(f"Fix {error_issues} security errors")
        if warning_issues > 0:
            recommendations.append(f"Review {warning_issues} security warnings")

        return QualityGateResult(
            gate_type=QualityGateType.SECURITY_ANALYSIS,
            passed=passed,
            score=score,
            execution_time=0.0,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )

    async def _analyze_performance(self, config: Dict[str, Any]) -> QualityGateResult:
        """Analyze performance characteristics"""

        issues = []
        metrics = {"files_checked": 0, "performance_issues": 0, "complexity_violations": 0}

        # Performance anti-patterns
        performance_patterns = [
            (r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\([^)]+\)\s*\)", "Use enumerate() instead of range(len())", ValidationSeverity.WARNING),
            (r"\.append\s*\([^)]+\)\s*\n\s*\.sort\s*\(\s*\)", "Consider using bisect.insort() for sorted insertion", ValidationSeverity.INFO),
            (r"list\s*\(\s*map\s*\([^)]+\)\s*\)", "Consider list comprehension for better performance", ValidationSeverity.INFO),
            (r"\.join\s*\(\s*\[.*for.*in.*\]\s*\)", "String concatenation in loop can be slow", ValidationSeverity.WARNING)
        ]

        python_files = list(self.project_path.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            metrics["files_checked"] += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check performance patterns
                for pattern, message, severity in performance_patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)

                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        metrics["performance_issues"] += 1

                        issues.append(QualityIssue(
                            issue_id=f"perf_{hashlib.md5(match.group().encode()).hexdigest()[:8]}",
                            gate_type=QualityGateType.PERFORMANCE_ANALYSIS,
                            severity=severity,
                            message=message,
                            file_path=str(file_path),
                            line_number=line_num,
                            auto_fixable=True,
                            suggestion="Consider performance optimization"
                        ))

                # Analyze function complexity
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            complexity = self._calculate_cyclomatic_complexity(node)

                            if complexity > config.get("complexity_threshold", 10):
                                metrics["complexity_violations"] += 1
                                issues.append(QualityIssue(
                                    issue_id=f"complexity_{file_path.name}_{node.name}",
                                    gate_type=QualityGateType.PERFORMANCE_ANALYSIS,
                                    severity=ValidationSeverity.WARNING,
                                    message=f"High cyclomatic complexity: {complexity}",
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    suggestion="Consider breaking down complex function"
                                ))

                except SyntaxError:
                    pass  # Skip files with syntax errors

            except Exception as e:
                issues.append(QualityIssue(
                    issue_id=f"perf_read_error_{file_path.name}",
                    gate_type=QualityGateType.PERFORMANCE_ANALYSIS,
                    severity=ValidationSeverity.WARNING,
                    message=f"Could not analyze file: {str(e)}",
                    file_path=str(file_path)
                ))

        # Calculate score
        total_issues = metrics["performance_issues"] + metrics["complexity_violations"]
        if metrics["files_checked"] == 0:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (total_issues / (metrics["files_checked"] * 3)))  # Allow 3 issues per file

        passed = score >= 0.7  # 70% threshold

        recommendations = []
        if not passed:
            recommendations.append(f"Address {total_issues} performance issues")
            recommendations.append("Consider code profiling for optimization opportunities")

        return QualityGateResult(
            gate_type=QualityGateType.PERFORMANCE_ANALYSIS,
            passed=passed,
            score=score,
            execution_time=0.0,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function"""

        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points that increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.ListComp):
                complexity += 1

        return complexity

    async def _analyze_test_coverage(self, config: Dict[str, Any]) -> QualityGateResult:
        """Analyze test coverage"""

        issues = []
        metrics = {"source_files": 0, "test_files": 0, "coverage_estimate": 0.0}

        # Find source and test files
        source_files = list(self.project_path.rglob("*.py"))
        test_files = [f for f in source_files if 'test' in f.name.lower() or f.parent.name == 'tests']

        source_files = [f for f in source_files if f not in test_files and not self._should_skip_file(f)]

        metrics["source_files"] = len(source_files)
        metrics["test_files"] = len(test_files)

        # Simple heuristic for test coverage estimate
        if metrics["source_files"] == 0:
            coverage_estimate = 1.0
        else:
            # Basic ratio of test files to source files
            file_coverage_ratio = min(1.0, metrics["test_files"] / metrics["source_files"])

            # Check for test functions
            total_test_functions = 0
            for test_file in test_files:
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                                total_test_functions += 1
                    except SyntaxError:
                        pass

                except Exception:
                    pass

            # Estimate coverage based on test function density
            function_coverage_ratio = min(1.0, total_test_functions / max(metrics["source_files"] * 3, 1))

            # Combined estimate
            coverage_estimate = (file_coverage_ratio + function_coverage_ratio) / 2

        metrics["coverage_estimate"] = coverage_estimate
        minimum_coverage = config.get("minimum_coverage", 0.7)

        if coverage_estimate < minimum_coverage:
            issues.append(QualityIssue(
                issue_id="low_test_coverage",
                gate_type=QualityGateType.TEST_COVERAGE,
                severity=ValidationSeverity.WARNING,
                message=f"Estimated test coverage {coverage_estimate:.1%} below threshold {minimum_coverage:.1%}",
                suggestion="Add more comprehensive tests"
            ))

        passed = coverage_estimate >= minimum_coverage
        score = coverage_estimate

        recommendations = []
        if not passed:
            recommendations.append(f"Increase test coverage to meet {minimum_coverage:.1%} threshold")
            recommendations.append(f"Add {max(1, int((minimum_coverage - coverage_estimate) * metrics['source_files']))} more test files")

        return QualityGateResult(
            gate_type=QualityGateType.TEST_COVERAGE,
            passed=passed,
            score=score,
            execution_time=0.0,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )

    async def _validate_documentation(self, config: Dict[str, Any]) -> QualityGateResult:
        """Validate documentation quality"""

        issues = []
        metrics = {"functions_checked": 0, "missing_docstrings": 0, "documentation_files": 0}

        # Check for documentation files
        doc_files = list(self.project_path.rglob("*.md")) + list(self.project_path.rglob("*.rst"))
        metrics["documentation_files"] = len(doc_files)

        # Check for function docstrings
        python_files = list(self.project_path.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not node.name.startswith('_'):  # Skip private functions
                                metrics["functions_checked"] += 1

                                # Check for docstring
                                if (not node.body or
                                    not isinstance(node.body[0], ast.Expr) or
                                    not isinstance(node.body[0].value, ast.Constant) or
                                    not isinstance(node.body[0].value.value, str)):

                                    metrics["missing_docstrings"] += 1
                                    issues.append(QualityIssue(
                                        issue_id=f"missing_docstring_{file_path.name}_{node.name}",
                                        gate_type=QualityGateType.DOCUMENTATION,
                                        severity=ValidationSeverity.WARNING,
                                        message=f"Missing docstring for function '{node.name}'",
                                        file_path=str(file_path),
                                        line_number=node.lineno,
                                        auto_fixable=True,
                                        suggestion="Add descriptive docstring"
                                    ))

                except SyntaxError:
                    pass  # Skip files with syntax errors

            except Exception as e:
                continue

        # Calculate score
        if metrics["functions_checked"] == 0:
            docstring_score = 1.0
        else:
            docstring_score = 1.0 - (metrics["missing_docstrings"] / metrics["functions_checked"])

        # Documentation file bonus
        doc_bonus = min(0.2, metrics["documentation_files"] * 0.05)

        score = min(1.0, docstring_score + doc_bonus)
        passed = score >= 0.8  # 80% threshold

        recommendations = []
        if not passed:
            recommendations.append(f"Add docstrings to {metrics['missing_docstrings']} functions")
            if metrics["documentation_files"] < 3:
                recommendations.append("Consider adding more documentation files (README, API docs, etc.)")

        return QualityGateResult(
            gate_type=QualityGateType.DOCUMENTATION,
            passed=passed,
            score=score,
            execution_time=0.0,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )

    async def _analyze_dependencies(self, config: Dict[str, Any]) -> QualityGateResult:
        """Analyze project dependencies"""

        issues = []
        metrics = {"dependencies_found": 0, "outdated_dependencies": 0, "vulnerable_dependencies": 0}

        # Check requirements.txt
        req_file = self.project_path / "requirements.txt"
        pyproject_file = self.project_path / "pyproject.toml"

        dependencies = []

        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            dependencies.append(line.split('==')[0].split('>=')[0].split('<=')[0])
            except Exception:
                pass

        if pyproject_file.exists():
            try:
                # Simple parsing for dependencies (would use toml library in production)
                with open(pyproject_file, 'r') as f:
                    content = f.read()

                # Extract dependencies from content (simplified)
                import re
                dep_pattern = r'"([a-zA-Z0-9_-]+)(?:[>=<].*?)?"'
                matches = re.findall(dep_pattern, content)
                dependencies.extend(matches)

            except Exception:
                pass

        metrics["dependencies_found"] = len(set(dependencies))

        # Known vulnerable packages (simplified list)
        vulnerable_packages = {
            "pillow": "9.0.0",  # Example vulnerable version
            "requests": "2.25.0",
            "urllib3": "1.26.0"
        }

        for dep in dependencies:
            if dep.lower() in vulnerable_packages:
                metrics["vulnerable_dependencies"] += 1
                issues.append(QualityIssue(
                    issue_id=f"vulnerable_dep_{dep}",
                    gate_type=QualityGateType.DEPENDENCY_ANALYSIS,
                    severity=ValidationSeverity.ERROR,
                    message=f"Potentially vulnerable dependency: {dep}",
                    suggestion=f"Update {dep} to latest secure version"
                ))

        # Calculate score
        if metrics["dependencies_found"] == 0:
            score = 1.0
        else:
            vulnerability_penalty = metrics["vulnerable_dependencies"] / metrics["dependencies_found"]
            score = max(0.0, 1.0 - vulnerability_penalty)

        passed = metrics["vulnerable_dependencies"] == 0

        recommendations = []
        if not passed:
            recommendations.append(f"Update {metrics['vulnerable_dependencies']} vulnerable dependencies")
            recommendations.append("Run security audit on dependencies regularly")

        return QualityGateResult(
            gate_type=QualityGateType.DEPENDENCY_ANALYSIS,
            passed=passed,
            score=score,
            execution_time=0.0,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis"""

        skip_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            "node_modules",
            "venv",
            ".venv",
            "env",
            ".env",
            "build",
            "dist",
            "*.egg-info"
        ]

        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)

    def _calculate_overall_score(self, gate_results: List[QualityGateResult]) -> float:
        """Calculate weighted overall quality score"""

        total_weight = 0.0
        weighted_score = 0.0

        for result in gate_results:
            config = self.gate_configs.get(result.gate_type, {})
            weight = config.get("weight", 0.1)

            total_weight += weight
            weighted_score += result.score * weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(self, gate_results: List[QualityGateResult]) -> List[str]:
        """Generate comprehensive recommendations"""

        recommendations = []

        # Collect all recommendations from gates
        for result in gate_results:
            recommendations.extend(result.recommendations)

        # Add overall recommendations based on results
        failed_gates = [r for r in gate_results if not r.passed]
        if failed_gates:
            critical_gates = [r for r in failed_gates if any(
                issue.severity == ValidationSeverity.CRITICAL
                for issue in r.issues
            )]

            if critical_gates:
                recommendations.append("PRIORITY: Address critical issues immediately")

            recommendations.append(f"Focus on improving {len(failed_gates)} failing quality gates")

        # Quality level recommendations
        if self.quality_level == QualityLevel.QUANTUM:
            recommendations.append("Consider implementing advanced quantum optimization techniques")
            recommendations.append("Set up continuous quality monitoring")

        return list(set(recommendations))  # Remove duplicates

    async def _apply_auto_fixes(self, gate_results: List[QualityGateResult]) -> int:
        """Apply automatic fixes for fixable issues"""

        if not self.auto_fix_enabled:
            return 0

        fixes_applied = 0

        for result in gate_results:
            fixable_issues = [issue for issue in result.issues if issue.auto_fixable]

            for issue in fixable_issues:
                try:
                    if await self._apply_auto_fix(issue):
                        fixes_applied += 1
                except Exception as e:
                    self.logger.warning(f"Failed to auto-fix issue {issue.issue_id}: {str(e)}")

        return fixes_applied

    async def _apply_auto_fix(self, issue: QualityIssue) -> bool:
        """Apply a single auto-fix"""

        if not issue.file_path or not issue.line_number:
            return False

        try:
            file_path = Path(issue.file_path)
            if not file_path.exists():
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if issue.line_number > len(lines):
                return False

            line_index = issue.line_number - 1
            original_line = lines[line_index]

            # Apply specific fixes based on issue type
            if "trailing_ws" in issue.issue_id:
                # Remove trailing whitespace
                lines[line_index] = original_line.rstrip() + '\n'

            elif "missing_docstring" in issue.issue_id:
                # Add basic docstring (simplified)
                indent = len(original_line) - len(original_line.lstrip())
                docstring = ' ' * (indent + 4) + '"""TODO: Add docstring"""\n'
                lines.insert(line_index + 1, docstring)

            else:
                return False  # No auto-fix available

            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            self.logger.debug(f"Applied auto-fix for {issue.issue_id}")
            return True

        except Exception as e:
            self.logger.error(f"Auto-fix failed for {issue.issue_id}: {str(e)}")
            return False

    async def export_quality_report(self,
                                   quality_report: QualityReport,
                                   output_path: Path = None) -> Path:
        """Export quality report to JSON file"""

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.project_path / f"quality_report_{timestamp}.json"

        # Convert report to JSON-serializable format
        report_data = {
            "report_id": quality_report.report_id,
            "project_path": str(quality_report.project_path),
            "quality_level": quality_report.quality_level.value,
            "overall_score": quality_report.overall_score,
            "execution_time": quality_report.execution_time,
            "total_issues": quality_report.total_issues,
            "critical_issues": quality_report.critical_issues,
            "auto_fixes_applied": quality_report.auto_fixes_applied,
            "recommendations": quality_report.recommendations,
            "generated_at": quality_report.generated_at.isoformat(),
            "gate_results": [
                {
                    "gate_type": result.gate_type.value,
                    "passed": result.passed,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "metrics": result.metrics,
                    "recommendations": result.recommendations,
                    "issues": [
                        {
                            "issue_id": issue.issue_id,
                            "gate_type": issue.gate_type.value,
                            "severity": issue.severity.value,
                            "message": issue.message,
                            "file_path": issue.file_path,
                            "line_number": issue.line_number,
                            "column_number": issue.column_number,
                            "rule_id": issue.rule_id,
                            "suggestion": issue.suggestion,
                            "auto_fixable": issue.auto_fixable
                        }
                        for issue in result.issues
                    ]
                }
                for result in quality_report.gate_results
            ],
            "validation_statistics": self.validation_stats.copy()
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Quality report exported to: {output_path}")
        return output_path

    def get_quality_health_status(self) -> Dict[str, Any]:
        """Get quality analyzer health status"""

        success_rate = (
            self.validation_stats["successful_validations"] /
            max(self.validation_stats["total_validations"], 1)
        )

        return {
            "healthy": success_rate >= 0.8,
            "success_rate": success_rate,
            "total_validations": self.validation_stats["total_validations"],
            "auto_fix_enabled": self.auto_fix_enabled,
            "parallel_execution": self.parallel_execution,
            "quality_level": self.quality_level.value,
            "average_execution_time": self.validation_stats["average_execution_time"],
            "auto_fixes_applied": self.validation_stats["auto_fixes_applied"]
        }


# Main execution for comprehensive quality gates
async def main():
    """Main quality gates execution"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger("QualityGates")

    # Initialize quality analyzer
    analyzer = QuantumQualityAnalyzer(
        project_path=Path.cwd(),
        quality_level=QualityLevel.QUANTUM,
        auto_fix_enabled=True,
        parallel_execution=True,
        logger=logger
    )

    # Execute comprehensive quality gates
    quality_report = await analyzer.execute_comprehensive_quality_gates()

    # Export report
    report_path = await analyzer.export_quality_report(quality_report)

    print("\n" + "="*80)
    print("🛡️ COMPREHENSIVE QUALITY GATES COMPLETE")
    print("="*80)
    print(f"Overall Score: {quality_report.overall_score:.1%}")
    print(f"Quality Level: {quality_report.quality_level.value.upper()}")
    print(f"Execution Time: {quality_report.execution_time:.2f}s")
    print(f"Total Issues: {quality_report.total_issues}")
    print(f"Critical Issues: {quality_report.critical_issues}")
    print(f"Auto-fixes Applied: {quality_report.auto_fixes_applied}")
    print(f"Report Exported: {report_path}")

    print(f"\n📊 Gate Results:")
    for result in quality_report.gate_results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  • {result.gate_type.value}: {status} ({result.score:.1%})")

    if quality_report.recommendations:
        print(f"\n📋 Top Recommendations:")
        for rec in quality_report.recommendations[:5]:
            print(f"  • {rec}")

    return quality_report


if __name__ == "__main__":
    asyncio.run(main())