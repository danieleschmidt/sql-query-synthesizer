"""Comprehensive Validation System for SQL Query Synthesizer.

This module implements multi-layered validation with security checks,
data integrity validation, and business rule enforcement.
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import sqlparse
from sqlparse import tokens as T

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validation checks."""

    SYNTAX = "syntax"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS_RULES = "business_rules"
    DATA_INTEGRITY = "data_integrity"
    COMPLIANCE = "compliance"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during checks."""

    issue_id: str
    validation_type: ValidationType
    severity: ValidationSeverity
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ValidationResult:
    """Result of comprehensive validation."""

    is_valid: bool
    issues: List[ValidationIssue]
    security_score: float
    performance_score: float
    compliance_score: float
    recommendations: List[str]
    execution_time_ms: float


class SQLSecurityValidator:
    """Advanced SQL security validation with multiple detection methods."""

    def __init__(self):
        # SQL injection patterns
        self.injection_patterns = [
            # Union-based injection
            r"(?i)\bunion\s+(?:all\s+)?select\b",
            # Boolean-based blind injection
            r"(?i)\b(?:and|or)\s+\d+\s*=\s*\d+",
            r'(?i)\b(?:and|or)\s+[\'"`]?[a-z]+[\'"`]?\s*=\s*[\'"`]?[a-z]+[\'"`]?',
            # Time-based blind injection
            r"(?i)\bwaitfor\s+delay\b",
            r"(?i)\bsleep\s*\(",
            r"(?i)\bbenchmark\s*\(",
            # Stacked queries
            r";\s*(?:drop|delete|insert|update|create|alter|truncate)\s+",
            # Comment-based evasion
            r"/\*.*?\*/",
            r"--.*$",
            r"#.*$",
            # Function-based injection
            r"(?i)\b(?:exec|execute|sp_executesql)\s*\(",
            r"(?i)\bxp_cmdshell\s*\(",
            # String concatenation patterns
            r"(?i)\bconcat\s*\(",
            r"\|\|",
            r'\+.*[\'"`]',
            # Encoded payloads
            r"(?i)0x[0-9a-f]+",
            r"(?i)char\s*\(\s*\d+",
            r"(?i)ascii\s*\(",
        ]

        # Dangerous functions
        self.dangerous_functions = {
            "exec",
            "execute",
            "sp_executesql",
            "xp_cmdshell",
            "openrowset",
            "opendatasource",
            "openquery",
            "openxml",
            "sp_oacreate",
            "load_file",
            "into outfile",
            "into dumpfile",
        }

        # Allowed tables/schemas (configurable)
        self.allowed_schemas = set()
        self.allowed_tables = set()
        self.forbidden_tables = {"information_schema", "mysql", "pg_catalog", "sys"}

    def validate_sql_security(self, sql: str) -> List[ValidationIssue]:
        """Perform comprehensive SQL security validation."""
        issues = []

        # Pattern-based detection
        issues.extend(self._check_injection_patterns(sql))

        # AST-based analysis
        issues.extend(self._ast_based_security_check(sql))

        # Semantic analysis
        issues.extend(self._semantic_security_check(sql))

        # Access control validation
        issues.extend(self._check_access_permissions(sql))

        return issues

    def _check_injection_patterns(self, sql: str) -> List[ValidationIssue]:
        """Check for SQL injection patterns using regex."""
        issues = []

        for i, pattern in enumerate(self.injection_patterns):
            matches = re.finditer(pattern, sql, re.MULTILINE)
            for match in matches:
                issues.append(
                    ValidationIssue(
                        issue_id=f"SEC_INJ_{i:03d}",
                        validation_type=ValidationType.SECURITY,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Potential SQL injection detected: {match.group()}",
                        location=f"Position {match.start()}-{match.end()}",
                        suggestion="Use parameterized queries instead of string concatenation",
                        metadata={"pattern": pattern, "match": match.group()},
                    )
                )

        return issues

    def _ast_based_security_check(self, sql: str) -> List[ValidationIssue]:
        """Perform AST-based security analysis."""
        issues = []

        try:
            parsed = sqlparse.parse(sql)[0]

            # Check for dangerous functions
            for token in parsed.flatten():
                if (
                    token.ttype is T.Name
                    and token.value.lower() in self.dangerous_functions
                ):
                    issues.append(
                        ValidationIssue(
                            issue_id="SEC_FUNC_001",
                            validation_type=ValidationType.SECURITY,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Dangerous function detected: {token.value}",
                            suggestion="Remove or replace with safer alternative",
                            metadata={"function": token.value},
                        )
                    )

            # Check for stacked queries
            statements = sqlparse.split(sql)
            if len(statements) > 1:
                non_empty_statements = [s.strip() for s in statements if s.strip()]
                if len(non_empty_statements) > 1:
                    issues.append(
                        ValidationIssue(
                            issue_id="SEC_STACK_001",
                            validation_type=ValidationType.SECURITY,
                            severity=ValidationSeverity.HIGH,
                            message="Multiple SQL statements detected (stacked queries)",
                            suggestion="Execute statements separately",
                            metadata={"statement_count": len(non_empty_statements)},
                        )
                    )

        except Exception as e:
            logger.warning(f"AST security check failed: {e}")

        return issues

    def _semantic_security_check(self, sql: str) -> List[ValidationIssue]:
        """Perform semantic security analysis."""
        issues = []

        sql_lower = sql.lower()

        # Check for information disclosure attempts
        info_disclosure_patterns = [
            "information_schema",
            "sys.tables",
            "mysql.user",
            "pg_tables",
            "sqlite_master",
        ]

        for pattern in info_disclosure_patterns:
            if pattern in sql_lower:
                issues.append(
                    ValidationIssue(
                        issue_id="SEC_INFO_001",
                        validation_type=ValidationType.SECURITY,
                        severity=ValidationSeverity.HIGH,
                        message=f"Information disclosure attempt: {pattern}",
                        suggestion="Remove access to system tables",
                        metadata={"pattern": pattern},
                    )
                )

        # Check for privilege escalation attempts
        if any(
            cmd in sql_lower for cmd in ["grant", "revoke", "create user", "drop user"]
        ):
            issues.append(
                ValidationIssue(
                    issue_id="SEC_PRIV_001",
                    validation_type=ValidationType.SECURITY,
                    severity=ValidationSeverity.CRITICAL,
                    message="Privilege escalation attempt detected",
                    suggestion="Remove privilege management statements",
                )
            )

        return issues

    def _check_access_permissions(self, sql: str) -> List[ValidationIssue]:
        """Check access permissions and table restrictions."""
        issues = []

        # Extract table names from SQL
        table_names = self._extract_table_names(sql)

        for table in table_names:
            # Check forbidden tables
            if any(forbidden in table.lower() for forbidden in self.forbidden_tables):
                issues.append(
                    ValidationIssue(
                        issue_id="SEC_ACCESS_001",
                        validation_type=ValidationType.SECURITY,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Access to forbidden table: {table}",
                        suggestion="Remove access to system tables",
                        metadata={"table": table},
                    )
                )

            # Check allowed tables (if configured)
            if self.allowed_tables and table.lower() not in self.allowed_tables:
                issues.append(
                    ValidationIssue(
                        issue_id="SEC_ACCESS_002",
                        validation_type=ValidationType.SECURITY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Access to non-whitelisted table: {table}",
                        suggestion="Ensure table access is authorized",
                        metadata={"table": table},
                    )
                )

        return issues

    def _extract_table_names(self, sql: str) -> Set[str]:
        """Extract table names from SQL query."""
        tables = set()

        try:
            parsed = sqlparse.parse(sql)[0]

            # Simple extraction - can be enhanced
            tokens = list(parsed.flatten())
            for i, token in enumerate(tokens):
                if token.ttype is T.Keyword and token.value.upper() in (
                    "FROM",
                    "JOIN",
                    "UPDATE",
                    "INTO",
                ):
                    # Look for table name in following tokens
                    for j in range(i + 1, min(i + 5, len(tokens))):
                        next_token = tokens[j]
                        if next_token.ttype is T.Name:
                            tables.add(next_token.value)
                            break
                        elif next_token.ttype not in (T.Whitespace, T.Punctuation):
                            break
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")

        return tables


class PerformanceValidator:
    """Validate queries for performance issues."""

    def __init__(self):
        self.performance_patterns = {
            "missing_where": r"(?i)select\s+.*\s+from\s+\w+(?:\s+(?!where|limit|order|group))",
            "select_star": r"(?i)select\s+\*\s+from",
            "cartesian_join": r"(?i)from\s+\w+\s*,\s*\w+(?:\s*,\s*\w+)*(?!\s+where)",
            "or_in_where": r"(?i)where\s+.*\bor\b",
            "function_in_where": r"(?i)where\s+.*(?:upper|lower|substring|trim)\s*\(",
            "like_leading_wildcard": r'(?i)like\s+[\'"]%',
        }

    def validate_performance(self, sql: str) -> List[ValidationIssue]:
        """Validate query for performance issues."""
        issues = []

        # Check for common performance anti-patterns
        for issue_type, pattern in self.performance_patterns.items():
            if re.search(pattern, sql):
                issues.append(self._create_performance_issue(issue_type, sql))

        # Check for complex joins
        join_count = len(re.findall(r"(?i)\bjoin\b", sql))
        if join_count > 5:
            issues.append(
                ValidationIssue(
                    issue_id="PERF_JOIN_001",
                    validation_type=ValidationType.PERFORMANCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Query has {join_count} joins, consider optimization",
                    suggestion="Consider denormalization or query restructuring",
                )
            )

        # Check for nested subqueries
        subquery_depth = self._count_subquery_depth(sql)
        if subquery_depth > 3:
            issues.append(
                ValidationIssue(
                    issue_id="PERF_SUBQ_001",
                    validation_type=ValidationType.PERFORMANCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Deep nesting detected: {subquery_depth} levels",
                    suggestion="Consider using CTEs or temporary tables",
                )
            )

        return issues

    def _create_performance_issue(self, issue_type: str, sql: str) -> ValidationIssue:
        """Create performance validation issue."""
        issues_map = {
            "missing_where": {
                "id": "PERF_WHERE_001",
                "severity": ValidationSeverity.WARNING,
                "message": "SELECT without WHERE clause detected",
                "suggestion": "Add WHERE clause to limit results",
            },
            "select_star": {
                "id": "PERF_SELECT_001",
                "severity": ValidationSeverity.INFO,
                "message": "SELECT * usage detected",
                "suggestion": "Specify only needed columns",
            },
            "cartesian_join": {
                "id": "PERF_CART_001",
                "severity": ValidationSeverity.ERROR,
                "message": "Potential cartesian product detected",
                "suggestion": "Add proper JOIN conditions",
            },
            "or_in_where": {
                "id": "PERF_OR_001",
                "severity": ValidationSeverity.INFO,
                "message": "OR condition in WHERE clause",
                "suggestion": "Consider using UNION or IN clause",
            },
            "function_in_where": {
                "id": "PERF_FUNC_001",
                "severity": ValidationSeverity.WARNING,
                "message": "Function in WHERE clause prevents index usage",
                "suggestion": "Rewrite to avoid functions on indexed columns",
            },
            "like_leading_wildcard": {
                "id": "PERF_LIKE_001",
                "severity": ValidationSeverity.WARNING,
                "message": "LIKE with leading wildcard prevents index usage",
                "suggestion": "Consider full-text search or different approach",
            },
        }

        issue_info = issues_map.get(
            issue_type,
            {
                "id": "PERF_UNKNOWN",
                "severity": ValidationSeverity.INFO,
                "message": f"Performance issue detected: {issue_type}",
                "suggestion": "Review query for optimization opportunities",
            },
        )

        return ValidationIssue(
            issue_id=issue_info["id"],
            validation_type=ValidationType.PERFORMANCE,
            severity=issue_info["severity"],
            message=issue_info["message"],
            suggestion=issue_info["suggestion"],
        )

    def _count_subquery_depth(self, sql: str) -> int:
        """Count the maximum depth of nested subqueries."""
        depth = 0
        max_depth = 0

        for char in sql:
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth = max(0, depth - 1)

        # Rough approximation - actual parsing would be more accurate
        return max_depth // 2


class BusinessRulesValidator:
    """Validate queries against business rules and policies."""

    def __init__(self):
        self.rules = {}
        self.table_policies = {}

    def add_business_rule(self, rule_id: str, rule_func: callable, description: str):
        """Add a custom business rule."""
        self.rules[rule_id] = {"function": rule_func, "description": description}

    def add_table_policy(self, table: str, policy: Dict[str, Any]):
        """Add access policy for a table."""
        self.table_policies[table.lower()] = policy

    def validate_business_rules(
        self, sql: str, context: Dict[str, Any] = None
    ) -> List[ValidationIssue]:
        """Validate query against business rules."""
        issues = []
        context = context or {}

        # Apply custom business rules
        for rule_id, rule_info in self.rules.items():
            try:
                result = rule_info["function"](sql, context)
                if not result:
                    issues.append(
                        ValidationIssue(
                            issue_id=f"BIZ_{rule_id}",
                            validation_type=ValidationType.BUSINESS_RULES,
                            severity=ValidationSeverity.ERROR,
                            message=f"Business rule violation: {rule_info['description']}",
                            suggestion="Modify query to comply with business rules",
                        )
                    )
            except Exception as e:
                logger.warning(f"Business rule {rule_id} failed: {e}")

        # Check table policies
        issues.extend(self._check_table_policies(sql, context))

        return issues

    def _check_table_policies(
        self, sql: str, context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check table-specific policies."""
        issues = []

        # Extract table names
        security_validator = SQLSecurityValidator()
        tables = security_validator._extract_table_names(sql)

        for table in tables:
            policy = self.table_policies.get(table.lower())
            if not policy:
                continue

            # Check time-based restrictions
            if "allowed_hours" in policy:
                current_hour = time.localtime().tm_hour
                if current_hour not in policy["allowed_hours"]:
                    issues.append(
                        ValidationIssue(
                            issue_id="BIZ_TIME_001",
                            validation_type=ValidationType.BUSINESS_RULES,
                            severity=ValidationSeverity.ERROR,
                            message=f"Access to {table} not allowed at this time",
                            suggestion=f"Access allowed during hours: {policy['allowed_hours']}",
                        )
                    )

            # Check user-based restrictions
            if "allowed_users" in policy:
                user_id = context.get("user_id")
                if user_id and user_id not in policy["allowed_users"]:
                    issues.append(
                        ValidationIssue(
                            issue_id="BIZ_USER_001",
                            validation_type=ValidationType.BUSINESS_RULES,
                            severity=ValidationSeverity.ERROR,
                            message=f"User {user_id} not authorized to access {table}",
                            suggestion="Contact administrator for access",
                        )
                    )

            # Check query type restrictions
            if "allowed_operations" in policy:
                sql_upper = sql.upper().strip()
                operation = sql_upper.split()[0] if sql_upper else ""
                if operation not in policy["allowed_operations"]:
                    issues.append(
                        ValidationIssue(
                            issue_id="BIZ_OP_001",
                            validation_type=ValidationType.BUSINESS_RULES,
                            severity=ValidationSeverity.ERROR,
                            message=f"Operation {operation} not allowed on {table}",
                            suggestion=f"Allowed operations: {policy['allowed_operations']}",
                        )
                    )

        return issues


class ComprehensiveValidator:
    """Main validator that orchestrates all validation types."""

    def __init__(self):
        self.security_validator = SQLSecurityValidator()
        self.performance_validator = PerformanceValidator()
        self.business_rules_validator = BusinessRulesValidator()

        # Validation weights for scoring
        self.validation_weights = {
            ValidationType.SECURITY: 0.4,
            ValidationType.PERFORMANCE: 0.3,
            ValidationType.BUSINESS_RULES: 0.2,
            ValidationType.COMPLIANCE: 0.1,
        }

        # Severity weights
        self.severity_weights = {
            ValidationSeverity.CRITICAL: 100,
            ValidationSeverity.ERROR: 75,
            ValidationSeverity.WARNING: 50,
            ValidationSeverity.INFO: 25,
        }

    def validate(self, sql: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Perform comprehensive validation of SQL query."""
        start_time = time.time()
        context = context or {}

        all_issues = []

        # Security validation
        security_issues = self.security_validator.validate_sql_security(sql)
        all_issues.extend(security_issues)

        # Performance validation
        performance_issues = self.performance_validator.validate_performance(sql)
        all_issues.extend(performance_issues)

        # Business rules validation
        business_issues = self.business_rules_validator.validate_business_rules(
            sql, context
        )
        all_issues.extend(business_issues)

        # Calculate scores
        security_score = self._calculate_score(security_issues, ValidationType.SECURITY)
        performance_score = self._calculate_score(
            performance_issues, ValidationType.PERFORMANCE
        )
        compliance_score = 100.0  # Placeholder - would integrate with compliance checks

        # Determine overall validity
        critical_issues = [
            i for i in all_issues if i.severity == ValidationSeverity.CRITICAL
        ]
        error_issues = [i for i in all_issues if i.severity == ValidationSeverity.ERROR]
        is_valid = len(critical_issues) == 0 and len(error_issues) == 0

        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues)

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            security_score=security_score,
            performance_score=performance_score,
            compliance_score=compliance_score,
            recommendations=recommendations,
            execution_time_ms=execution_time,
        )

    def _calculate_score(
        self, issues: List[ValidationIssue], validation_type: ValidationType
    ) -> float:
        """Calculate score for a specific validation type."""
        if not issues:
            return 100.0

        total_penalty = 0
        for issue in issues:
            if issue.validation_type == validation_type:
                total_penalty += self.severity_weights[issue.severity]

        # Maximum penalty is 500 (5 critical issues)
        max_penalty = 500
        penalty_ratio = min(total_penalty / max_penalty, 1.0)

        return max(0.0, 100.0 - (penalty_ratio * 100))

    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable recommendations based on issues."""
        recommendations = []

        # Group issues by type
        by_type = {}
        for issue in issues:
            if issue.validation_type not in by_type:
                by_type[issue.validation_type] = []
            by_type[issue.validation_type].append(issue)

        # Generate type-specific recommendations
        if ValidationType.SECURITY in by_type:
            recommendations.append(
                "Implement parameterized queries to prevent SQL injection"
            )
            recommendations.append("Review and restrict database permissions")

        if ValidationType.PERFORMANCE in by_type:
            recommendations.append("Add appropriate indexes for query optimization")
            recommendations.append(
                "Consider query restructuring for better performance"
            )

        if ValidationType.BUSINESS_RULES in by_type:
            recommendations.append("Review business rule compliance")
            recommendations.append("Ensure proper authorization for data access")

        # Add specific suggestions from issues
        for issue in issues:
            if issue.suggestion and issue.suggestion not in recommendations:
                recommendations.append(issue.suggestion)

        return recommendations[:10]  # Limit to top 10 recommendations


# Global validator instance
comprehensive_validator = ComprehensiveValidator()
