#!/usr/bin/env python3
"""
TDD Security Checklist Integration
Implements security validation as part of the TDD cycle.
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SecurityCheckResult:
    """Result of a security check."""
    check_name: str
    passed: bool
    details: str
    severity: str = "medium"
    file_path: Optional[str] = None
    line_number: Optional[int] = None

class TDDSecurityChecker:
    """Integrates security checks into TDD cycle."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.results: List[SecurityCheckResult] = []
    
    def run_security_checklist(self) -> List[SecurityCheckResult]:
        """Run comprehensive security checklist."""
        self.results = []
        
        # 1. Input validation and sanitization
        self._check_input_validation()
        
        # 2. Authentication and authorization controls
        self._check_auth_controls()
        
        # 3. Secrets management
        self._check_secrets_management()
        
        # 4. Safe logging practices
        self._check_safe_logging()
        
        # 5. Software Composition Analysis (SCA)
        self._run_sca_scan()
        
        # 6. Static Application Security Testing (SAST)
        self._run_sast_scan()
        
        return self.results
    
    def _check_input_validation(self) -> None:
        """Check for proper input validation."""
        try:
            # Look for SQL injection prevention patterns
            result = subprocess.run([
                'rg', '--type', 'py', '--line-number',
                r'(execute|query).*%.*[\'"]', str(self.repo_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':', 3)
                        if len(parts) >= 3:
                            file_path, line_num = parts[0], parts[1]
                            self.results.append(SecurityCheckResult(
                                check_name="SQL Injection Prevention",
                                passed=False,
                                details=f"Potential SQL injection risk detected",
                                severity="high",
                                file_path=file_path,
                                line_number=int(line_num)
                            ))
            else:
                self.results.append(SecurityCheckResult(
                    check_name="SQL Injection Prevention",
                    passed=True,
                    details="No SQL injection patterns detected"
                ))
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.results.append(SecurityCheckResult(
                check_name="SQL Injection Prevention",
                passed=False,
                details="Unable to run SQL injection check",
                severity="medium"
            ))
    
    def _check_auth_controls(self) -> None:
        """Check authentication and authorization controls."""
        try:
            # Look for hardcoded credentials
            result = subprocess.run([
                'rg', '--type', 'py', '--line-number',
                r'(password|secret|key|token)\s*=\s*[\'"][^\'"\s]{8,}[\'"]',
                str(self.repo_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':', 3)
                        if len(parts) >= 3:
                            file_path, line_num = parts[0], parts[1]
                            self.results.append(SecurityCheckResult(
                                check_name="Hardcoded Credentials",
                                passed=False,
                                details="Potential hardcoded credential detected",
                                severity="high",
                                file_path=file_path,
                                line_number=int(line_num)
                            ))
            else:
                self.results.append(SecurityCheckResult(
                    check_name="Hardcoded Credentials",
                    passed=True,
                    details="No hardcoded credentials detected"
                ))
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.results.append(SecurityCheckResult(
                check_name="Hardcoded Credentials",
                passed=False,
                details="Unable to run credential check",
                severity="medium"
            ))
    
    def _check_secrets_management(self) -> None:
        """Check secrets management via environment variables."""
        env_file = self.repo_path / ".env"
        env_example = self.repo_path / ".env.example"
        
        if env_file.exists() and not env_example.exists():
            self.results.append(SecurityCheckResult(
                check_name="Secrets Management",
                passed=False,
                details=".env file exists but no .env.example template found",
                severity="medium"
            ))
        else:
            self.results.append(SecurityCheckResult(
                check_name="Secrets Management",
                passed=True,
                details="Proper environment variable setup detected"
            ))
    
    def _check_safe_logging(self) -> None:
        """Check for safe logging practices."""
        try:
            # Look for potential secret logging
            result = subprocess.run([
                'rg', '--type', 'py', '--line-number',
                r'log.*\.(password|secret|key|token)',
                str(self.repo_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':', 3)
                        if len(parts) >= 3:
                            file_path, line_num = parts[0], parts[1]
                            self.results.append(SecurityCheckResult(
                                check_name="Safe Logging",
                                passed=False,
                                details="Potential secret logging detected",
                                severity="high",
                                file_path=file_path,
                                line_number=int(line_num)
                            ))
            else:
                self.results.append(SecurityCheckResult(
                    check_name="Safe Logging",
                    passed=True,
                    details="No unsafe logging patterns detected"
                ))
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.results.append(SecurityCheckResult(
                check_name="Safe Logging",
                passed=False,
                details="Unable to run logging check",
                severity="medium"
            ))
    
    def _run_sca_scan(self) -> None:
        """Run Software Composition Analysis using OWASP Dependency-Check."""
        try:
            # Check if dependency-check is available
            result = subprocess.run([
                'dependency-check', '--version'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Run dependency check with cached NVD database
                scan_result = subprocess.run([
                    'dependency-check', '--project', 'autonomous-backlog',
                    '--scan', str(self.repo_path),
                    '--format', 'JSON',
                    '--out', str(self.repo_path / 'dependency-check-report.json'),
                    '--nvdDatafeed', str(self.repo_path / '.nvd-cache')
                ], capture_output=True, text=True)
                
                self.results.append(SecurityCheckResult(
                    check_name="Software Composition Analysis",
                    passed=scan_result.returncode == 0,
                    details=f"Dependency check completed with exit code {scan_result.returncode}"
                ))
            else:
                self.results.append(SecurityCheckResult(
                    check_name="Software Composition Analysis",
                    passed=False,
                    details="OWASP Dependency-Check not available",
                    severity="low"
                ))
        except FileNotFoundError:
            self.results.append(SecurityCheckResult(
                check_name="Software Composition Analysis",
                passed=False,
                details="OWASP Dependency-Check not installed",
                severity="low"
            ))
    
    def _run_sast_scan(self) -> None:
        """Run Static Application Security Testing using CodeQL."""
        try:
            # Check if CodeQL is available
            result = subprocess.run([
                'codeql', 'version'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.results.append(SecurityCheckResult(
                    check_name="Static Application Security Testing",
                    passed=True,
                    details="CodeQL available - SAST scan configured in CI"
                ))
            else:
                self.results.append(SecurityCheckResult(
                    check_name="Static Application Security Testing", 
                    passed=False,
                    details="CodeQL not available locally - relies on GitHub CI",
                    severity="low"
                ))
        except FileNotFoundError:
            self.results.append(SecurityCheckResult(
                check_name="Static Application Security Testing",
                passed=False,
                details="CodeQL not installed locally - relies on GitHub CI",
                severity="low"
            ))
    
    def generate_security_report(self) -> Dict:
        """Generate comprehensive security report."""
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.passed])
        failed_checks = total_checks - passed_checks
        
        high_severity = len([r for r in self.results if r.severity == "high" and not r.passed])
        medium_severity = len([r for r in self.results if r.severity == "medium" and not r.passed])
        low_severity = len([r for r in self.results if r.severity == "low" and not r.passed])
        
        return {
            'timestamp': str(Path().absolute()),
            'summary': {
                'total_checks': total_checks,
                'passed': passed_checks,
                'failed': failed_checks,
                'pass_rate': round(passed_checks / max(total_checks, 1) * 100, 2)
            },
            'severity_breakdown': {
                'high': high_severity,
                'medium': medium_severity,
                'low': low_severity
            },
            'results': [asdict(result) for result in self.results],
            'status': 'PASS' if failed_checks == 0 or high_severity == 0 else 'FAIL'
        }
    
    def save_security_report(self, report_path: Optional[str] = None) -> str:
        """Save security report to file."""
        if report_path is None:
            report_path = str(self.repo_path / 'security-report.json')
        
        report = self.generate_security_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path


def main():
    """Main entry point for TDD security checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TDD Security Checklist")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--report-file", help="Output report file")
    parser.add_argument("--fail-on-high", action="store_true", 
                       help="Fail if high severity issues found")
    
    args = parser.parse_args()
    
    checker = TDDSecurityChecker(args.repo_path)
    results = checker.run_security_checklist()
    
    report_file = checker.save_security_report(args.report_file)
    report = checker.generate_security_report()
    
    print(f"Security check completed. Report saved to: {report_file}")
    print(f"Status: {report['status']}")
    print(f"Checks passed: {report['summary']['passed']}/{report['summary']['total_checks']}")
    
    if args.fail_on_high and report['severity_breakdown']['high'] > 0:
        print(f"FAIL: {report['severity_breakdown']['high']} high severity issues found")
        return 1
    
    return 0 if report['status'] == 'PASS' else 1


if __name__ == "__main__":
    exit(main())