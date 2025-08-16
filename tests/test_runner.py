#!/usr/bin/env python3
"""Comprehensive test runner for SQL Query Synthesizer."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


class TestRunner:
    """Manages and executes different types of tests with reporting."""

    def __init__(self, project_root: Path):
        """Initialize test runner."""
        self.project_root = project_root
        self.test_results = {}

    def run_unit_tests(self, verbose: bool = False, coverage: bool = False) -> bool:
        """Run unit tests."""
        print("🧪 Running unit tests...")

        cmd = ["python3", "-m", "pytest"]

        # Add test directories
        cmd.extend(
            [
                "tests/test_core.py",
                "tests/test_utils.py",
                "tests/test_database_layer.py",
                "tests/unit/",
            ]
        )

        # Add options
        if verbose:
            cmd.append("-v")

        if coverage:
            cmd.extend(
                [
                    "--cov=sql_synthesizer",
                    "--cov-report=term-missing",
                    "--cov-report=html:htmlcov",
                ]
            )

        cmd.extend(["--tb=short", "--strict-markers", "--disable-warnings"])

        return self._run_command(cmd, "unit_tests")

    def run_integration_tests(self, verbose: bool = False) -> bool:
        """Run integration tests."""
        print("🔗 Running integration tests...")

        cmd = ["python3", "-m", "pytest"]
        cmd.extend(["tests/integration/", "-m", "integration"])

        if verbose:
            cmd.append("-v")

        cmd.extend(["--tb=short", "--disable-warnings"])

        return self._run_command(cmd, "integration_tests")

    def run_security_tests(self, verbose: bool = False) -> bool:
        """Run security-focused tests."""
        print("🔒 Running security tests...")

        cmd = ["python3", "-m", "pytest"]
        cmd.extend(
            [
                "tests/test_security.py",
                "tests/test_security_audit.py",
                "tests/test_enhanced_sql_injection.py",
                "tests/test_webapp_security.py",
                "-m",
                "security",
            ]
        )

        if verbose:
            cmd.append("-v")

        cmd.extend(["--tb=short", "--disable-warnings"])

        return self._run_command(cmd, "security_tests")

    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance tests."""
        print("⚡ Running performance tests...")

        cmd = ["python3", "-m", "pytest"]
        cmd.extend(
            [
                "tests/test_cache_metrics.py",
                "tests/test_comprehensive_metrics.py",
                "-m",
                "performance",
            ]
        )

        if verbose:
            cmd.append("-v")

        cmd.extend(["--tb=short", "--disable-warnings"])

        return self._run_command(cmd, "performance_tests")

    def run_code_quality_checks(self) -> bool:
        """Run code quality checks."""
        print("📏 Running code quality checks...")

        success = True

        # Run linting with ruff
        print("  → Running ruff linter...")
        lint_success = self._run_command(
            ["python3", "-m", "ruff", "check", "sql_synthesizer/", "tests/"],
            "ruff_lint",
        )
        success = success and lint_success

        # Run formatting check with black
        print("  → Checking code formatting...")
        format_success = self._run_command(
            ["python3", "-m", "black", "sql_synthesizer/", "tests/", "--check"],
            "black_format",
        )
        success = success and format_success

        # Run import sorting check
        print("  → Checking import sorting...")
        isort_success = self._run_command(
            ["python3", "-m", "isort", "sql_synthesizer/", "tests/", "--check-only"],
            "isort_check",
        )
        success = success and isort_success

        return success

    def run_security_scan(self) -> bool:
        """Run security scanning with bandit."""
        print("🛡️  Running security scan...")

        cmd = [
            "python3",
            "-m",
            "bandit",
            "-r",
            "sql_synthesizer/",
            "-f",
            "json",
            "-o",
            "security-report.json",
            "--skip",
            "B101,B601",  # Skip assert and shell injection for tests
        ]

        return self._run_command(cmd, "security_scan")

    def run_type_checking(self) -> bool:
        """Run type checking with mypy."""
        print("🏷️  Running type checking...")

        cmd = [
            "python3",
            "-m",
            "mypy",
            "sql_synthesizer/",
            "--ignore-missing-imports",
            "--no-strict-optional",
        ]

        return self._run_command(cmd, "type_checking")

    def run_all_tests(self, verbose: bool = False, coverage: bool = False) -> bool:
        """Run all test suites."""
        print("🚀 Running complete test suite...")

        test_suites = [
            ("Unit Tests", lambda: self.run_unit_tests(verbose, coverage)),
            ("Integration Tests", lambda: self.run_integration_tests(verbose)),
            ("Security Tests", lambda: self.run_security_tests(verbose)),
            ("Performance Tests", lambda: self.run_performance_tests(verbose)),
            ("Code Quality", self.run_code_quality_checks),
            ("Security Scan", self.run_security_scan),
            ("Type Checking", self.run_type_checking),
        ]

        all_passed = True

        for suite_name, test_func in test_suites:
            print(f"\n{'='*60}")
            print(f"Running {suite_name}")
            print("=" * 60)

            try:
                suite_passed = test_func()
                all_passed = all_passed and suite_passed

                status = "✅ PASSED" if suite_passed else "❌ FAILED"
                print(f"{suite_name}: {status}")

            except Exception as e:
                print(f"❌ {suite_name} failed with error: {e}")
                all_passed = False

        return all_passed

    def _run_command(self, cmd: List[str], test_name: str) -> bool:
        """Run a command and capture results."""
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            duration = time.time() - start_time

            self.test_results[test_name] = {
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }

            if result.returncode != 0:
                print(f"❌ {test_name} failed (exit code {result.returncode})")
                if result.stderr:
                    print(f"Error output:\n{result.stderr}")
                return False

            print(f"✅ {test_name} passed ({duration:.2f}s)")
            return True

        except subprocess.TimeoutExpired:
            print(f"❌ {test_name} timed out after 5 minutes")
            return False
        except FileNotFoundError:
            print(f"❌ {test_name} - command not found: {cmd[0]}")
            return False
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            return False

    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate a test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(
            1 for result in self.test_results.values() if result["success"]
        )
        failed_tests = total_tests - passed_tests
        total_duration = sum(
            result["duration"] for result in self.test_results.values()
        )

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / max(total_tests, 1)) * 100,
                "total_duration": total_duration,
            },
            "results": self.test_results,
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"📊 Test report saved to {output_file}")

        return report

    def print_summary(self):
        """Print test summary."""
        if not self.test_results:
            print("No tests run.")
            return

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r["success"])
        failed = total - passed
        duration = sum(r["duration"] for r in self.test_results.values())

        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total test suites: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success rate: {(passed/max(total,1)*100):.1f}%")
        print(f"Total duration: {duration:.2f}s")

        if failed > 0:
            print("\nFailed test suites:")
            for name, result in self.test_results.items():
                if not result["success"]:
                    print(f"  - {name}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="SQL Query Synthesizer Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--security", action="store_true", help="Run security tests only"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests only"
    )
    parser.add_argument(
        "--quality", action="store_true", help="Run code quality checks only"
    )
    parser.add_argument("--scan", action="store_true", help="Run security scan only")
    parser.add_argument("--types", action="store_true", help="Run type checking only")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("--report", help="Save test report to file")

    args = parser.parse_args()

    # Default to running all tests if no specific test type is specified
    if not any(
        [
            args.unit,
            args.integration,
            args.security,
            args.performance,
            args.quality,
            args.scan,
            args.types,
        ]
    ):
        args.all = True

    project_root = Path(__file__).parent.parent
    runner = TestRunner(project_root)

    success = True

    try:
        if args.all:
            success = runner.run_all_tests(args.verbose, args.coverage)
        else:
            if args.unit:
                success = success and runner.run_unit_tests(args.verbose, args.coverage)
            if args.integration:
                success = success and runner.run_integration_tests(args.verbose)
            if args.security:
                success = success and runner.run_security_tests(args.verbose)
            if args.performance:
                success = success and runner.run_performance_tests(args.verbose)
            if args.quality:
                success = success and runner.run_code_quality_checks()
            if args.scan:
                success = success and runner.run_security_scan()
            if args.types:
                success = success and runner.run_type_checking()

        # Generate report
        if args.report:
            runner.generate_report(args.report)

        # Print summary
        runner.print_summary()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
