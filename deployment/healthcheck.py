#!/usr/bin/env python3
"""Health check script for SQL Query Synthesizer production deployment."""

import json
import logging
import os
import sys
import time
from typing import Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [HEALTHCHECK] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health check for SQL Query Synthesizer."""

    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.timeout = 10
        self.checks_passed = 0
        self.checks_failed = 0

    def check_basic_connectivity(self) -> Tuple[bool, str]:
        """Check basic HTTP connectivity."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
                headers={'User-Agent': 'HealthChecker/1.0'}
            )

            if response.status_code == 200:
                return True, "Basic connectivity OK"
            else:
                return False, f"HTTP {response.status_code}: {response.text[:100]}"

        except requests.exceptions.ConnectionError:
            return False, "Connection refused - service not running"
        except requests.exceptions.Timeout:
            return False, f"Timeout after {self.timeout} seconds"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def check_application_health(self) -> Tuple[bool, str]:
        """Check detailed application health."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
                headers={'Accept': 'application/json'}
            )

            if response.status_code != 200:
                return False, f"Health endpoint returned {response.status_code}"

            try:
                health_data = response.json()
            except json.JSONDecodeError:
                return False, "Health endpoint returned invalid JSON"

            # Check overall health status
            overall_healthy = health_data.get('overall_healthy', False)
            if not overall_healthy:
                failed_services = []
                for service, status in health_data.items():
                    if isinstance(status, dict) and not status.get('healthy', True):
                        failed_services.append(service)
                return False, f"Unhealthy services: {', '.join(failed_services)}"

            return True, "Application health OK"

        except Exception as e:
            return False, f"Health check failed: {str(e)}"

    def check_database_connectivity(self) -> Tuple[bool, str]:
        """Check database connectivity through health endpoint."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False, "Cannot reach health endpoint"

            health_data = response.json()
            db_status = health_data.get('database', {})

            if not db_status.get('healthy', False):
                error_msg = db_status.get('error', 'Unknown database error')
                return False, f"Database unhealthy: {error_msg}"

            return True, "Database connectivity OK"

        except Exception as e:
            return False, f"Database check failed: {str(e)}"

    def check_cache_connectivity(self) -> Tuple[bool, str]:
        """Check cache connectivity through health endpoint."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False, "Cannot reach health endpoint"

            health_data = response.json()
            cache_status = health_data.get('cache', {})

            if not cache_status.get('healthy', False):
                error_msg = cache_status.get('error', 'Unknown cache error')
                return False, f"Cache unhealthy: {error_msg}"

            return True, "Cache connectivity OK"

        except Exception as e:
            return False, f"Cache check failed: {str(e)}"

    def check_api_functionality(self) -> Tuple[bool, str]:
        """Test basic API functionality."""
        try:
            # Test metrics endpoint
            response = requests.get(
                f"{self.base_url}/metrics",
                timeout=self.timeout,
                headers={'Accept': 'text/plain'}
            )

            if response.status_code == 200:
                # Check if response contains Prometheus metrics
                if 'sql_synthesizer_' in response.text:
                    return True, "API functionality OK"
                else:
                    return False, "Metrics endpoint not returning expected data"
            else:
                return False, f"Metrics endpoint returned {response.status_code}"

        except Exception as e:
            return False, f"API functionality check failed: {str(e)}"

    def check_openai_integration(self) -> Tuple[bool, str]:
        """Check OpenAI integration if configured."""
        try:
            # Only check if OpenAI is configured
            if not os.getenv('OPENAI_API_KEY'):
                return True, "OpenAI not configured (skipped)"

            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False, "Cannot reach health endpoint"

            health_data = response.json()
            openai_status = health_data.get('openai', {})

            if not openai_status.get('healthy', True):
                error_msg = openai_status.get('error', 'Unknown OpenAI error')
                return False, f"OpenAI integration unhealthy: {error_msg}"

            return True, "OpenAI integration OK"

        except Exception as e:
            return False, f"OpenAI integration check failed: {str(e)}"

    def check_performance_metrics(self) -> Tuple[bool, str]:
        """Check if performance metrics are being collected."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False, "Cannot reach health endpoint"

            health_data = response.json()

            # Check if system info is present
            system_info = health_data.get('system_info', {})
            if not system_info:
                return False, "System metrics not available"

            # Check for key metrics
            required_metrics = ['uptime', 'total_queries', 'cache_hit_rate']
            missing_metrics = []

            for metric in required_metrics:
                if metric not in system_info:
                    missing_metrics.append(metric)

            if missing_metrics:
                return False, f"Missing metrics: {', '.join(missing_metrics)}"

            return True, "Performance metrics OK"

        except Exception as e:
            return False, f"Performance metrics check failed: {str(e)}"

    def check_security_features(self) -> Tuple[bool, str]:
        """Check security features are enabled."""
        try:
            # Test that security headers are present
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False, "Cannot reach application"

            headers = response.headers

            # Check for basic security headers
            expected_headers = [
                'X-Content-Type-Options',
                'X-Frame-Options',
                'X-XSS-Protection'
            ]

            missing_headers = []
            for header in expected_headers:
                if header not in headers:
                    missing_headers.append(header)

            if missing_headers:
                return False, f"Missing security headers: {', '.join(missing_headers)}"

            # Check HSTS if enabled
            if os.getenv('QUERY_AGENT_ENABLE_HSTS') == 'true':
                if 'Strict-Transport-Security' not in headers:
                    return False, "HSTS enabled but header missing"

            return True, "Security features OK"

        except Exception as e:
            return False, f"Security features check failed: {str(e)}"

    def run_all_checks(self) -> bool:
        """Run all health checks and return overall status."""
        checks = [
            ("Basic Connectivity", self.check_basic_connectivity),
            ("Application Health", self.check_application_health),
            ("Database Connectivity", self.check_database_connectivity),
            ("Cache Connectivity", self.check_cache_connectivity),
            ("API Functionality", self.check_api_functionality),
            ("OpenAI Integration", self.check_openai_integration),
            ("Performance Metrics", self.check_performance_metrics),
            ("Security Features", self.check_security_features),
        ]

        logger.info("Starting comprehensive health check...")

        results = []
        for check_name, check_func in checks:
            try:
                success, message = check_func()
                if success:
                    logger.info(f"✅ {check_name}: {message}")
                    self.checks_passed += 1
                else:
                    logger.error(f"❌ {check_name}: {message}")
                    self.checks_failed += 1

                results.append({
                    'check': check_name,
                    'success': success,
                    'message': message
                })

            except Exception as e:
                logger.error(f"❌ {check_name}: Exception - {str(e)}")
                self.checks_failed += 1
                results.append({
                    'check': check_name,
                    'success': False,
                    'message': f"Exception: {str(e)}"
                })

        # Print summary
        total_checks = self.checks_passed + self.checks_failed
        logger.info(f"Health check summary: {self.checks_passed}/{total_checks} checks passed")

        # Return overall health status
        return self.checks_failed == 0

    def wait_for_readiness(self, max_wait: int = 120) -> bool:
        """Wait for the application to be ready."""
        logger.info(f"Waiting for application readiness (max {max_wait} seconds)...")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            success, message = self.check_basic_connectivity()
            if success:
                logger.info("Application is ready!")
                return True

            logger.info(f"Not ready yet: {message}")
            time.sleep(5)

        logger.error(f"Application did not become ready within {max_wait} seconds")
        return False


def main():
    """Main health check function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="SQL Query Synthesizer Health Check")
    parser.add_argument(
        '--wait', '-w',
        type=int,
        default=0,
        help='Wait for application to be ready (seconds)'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=10,
        help='HTTP timeout for checks (seconds)'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results in JSON format'
    )
    args = parser.parse_args()

    # Create health checker
    checker = HealthChecker()
    checker.timeout = args.timeout

    # Wait for readiness if requested
    if args.wait > 0:
        if not checker.wait_for_readiness(args.wait):
            sys.exit(1)

    # Run health checks
    healthy = checker.run_all_checks()

    # Output results
    if args.json:
        result = {
            'healthy': healthy,
            'checks_passed': checker.checks_passed,
            'checks_failed': checker.checks_failed,
            'timestamp': time.time()
        }
        print(json.dumps(result))

    # Exit with appropriate code
    if healthy:
        logger.info("🎉 All health checks passed!")
        sys.exit(0)
    else:
        logger.error("💥 Some health checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
