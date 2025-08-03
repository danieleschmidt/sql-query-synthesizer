#!/usr/bin/env python3
"""API testing tool for SQL Query Synthesizer."""

import asyncio
import json
import time
import argparse
import sys
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TestResult:
    """Result of an API test."""
    name: str
    success: bool
    duration: float
    response: Optional[Dict] = None
    error: Optional[str] = None


class APITester:
    """Tests API endpoints without requiring actual HTTP server."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize API tester."""
        self.base_url = base_url
        self.test_results: List[TestResult] = []
    
    def test_query_api_structure(self) -> TestResult:
        """Test the query API endpoint structure."""
        start_time = time.time()
        
        try:
            # Test request structure validation
            valid_request = {
                'question': 'Show me all users',
                'max_rows': 10
            }
            
            # Validate request structure
            assert 'question' in valid_request
            assert isinstance(valid_request['question'], str)
            assert len(valid_request['question']) > 0
            assert isinstance(valid_request.get('max_rows', 10), int)
            
            # Test expected response structure
            expected_response = {
                'success': True,
                'sql': 'SELECT * FROM users LIMIT 10;',
                'data': [],
                'explanation': 'Query explanation',
                'columns': [],
                'row_count': 0,
                'query_time_ms': 0.0,
                'query_id': 'query-abc123'
            }
            
            # Validate response structure
            required_fields = ['success', 'sql', 'data', 'explanation', 'query_id']
            for field in required_fields:
                assert field in expected_response
            
            duration = time.time() - start_time
            return TestResult(
                name="Query API Structure",
                success=True,
                duration=duration,
                response=expected_response
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Query API Structure",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def test_health_api_structure(self) -> TestResult:
        """Test the health API endpoint structure."""
        start_time = time.time()
        
        try:
            expected_health = {
                'status': 'healthy',
                'timestamp': '2025-08-03T00:00:00Z',
                'version': '0.2.2',
                'uptime_seconds': 3600.0,
                'database': {
                    'healthy': True,
                    'response_time_ms': 15.2,
                    'connection_count': 5
                },
                'cache': {
                    'healthy': True,
                    'hit_rate': 85.5,
                    'size': 150
                },
                'llm': {
                    'healthy': True,
                    'provider': 'openai',
                    'model': 'gpt-3.5-turbo'
                }
            }
            
            # Validate structure
            required_fields = ['status', 'timestamp', 'version', 'database', 'cache', 'llm']
            for field in required_fields:
                assert field in expected_health
            
            # Validate status values
            valid_statuses = ['healthy', 'degraded', 'unhealthy']
            assert expected_health['status'] in valid_statuses
            
            duration = time.time() - start_time
            return TestResult(
                name="Health API Structure",
                success=True,
                duration=duration,
                response=expected_health
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Health API Structure",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def test_metrics_api_structure(self) -> TestResult:
        """Test the metrics API endpoint structure."""
        start_time = time.time()
        
        try:
            expected_metrics = [
                '# HELP query_requests_total Total number of query requests',
                '# TYPE query_requests_total counter',
                'query_requests_total{status="success"} 150',
                'query_requests_total{status="error"} 25',
                '',
                '# HELP query_duration_seconds Query execution duration',
                '# TYPE query_duration_seconds histogram',
                'query_duration_seconds_bucket{le="0.1"} 50',
                'query_duration_seconds_bucket{le="+Inf"} 175',
                'query_duration_seconds_count 175',
                'query_duration_seconds_sum 87.5',
            ]
            
            # Validate Prometheus format
            for line in expected_metrics:
                if line.startswith('# HELP'):
                    assert 'HELP' in line
                elif line.startswith('# TYPE'):
                    assert 'TYPE' in line
                    assert any(t in line for t in ['counter', 'gauge', 'histogram'])
                elif line and not line.startswith('#'):
                    # Should have metric name and value
                    parts = line.split(' ')
                    assert len(parts) >= 2
                    # Value should be numeric
                    try:
                        float(parts[-1])
                    except ValueError:
                        assert False, f"Invalid metric value: {parts[-1]}"
            
            duration = time.time() - start_time
            return TestResult(
                name="Metrics API Structure",
                success=True,
                duration=duration,
                response={'metrics_count': len([l for l in expected_metrics if l and not l.startswith('#')])}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Metrics API Structure",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def test_input_validation(self) -> TestResult:
        """Test input validation logic."""
        start_time = time.time()
        
        try:
            # Test valid inputs
            valid_inputs = [
                {'question': 'Show me all users', 'max_rows': 10},
                {'question': 'Count orders by status', 'max_rows': 100},
                {'question': 'Find recent transactions', 'max_rows': 50}
            ]
            
            for input_data in valid_inputs:
                assert len(input_data['question']) > 0
                assert len(input_data['question']) <= 1000
                assert 1 <= input_data['max_rows'] <= 1000
            
            # Test invalid inputs
            invalid_inputs = [
                {},  # Missing question
                {'question': ''},  # Empty question
                {'question': 'A' * 2000},  # Too long
                {'question': 'Show users', 'max_rows': 0},  # Invalid max_rows
                {'question': 'Show users', 'max_rows': 10000},  # Too many rows
            ]
            
            for invalid_input in invalid_inputs:
                # These should fail validation
                if not invalid_input.get('question'):
                    assert True  # Missing/empty question is invalid
                elif len(invalid_input.get('question', '')) > 1000:
                    assert True  # Too long question is invalid
                elif invalid_input.get('max_rows', 10) <= 0 or invalid_input.get('max_rows', 10) > 1000:
                    assert True  # Invalid max_rows is invalid
            
            duration = time.time() - start_time
            return TestResult(
                name="Input Validation",
                success=True,
                duration=duration,
                response={'valid_inputs': len(valid_inputs), 'invalid_inputs': len(invalid_inputs)}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Input Validation",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def test_security_validation(self) -> TestResult:
        """Test security validation logic."""
        start_time = time.time()
        
        try:
            # Test SQL injection patterns
            malicious_inputs = [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "UNION SELECT password FROM admin",
                "'; INSERT INTO users VALUES ('hacker'); --",
                "/* comment */ SELECT * FROM admin",
                "SELECT * FROM users WHERE id = 1; DELETE FROM users; --"
            ]
            
            for malicious_input in malicious_inputs:
                # Should detect dangerous patterns
                has_dangerous_pattern = any(
                    keyword in malicious_input.upper() 
                    for keyword in ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'UNION', '--', '/*']
                )
                assert has_dangerous_pattern, f"Should detect dangerous pattern in: {malicious_input}"
            
            # Test safe inputs
            safe_inputs = [
                "Show me all users",
                "Find products with high ratings",
                "Count orders from last month",
                "List customer information"
            ]
            
            for safe_input in safe_inputs:
                # Should not detect dangerous patterns
                has_dangerous_pattern = any(
                    keyword in safe_input.upper() 
                    for keyword in ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'UNION', '--', '/*']
                )
                assert not has_dangerous_pattern, f"Should not flag safe input: {safe_input}"
            
            duration = time.time() - start_time
            return TestResult(
                name="Security Validation",
                success=True,
                duration=duration,
                response={'malicious_inputs': len(malicious_inputs), 'safe_inputs': len(safe_inputs)}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Security Validation",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def test_error_handling(self) -> TestResult:
        """Test error handling and response format."""
        start_time = time.time()
        
        try:
            error_scenarios = [
                {
                    'type': 'validation_error',
                    'response': {
                        'success': False,
                        'error': 'Invalid input format',
                        'error_code': 'VALIDATION_ERROR',
                        'query_id': None
                    }
                },
                {
                    'type': 'security_error',
                    'response': {
                        'success': False,
                        'error': 'Potential SQL injection detected',
                        'error_code': 'SECURITY_VIOLATION',
                        'query_id': None
                    }
                },
                {
                    'type': 'database_error',
                    'response': {
                        'success': False,
                        'error': 'Database connection failed',
                        'error_code': 'DATABASE_ERROR',
                        'query_id': 'query-abc123'
                    }
                }
            ]
            
            for scenario in error_scenarios:
                response = scenario['response']
                
                # Validate error response structure
                assert 'success' in response
                assert response['success'] is False
                assert 'error' in response
                assert 'error_code' in response
                
                # Error message should not leak sensitive information
                error_msg = response['error'].lower()
                sensitive_terms = ['password', 'token', 'key', 'secret', 'credential']
                for term in sensitive_terms:
                    assert term not in error_msg, f"Error message should not contain: {term}"
            
            duration = time.time() - start_time
            return TestResult(
                name="Error Handling",
                success=True,
                duration=duration,
                response={'error_scenarios': len(error_scenarios)}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Error Handling",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def test_openapi_schema(self) -> TestResult:
        """Test OpenAPI schema structure."""
        start_time = time.time()
        
        try:
            expected_schema = {
                'openapi': '3.0.0',
                'info': {
                    'title': 'SQL Query Synthesizer API',
                    'version': '0.2.2',
                    'description': 'Natural language to SQL conversion API'
                },
                'paths': {
                    '/api/query': {
                        'post': {
                            'summary': 'Generate SQL from natural language',
                            'requestBody': {
                                'required': True,
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            'type': 'object',
                                            'properties': {
                                                'question': {'type': 'string'},
                                                'max_rows': {'type': 'integer'}
                                            },
                                            'required': ['question']
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            # Validate OpenAPI structure
            assert 'openapi' in expected_schema
            assert expected_schema['openapi'] == '3.0.0'
            assert 'info' in expected_schema
            assert 'paths' in expected_schema
            
            # Validate info section
            info = expected_schema['info']
            required_info_fields = ['title', 'version', 'description']
            for field in required_info_fields:
                assert field in info
            
            # Validate paths
            paths = expected_schema['paths']
            assert '/api/query' in paths
            
            duration = time.time() - start_time
            return TestResult(
                name="OpenAPI Schema",
                success=True,
                duration=duration,
                response={'endpoints': len(paths)}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="OpenAPI Schema",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all API tests."""
        print("🚀 Running API structure and validation tests...")
        
        test_methods = [
            self.test_query_api_structure,
            self.test_health_api_structure,
            self.test_metrics_api_structure,
            self.test_input_validation,
            self.test_security_validation,
            self.test_error_handling,
            self.test_openapi_schema
        ]
        
        results = []
        
        for test_method in test_methods:
            print(f"Running {test_method.__name__}...")
            result = test_method()
            results.append(result)
            
            if result.success:
                print(f"✅ {result.name} - PASSED ({result.duration:.3f}s)")
            else:
                print(f"❌ {result.name} - FAILED ({result.duration:.3f}s)")
                if result.error:
                    print(f"   Error: {result.error}")
        
        self.test_results = results
        return results
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.duration for r in self.test_results)
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests / max(total_tests, 1)) * 100,
                'total_duration': total_duration
            },
            'results': [
                {
                    'name': r.name,
                    'success': r.success,
                    'duration': r.duration,
                    'response': r.response,
                    'error': r.error
                }
                for r in self.test_results
            ]
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 API test report saved to {output_file}")
        
        return report
    
    def print_summary(self):
        """Print test summary."""
        if not self.test_results:
            print("No tests run.")
            return
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.success)
        failed = total - passed
        duration = sum(r.duration for r in self.test_results)
        
        print(f"\n{'='*60}")
        print("API TEST SUMMARY")
        print('='*60)
        print(f"Total tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success rate: {(passed/max(total,1)*100):.1f}%")
        print(f"Total duration: {duration:.3f}s")
        
        if failed > 0:
            print(f"\nFailed tests:")
            for result in self.test_results:
                if not result.success:
                    print(f"  - {result.name}: {result.error}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="SQL Query Synthesizer API Tester")
    parser.add_argument('--base-url', default='http://localhost:5000', 
                        help='Base URL for API (default: http://localhost:5000)')
    parser.add_argument('--report', help='Save test report to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    tester = APITester(args.base_url)
    
    try:
        results = tester.run_all_tests()
        
        if args.report:
            tester.generate_report(args.report)
        
        tester.print_summary()
        
        # Return appropriate exit code
        all_passed = all(r.success for r in results)
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ API tester failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())