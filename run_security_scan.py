#!/usr/bin/env python3
"""Security scan runner for the SQL Query Synthesizer project."""

import os
import subprocess
import sys
import json
from pathlib import Path


def run_bandit_scan():
    """Run Bandit security scan."""
    print("🔍 Running Bandit security scan...")
    
    try:
        result = subprocess.run([
            'python', '-m', 'bandit', '-r', 'sql_synthesizer/', 
            '-f', 'json', '-o', 'bandit-report.json'
        ], capture_output=True, text=True, cwd='/root/repo')
        
        if result.returncode == 0:
            print("✅ Bandit scan completed successfully - No security issues found")
        else:
            print(f"⚠️  Bandit scan found potential security issues")
            if os.path.exists('/root/repo/bandit-report.json'):
                with open('/root/repo/bandit-report.json', 'r') as f:
                    report = json.load(f)
                    print(f"   Issues found: {len(report.get('results', []))}")
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("❌ Bandit not installed - installing...")
        subprocess.run(['pip', 'install', 'bandit[toml]'], check=True)
        return run_bandit_scan()
    except Exception as e:
        print(f"❌ Bandit scan failed: {e}")
        return False


def run_safety_check():
    """Run Safety vulnerability check."""
    print("🔍 Running Safety vulnerability check...")
    
    try:
        result = subprocess.run([
            'python', '-m', 'safety', 'check', '--json'
        ], capture_output=True, text=True, cwd='/root/repo')
        
        if result.returncode == 0:
            print("✅ Safety check completed - No known vulnerabilities in dependencies")
            return True
        else:
            print("⚠️  Safety check found potential vulnerabilities:")
            if result.stdout:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    for vuln in vulnerabilities:
                        print(f"   - {vuln.get('package', 'Unknown')}: {vuln.get('advisory', 'No details')}")
                except json.JSONDecodeError:
                    print(f"   Raw output: {result.stdout}")
            return False
            
    except FileNotFoundError:
        print("❌ Safety not installed - installing...")
        subprocess.run(['pip', 'install', 'safety'], check=True)
        return run_safety_check()
    except Exception as e:
        print(f"❌ Safety check failed: {e}")
        return False


def check_secrets():
    """Check for potential secrets in code."""
    print("🔍 Checking for potential secrets...")
    
    patterns_to_check = [
        (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'][^"\']{3,}["\']', 'Potential password in code'),
        (r'(?i)(api_?key|secret|token)\s*[=:]\s*["\'][^"\']{10,}["\']', 'Potential API key/secret'),
        (r'(?i)(private_?key)\s*[=:]\s*["\'][^"\']{10,}["\']', 'Potential private key'),
        (r'["\'][A-Za-z0-9]{32,}["\']', 'Potential hardcoded token/hash'),
    ]
    
    issues_found = []
    
    for root, dirs, files in os.walk('/root/repo/sql_synthesizer'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache']]
        
        for file in files:
            if not file.endswith(('.py', '.yaml', '.yml', '.json')):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern, description in patterns_to_check:
                        import re
                        if re.search(pattern, line):
                            # Skip test files and comments
                            if 'test' in file.lower() or line.strip().startswith('#'):
                                continue
                            
                            issues_found.append({
                                'file': file_path,
                                'line': i,
                                'description': description,
                                'content': line.strip()[:100] + '...' if len(line.strip()) > 100 else line.strip()
                            })
                            
            except Exception as e:
                print(f"   Warning: Could not scan {file_path}: {e}")
    
    if issues_found:
        print(f"⚠️  Found {len(issues_found)} potential secret issues:")
        for issue in issues_found[:5]:  # Show first 5
            print(f"   - {issue['file']}:{issue['line']}: {issue['description']}")
        return False
    else:
        print("✅ No potential secrets found in code")
        return True


def check_sql_injection_patterns():
    """Check for potential SQL injection vulnerabilities."""
    print("🔍 Checking for SQL injection vulnerabilities...")
    
    dangerous_patterns = [
        (r'["\'].*\+.*["\'].*sql|sql.*\+.*["\']', 'String concatenation in SQL'),
        (r'f["\'].*\{.*\}.*["\'].*(?:SELECT|INSERT|UPDATE|DELETE)', 'F-string formatting in SQL'),
        (r'%.*%.*(?:SELECT|INSERT|UPDATE|DELETE)', 'String formatting in SQL'),
        (r'\.format\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)', 'String format() in SQL'),
    ]
    
    issues_found = []
    
    for root, dirs, files in os.walk('/root/repo/sql_synthesizer'):
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache']]
        
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern, description in dangerous_patterns:
                        import re
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip comments and test files
                            if line.strip().startswith('#') or 'test' in file.lower():
                                continue
                            
                            issues_found.append({
                                'file': file_path,
                                'line': i,
                                'description': description,
                                'content': line.strip()
                            })
                            
            except Exception as e:
                print(f"   Warning: Could not scan {file_path}: {e}")
    
    if issues_found:
        print(f"⚠️  Found {len(issues_found)} potential SQL injection issues:")
        for issue in issues_found[:3]:  # Show first 3
            print(f"   - {issue['file']}:{issue['line']}: {issue['description']}")
        return False
    else:
        print("✅ No obvious SQL injection patterns found")
        return True


def run_comprehensive_security_scan():
    """Run comprehensive security scan."""
    print("🚀 Starting comprehensive security scan...")
    print("=" * 60)
    
    results = {
        'bandit': run_bandit_scan(),
        'safety': run_safety_check(),
        'secrets': check_secrets(),
        'sql_injection': check_sql_injection_patterns(),
    }
    
    print("=" * 60)
    print("📊 Security Scan Results:")
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {check.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("🏆 All security checks passed!")
        return True
    else:
        print(f"⚠️  {total - passed} security checks failed - please review and fix")
        return False


if __name__ == "__main__":
    success = run_comprehensive_security_scan()
    sys.exit(0 if success else 1)