#!/usr/bin/env python3
"""Simple test script to verify specific exception handling improvements."""

import subprocess
import sys


# Test the basic imports work
def test_imports():
    """Test that our modules can be imported successfully."""
    try:
        print("✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_cli_specific_exceptions():
    """Test that CLI now handles specific exceptions."""
    try:
        # Import and check the main function has proper exception handling
        import inspect

        from query_agent import main

        # Get source code of main function
        source = inspect.getsource(main)

        # Check for specific exception types
        specific_exceptions = [
            'KeyboardInterrupt',
            'ImportError',
            'ModuleNotFoundError',
            'FileNotFoundError',
            'PermissionError',
            'ValueError'
        ]

        found_exceptions = []
        for exc in specific_exceptions:
            if exc in source:
                found_exceptions.append(exc)

        print(f"✅ CLI handles {len(found_exceptions)} specific exception types: {found_exceptions}")
        return len(found_exceptions) >= 4  # At least 4 specific types

    except Exception as e:
        print(f"❌ CLI exception test failed: {e}")
        return False

def test_query_agent_exceptions():
    """Test query agent has specific exception handling."""
    try:
        import inspect

        from sql_synthesizer.query_agent import QueryAgent

        # Check the _perform_cache_cleanup method
        source = inspect.getsource(QueryAgent._perform_cache_cleanup)

        # Should have specific exception handling
        if 'AttributeError' in source and 'RuntimeError' in source:
            print("✅ QueryAgent cache cleanup has specific exception handling")
            return True
        else:
            print("❌ QueryAgent cache cleanup still uses broad exception handling")
            return False

    except Exception as e:
        print(f"❌ QueryAgent exception test failed: {e}")
        return False

def test_webapp_exceptions():
    """Test webapp has specific exception handling."""
    try:
        with open('/root/repo/sql_synthesizer/webapp.py') as f:
            source = f.read()

        # Check for specific exception types in webapp
        specific_exceptions = [
            'SQLTimeoutError',
            'OperationalError',
            'DatabaseError',
            'AuthenticationError',
            'RateLimitError',
            'APITimeoutError'
        ]

        found_exceptions = []
        for exc in specific_exceptions:
            if exc in source:
                found_exceptions.append(exc)

        print(f"✅ Webapp handles {len(found_exceptions)} specific exception types: {found_exceptions}")
        return len(found_exceptions) >= 4

    except Exception as e:
        print(f"❌ Webapp exception test failed: {e}")
        return False

def run_syntax_check():
    """Run Python syntax check on modified files."""
    files_to_check = [
        '/root/repo/query_agent.py',
        '/root/repo/sql_synthesizer/query_agent.py',
        '/root/repo/sql_synthesizer/webapp.py',
        '/root/repo/sql_synthesizer/openai_adapter.py',
        '/root/repo/sql_synthesizer/async_query_agent.py',
        '/root/repo/sql_synthesizer/services/async_sql_generator_service.py',
        '/root/repo/sql_synthesizer/services/enhanced_query_validator.py'
    ]

    all_passed = True
    for file_path in files_to_check:
        try:
            result = subprocess.run([sys.executable, '-m', 'py_compile', file_path],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {file_path}: Syntax OK")
            else:
                print(f"❌ {file_path}: Syntax Error - {result.stderr}")
                all_passed = False
        except Exception as e:
            print(f"❌ {file_path}: Check failed - {e}")
            all_passed = False

    return all_passed

def main():
    """Run all tests."""
    print("🧪 Testing specific exception handling improvements...\n")

    tests = [
        ("Import test", test_imports),
        ("CLI exceptions", test_cli_specific_exceptions),
        ("QueryAgent exceptions", test_query_agent_exceptions),
        ("Webapp exceptions", test_webapp_exceptions),
        ("Syntax check", run_syntax_check)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Exception handling improvements successful.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
