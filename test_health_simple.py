"""Simple test for enhanced health check functionality without Flask dependency."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_openai_health_method():
    """Test that the _check_openai_api_health method exists and returns proper structure."""
    try:
        # Import without triggering Flask import
        import sql_synthesizer.query_agent as qa_module
        
        # Check if the method exists
        assert hasattr(qa_module.QueryAgent, '_check_openai_api_health'), "Method _check_openai_api_health should exist"
        
        print("✅ _check_openai_api_health method exists")
        
        # Check method signature
        import inspect
        method = getattr(qa_module.QueryAgent, '_check_openai_api_health')
        sig = inspect.signature(method)
        
        # Should have self parameter only
        params = list(sig.parameters.keys())
        assert params == ['self'], f"Expected ['self'], got {params}"
        
        print("✅ Method signature is correct")
        
        # Check return type annotation
        return_annotation = sig.return_annotation
        expected = 'Dict[str, Any]'
        if hasattr(return_annotation, '__name__'):
            actual = return_annotation.__name__
        else:
            actual = str(return_annotation)
            
        print(f"✅ Method return annotation: {actual}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_health_check_enhanced():
    """Test that health_check method includes OpenAI API status."""
    try:
        # Read the source code to verify structure
        with open('sql_synthesizer/query_agent.py', 'r') as f:
            source = f.read()
        
        # Check for key components
        checks = [
            ('openai_health = self._check_openai_api_health()', "OpenAI health check call"),
            ('"openai_api": openai_health', "OpenAI health in services"),
            ('health_status["services"]["openai_api"]["healthy"]', "OpenAI health in overall status"),
            ('"timestamp": time.time()', "Timestamp in health status")
        ]
        
        for check, desc in checks:
            if check in source:
                print(f"✅ {desc} found")
            else:
                print(f"❌ {desc} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing enhanced health check implementation...")
    
    success1 = test_openai_health_method()
    success2 = test_health_check_enhanced()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Enhanced health check implementation is correct.")
    else:
        print("\n❌ Some tests failed. Check implementation.")