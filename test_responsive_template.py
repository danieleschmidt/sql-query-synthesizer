#!/usr/bin/env python3
"""Test responsive HTML template functionality."""

import re
import sys

sys.path.insert(0, '/root/repo')

def test_html_template_syntax():
    """Test that the HTML template has valid syntax."""
    try:
        with open('/root/repo/sql_synthesizer/templates/index.html') as f:
            content = f.read()

        # Check basic HTML structure
        assert '<!DOCTYPE html>' in content, "Missing DOCTYPE declaration"
        assert '<html lang="en">' in content, "Missing language attribute"
        assert '<meta name="viewport"' in content, "Missing viewport meta tag"
        print("✅ HTML template has valid basic structure")

        # Check responsive features
        assert '@media (max-width: 768px)' in content, "Missing tablet responsive breakpoint"
        assert '@media (max-width: 480px)' in content, "Missing mobile responsive breakpoint"
        assert 'flex-direction: column' in content, "Missing mobile layout changes"
        print("✅ HTML template has responsive CSS breakpoints")

        # Check accessibility features
        assert 'input:focus' in content, "Missing focus styles for accessibility"
        assert '@media (prefers-contrast: high)' in content, "Missing high contrast support"
        assert '@media (prefers-reduced-motion: reduce)' in content, "Missing reduced motion support"
        print("✅ HTML template has accessibility features")

        # Check modern CSS features
        assert 'box-sizing: border-box' in content, "Missing modern box model"
        assert 'input-group' in content, "Missing responsive form layout"
        assert 'font-size: 18px; /* Prevents zoom on iOS */' in content, "Missing iOS zoom prevention"
        print("✅ HTML template has modern CSS practices")

        return True

    except Exception as e:
        print(f"❌ HTML template test failed: {e}")
        return False

def test_template_structure():
    """Test template structure and classes."""
    try:
        with open('/root/repo/sql_synthesizer/templates/index.html') as f:
            content = f.read()

        # Check that form structure is updated
        assert 'form-container' in content, "Missing form container class"
        assert 'input-group' in content, "Missing input group class"
        assert 'error-block' in content, "Missing error block class"
        print("✅ HTML template has updated responsive structure")

        # Check that old inline styles are removed
        assert 'style="color: #e74c3c; background: #fdf2f2; border-left: 4px solid #e74c3c;"' not in content, "Old inline error styles still present"
        print("✅ HTML template removed old inline styles")

        # Check form input improvements
        assert 'autocomplete="off"' in content, "Missing autocomplete attribute"
        assert 'maxlength="1000"' in content, "Missing maxlength validation"
        print("✅ HTML template has improved form inputs")

        return True

    except Exception as e:
        print(f"❌ Template structure test failed: {e}")
        return False

def test_css_responsiveness():
    """Test CSS responsive design patterns."""
    try:
        with open('/root/repo/sql_synthesizer/templates/index.html') as f:
            content = f.read()

        # Check for flexible layouts
        assert 'flex: 1' in content, "Missing flexible input sizing"
        assert 'width: 100%' in content, "Missing full-width mobile inputs"
        print("✅ CSS has flexible layout patterns")

        # Check for touch-friendly sizing
        assert 'min-height: 44px' in content, "Missing touch-friendly button sizing"
        assert 'padding: 15px' in content, "Missing adequate touch padding"
        print("✅ CSS has touch-friendly sizing")

        # Check for proper font scaling
        mobile_font_sizes = re.findall(r'font-size: (\d+(?:\.\d+)?)(?:rem|px)', content)
        assert len(mobile_font_sizes) > 5, "Missing responsive font sizes"
        print("✅ CSS has responsive font sizing")

        return True

    except Exception as e:
        print(f"❌ CSS responsiveness test failed: {e}")
        return False

def test_security_features():
    """Test that security features are maintained."""
    try:
        with open('/root/repo/sql_synthesizer/templates/index.html') as f:
            content = f.read()

        # Check CSP header is still present
        assert 'Content-Security-Policy' in content, "Missing Content Security Policy"
        assert "default-src 'self'" in content, "Missing CSP default-src"
        print("✅ Security features maintained")

        # Check template escaping
        assert '{{ error|e }}' in content, "Missing error escaping"
        assert '{{ sql|e }}' in content, "Missing SQL escaping"
        assert '{{ data|e }}' in content, "Missing data escaping"
        print("✅ Template escaping maintained")

        return True

    except Exception as e:
        print(f"❌ Security features test failed: {e}")
        return False

def main():
    """Run all template tests."""
    print("🧪 Testing responsive HTML template...\n")

    tests = [
        ("HTML template syntax", test_html_template_syntax),
        ("Template structure", test_template_structure),
        ("CSS responsiveness", test_css_responsiveness),
        ("Security features", test_security_features)
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
        print("🎉 All responsive template tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
