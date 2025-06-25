from pathlib import Path


def test_readme_includes_validation_usage_example():
    text = Path("README.md").read_text()
    assert "ValidationResult" in text


def test_contributing_mentions_validation_rules():
    contrib = Path("CONTRIBUTING.md")
    assert contrib.exists(), "CONTRIBUTING.md should exist"
    content = contrib.read_text().lower()
    assert "validation rules" in content
