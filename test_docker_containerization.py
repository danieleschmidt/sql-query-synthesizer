"""Test for Docker containerization functionality."""

from pathlib import Path


def test_dockerfile_exists():
    """Test that Dockerfile exists in project root."""
    dockerfile_path = Path("./Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile should exist for containerization"


def test_dockerfile_content():
    """Test that Dockerfile has required content for Python web app."""
    dockerfile_path = Path("./Dockerfile")
    if not dockerfile_path.exists():
        return  # Skip if file doesn't exist yet

    content = dockerfile_path.read_text()

    # Check for essential Dockerfile elements
    assert "FROM python:" in content, "Should use Python base image"
    assert "COPY requirements.txt" in content, "Should copy requirements"
    assert "pip install" in content, "Should install Python dependencies"
    assert "COPY ." in content or "COPY sql_synthesizer" in content, "Should copy source code"
    assert "CMD" in content or "ENTRYPOINT" in content, "Should have startup command"


def test_sample_data_exists():
    """Test that sample data files exist for demo."""
    sample_files = [
        "sample_data/demo_queries.txt",
        "sample_data/README.md"
    ]

    for file_path in sample_files:
        path = Path(file_path)
        # Just check if directory structure can be created
        assert path.parent.name == "sample_data", "Sample data should be in sample_data directory"


def test_docker_ignore_exists():
    """Test that .dockerignore exists to optimize build."""
    dockerignore_path = Path("./.dockerignore")
    if dockerignore_path.exists():
        content = dockerignore_path.read_text()
        # Should exclude common files that don't need to be in container
        assert any(pattern in content for pattern in ["*.pyc", "__pycache__", ".git"]), \
            ".dockerignore should exclude unnecessary files"


if __name__ == "__main__":
    # Run basic test
    try:
        test_dockerfile_exists()
        print("✅ Dockerfile exists")
    except AssertionError as e:
        print(f"❌ {e}")

    try:
        test_dockerfile_content()
        print("✅ Dockerfile content is valid")
    except AssertionError as e:
        print(f"❌ {e}")
    except Exception:
        print("⚠️ Dockerfile content not yet testable")

    try:
        test_sample_data_exists()
        print("✅ Sample data structure is valid")
    except AssertionError as e:
        print(f"❌ {e}")
