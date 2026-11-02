"""Smoke test to verify test harness and imports."""
import sys


def test_python_version():
    """Verify supported Python runtime environment."""
    assert sys.version_info >= (3, 10), "Python version must be 3.10 or higher"


def test_environment_smoke():
    """Basic sanity check for test discovery."""
    assert True
