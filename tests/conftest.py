"""
Pytest configuration file for the Research Agent tests.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create test fixtures that can be reused across tests
@pytest.fixture
def test_config():
    """Return a test configuration dictionary."""
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7
        },
        "tools": {
            "search": {
                "enabled": True,
                "api_key_env": "SERPAPI_API_KEY",
                "engine": "google",
                "max_results": 5
            },
            "url_fetcher": {
                "enabled": True,
                "timeout": 10,
                "max_size": 524288  # 512KB
            },
            "pdf_reader": {
                "enabled": True,
                "max_pages": 10
            },
            "data_analysis": {
                "enabled": True,
                "packages": ["pandas", "numpy"]
            },
            "visualization": {
                "enabled": True,
                "max_width": 80,
                "use_unicode": True
            },
            "database": {
                "enabled": True,
                "path": "tests/data/test_research_data.db",
                "cache_enabled": True,
                "cache_ttl": 3600  # 1 hour
            }
        },
        "planning": {
            "max_steps": 5,
            "step_timeout": 60,
            "research_depth": "shallow"
        },
        "language": {
            "primary": "en",
            "supported": ["en", "uk"],
            "auto_detect": True
        }
    }

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test_serpapi_key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test_brave_key")

@pytest.fixture
def test_data_dir():
    """Create and return a test data directory."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir
