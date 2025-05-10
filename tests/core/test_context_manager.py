"""
Unit tests for the ContextManager class.
"""

from unittest.mock import patch, mock_open
from datetime import datetime, timedelta

from core.context_manager import ContextManager


class TestContextManager:
    """Test cases for the ContextManager class."""

    def test_initialization(self):
        """Test that the context manager initializes with default values."""
        context_manager = ContextManager()
        
        # Check that the context has the expected structure
        assert "metadata" in context_manager.context
        assert "research" in context_manager.context
        assert "sources" in context_manager.context
        assert "cache" in context_manager.context
        assert "session_history" in context_manager.context
        
        # Check that metadata has created_at and updated_at
        assert "created_at" in context_manager.context["metadata"]
        assert "updated_at" in context_manager.context["metadata"]
    
    def test_add_to_context(self):
        """Test adding values to the context."""
        context_manager = ContextManager()
        
        # Add a simple value
        context_manager.add_to_context("test_key", "test_value")
        assert context_manager.context["test_key"] == "test_value"
        
        # Add a nested value using dot notation
        context_manager.add_to_context("nested.key", "nested_value")
        assert context_manager.context["nested"]["key"] == "nested_value"
        
        # Add to an existing nested structure
        context_manager.add_to_context("nested.another_key", "another_value")
        assert context_manager.context["nested"]["another_key"] == "another_value"
        
        # Check that metadata updated_at was updated
        assert context_manager.context["metadata"]["updated_at"] != context_manager.context["metadata"]["created_at"]
    
    def test_get_from_context(self):
        """Test retrieving values from the context."""
        context_manager = ContextManager()
        
        # Add values to retrieve
        context_manager.add_to_context("test_key", "test_value")
        context_manager.add_to_context("nested.key", "nested_value")
        
        # Retrieve simple value
        assert context_manager.get_from_context("test_key") == "test_value"
        
        # Retrieve nested value
        assert context_manager.get_from_context("nested.key") == "nested_value"
        
        # Retrieve non-existent value with default
        assert context_manager.get_from_context("non_existent", "default") == "default"
        
        # Retrieve non-existent nested value with default
        assert context_manager.get_from_context("nested.non_existent", "default") == "default"
    
    def test_add_source(self):
        """Test adding a source to the context."""
        context_manager = ContextManager()
        
        # Add a source
        source = {
            "url": "https://example.com",
            "title": "Example Source",
            "type": "website"
        }
        
        context_manager.add_source(source)
        
        # Check that the source was added to the sources list
        assert len(context_manager.context["sources"]) == 1
        assert context_manager.context["sources"][0]["url"] == "https://example.com"
        assert context_manager.context["sources"][0]["title"] == "Example Source"
        
        # Check that timestamp and id were added
        assert "timestamp" in context_manager.context["sources"][0]
        assert "id" in context_manager.context["sources"][0]
        
        # Test adding a source with missing required fields
        incomplete_source = {
            "title": "Incomplete Source"
        }
        
        # Should not add the incomplete source
        context_manager.add_source(incomplete_source)
        assert len(context_manager.context["sources"]) == 1  # Still only one source
    
    def test_get_sources(self):
        """Test retrieving sources from the context."""
        context_manager = ContextManager()
        
        # Add sources
        source1 = {
            "url": "https://example1.com",
            "title": "Example Source 1",
            "type": "website"
        }
        
        source2 = {
            "url": "https://example2.com",
            "title": "Example Source 2",
            "type": "pdf"
        }
        
        context_manager.add_source(source1)
        context_manager.add_source(source2)
        
        # Get all sources
        sources = context_manager.get_sources()
        assert len(sources) == 2
        
        # Get sources filtered by type
        website_sources = context_manager.get_sources({"type": "website"})
        assert len(website_sources) == 1
        assert website_sources[0]["url"] == "https://example1.com"
        
        pdf_sources = context_manager.get_sources({"type": "pdf"})
        assert len(pdf_sources) == 1
        assert pdf_sources[0]["url"] == "https://example2.com"
    
    def test_cache_operations(self):
        """Test adding and retrieving values from the cache."""
        context_manager = ContextManager()
        
        # Add a value to the cache with no TTL
        context_manager.add_to_cache("cache_key", "cache_value")
        assert context_manager.get_from_cache("cache_key") == "cache_value"
        
        # Add a value with TTL
        context_manager.add_to_cache("expiring_key", "expiring_value", ttl=1)  # 1 second TTL
        assert context_manager.get_from_cache("expiring_key") == "expiring_value"
        
        # Simulate expiration by modifying the expires_at directly
        cache_entry = context_manager.context["cache"]["expiring_key"]
        expired_time = (datetime.now() - timedelta(seconds=10)).isoformat()
        cache_entry["expires_at"] = expired_time
        
        # Should return default value for expired entry
        assert context_manager.get_from_cache("expiring_key", "default") == "default"
        
        # Check that the expired entry was removed
        assert "expiring_key" not in context_manager.context["cache"]
    
    def test_clear_context(self):
        """Test clearing the context."""
        context_manager = ContextManager()
        
        # Add some data to the context
        context_manager.add_to_context("test_key", "test_value")
        context_manager.add_source({
            "url": "https://example.com",
            "title": "Example Source",
            "type": "website"
        })
        context_manager.add_to_cache("cache_key", "cache_value")
        
        # Clear the context
        context_manager.clear_context()
        
        # Check that the context was reset to default structure
        assert "metadata" in context_manager.context
        assert "research" in context_manager.context
        assert "sources" in context_manager.context
        assert "cache" in context_manager.context
        assert "session_history" in context_manager.context
        
        # Check that the data was cleared
        assert "test_key" not in context_manager.context
        assert len(context_manager.context["sources"]) == 0
        assert len(context_manager.context["cache"]) == 0
        
        # Check that metadata was preserved
        assert "created_at" in context_manager.context["metadata"]
        assert "updated_at" in context_manager.context["metadata"]
    
    @patch("builtins.open", new_callable=mock_open, read_data='{"metadata": {"test": "value"}}')
    @patch("pathlib.Path.exists")
    def test_load_context(self, mock_exists, mock_file):
        """Test loading context from a file."""
        mock_exists.return_value = True
        
        # Initialize with a context file
        context_manager = ContextManager("test_context.json")
        
        # Check that the file was opened (Path object is converted to string internally)
        mock_file.assert_called_once()
        # Get the first positional argument of the call
        call_args = mock_file.call_args[0]
        assert str(call_args[0]) == "test_context.json"
        assert call_args[1] == 'r'
        assert mock_file.call_args[1]['encoding'] == 'utf-8'
        
        # Check that the loaded value is in the context
        assert "test" in context_manager.context["metadata"]
        assert context_manager.context["metadata"]["test"] == "value"
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.parent")
    def test_save_context(self, mock_parent, mock_file):
        """Test saving context to a file."""
        # Mock the parent directory
        mock_parent.mkdir.return_value = None
        
        # Initialize with a context file
        context_manager = ContextManager("test_context.json")
        
        # Add some data to the context
        context_manager.add_to_context("test_key", "test_value")
        
        # Check that the file was opened for writing (Path object is converted to string internally)
        mock_file.assert_called()
        # Get the first positional argument of the call
        call_args = mock_file.call_args[0]
        assert str(call_args[0]) == "test_context.json"
        assert call_args[1] == 'w'
        assert mock_file.call_args[1]['encoding'] == 'utf-8'
        
        # Check that json.dump was called with the context
        handle = mock_file()
        assert handle.write.called
